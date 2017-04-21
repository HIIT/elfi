import logging
from math import ceil
from operator import mul
from functools import reduce, partial
from toolz.functoolz import compose

import numpy as np

import elfi.client
from elfi.utils import args_to_tuple
from elfi.store import OutputPool
from elfi.bo.gpy_regression import GPyRegression
from elfi.bo.acquisition import LCBSC
from elfi.bo.utils import stochastic_optimization
from elfi.methods.utils import GMDistribution, weighted_var
from elfi.methods.posteriors import BolfiPosterior
from elfi.model.elfi_model import NodeReference, Operation, ElfiModel

logger = logging.getLogger(__name__)

__all__ = ['Rejection', 'SMC', 'BayesianOptimization', 'BOLFI']


"""

Implementing a new inference method
-----------------------------------

You can implement your own algorithm by subclassing the `InferenceMethod` class. The
methods that must be implemented raise `NotImplementedError`. In addition, you will
probably also want to override the `__init__` method. It can be useful to read through
`Rejection`, `SMC` and/or `BayesianOptimization` class implementations below to get you
going. The reason for the imposed structure in `InferenceMethod` is to encourage a design
where one can advance the inference iteratively, that is, to stop at any point, check the
current state and to be able to continue. This makes it possible to effectively tune the
inference as there are usually many moving parts, such as summary statistic choices or
deciding the best discrepancy function.

ELFI operates through batches. A batch is an indexed collection of one or more successive
outputs from the generative model (`ElfiModel`). The rule of thumb is that it should take
a significant amount of time to compute a batch. This ensures that it is worthwhile to
send a batch over the network to a remote worker to be computed. A batch also needs to fit
into memory.

ELFI guarantees that computing a batch with the same index will always produce the same
output given the same model and `ComputationContext` object. The `ComputationContext`
object holds the batch size, seed for the PRNG, and a pool of precomputed batches of nodes
and the observed values of the nodes.

When a new `InferenceMethod` is constructed, it will make a copy of the user provided
`ElfiModel` and make a new `ComputationContext` object for it. The user's model will stay
intact and the algorithm is free to modify it's copy as it needs to.


### Implementing the `__init__` method

You will need to call the `InferenceMethod.__init__` with a list of outputs, e.g. names of
nodes that you need the data for in each batch. For example, the rejection algorithm needs
the parameters and the discrepancy node output.

The first parameter to your `__init__` can be either the ElfiModel object or directly a
"target" node, e.g. discrepancy in rejection sampling. Assuming your `__init__` takes an
optional discrepancy parameter, you can detect which one was passed by using
`_resolve_model` method:

```
def __init__(model, discrepancy, ...):
    model, discrepancy = self._resolve_model(model, discrepancy)
```

In case you need multiple target nodes, you will need to write your own resolver.


### Explanations for some members of `InferenceMethod`

- objective : dict
    Holds the data for the algorithm to internally determine how many batches are still
    needed. You must have a key `n_batches` here. This information is used to determine
    when the algorithm is finished.

- state : dict
    Stores any temporal data related to achieving the objective. Must include a key
    `n_batches` for determining when the inference is finished.


### Good to know

#### `BatchHandler`

`InferenceMethod` class instantiates a `elfi.client.BatchHandler` helper class for you and
assigns it to `self.batches`. This object is in essence a wrapper to the `Client`
interface making it easier to work with batches that are in computation. Some of the
duties of `BatchHandler` is to keep track of the current batch_index and of the status of
the batches that have been submitted. You may however may not need to interact with it
directly.

#### `OutputPool`

`elfi.store.OutputPool` serves a dual purpose:
1. It stores the computed outputs of selected nodes
2. It provides those outputs when a batch is recomputed saving the need to recompute them.

If you want to provide values for outputs of certain nodes from outside the generative
model, you can return then in `prepare_new_batch` method. They will be inserted into to
the `OutputPool` and will replace any value that would have otherwise been generated from
the node. This is used e.g. in `BOLFI` where values from the acquisition function replace
values coming from the prior in the Bayesian optimization phase.

"""

# TODO: use only either n_batches or n_sim in state dict
# TODO: plan how continuing the inference is standardized


class InferenceMethod(object):
    """
    """

    def __init__(self, model, outputs, batch_size=1000, seed=None, pool=None,
                 max_parallel_batches=None):
        """Construct the inference algorithm object.

        If you are implementing your own algorithm do not forget to call `super`.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        outputs : list
            Contains the node names for which the algorithm needs to receive the outputs
            in every batch.
        batch_size : int
        seed : int
            Seed for the data generation from the ElfiModel
        pool : OutputPool
            OutputPool both stores and provides precomputed values for batches.
        max_parallel_batches : int
            Maximum number of batches allowed to be in computation at the same time.
            Defaults to number of cores in the client


        """
        model = model.model if isinstance(model, NodeReference) else model

        if not model.parameters:
            raise ValueError('Model {} defines no parameters'.format(model))

        self.model = model.copy()
        self.outputs = outputs
        self.batch_size = batch_size

        # Prepare the computation_context
        context = model.computation_context.copy()
        if seed is not None:
            context.seed = seed
        context.batch_size = self.batch_size
        context.pool = pool

        self.model.computation_context = context

        self.client = elfi.client.get()
        self.batches = elfi.client.BatchHandler(self.model, outputs=outputs, client=self.client)

        self.max_parallel_batches = max_parallel_batches or self.client.num_cores

        # State and objective should contain all information needed to continue the
        # inference after an iteration.
        self.state = dict(n_sim=0, n_batches=0)
        self.objective = dict(n_batches=0)

    @property
    def pool(self):
        return self.model.computation_context.pool

    @property
    def seed(self):
        return self.model.computation_context.seed

    @property
    def parameters(self):
        return self.model.parameters

    def init_inference(self, *args, **kwargs):
        """This method is called when one wants to begin the inference. Set `self.state`
        and `self.objective` here for the inference.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def extract_result(self):
        """This method is called when one wants to receive the result from the inference.
        You should prepare the output here and return it.

        Returns
        -------
        result : dict
        """
        raise NotImplementedError

    def update(self, batch, batch_index):
        """ELFI calls this method when a new batch has been computed and the state of
        the inference should be updated with it.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        Returns
        -------
        None
        """
        raise NotImplementedError

    def prepare_new_batch(self, batch_index):
        """ELFI calls this method before submitting a new batch with an increasing index
        `batch_index`. This is an optional method to override. Use this if you have a need
        do do preparations, e.g. in Bayesian optimization algorithm, the next acquisition
        points would be acquired here.

        If you need provide values for certain nodes, you can do so by constructing a
        batch dictionary and returning it. See e.g. BayesianOptimization for an example.

        Parameters
        ----------
        batch_index : int
            next batch_index to be submitted

        Returns
        -------
        batch : dict or None

        """
        pass

    def infer(self, *args, **kwargs):
        """Init the inference and start the iterate loop until the inference is finished.

        Returns
        -------
        result : dict
        """

        self.init_inference(*args, **kwargs)

        while not self.finished:
            self.iterate()

        self.batches.cancel_pending()
        return self.extract_result()

    def iterate(self):
        """Forward the inference one iteration. One iteration consists of processing the
        the result of the next batch in succession.

        If the next batch is ready, it will be processed immediately and no new batches
        are submitted.

        If the next batch is not ready, new batches will be submitted up to the
        `_n_total_batches` or `max_parallel_batches` or until the next batch is complete.

        If there are no more submissions to do and the next batch has still not finished,
        the method will wait for it's result.

        Returns
        -------
        None

        """

        # Submit new batches if allowed
        while self._allow_submit:
            batch_index = self.batches.next_index
            batch = self.prepare_new_batch(batch_index)
            self.batches.submit(batch)

        # Handle the next batch in succession
        batch, batch_index = self.batches.wait_next()
        self.update(batch, batch_index)

    @property
    def finished(self):
        return self.objective['n_batches'] <= self.state['n_batches']

    @property
    def _allow_submit(self):
        return self.max_parallel_batches > self.batches.num_pending and \
               self._has_batches_to_submit and \
               (not self.batches.has_ready)

    @property
    def _has_batches_to_submit(self):
        return self.objective['n_batches'] > self.state['n_batches'] + self.batches.num_pending

    def _to_array(self, batch, outputs=None):
        """Helper method to turn batches into numpy array
        
        Parameters
        ----------
        batch : dict or list
           Batch or list of batches
        outputs : list, optional
           Name of outputs to include in the array. Default is the `self.outputs`

        Returns
        -------
        np.array
            2d, where columns are batch outputs
        
        """

        if not batch:
            return []
        if not isinstance(batch, list):
            batch = [batch]
        outputs = outputs or self.outputs

        rows = []
        for batch_ in batch:
            rows.append(np.column_stack([batch_[output] for output in outputs]))

        return np.vstack(rows)

    @staticmethod
    def _resolve_model(model, target, default_reference_class=NodeReference):
        # TODO: extract the default_reference_class from the model?

        if isinstance(model, ElfiModel) and target is None:
            raise NotImplementedError("Please specify the target node of the inference method")

        if isinstance(model, NodeReference):
            target = model
            model = target.model

        if isinstance(target, str):
            target = model[target]

        if not isinstance(target, default_reference_class):
            raise ValueError('Unknown target node class')

        return model, target.name

    @staticmethod
    def _ensure_outputs(outputs, required_outputs):
        outputs = outputs or []
        for out in required_outputs:
            if out not in outputs:
                outputs.append(out)
        return outputs


class Sampler(InferenceMethod):
    def sample(self, n_samples, *args, **kwargs):
        """
        Parameters
        ----------
        n_samples : int
            Number of samples to generate from the (approximate) posterior

        Returns
        -------
        result : dict
            A dictionary with at least the following items:
            samples : dict
                Dictionary of samples from the posterior distribution for each parameter.
        """

        return self.infer(n_samples, *args, **kwargs)


class Rejection(Sampler):
    """Parallel ABC rejection sampler.

    For a description of the rejection sampler and a general introduction to ABC, see e.g.
    Lintusaari et al. 2016.

    References
    ----------
    Lintusaari J, Gutmann M U, Dutta R, Kaski S, Corander J (2016). Fundamentals and
    Recent Developments in Approximate Bayesian Computation. Systematic Biology.
    http://dx.doi.org/10.1093/sysbio/syw077.
    """

    def __init__(self, model, discrepancy=None, outputs=None, **kwargs):
        """

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy : str or NodeReference
            Only needed if model is an ElfiModel
        kwargs:
            See InferenceMethod
        """

        model, self.discrepancy = self._resolve_model(model, discrepancy)
        outputs = self._ensure_outputs(outputs, model.parameters + [self.discrepancy])
        super(Rejection, self).__init__(model, outputs, **kwargs)

    def init_inference(self, n_samples, threshold=None, quantile=None, n_sim=None):
        """

        Parameters
        ----------
        n_samples : int
            number of samples to generate
        threshold : float
            Acceptance threshold
        quantile : float
            In between (0,1). Define the threshold as the p-quantile of all the
            simulations. n_sim = n_samples/quantile.
        n_sim : int
            Total number of simulations. The threshold will be the n_samples smallest
            distance among n_sim simulations.

        Returns
        -------

        """
        if quantile is None and threshold is None and n_sim is None:
            quantile = .01
        self.state = dict(samples=None, threshold=np.Inf, n_sim=0, accept_rate=1,
                          n_batches=0)

        if quantile: n_sim = ceil(n_samples/quantile)

        # Set initial n_batches estimate
        if n_sim:
            n_batches = ceil(n_sim/self.batch_size)
        else:
            n_batches = self.max_parallel_batches

        self.objective = dict(n_samples=n_samples, threshold=threshold,
                              n_batches=n_batches)

        # Reset the inference
        self.batches.reset()

    def update(self, batch, batch_index):
        if self.state['samples'] is None:
            # Lazy initialization of the outputs dict
            self._init_samples_lazy(batch)
        self._merge_batch(batch)
        self._update_state_meta()
        self._update_objective()

    def extract_result(self):
        """Extracts the result from the current state"""
        if self.state['samples'] is None:
            raise ValueError('Nothing to extract')

        # Take out the correct number of samples
        samples = dict()
        for k, v in self.state['samples'].items():
            samples[k] = v[:self.objective['n_samples']]

        result = self.state.copy()
        result['samples'] = samples
        result['n_samples'] = self.objective['n_samples']
        return result

    def _init_samples_lazy(self, batch):
        # Initialize the outputs dict based on the received batch
        samples = {}
        for node in self.outputs:
            shape = (self.objective['n_samples'] + self.batch_size,) + batch[node].shape[1:]
            samples[node] = np.ones(shape) * np.inf
        self.state['samples'] = samples

    def _merge_batch(self, batch):
        # TODO: add index vector so that you can recover the original order, also useful
        #       for async

        samples = self.state['samples']

        # Put the acquired samples to the end
        for node, v in samples.items():
            v[self.objective['n_samples']:] = batch[node]

        # Sort the smallest to the beginning
        sort_mask = np.argsort(samples[self.discrepancy], axis=0).ravel()
        for k, v in samples.items():
            v[:] = v[sort_mask]

    def _update_state_meta(self):
        """Updates n_sim, threshold, and accept_rate
        """
        o = self.objective
        s = self.state
        s['n_batches'] += 1
        s['n_sim'] += self.batch_size
        s['threshold'] = s['samples'][self.discrepancy][o['n_samples'] - 1]
        s['accept_rate'] = min(1, o['n_samples']/s['n_sim'])

    def _update_objective(self):
        """Updates the objective n_batches if applicable"""
        if not self.objective.get('threshold'): return

        s = self.state
        t, n_samples = [self.objective.get(k) for k in ('threshold', 'n_samples')]

        # noinspection PyTypeChecker
        n_acceptable = np.sum(s['samples'][self.discrepancy] <= t) if s['samples'] else 0
        if n_acceptable == 0: return

        accept_rate_t = n_acceptable / s['n_sim']
        # Add some margin to estimated batches_total. One could use confidence bounds here
        margin = .2 * self.batch_size * int(n_acceptable < n_samples)
        n_batches = (n_samples / accept_rate_t + margin) / self.batch_size

        self.objective['n_batches'] = ceil(n_batches)
        logger.debug('Estimated objective n_batches=%d' % self.objective['n_batches'])


class SMC(Sampler):
    def __init__(self, model, discrepancy=None, outputs=None, **kwargs):
        model, self.discrepancy = self._resolve_model(model, discrepancy)
        outputs = self._ensure_outputs(outputs, model.parameters + [self.discrepancy])
        model, added_nodes = self._augment_model(model)

        super(SMC, self).__init__(model, outputs + added_nodes, **kwargs)

        self.state['round'] = 0
        self._populations = []
        self._rejection = None

    def init_inference(self, n_samples, thresholds):
        self.objective.update(dict(n_samples=n_samples,
                                   n_batches=self.max_parallel_batches,
                                   round=len(thresholds) - 1,
                                   thresholds=thresholds))
        self._new_round()

    def extract_result(self):
        # TODO: make a better result object
        pop = self._extract_population()
        result = self.state.copy()
        result['populations'] = self._populations.copy()
        result['populations'].append(pop)
        result['samples'] = pop['samples']
        return result

    def update(self, batch, batch_index):
        self._rejection.update(batch, batch_index)

        if self._rejection.finished:
            self.batches.reset(self.state['n_batches'] + 1)
            if self.state['round'] < self.objective['round']:
                self._populations.append(self._extract_population())
                self.state['round'] += 1
                self._new_round()

        self._update_state()
        self._update_objective()

    def prepare_new_batch(self, batch_index):
        # Use the actual prior
        if self.state['round'] == 0:
            return

        # Sample from the proposal
        params = GMDistribution.rvs(*self._gm_params, size=self.batch_size)
        # TODO: support vector parameter nodes
        batch = {p:params[:,i] for i, p in enumerate(self.parameters)}
        return batch

    def _new_round(self):
        dashes = '-'*16
        logger.info('%s Starting round %d %s' % (dashes, self.state['round'], dashes))

        self._rejection = Rejection(self.model,
                                    discrepancy=self.discrepancy,
                                    outputs=self.outputs,
                                    batch_size=self.batch_size,
                                    seed=self.seed,
                                    max_parallel_batches=self.max_parallel_batches)

        self._rejection.init_inference(self.objective['n_samples'],
                                       threshold=self.current_population_threshold)

    def _extract_population(self):
        pop = self._rejection.extract_result()
        w, cov = self._compute_weights_and_cov(pop)
        pop['samples']['weights'] = w
        pop['cov'] = cov
        return pop

    def _compute_weights_and_cov(self, pop):
        samples = pop['samples']
        params = np.column_stack(tuple([samples[p] for p in self.parameters]))

        if self._populations:
            q_densities = GMDistribution.pdf(params, *self._gm_params)
            w = samples['_prior_pdf'] / q_densities
        else:
            w = np.ones(pop['n_samples'])

        # New covariance
        cov = 2 * np.diag(weighted_var(params, w))
        return w, cov

    def _update_state(self):
        """Updates n_sim, threshold, and accept_rate
        """
        s = self.state
        s['n_batches'] += 1
        s['n_sim'] += self.batch_size
        # TODO: use overall estimates
        s['threshold'] = self._rejection.state['threshold']
        s['accept_rate'] = self._rejection.state['accept_rate']

    def _update_objective(self):
        """Updates the objective n_batches"""
        n_batches = sum([pop['n_batches'] for pop in self._populations])
        self.objective['n_batches'] = n_batches + self._rejection.objective['n_batches']

    @staticmethod
    def _augment_model(model):
        # Add nodes to the model for computing the prior density
        model = model.copy()
        pdfs = []
        for p in model.parameters:
            param = model[p]
            pdfs.append(Operation(param.distribution.pdf, *([param] + param.parents),
                                  model=model))
        # Multiply the individual pdfs
        Operation(compose(partial(reduce, mul), args_to_tuple), *pdfs, model=model,
                  name='_prior_pdf')
        return model, ['_prior_pdf']

    @property
    def _gm_params(self):
        pop_ = self._populations[-1]
        params_ = np.column_stack(tuple([pop_['samples'][p] for p in self.parameters]))
        return params_, pop_['cov'], pop_['samples']['weights']

    @property
    def current_population_threshold(self):
        return self.objective['thresholds'][self.state['round']]


class BayesianOptimization(InferenceMethod):
    """Bayesian Optimization of an unknown target function."""

    def __init__(self, model, target=None, outputs=None, batch_size=1, n_acq=150,
                 initial_evidence=10, update_interval=10, bounds=None, target_model=None,
                 acquisition_method=None, **kwargs):
        """
        Parameters
        ----------
        model : ElfiModel or NodeReference
        target : str or NodeReference
            Only needed if model is an ElfiModel
        target_model : GPyRegression, optional
        acquisition_method : Acquisition, optional
        bounds : list
            The region where to estimate the posterior for each parameter in
            model.parameters.
            `[(lower, upper), ... ]`
        initial_evidence : int, dict
            Number of initial evidence or a precomputed batch dict containing parameter 
            and discrepancy values
        n_evidence : int
            The total number of evidence to acquire for the target_model regression
        update_interval : int
            How often to update the GP hyperparameters of the target_model
        exploration_rate : float
            Exploration rate of the acquisition method
        """

        model, self.target = self._resolve_model(model, target)
        outputs = self._ensure_outputs(outputs, model.parameters + [self.target])
        super(BayesianOptimization, self).\
            __init__(model, outputs=outputs, batch_size=batch_size, **kwargs)

        target_model = \
            target_model or GPyRegression(len(self.model.parameters), bounds=bounds)

        if not isinstance(initial_evidence, int):
            # Add precomputed batch data
            params = self._to_array(initial_evidence, self.parameters)
            target_model.update(params, initial_evidence[self.target])
            initial_evidence = len(params)

        # TODO: check if this can be removed
        if initial_evidence % self.batch_size != 0:
            raise ValueError('Initial evidence must be divisible by the batch size')

        self.acquisition_method = acquisition_method or LCBSC(target_model)

        # TODO: move some of these to objective
        self.target_model = target_model
        self.n_initial_evidence = initial_evidence
        self.n_acq = n_acq
        self.update_interval = update_interval

    def init_inference(self, n_acq=None):
        """You can continue BO by giving a larger n_acq"""
        self.state['last_update'] = self.state.get('last_update') or 0

        if n_acq and self.n_acq > n_acq:
            raise ValueError('New n_acq must be greater than the earlier')

        self.n_acq = n_acq or self.n_acq
        self.objective['n_batches'] = \
            ceil((self.n_acq + self.n_initial_evidence) / self.batch_size)

    def extract_result(self):
        param, min_value = stochastic_optimization(self.target_model.predict_mean,
                                                   self.target_model.bounds)

        param_hat = {}
        for i, p in enumerate(self.model.parameters):
            # Preserve as array
            param_hat[p] = param[i]

        return dict(samples=param_hat)

    def update(self, batch, batch_index):
        """Update the GP regression model of the target node.
        """
        params = self._to_array(batch, self.parameters)
        self._report_batch(batch_index, params, batch[self.target])

        optimize = self._should_optimize()
        self.target_model.update(params, batch[self.target], optimize)

        if optimize:
            self.state['last_update'] = self.target_model.n_evidence

        self.state['n_batches'] += 1

    def prepare_new_batch(self, batch_index):
        if self._n_submitted_evidence < self.n_initial_evidence:
            return

        pending_params = self._get_pending_params()
        t = self.batches.total - int(self.n_initial_evidence/self.batch_size)
        new_param = self.acquisition_method.acquire(self.batch_size, pending_params, t)

        # Add the next evaluation location to the pool
        # TODO: make to_batch method
        batch = {p: new_param[:,i:i+1] for i, p in enumerate(self.parameters)}
        return batch

    # TODO: use state dict
    @property
    def _n_submitted_evidence(self):
        return self.batches.total*self.batch_size

    @property
    def _allow_submit(self):
        # Do not start acquisition unless all of the initial evidence is ready
        prevent = self._n_submitted_evidence >= self.n_initial_evidence and \
                  self.target_model.n_evidence < self.n_initial_evidence
        return super(BayesianOptimization, self)._allow_submit and not prevent

    def _should_optimize(self):
        current = self.target_model.n_evidence + self.batch_size
        next_update = self.state['last_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update

    def _get_pending_params(self):
        # Prepare pending locations for the acquisition
        pending_batches = [self.pool.get_batch(i, self.parameters) for i in
                           self.batches.pending_indices]
        return self._to_array(pending_batches, self.parameters)

    def _report_batch(self, batch_index, params, distances):
        str = "Received batch {}:\n".format(batch_index)
        fill = 6 * ' '
        for i in range(self.batch_size):
            str += "{}{} at {}\n".format(fill, distances[i].item(), params[i])
        logger.debug(str)


class BOLFI(InferenceMethod):
    """Bayesian Optimization for Likelihood-Free Inference (BOLFI).

    Approximates the discrepancy function by a stochastic regression model.
    Discrepancy model is fit by sampling the discrepancy function at points decided by
    the acquisition function.

    The implementation follows that of Gutmann & Corander, 2016.

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

    """

    def __init__(self, model, batch_size=1, discrepancy=None, bounds=None, **kwargs):
        """
        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy : str or NodeReference
            Only needed if model is an ElfiModel
        discrepancy_model : GPRegression, optional
        acquisition_method : Acquisition, optional
        bounds : dict
            The region where to estimate the posterior for each parameter;
            `dict(param0: (lower, upper), param2: ... )`
        initial_evidence : int, dict
            Number of initial evidence or a precomputed dict containing parameter and
            discrepancy values
        n_evidence : int
            The total number of evidence to acquire for the discrepancy_model regression
        update_interval : int
            How often to update the GP hyperparameters of the discrepancy_model
        exploration_rate : float
            Exploration rate of the acquisition method
        """

    def get_posterior(self, threshold):
        """Returns the posterior.

        Parameters
        ----------
        threshold: float
            discrepancy threshold for creating the posterior

        Returns
        -------
        BolfiPosterior object
        """
        return BolfiPosterior(self.discrepancy_model, threshold)

