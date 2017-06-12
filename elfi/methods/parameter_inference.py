import logging
from collections import OrderedDict
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

import elfi.client
import elfi.visualization.visualization as vis
import elfi.visualization.interactive as visin
import elfi.methods.mcmc as mcmc
import elfi.model.augmenter as augmenter

from elfi.loader import get_sub_seed
from elfi.methods.bo.acquisition import LCBSC
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.utils import stochastic_optimization
from elfi.methods.results import Result, ResultSMC, ResultBOLFI
from elfi.methods.posteriors import BolfiPosterior
from elfi.methods.utils import GMDistribution, weighted_var, ModelPrior
from elfi.model.elfi_model import ComputationContext, NodeReference, ElfiModel

logger = logging.getLogger(__name__)

__all__ = ['Rejection', 'SMC', 'BayesianOptimization', 'BOLFI']


# TODO: use only either n_batches or n_sim in state dict
# TODO: plan how continuing the inference is standardized
class ParameterInference:
    """A base class for parameter inference methods.
    """

    def __init__(self, model, outputs, batch_size=1000, seed=None, pool=None,
                 max_parallel_batches=None):
        """Construct the inference algorithm object.

        If you are implementing your own algorithm do not forget to call `super`.

        Parameters
        ----------
        model : ElfiModel
            Model to perform the inference with.
        outputs : list
            Contains the node names for which the algorithm needs to receive the outputs
            in every batch.
        batch_size : int
        seed : int, optional
            Seed for the data generation from the ElfiModel
        pool : OutputPool
            OutputPool both stores and provides precomputed values for batches.
        max_parallel_batches : int
            Maximum number of batches allowed to be in computation at the same time.
            Defaults to number of cores in the client


        """

        if not model.parameters:
            raise ValueError('Model {} defines no parameters'.format(model))

        self.model = model.copy()
        self.outputs = outputs

        # Prepare the computation_context
        context = ComputationContext(
            batch_size=batch_size,
            seed=seed,
            observed=model.computation_context.observed,
            pool=pool
        )
        self.model.computation_context = context
        self.client = elfi.client.get_client()
        self.batches = elfi.client.BatchHandler(self.model, outputs=outputs, client=self.client)
        self.max_parallel_batches = max_parallel_batches or self.client.num_cores

        if self.max_parallel_batches <= 0:
            msg = 'Value for max_parallel_batches ({}) must be at least one.'.format(
                self.max_parallel_batches)
            if self.client.num_cores == 0:
                msg += ' Client has currently no workers available. Please make sure ' \
                       'the cluster has fully started or set the max_parallel_batches ' \
                       'parameter by hand.'
            raise ValueError(msg)

        # State and objective should contain all information needed to continue the
        # inference after an iteration.
        self.state = dict(n_sim=0, n_batches=0)
        self.objective = dict(n_batches=0)

    @property
    def pool(self):
        """Return the output pool of the inference."""
        return self.model.computation_context.pool

    @property
    def seed(self):
        """Return the seed of the inference."""
        return self.model.computation_context.seed

    @property
    def parameters(self):
        """Return the parameters to be inferred."""
        return self.model.parameters

    @property
    def batch_size(self):
        """Return the current batch_size."""
        return self.model.computation_context.batch_size

    def set_objective(self, *args, **kwargs):
        """Set the objective of the inference.

        This method sets the objective of the inference (values typically stored in the
        `self.objective` dict).

        Returns
        -------
        None
        """
        raise NotImplementedError

    def extract_result(self):
        """Prepare the result from the current state of the inference.

        ELFI calls this method in the end of the inference to return the result.

        Returns
        -------
        result : elfi.methods.result.Result
        """
        raise NotImplementedError

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        ELFI calls this method when a new batch has been computed and the state of
        the inference should be updated with it. It is also possible to bypass ELFI and
        call this directly to update the inference.

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
        logger.info('Received batch %d' % batch_index)
        self.state['n_batches'] += 1

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch

        ELFI calls this method before submitting a new batch with an increasing index
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
            Keys should match to node names in the model. These values will override any
            default values or operations in those nodes.

        """
        pass

    def _init_model(self, model):
        """Initialize the model.

        If your algorithm needs to modify the model, you may do so here. ELFI will call
        this method before compiling the model.

        Parameters
        ----------
        model : elfi.ElfiModel
            A copy of the original model.

        """
        return model

    def plot_state(self, **kwargs):
        """Plot the current state of the algorithm.

        Parameters
        ----------
        axes : matplotlib.axes.Axes (optional)
        figure : matplotlib.figure.Figure (optional)
        xlim
            x-axis limits
        ylim
            y-axis limits
        interactive : bool (default False)
            If true, uses IPython.display to update the cell figure
        close
            Close figure in the end of plotting. Used in the end of interactive mode.

        Returns
        -------
        None

        """
        raise NotImplementedError

    def infer(self, *args, vis=None, **kwargs):
        """Set the objective and start the iterate loop until the inference is finished.

        Returns
        -------
        result : Result
        """
        vis_opt = vis if isinstance(vis, dict) else {}

        self.set_objective(*args, **kwargs)

        while not self.finished:
            self.iterate()
            if vis:
                self.plot_state(interactive=True, **vis_opt)

        self.batches.cancel_pending()
        if vis:
            self.plot_state(close=True, **vis_opt)
        return self.extract_result()

    def iterate(self):
        """Forward the inference one iteration.

        This is a way to manually progress the inference. One iteration consists of
        waiting and processing the result of the next batch in succession and possibly
        submitting new batches.

        Notes
        -----
        If the next batch is ready, it will be processed immediately and no new batches
        are submitted.

        New batches are submitted only while waiting for the next one to complete. There
        will never be more batches submitted in parallel than the `max_parallel_batches`
        setting allows.

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

    def _to_array(self, batches, outputs=None):
        """Helper method to turn batches into numpy array
        
        Parameters
        ----------
        batches : list or dict
           A list of batches or as single batch
        outputs : list, optional
           Name of outputs to include in the array. Default is the `self.outputs`

        Returns
        -------
        np.array
            2d, where columns are batch outputs
        
        """

        if not batches:
            return []
        if not isinstance(batches, list):
            batches = [batches]
        outputs = outputs or self.outputs

        rows = []
        for batch_ in batches:
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
    def _compose_outputs(*output_args):
        outputs = []
        for arg in output_args:
            if arg is None:
                continue

            if isinstance(arg, str):
                arg = [arg]
            elif not isinstance(arg, (list, tuple)):
                raise ValueError('Unknown output argument type: {}'.format(arg))

            for output in arg:
                if output not in outputs:
                    outputs.append(output)

        return outputs


class Sampler(ParameterInference):
    def sample(self, n_samples, *args, **kwargs):
        """
        Parameters
        ----------
        n_samples : int
            Number of samples to generate from the (approximate) posterior

        Returns
        -------
        result : Result
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
        outputs = self._compose_outputs(outputs, model.parameters, self.discrepancy)
        super(Rejection, self).__init__(model, outputs, **kwargs)

    def set_objective(self, n_samples, threshold=None, quantile=None, n_sim=None):
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
            discrepancy among n_sim simulations.

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
        """Extracts the result from the current state

        Returns
        -------
        result : Result
        """
        if self.state['samples'] is None:
            raise ValueError('Nothing to extract')

        # Take out the correct number of samples
        n_samples = self.objective['n_samples']
        outputs = dict()
        for k, v in self.state['samples'].items():
            outputs[k] = v[:n_samples]

        result = Result(method_name=self.__class__.__name__,
                        outputs=outputs,
                        parameter_names=self.parameters,
                        discrepancy_name=self.discrepancy,
                        threshold=self.state['threshold'],
                        n_sim=self.state['n_sim'],
                        accept_rate=self.state['accept_rate'],
                        seed=self.seed
                        )

        return result

    def _init_samples_lazy(self, batch):
        # Initialize the outputs dict based on the received batch
        samples = {}
        for node in self.outputs:
            # Check the requested outputs
            try:
                if len(batch[node]) != self.batch_size:
                    raise ValueError("Node {} output length was {}. It should be equal "
                                     "to the batch size {}.".format(node,
                                                                    len(batch[node]),
                                                                    self.batch_size))
            except TypeError:
                raise ValueError("Node {} output has no length. It should be equal to "
                                 "the batch size {}.".format(node, self.batch_size))
            except KeyError:
                raise KeyError("Did not receive outputs for node {}".format(node))

            # Prepare samples
            shape = (self.objective['n_samples'] + self.batch_size,) + batch[node].shape[1:]
            # FIXME: add the correct dtype from batch. The inf initialization only for the
            #        distance
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
        s['threshold'] = s['samples'][self.discrepancy][o['n_samples'] - 1].item()
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

    def plot_state(self, **options):
        displays = []
        if options.get('interactive'):
            from IPython import display
            displays.append(display.HTML(
                    '<span>Threshold: {}</span>'.format(self.state['threshold'])))

        visin.plot_sample(self.state['samples'], nodes=self.parameters,
                    n=self.objective['n_samples'], displays=displays, **options)


class SMC(Sampler):
    """Sequential Monte Carlo ABC sampler"""
    def __init__(self, model, discrepancy=None, outputs=None, **kwargs):
        model, discrepancy = self._resolve_model(model, discrepancy)

        # Add the prior pdf nodes to the model
        model = model.copy()
        pdf = augmenter.add_pdf_nodes(model)[0]
        outputs = self._compose_outputs(outputs, model.parameters, discrepancy, pdf)

        super(SMC, self).__init__(model, outputs, **kwargs)

        self.discrepancy = discrepancy
        self.prior_pdf = pdf
        self.state['round'] = 0
        self._populations = []
        self._rejection = None

    def set_objective(self, n_samples, thresholds):
        self.objective.update(dict(n_samples=n_samples,
                                   n_batches=self.max_parallel_batches,
                                   round=len(thresholds) - 1,
                                   thresholds=thresholds))
        self._new_round()

    def extract_result(self):
        pop = self._extract_population()

        result = ResultSMC(method_name="SMC-ABC",
                           outputs=pop.outputs,
                           parameter_names=self.parameters,
                           discrepancy_name=self.discrepancy,
                           threshold=self.state['threshold'],
                           n_sim=self.state['n_sim'],
                           accept_rate=self.state['accept_rate'],
                           seed=self.seed,
                           populations=self._populations.copy() + [pop]
                           )

        return result

    def update(self, batch, batch_index):
        self._rejection.update(batch, batch_index)

        if self._rejection.finished:
            self.batches.cancel_pending()
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

        self._rejection.set_objective(self.objective['n_samples'],
                                      threshold=self.current_population_threshold)

    def _extract_population(self):
        pop = self._rejection.extract_result()
        pop.method_name = "Rejection within SMC-ABC"
        w, cov = self._compute_weights_and_cov(pop)
        pop.weights = w
        pop.cov = cov
        pop.n_batches = self._rejection.state['n_batches']
        return pop

    def _compute_weights_and_cov(self, pop):
        samples = pop.outputs
        params = np.column_stack(tuple([samples[p] for p in self.parameters]))

        if self._populations:
            q_densities = GMDistribution.pdf(params, *self._gm_params)
            w = samples[self.prior_pdf] / q_densities
        else:
            w = np.ones(pop.n_samples)

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
        n_batches = sum([pop.n_batches for pop in self._populations])
        self.objective['n_batches'] = n_batches + self._rejection.objective['n_batches']

    @property
    def _gm_params(self):
        pop_ = self._populations[-1]
        params_ = np.column_stack(tuple([pop_.samples[p] for p in self.parameters]))
        return params_, pop_.cov, pop_.weights

    @property
    def current_population_threshold(self):
        return self.objective['thresholds'][self.state['round']]


class BayesianOptimization(ParameterInference):
    """Bayesian Optimization of an unknown target function."""

    def __init__(self, model, target=None, outputs=None, batch_size=1,
                 initial_evidence=None, update_interval=10, bounds=None, target_model=None,
                 acquisition_method=None, acq_noise_cov=0, **kwargs):
        """
        Parameters
        ----------
        model : ElfiModel or NodeReference
        target : str or NodeReference
            Only needed if model is an ElfiModel
        target_model : GPyRegression, optional
        acquisition_method : Acquisition, optional
            Method of acquiring evidence points. Defaults to LCBSC.
        acq_noise_cov : float, or np.array of shape (n_params, n_params), optional
            Covariance of the noise added in the default LCBSC acquisition method.
        bounds : list
            The region where to estimate the posterior for each parameter in
            model.parameters.
            `[(lower, upper), ... ]`
        initial_evidence : int, dict, optional
            Number of initial evidence or a precomputed batch dict containing parameter 
            and discrepancy values
        update_interval : int
            How often to update the GP hyperparameters of the target_model
        exploration_rate : float
            Exploration rate of the acquisition method
        """

        model, self.target = self._resolve_model(model, target)
        outputs = self._compose_outputs(outputs, model.parameters, self.target)

        super(BayesianOptimization, self).\
            __init__(model, outputs=outputs, batch_size=batch_size, **kwargs)

        target_model = \
            target_model or GPyRegression(len(self.model.parameters), bounds=bounds)

        # Some sensibility limit for starting GP regression
        n_initial_required = max(10, 2**target_model.input_dim + 1)
        self._n_precomputed = 0

        if initial_evidence is None:
            initial_evidence = n_initial_required
        elif not isinstance(initial_evidence, int):
            # Add precomputed batch data
            params = self._to_array(initial_evidence, self.parameters)
            target_model.update(params, initial_evidence[self.target])
            initial_evidence = len(params)
            self._n_precomputed = initial_evidence

        if initial_evidence < n_initial_required:
            raise ValueError('Need at least {} initialization points'.format(n_initial_required))

        # TODO: check if this can be removed
        if initial_evidence % self.batch_size != 0:
            raise ValueError('Initial evidence must be divisible by the batch size')

        self.acquisition_method = acquisition_method or \
                                  LCBSC(target_model, prior=ModelPrior(self.model),
                                        noise_cov=acq_noise_cov, seed=self.seed)

        # TODO: move some of these to objective
        self.n_evidence = initial_evidence
        self.target_model = target_model
        self.n_initial_evidence = initial_evidence
        self.update_interval = update_interval

    def set_objective(self, n_evidence):
        """You can continue BO by giving a larger n_evidence"""
        self.state['pending'] = OrderedDict()
        self.state['last_update'] = self.state.get('last_update') or self._n_precomputed

        if n_evidence and self.n_evidence > n_evidence:
            raise ValueError('New n_evidence must be greater than the earlier')

        self.n_evidence = n_evidence or self.n_evidence
        self.objective['n_batches'] = ceil((self.n_evidence - self._n_precomputed) / self.batch_size)

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
        self.state['pending'].pop(batch_index, None)

        params = self._to_array(batch, self.parameters)
        self._report_batch(batch_index, params, batch[self.target])

        optimize = self._should_optimize()
        self.target_model.update(params, batch[self.target], optimize)

        if optimize:
            self.state['last_update'] = self.target_model.n_evidence

        self.state['n_batches'] += 1
        self.state['n_sim'] += self.batch_size

    def prepare_new_batch(self, batch_index):
        if self._n_submitted_evidence < self.n_initial_evidence - self._n_precomputed:
            return

        pending_params = self._to_array(list(self.state['pending'].values()),
                                        self.parameters)
        t = self.batches.total - int(self.n_initial_evidence / self.batch_size)
        new_param = self.acquisition_method.acquire(self.batch_size, pending_params, t)

        # TODO: implement self._to_batch method?
        batch = {p: new_param[:,i] for i, p in enumerate(self.parameters)}
        self.state['pending'][batch_index] = batch

        return batch

    # TODO: use state dict
    @property
    def _n_submitted_evidence(self):
        return self.batches.total*self.batch_size

    @property
    def _allow_submit(self):
        # TODO: replace this by handling the objective['n_batches']
        # Do not start acquisitions unless all of the initial evidence is ready
        prevent = self.target_model.n_evidence < self.n_initial_evidence <= \
            self._n_submitted_evidence + self._n_precomputed
        return not prevent and super(BayesianOptimization, self)._allow_submit

    def _should_optimize(self):
        current = self.target_model.n_evidence + self.batch_size
        next_update = self.state['last_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update

    def _report_batch(self, batch_index, params, distances):
        str = "Received batch {}:\n".format(batch_index)
        fill = 6 * ' '
        for i in range(self.batch_size):
            str += "{}{} at {}\n".format(fill, distances[i].item(), params[i])
        logger.debug(str)

    def plot_state(self, **options):
        # TODO: Refactor

        # Plot the GP surface
        f = plt.gcf()
        if len(f.axes) < 2:
            f, _ = plt.subplots(1,2, figsize=(13,6), sharex='row', sharey='row')

        gp = self.target_model

        # Draw the GP surface
        visin.draw_contour(gp.predict_mean,
                           gp.bounds,
                           self.parameters,
                           title='GP target surface',
                           points = gp._gp.X,
                           axes=f.axes[0], **options)

        # Draw the latest acquisitions
        point = gp._gp.X[-1, :]
        if len(gp._gp.X) > 1:
            f.axes[1].scatter(*point, color='red')

        displays = [gp._gp]

        if options.get('interactive'):
            from IPython import display
            displays.insert(0, display.HTML(
                    '<span><b>Iteration {}:</b> Acquired {} at {}</span>'.format(
                        len(gp._gp.Y), gp._gp.Y[-1][0], point)))

        # Update
        visin._update_interactive(displays, options)

        acq = lambda x : self.acquisition_method.evaluate(x, len(gp._gp.X))
        # Draw the acquisition surface
        visin.draw_contour(acq,
                           gp.bounds,
                           self.parameters,
                           title='Acquisition surface',
                           points = None,
                           axes=f.axes[1], **options)

        if options.get('close'):
            plt.close()

    def plot_discrepancy(self, axes=None, **kwargs):
        """Plot acquired parameters vs. resulting discrepancy.

        TODO: refactor
        """
        n_plots = self.target_model.input_dim
        ncols = kwargs.pop('ncols', 5)
        nrows = kwargs.pop('nrows', 1)
        kwargs['sharey'] = kwargs.get('sharey', True)
        shape = (max(1, n_plots // ncols), min(n_plots, ncols))
        axes, kwargs = vis._create_axes(axes, shape, **kwargs)
        axes = axes.ravel()

        for ii in range(n_plots):
            axes[ii].scatter(self.target_model._gp.X[:, ii], self.target_model._gp.Y[:, 0])
            axes[ii].set_xlabel(self.parameters[ii])

        axes[0].set_ylabel('Discrepancy')

        return axes


class BOLFI(BayesianOptimization):
    """Bayesian Optimization for Likelihood-Free Inference (BOLFI).

    Approximates the discrepancy function by a stochastic regression model.
    Discrepancy model is fit by sampling the discrepancy function at points decided by
    the acquisition function.

    The method implements the framework introduced in Gutmann & Corander, 2016.

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

    """

    def fit(self, *args, **kwargs):
        """Fit the surrogate model (e.g. Gaussian process) to generate a regression
        model between the priors and the resulting discrepancy.


        """
        logger.info("BOLFI: Fitting the surrogate model...")
        self.infer(*args, **kwargs)

    def infer_posterior(self, threshold=None):
        """Returns an object representing the approximate posterior based on
        surrogate model regression.

        Parameters
        ----------
        threshold: float
            Discrepancy threshold for creating the posterior (log with log discrepancy).

        Returns
        -------
        posterior : elfi.methods.posteriors.BolfiPosterior
        """
        if self.state['n_batches'] == 0:
            self.fit()

        return BolfiPosterior(self.target_model, threshold=threshold, prior=ModelPrior(self.model))


    def sample(self, n_samples, warmup=None, n_chains=4, threshold=None, initials=None,
               algorithm='nuts', **kwargs):
        """Sample the posterior distribution of BOLFI, where the likelihood is defined through
        the cumulative density function of standard normal distribution:

        L(\theta) \propto F((h-\mu(\theta)) / \sigma(\theta))

        where h is the threshold, and \mu(\theta) and \sigma(\theta) are the posterior mean and
        (noisy) standard deviation of the associated Gaussian process.

        The sampling is performed with an MCMC sampler (the No-U-Turn Sampler, NUTS).

        Parameters
        ----------
        n_samples : int
            Number of requested samples from the posterior for each chain. This includes warmup,
            and note that the effective sample size is usually considerably smaller.
        warmpup : int, optional
            Length of warmup sequence in MCMC sampling. Defaults to n_samples//2.
        n_chains : int, optional
            Number of independent chains.
        threshold : float, optional
            The threshold (bandwidth) for posterior (give as log if log discrepancy).
        initials : np.array of shape (n_chains, n_params), optional
            Initial values for the sampled parameters for each chain. Defaults to best evidence points.
        algorithm : string, optional
            Sampling algorithm to use. Currently only 'nuts' is supported.

        Returns
        -------
        np.array
        """
        #TODO: other MCMC algorithms

        posterior = self.infer_posterior(threshold)
        warmup = warmup or n_samples // 2

        # Unless given, select the evidence points with smallest discrepancy
        if initials is not None:
            if np.asarray(initials).shape != (n_chains, self.target_model.input_dim):
                raise ValueError("The shape of initials must be (n_chains, n_params).")
        else:
            # TODO: now GPy specific
            inds = np.argsort(self.target_model._gp.Y[:,0])[:n_chains]
            initials = np.asarray(self.target_model._gp.X[inds])

        self.target_model.is_sampling = True  # enables caching for default RBF kernel

        random_state = np.random.RandomState(self.seed)
        tasks_ids = []

        # sampling is embarrassingly parallel, so depending on self.client this may parallelize
        for ii in range(n_chains):
            seed = get_sub_seed(random_state, ii)
            tasks_ids.append(self.client.apply(mcmc.nuts, n_samples, initials[ii], posterior.logpdf,
                                               posterior.gradient_logpdf, n_adapt=warmup, seed=seed, **kwargs))

        # get results from completed tasks or run sampling (client-specific)
        # TODO: support async
        chains = []
        for id in tasks_ids:
            chains.append(self.client.get(id))

        chains = np.asarray(chains)

        print("{} chains of {} iterations acquired. Effective sample size and Rhat for each parameter:"
              .format(n_chains, n_samples))
        for ii, node in enumerate(self.parameters):
            print(node, mcmc.eff_sample_size(chains[:, :, ii]), mcmc.gelman_rubin(chains[:, :, ii]))

        self.target_model.is_sampling = False

        return ResultBOLFI(method_name='BOLFI',
                           chains=chains,
                           parameter_names=self.parameters,
                           warmup=warmup,
                           threshold=float(posterior.threshold),
                           n_sim=self.state['n_sim'],
                           seed=self.seed
                           )