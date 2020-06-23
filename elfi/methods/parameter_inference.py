"""This module contains common inference methods."""

__all__ = ['Rejection', 'SMC', 'BayesianOptimization', 'BOLFI', 'ROMC']

import logging
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import scipy.optimize as optim

import elfi.client
import elfi.methods.mcmc as mcmc
import elfi.visualization.interactive as visin
import elfi.visualization.visualization as vis
from elfi.loader import get_sub_seed
from elfi.methods.bo.acquisition import LCBSC
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.utils import stochastic_optimization
from elfi.methods.posteriors import BolfiPosterior
from elfi.methods.results import BolfiSample, OptimizationResult, Sample, SmcSample
from elfi.methods.utils import (GMDistribution, ModelPrior, arr2d_to_batch,
                                batch_to_arr2d, ceil_to_batch_size, weighted_var)
from elfi.model.elfi_model import ComputationContext, ElfiModel, NodeReference
from elfi.utils import is_array
from elfi.visualization.visualization import progress_bar

logger = logging.getLogger(__name__)


# TODO: refactor the plotting functions


class ParameterInference:
    """A base class for parameter inference methods.

    Attributes
    ----------
    model : elfi.ElfiModel
        The ELFI graph used by the algorithm
    output_names : list
        Names of the nodes whose outputs are included in the batches
    client : elfi.client.ClientBase
        The batches are computed in the client
    max_parallel_batches : int
    state : dict
        Stores any changing data related to achieving the objective. Must include a key
        ``n_batches`` for determining when the inference is finished.
    objective : dict
        Holds the data for the algorithm to internally determine how many batches are still
        needed. You must have a key ``n_batches`` here. By default the algorithm finished when
        the ``n_batches`` in the state dictionary is equal or greater to the corresponding
        objective value.
    batches : elfi.client.BatchHandler
        Helper class for submitting batches to the client and keeping track of their
        indexes.
    pool : elfi.store.OutputPool
        Pool object for storing and reusing node outputs.


    """

    def __init__(self,
                 model,
                 output_names,
                 batch_size=1,
                 seed=None,
                 pool=None,
                 max_parallel_batches=None):
        """Construct the inference algorithm object.

        If you are implementing your own algorithm do not forget to call `super`.

        Parameters
        ----------
        model : ElfiModel
            Model to perform the inference with.
        output_names : list
            Names of the nodes whose outputs will be requested from the ELFI graph.
        batch_size : int, optional
            The number of parameter evaluations in each pass through the ELFI graph.
            When using a vectorized simulator, using a suitably large batch_size can provide
            a significant performance boost.
        seed : int, optional
            Seed for the data generation from the ElfiModel
        pool : OutputPool, optional
            OutputPool both stores and provides precomputed values for batches.
        max_parallel_batches : int, optional
            Maximum number of batches allowed to be in computation at the same time.
            Defaults to number of cores in the client


        """
        model = model.model if isinstance(model, NodeReference) else model
        if not model.parameter_names:
            raise ValueError('Model {} defines no parameters'.format(model))

        self.model = model.copy()
        self.output_names = self._check_outputs(output_names)

        self.client = elfi.client.get_client()

        # Prepare the computation_context
        context = ComputationContext(batch_size=batch_size, seed=seed, pool=pool)
        self.batches = elfi.client.BatchHandler(
            self.model, context=context, output_names=output_names, client=self.client)
        self.computation_context = context
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
        self.objective = dict()

    @property
    def pool(self):
        """Return the output pool of the inference."""
        return self.computation_context.pool

    @property
    def seed(self):
        """Return the seed of the inference."""
        return self.computation_context.seed

    @property
    def parameter_names(self):
        """Return the parameters to be inferred."""
        return self.model.parameter_names

    @property
    def batch_size(self):
        """Return the current batch_size."""
        return self.computation_context.batch_size

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
        self.state['n_batches'] += 1
        self.state['n_sim'] += self.batch_size

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

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

    def infer(self, *args, vis=None, bar=True, **kwargs):
        """Set the objective and start the iterate loop until the inference is finished.

        See the other arguments from the `set_objective` method.

        Parameters
        ----------
        vis : dict, optional
            Plotting options. More info in self.plot_state method
        bar : bool, optional
            Flag to remove (False) or keep (True) the progress bar from/in output.

        Returns
        -------
        result : Sample

        """
        vis_opt = vis if isinstance(vis, dict) else {}

        self.set_objective(*args, **kwargs)

        if bar:
            progress_bar(0, self._objective_n_batches, prefix='Progress:',
                         suffix='Complete', length=50)

        while not self.finished:
            self.iterate()
            if vis:
                self.plot_state(interactive=True, **vis_opt)

            if bar:
                progress_bar(self.state['n_batches'], self._objective_n_batches,
                             prefix='Progress:', suffix='Complete', length=50)

        self.batches.cancel_pending()
        if vis:
            self.plot_state(close=True, **vis_opt)

        return self.extract_result()

    def iterate(self):
        """Advance the inference by one iteration.

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
        while self._allow_submit(self.batches.next_index):
            next_batch = self.prepare_new_batch(self.batches.next_index)
            logger.debug("Submitting batch %d" % self.batches.next_index)
            self.batches.submit(next_batch)

        # Handle the next ready batch in succession
        batch, batch_index = self.batches.wait_next()
        logger.debug('Received batch %d' % batch_index)
        self.update(batch, batch_index)

    @property
    def finished(self):
        return self._objective_n_batches <= self.state['n_batches']

    def _allow_submit(self, batch_index):
        return (self.max_parallel_batches > self.batches.num_pending
                and self._has_batches_to_submit and (not self.batches.has_ready()))

    @property
    def _has_batches_to_submit(self):
        return self._objective_n_batches > self.state['n_batches'] + self.batches.num_pending

    @property
    def _objective_n_batches(self):
        """Check that n_batches can be computed from the objective."""
        if 'n_batches' in self.objective:
            n_batches = self.objective['n_batches']
        elif 'n_sim' in self.objective:
            n_batches = ceil(self.objective['n_sim'] / self.batch_size)
        else:
            raise ValueError('Objective must define either `n_batches` or `n_sim`.')
        return n_batches

    def _extract_result_kwargs(self):
        """Extract common arguments for the ParameterInferenceResult object."""
        return {
            'method_name': self.__class__.__name__,
            'parameter_names': self.parameter_names,
            'seed': self.seed,
            'n_sim': self.state['n_sim'],
            'n_batches': self.state['n_batches']
        }

    @staticmethod
    def _resolve_model(model, target, default_reference_class=NodeReference):
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

    def _check_outputs(self, output_names):
        """Filter out duplicates and check that corresponding nodes exist.

        Preserves the order.
        """
        output_names = output_names or []
        checked_names = []
        seen = set()
        for name in output_names:
            if isinstance(name, NodeReference):
                name = name.name

            if name in seen:
                continue
            elif not isinstance(name, str):
                raise ValueError(
                    'All output names must be strings, object {} was given'.format(name))
            elif not self.model.has_node(name):
                raise ValueError('Node {} output was requested, but it is not in the model.')

            seen.add(name)
            checked_names.append(name)

        return checked_names


class Sampler(ParameterInference):
    def sample(self, n_samples, *args, **kwargs):
        """Sample from the approximate posterior.

        See the other arguments from the `set_objective` method.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate from the (approximate) posterior
        *args
        **kwargs

        Returns
        -------
        result : Sample

        """
        bar = kwargs.pop('bar', True)

        return self.infer(n_samples, *args, bar=bar, **kwargs)

    def _extract_result_kwargs(self):
        kwargs = super(Sampler, self)._extract_result_kwargs()
        for state_key in ['threshold', 'accept_rate']:
            if state_key in self.state:
                kwargs[state_key] = self.state[state_key]
        if hasattr(self, 'discrepancy_name'):
            kwargs['discrepancy_name'] = self.discrepancy_name
        return kwargs


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

    def __init__(self, model, discrepancy_name=None, output_names=None, **kwargs):
        """Initialize the Rejection sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        kwargs:
            See InferenceMethod

        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)
        output_names = [discrepancy_name] + model.parameter_names + (output_names or [])
        super(Rejection, self).__init__(model, output_names, **kwargs)

        self.discrepancy_name = discrepancy_name

    def set_objective(self, n_samples, threshold=None, quantile=None, n_sim=None):
        """Set objective for inference.

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
            Total number of simulations. The threshold will be the n_samples-th smallest
            discrepancy among n_sim simulations.

        """
        if quantile is None and threshold is None and n_sim is None:
            quantile = .01
        self.state = dict(samples=None, threshold=np.Inf, n_sim=0, accept_rate=1, n_batches=0)

        if quantile:
            n_sim = ceil(n_samples / quantile)

        # Set initial n_batches estimate
        if n_sim:
            n_batches = ceil(n_sim / self.batch_size)
        else:
            n_batches = self.max_parallel_batches

        self.objective = dict(n_samples=n_samples, threshold=threshold, n_batches=n_batches)

        # Reset the inference
        self.batches.reset()

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(Rejection, self).update(batch, batch_index)
        if self.state['samples'] is None:
            # Lazy initialization of the outputs dict
            self._init_samples_lazy(batch)
        self._merge_batch(batch)
        self._update_state_meta()
        self._update_objective_n_batches()

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : Sample

        """
        if self.state['samples'] is None:
            raise ValueError('Nothing to extract')

        # Take out the correct number of samples
        outputs = dict()
        for k, v in self.state['samples'].items():
            outputs[k] = v[:self.objective['n_samples']]

        return Sample(outputs=outputs, **self._extract_result_kwargs())

    def _init_samples_lazy(self, batch):
        """Initialize the outputs dict based on the received batch."""
        samples = {}
        e_noarr = "Node {} output must be in a numpy array of length {} (batch_size)."
        e_len = "Node {} output has array length {}. It should be equal to the batch size {}."

        for node in self.output_names:
            # Check the requested outputs
            if node not in batch:
                raise KeyError("Did not receive outputs for node {}".format(node))

            nbatch = batch[node]
            if not is_array(nbatch):
                raise ValueError(e_noarr.format(node, self.batch_size))
            elif len(nbatch) != self.batch_size:
                raise ValueError(e_len.format(node, len(nbatch), self.batch_size))

            # Prepare samples
            shape = (self.objective['n_samples'] + self.batch_size, ) + nbatch.shape[1:]
            dtype = nbatch.dtype

            if node == self.discrepancy_name:
                # Initialize the distances to inf
                samples[node] = np.ones(shape, dtype=dtype) * np.inf
            else:
                samples[node] = np.empty(shape, dtype=dtype)

        self.state['samples'] = samples

    def _merge_batch(self, batch):
        # TODO: add index vector so that you can recover the original order
        samples = self.state['samples']
        # Put the acquired samples to the end
        for node, v in samples.items():
            v[self.objective['n_samples']:] = batch[node]

        # Sort the smallest to the beginning
        sort_mask = np.argsort(samples[self.discrepancy_name], axis=0).ravel()
        for k, v in samples.items():
            v[:] = v[sort_mask]

    def _update_state_meta(self):
        """Update `n_sim`, `threshold`, and `accept_rate`."""
        o = self.objective
        s = self.state
        s['threshold'] = s['samples'][self.discrepancy_name][o['n_samples'] - 1].item()
        s['accept_rate'] = min(1, o['n_samples'] / s['n_sim'])

    def _update_objective_n_batches(self):
        # Only in the case that the threshold is used
        if self.objective.get('threshold') is None:
            return

        s = self.state
        t, n_samples = [self.objective.get(k) for k in ('threshold', 'n_samples')]

        # noinspection PyTypeChecker
        n_acceptable = np.sum(s['samples'][self.discrepancy_name] <= t) if s['samples'] else 0
        if n_acceptable == 0:
            # No acceptable samples found yet, increase n_batches of objective by one in
            # order to keep simulating
            n_batches = self.objective['n_batches'] + 1
        else:
            accept_rate_t = n_acceptable / s['n_sim']
            # Add some margin to estimated n_batches. One could also use confidence
            # bounds here
            margin = .2 * self.batch_size * int(n_acceptable < n_samples)
            n_batches = (n_samples / accept_rate_t + margin) / self.batch_size
            n_batches = ceil(n_batches)

        self.objective['n_batches'] = n_batches
        logger.debug('Estimated objective n_batches=%d' % self.objective['n_batches'])

    def plot_state(self, **options):
        """Plot the current state of the inference algorithm.

        This feature is still experimental and only supports 1d or 2d cases.
        """
        displays = []
        if options.get('interactive'):
            from IPython import display
            displays.append(
                display.HTML('<span>Threshold: {}</span>'.format(self.state['threshold'])))

        visin.plot_sample(
            self.state['samples'],
            nodes=self.parameter_names,
            n=self.objective['n_samples'],
            displays=displays,
            **options)


class SMC(Sampler):
    """Sequential Monte Carlo ABC sampler."""

    def __init__(self, model, discrepancy_name=None, output_names=None, **kwargs):
        """Initialize the SMC-ABC sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        kwargs:
            See InferenceMethod

        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)

        super(SMC, self).__init__(model, output_names, **kwargs)

        self._prior = ModelPrior(self.model)
        self.discrepancy_name = discrepancy_name
        self.state['round'] = 0
        self._populations = []
        self._rejection = None
        self._round_random_state = None

    def set_objective(self, n_samples, thresholds):
        """Set the objective of the inference."""
        self.objective.update(
            dict(
                n_samples=n_samples,
                n_batches=self.max_parallel_batches,
                round=len(thresholds) - 1,
                thresholds=thresholds))
        self._init_new_round()

    def extract_result(self):
        """Extract the result from the current state.

         Returns
        -------
        SmcSample

        """
        # Extract information from the population
        pop = self._extract_population()
        return SmcSample(
            outputs=pop.outputs,
            populations=self._populations.copy() + [pop],
            weights=pop.weights,
            threshold=pop.threshold,
            **self._extract_result_kwargs())

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(SMC, self).update(batch, batch_index)
        self._rejection.update(batch, batch_index)

        if self._rejection.finished:
            self.batches.cancel_pending()
            if self.state['round'] < self.objective['round']:
                self._populations.append(self._extract_population())
                self.state['round'] += 1
                self._init_new_round()

        self._update_objective()

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

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
        if self.state['round'] == 0:
            # Use the actual prior
            return

        # Sample from the proposal, condition on actual prior
        params = GMDistribution.rvs(*self._gm_params, size=self.batch_size,
                                    prior_logpdf=self._prior.logpdf,
                                    random_state=self._round_random_state)

        batch = arr2d_to_batch(params, self.parameter_names)
        return batch

    def _init_new_round(self):
        round = self.state['round']

        dashes = '-' * 16
        logger.info('%s Starting round %d %s' % (dashes, round, dashes))

        # Get a subseed for this round for ensuring consistent results for the round
        seed = self.seed if round == 0 else get_sub_seed(self.seed, round)
        self._round_random_state = np.random.RandomState(seed)

        self._rejection = Rejection(
            self.model,
            discrepancy_name=self.discrepancy_name,
            output_names=self.output_names,
            batch_size=self.batch_size,
            seed=seed,
            max_parallel_batches=self.max_parallel_batches)

        self._rejection.set_objective(
            self.objective['n_samples'], threshold=self.current_population_threshold)

    def _extract_population(self):
        sample = self._rejection.extract_result()
        # Append the sample object
        sample.method_name = "Rejection within SMC-ABC"
        w, cov = self._compute_weights_and_cov(sample)
        sample.weights = w
        sample.meta['cov'] = cov
        return sample

    def _compute_weights_and_cov(self, pop):
        params = np.column_stack(tuple([pop.outputs[p] for p in self.parameter_names]))

        if self._populations:
            q_logpdf = GMDistribution.logpdf(params, *self._gm_params)
            p_logpdf = self._prior.logpdf(params)
            w = np.exp(p_logpdf - q_logpdf)
        else:
            w = np.ones(pop.n_samples)

        if np.count_nonzero(w) == 0:
            raise RuntimeError("All sample weights are zero. If you are using a prior "
                               "with a bounded support, this may be caused by specifying "
                               "a too small sample size.")

        # New covariance
        cov = 2 * np.diag(weighted_var(params, w))

        if not np.all(np.isfinite(cov)):
            logger.warning("Could not estimate the sample covariance. This is often "
                           "caused by majority of the sample weights becoming zero."
                           "Falling back to using unit covariance.")
            cov = np.diag(np.ones(params.shape[1]))

        return w, cov

    def _update_objective(self):
        """Update the objective n_batches."""
        n_batches = sum([pop.n_batches for pop in self._populations])
        self.objective['n_batches'] = n_batches + self._rejection.objective['n_batches']

    @property
    def _gm_params(self):
        sample = self._populations[-1]
        params = sample.samples_array
        return params, sample.cov, sample.weights

    @property
    def current_population_threshold(self):
        """Return the threshold for current population."""
        return self.objective['thresholds'][self.state['round']]


class BayesianOptimization(ParameterInference):
    """Bayesian Optimization of an unknown target function."""

    def __init__(self,
                 model,
                 target_name=None,
                 bounds=None,
                 initial_evidence=None,
                 update_interval=10,
                 target_model=None,
                 acquisition_method=None,
                 acq_noise_var=0,
                 exploration_rate=10,
                 batch_size=1,
                 batches_per_acquisition=None,
                 async_acq=False,
                 **kwargs):
        """Initialize Bayesian optimization.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        target_name : str or NodeReference
            Only needed if model is an ElfiModel
        bounds : dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters: dict('parameter_name':(lower, upper), ... )`. Not used if
            custom target_model is given.
        initial_evidence : int, dict, optional
            Number of initial evidence or a precomputed batch dict containing parameter
            and discrepancy values. Default value depends on the dimensionality.
        update_interval : int, optional
            How often to update the GP hyperparameters of the target_model
        target_model : GPyRegression, optional
        acquisition_method : Acquisition, optional
            Method of acquiring evidence points. Defaults to LCBSC.
        acq_noise_var : float or np.array, optional
            Variance(s) of the noise added in the default LCBSC acquisition method.
            If an array, should be 1d specifying the variance for each dimension.
        exploration_rate : float, optional
            Exploration rate of the acquisition method
        batch_size : int, optional
            Elfi batch size. Defaults to 1.
        batches_per_acquisition : int, optional
            How many batches will be requested from the acquisition function at one go.
            Defaults to max_parallel_batches.
        async_acq : bool, optional
            Allow acquisitions to be made asynchronously, i.e. do not wait for all the
            results from the previous acquisition before making the next. This can be more
            efficient with a large amount of workers (e.g. in cluster environments) but
            forgoes the guarantee for the exactly same result with the same initial
            conditions (e.g. the seed). Default False.
        **kwargs

        """
        model, target_name = self._resolve_model(model, target_name)
        output_names = [target_name] + model.parameter_names
        super(BayesianOptimization, self).__init__(
            model, output_names, batch_size=batch_size, **kwargs)

        target_model = target_model or GPyRegression(self.model.parameter_names, bounds=bounds)

        self.target_name = target_name
        self.target_model = target_model

        n_precomputed = 0
        n_initial, precomputed = self._resolve_initial_evidence(initial_evidence)
        if precomputed is not None:
            params = batch_to_arr2d(precomputed, self.parameter_names)
            n_precomputed = len(params)
            self.target_model.update(params, precomputed[target_name])

        self.batches_per_acquisition = batches_per_acquisition or self.max_parallel_batches
        self.acquisition_method = acquisition_method or LCBSC(self.target_model,
                                                              prior=ModelPrior(self.model),
                                                              noise_var=acq_noise_var,
                                                              exploration_rate=exploration_rate,
                                                              seed=self.seed)

        self.n_initial_evidence = n_initial
        self.n_precomputed_evidence = n_precomputed
        self.update_interval = update_interval
        self.async_acq = async_acq

        self.state['n_evidence'] = self.n_precomputed_evidence
        self.state['last_GP_update'] = self.n_initial_evidence
        self.state['acquisition'] = []

    def _resolve_initial_evidence(self, initial_evidence):
        # Some sensibility limit for starting GP regression
        precomputed = None
        n_required = max(10, 2**self.target_model.input_dim + 1)
        n_required = ceil_to_batch_size(n_required, self.batch_size)

        if initial_evidence is None:
            n_initial_evidence = n_required
        elif isinstance(initial_evidence, (int, np.int, float)):
            n_initial_evidence = int(initial_evidence)
        else:
            precomputed = initial_evidence
            n_initial_evidence = len(precomputed[self.target_name])

        if n_initial_evidence < 0:
            raise ValueError('Number of initial evidence must be positive or zero '
                             '(was {})'.format(initial_evidence))
        elif n_initial_evidence < n_required:
            logger.warning('We recommend having at least {} initialization points for '
                           'the initialization (now {})'.format(n_required, n_initial_evidence))

        if precomputed is None and (n_initial_evidence % self.batch_size != 0):
            logger.warning('Number of initial_evidence %d is not divisible by '
                           'batch_size %d. Rounding it up...' % (n_initial_evidence,
                                                                 self.batch_size))
            n_initial_evidence = ceil_to_batch_size(n_initial_evidence, self.batch_size)

        return n_initial_evidence, precomputed

    @property
    def n_evidence(self):
        """Return the number of acquired evidence points."""
        return self.state.get('n_evidence', 0)

    @property
    def acq_batch_size(self):
        """Return the total number of acquisition per iteration."""
        return self.batch_size * self.batches_per_acquisition

    def set_objective(self, n_evidence=None):
        """Set objective for inference.

        You can continue BO by giving a larger n_evidence.

        Parameters
        ----------
        n_evidence : int
            Number of total evidence for the GP fitting. This includes any initial
            evidence.

        """
        if n_evidence is None:
            n_evidence = self.objective.get('n_evidence', self.n_evidence)

        if n_evidence < self.n_evidence:
            logger.warning('Requesting less evidence than there already exists')

        self.objective['n_evidence'] = n_evidence
        self.objective['n_sim'] = n_evidence - self.n_precomputed_evidence

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        OptimizationResult

        """
        x_min, _ = stochastic_optimization(
            self.target_model.predict_mean, self.target_model.bounds, seed=self.seed)

        batch_min = arr2d_to_batch(x_min, self.parameter_names)
        outputs = arr2d_to_batch(self.target_model.X, self.parameter_names)
        outputs[self.target_name] = self.target_model.Y

        return OptimizationResult(
            x_min=batch_min, outputs=outputs, **self._extract_result_kwargs())

    def update(self, batch, batch_index):
        """Update the GP regression model of the target node with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(BayesianOptimization, self).update(batch, batch_index)
        self.state['n_evidence'] += self.batch_size

        params = batch_to_arr2d(batch, self.parameter_names)
        self._report_batch(batch_index, params, batch[self.target_name])

        optimize = self._should_optimize()
        self.target_model.update(params, batch[self.target_name], optimize)
        if optimize:
            self.state['last_GP_update'] = self.target_model.n_evidence

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

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
        t = self._get_acquisition_index(batch_index)

        # Check if we still should take initial points from the prior
        if t < 0:
            return

        # Take the next batch from the acquisition_batch
        acquisition = self.state['acquisition']
        if len(acquisition) == 0:
            acquisition = self.acquisition_method.acquire(self.acq_batch_size, t=t)

        batch = arr2d_to_batch(acquisition[:self.batch_size], self.parameter_names)
        self.state['acquisition'] = acquisition[self.batch_size:]

        return batch

    def _get_acquisition_index(self, batch_index):
        acq_batch_size = self.batch_size * self.batches_per_acquisition
        initial_offset = self.n_initial_evidence - self.n_precomputed_evidence
        starting_sim_index = self.batch_size * batch_index

        t = (starting_sim_index - initial_offset) // acq_batch_size
        return t

    # TODO: use state dict
    @property
    def _n_submitted_evidence(self):
        return self.batches.total * self.batch_size

    def _allow_submit(self, batch_index):
        if not super(BayesianOptimization, self)._allow_submit(batch_index):
            return False

        if self.async_acq:
            return True

        # Allow submitting freely as long we are still submitting initial evidence
        t = self._get_acquisition_index(batch_index)
        if t < 0:
            return True

        # Do not allow acquisition until previous acquisitions are ready (as well
        # as all initial acquisitions)
        acquisitions_left = len(self.state['acquisition'])
        if acquisitions_left == 0 and self.batches.has_pending:
            return False

        return True

    def _should_optimize(self):
        current = self.target_model.n_evidence + self.batch_size
        next_update = self.state['last_GP_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update

    def _report_batch(self, batch_index, params, distances):
        str = "Received batch {}:\n".format(batch_index)
        fill = 6 * ' '
        for i in range(self.batch_size):
            str += "{}{} at {}\n".format(fill, distances[i].item(), params[i])
        logger.debug(str)

    def plot_state(self, **options):
        """Plot the GP surface.

        This feature is still experimental and currently supports only 2D cases.
        """
        f = plt.gcf()
        if len(f.axes) < 2:
            f, _ = plt.subplots(1, 2, figsize=(13, 6), sharex='row', sharey='row')

        gp = self.target_model

        # Draw the GP surface
        visin.draw_contour(
            gp.predict_mean,
            gp.bounds,
            self.parameter_names,
            title='GP target surface',
            points=gp.X,
            axes=f.axes[0],
            **options)

        # Draw the latest acquisitions
        if options.get('interactive'):
            point = gp.X[-1, :]
            if len(gp.X) > 1:
                f.axes[1].scatter(*point, color='red')

        displays = [gp._gp]

        if options.get('interactive'):
            from IPython import display
            displays.insert(
                0,
                display.HTML('<span><b>Iteration {}:</b> Acquired {} at {}</span>'.format(
                    len(gp.Y), gp.Y[-1][0], point)))

        # Update
        visin._update_interactive(displays, options)

        def acq(x):
            return self.acquisition_method.evaluate(x, len(gp.X))

        # Draw the acquisition surface
        visin.draw_contour(
            acq,
            gp.bounds,
            self.parameter_names,
            title='Acquisition surface',
            points=None,
            axes=f.axes[1],
            **options)

        if options.get('close'):
            plt.close()

    def plot_discrepancy(self, axes=None, **kwargs):
        """Plot acquired parameters vs. resulting discrepancy.

        Parameters
        ----------
        axes : plt.Axes or arraylike of plt.Axes

        Return
        ------
        axes : np.array of plt.Axes

        """
        return vis.plot_discrepancy(self.target_model, self.parameter_names, axes=axes, **kwargs)

    def plot_gp(self, axes=None, resol=50, const=None, bounds=None, true_params=None, **kwargs):
        """Plot pairwise relationships as a matrix with parameters vs. discrepancy.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
        resol : int, optional
            Resolution of the plotted grid.
        const : np.array, optional
            Values for parameters in plots where held constant. Defaults to minimum evidence.
        bounds: list of tuples, optional
            List of tuples for axis boundaries.
        true_params : dict, optional
            Dictionary containing parameter names with corresponding true parameter values.

        Returns
        -------
        axes : np.array of plt.Axes

        """
        return vis.plot_gp(self.target_model, self.parameter_names, axes,
                           resol, const, bounds, true_params, **kwargs)


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

    def fit(self, n_evidence, threshold=None, bar=True):
        """Fit the surrogate model.

        Generates a regression model for the discrepancy given the parameters.

        Currently only Gaussian processes are supported as surrogate models.

        Parameters
        ----------
        n_evidence : int, required
            Number of evidence for fitting
        threshold : float, optional
            Discrepancy threshold for creating the posterior (log with log discrepancy).
        bar : bool, optional
            Flag to remove (False) the progress bar from output.

        """
        logger.info("BOLFI: Fitting the surrogate model...")

        if n_evidence is None:
            raise ValueError(
                'You must specify the number of evidence (n_evidence) for the fitting')

        self.infer(n_evidence, bar=bar)
        return self.extract_posterior(threshold)

    def extract_posterior(self, threshold=None):
        """Return an object representing the approximate posterior.

        The approximation is based on surrogate model regression.

        Parameters
        ----------
        threshold: float, optional
            Discrepancy threshold for creating the posterior (log with log discrepancy).

        Returns
        -------
        posterior : elfi.methods.posteriors.BolfiPosterior

        """
        if self.state['n_evidence'] == 0:
            raise ValueError('Model is not fitted yet, please see the `fit` method.')

        return BolfiPosterior(self.target_model, threshold=threshold, prior=ModelPrior(self.model))

    def sample(self,
               n_samples,
               warmup=None,
               n_chains=4,
               threshold=None,
               initials=None,
               algorithm='nuts',
               sigma_proposals=None,
               n_evidence=None,
               **kwargs):
        r"""Sample the posterior distribution of BOLFI.

        Here the likelihood is defined through the cumulative density function
        of the standard normal distribution:

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
            Initial values for the sampled parameters for each chain.
            Defaults to best evidence points.
        algorithm : string, optional
            Sampling algorithm to use. Currently 'nuts'(default) and 'metropolis' are supported.
        sigma_proposals : np.array
            Standard deviations for Gaussian proposals of each parameter for Metropolis
            Markov Chain sampler.
        n_evidence : int
            If the regression model is not fitted yet, specify the amount of evidence

        Returns
        -------
        BolfiSample

        """
        if self.state['n_batches'] == 0:
            self.fit(n_evidence)

        # TODO: add more MCMC algorithms
        if algorithm not in ['nuts', 'metropolis']:
            raise ValueError("Unknown posterior sampler.")

        posterior = self.extract_posterior(threshold)
        warmup = warmup or n_samples // 2

        # Unless given, select the evidence points with smallest discrepancy
        if initials is not None:
            if np.asarray(initials).shape != (n_chains, self.target_model.input_dim):
                raise ValueError("The shape of initials must be (n_chains, n_params).")
        else:
            inds = np.argsort(self.target_model.Y[:, 0])
            initials = np.asarray(self.target_model.X[inds])

        self.target_model.is_sampling = True  # enables caching for default RBF kernel

        tasks_ids = []
        ii_initial = 0
        if algorithm == 'metropolis':
            if sigma_proposals is None:
                raise ValueError("Gaussian proposal standard deviations "
                                 "have to be provided for Metropolis-sampling.")
            elif sigma_proposals.shape[0] != self.target_model.input_dim:
                raise ValueError("The length of Gaussian proposal standard "
                                 "deviations must be n_params.")

        # sampling is embarrassingly parallel, so depending on self.client this may parallelize
        for ii in range(n_chains):
            seed = get_sub_seed(self.seed, ii)
            # discard bad initialization points
            while np.isinf(posterior.logpdf(initials[ii_initial])):
                ii_initial += 1
                if ii_initial == len(inds):
                    raise ValueError(
                        "BOLFI.sample: Cannot find enough acceptable initialization points!")

            if algorithm == 'nuts':
                tasks_ids.append(
                    self.client.apply(
                        mcmc.nuts,
                        n_samples,
                        initials[ii_initial],
                        posterior.logpdf,
                        posterior.gradient_logpdf,
                        n_adapt=warmup,
                        seed=seed,
                        **kwargs))

            elif algorithm == 'metropolis':
                tasks_ids.append(
                    self.client.apply(
                        mcmc.metropolis,
                        n_samples,
                        initials[ii_initial],
                        posterior.logpdf,
                        sigma_proposals,
                        warmup,
                        seed=seed,
                        **kwargs))

            ii_initial += 1

        # get results from completed tasks or run sampling (client-specific)
        chains = []
        for id in tasks_ids:
            chains.append(self.client.get_result(id))

        chains = np.asarray(chains)
        print(
            "{} chains of {} iterations acquired. Effective sample size and Rhat for each "
            "parameter:".format(n_chains, n_samples))
        for ii, node in enumerate(self.parameter_names):
            print(node, mcmc.eff_sample_size(chains[:, :, ii]),
                  mcmc.gelman_rubin(chains[:, :, ii]))
        self.target_model.is_sampling = False

        return BolfiSample(
            method_name='BOLFI',
            chains=chains,
            parameter_names=self.parameter_names,
            warmup=warmup,
            threshold=float(posterior.threshold),
            n_sim=self.state['n_sim'],
            seed=self.seed)


class ROMC(ParameterInference):

    def __init__(self, model, prior, discrepancy_name=None, output_names=None, **kwargs):
        """

        Parameters
        ----------
        model: elfi.model or elfi.ReferenceNode
        prior: callable, theta_flat -> np.array (pdf of the prior)
        discrepancy_name: if elfi.model is passed as model, then this is the name of the output node
        output_names: list of names to store during the procedure
        kwargs: other named parameters
        """
        # define model, output names asked by the romc method TODO check the purpose of output names
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)
        output_names = [discrepancy_name] + model.parameter_names + (output_names or [])
        self.discrepancy_name = discrepancy_name

        # set model as attribute
        self.model = model

        # set prior as attribute
        self.prior = prior

        # self.D = 1

        # dict of binary/values indicating which parts of the inference process have been obtained
        self.method_state = {"_has_gen_nuisance": False,
                             "_has_defined_optim_problems": False,
                             "_has_solved_optim_problems": False,
                             "_has_filtered_solutions": False,
                             "_has_chosen_solutions": False,
                             "_has_estimated_proposals": False}

        # inputs passed to the inference method
        self.inference_args = {}

        # state of the inference procedure. This is where the values reached along the inference process are stores.
        self.inference_state = {}
        super(ROMC, self).__init__(model, output_names, **kwargs)

    def sample_nuisance(self, n1, seed=None):
        """
        Draws n1 nuisance variables (i.e. seeds) and stores them in the inference_state dict.

        Parameters
        ----------
        n1: int, nof nuisance samples
        seed: int, the seed used for sampling the nuisance variables
        """
        # procedure
        up_lim = 2**32 - 1
        u = np.random.default_rng(seed=seed).choice(up_lim, size=n1, replace=False)
        # u = ss.randint(1, up_lim).rvs(size=(n1,), random_state=seed)

        # update method state
        self.method_state["_has_gen_nuisance"] = True

        # update inference state
        self.inference_state["u"] = u

        # update inference args
        self.inference_args["N1"] = n1
        self.inference_args["initial_seed"] = seed

        # display state
        print("NOF u points        : %d " % n1)

    def define_optim_problems(self):
        """Defines an optimization problem for each nuisance variable
        and computes the dimensionality of the flatten theta array.

        Returns
        -------
        None
        """
        def create_deterministic_generator(u):
            """
            Parameters
            __________
            u: int, seed passed to model.generate

            Returns
            -------
            func: deterministic generator
            """

            def deterministic_generator(theta):
                """

                Parameters
                ----------
                theta: np.ndarray (D,) flattened parameters; follows the order of the parameters

                Returns
                -------
                float: the output node sample, with frozen seed, given theta
                """
                # Map flattened array of parameters to parameter names with correct shape
                res = self.model.generate(batch_size=1)
                param_dict = {}
                cur_ind = 0
                for param_name in self.model.parameter_names:
                    tensor = res[param_name]
                    assert isinstance(tensor, np.ndarray)
                    if tensor.ndim == 2:
                        dim = tensor.shape[1]
                        val = theta[cur_ind:cur_ind + dim]
                        cur_ind += dim
                        assert isinstance(val, np.ndarray)
                        assert val.ndim == 1
                        param_dict[param_name] = np.expand_dims(val, 0)

                    else:
                        dim = 1
                        val = theta[cur_ind:cur_ind + dim]
                        cur_ind += dim
                        assert isinstance(val, np.ndarray)
                        assert val.ndim == 1
                        param_dict[param_name] = val

                # "generate" a single sample deterministically, with theta passed as parameters
                # and frozen seed
                return float(self.model.generate(batch_size=1,
                                                 with_values=param_dict,
                                                 seed=int(u))['distance'])
            return deterministic_generator

        # getters
        u = self.inference_state["u"]
        
        # create deterministic generators for each nuisance (seed)
        deterministic_funcs = []
        for i in range(len(u)):
            deterministic_funcs.append(create_deterministic_generator(u[i]))

        # find dimension of theta_flat
        nof_elem = 0
        res = self.model.generate(batch_size=1)
        for param_name in self.model.parameter_names:
            tensor = res[param_name]
            assert isinstance(tensor, np.ndarray)
            if tensor.ndim == 2:
                dim = tensor.shape[1]
                nof_elem += dim
            else:
                dim = 1
                nof_elem += dim

        # store list with deterministic functions and dimensionality of flat theta
        self.inference_state["deterministic_funcs"] = deterministic_funcs
        self.inference_state["flat_theta_dim"] = nof_elem
        
        # update method state
        self.method_state["_has_defined_optim_problems"] = True

    def solve_optim_problems(self, seed):
        # getters
        N1 = self.inference_args["N1"]
        optim_funcs = self.inference_state["deterministic_funcs"]
        dim = self.inference_state["flat_theta_dim"]

        solutions_obtained_mask = []
        solutions_obtained_mapping = []
        solutions = []
        initial_points = ss.norm(loc=0, scale=3).rvs(size=(N1, dim), random_state=seed)
        for i in range(N1):
            cur_func = optim_funcs[i]
            init_point = initial_points[i]
            res = optim.minimize(cur_func,
                                 init_point,
                                 method="Nelder-Mead",
                                 tol=.01)
            if res.success:
                solutions_obtained_mask.append(True)
                solutions.append(res)
                solutions_obtained_mapping.append(i)
            else:
                solutions_obtained_mask.append(False)

        # update inference_state
        self.inference_state["solutions_obtained_mask"] = solutions_obtained_mask
        self.inference_state["solutions_obtained_mapping"] = solutions_obtained_mapping
        self.inference_state["solutions"] = solutions
        self.inference_state["initial_points"] = initial_points

        # update state
        self.method_state["_has_solved_optim_problems"] = True

        # print state
        print("NOF solutions obtained : %d " % np.sum(solutions_obtained_mask))

    def filter_solutions(self, eps):
        # checks
        assert self.method_state["_has_solved_optim_problems"]

        # getters
        N1 = self.inference_args["N1"]
        optim_funcs = self.inference_state["deterministic_funcs"]
        dim = self.inference_state["flat_theta_dim"]

        solutions_obtained_mask = self.inference_state["solutions_obtained_mask"]
        solutions_obtained_mapping = self.inference_state["solutions_obtained_mapping"]
        solutions = self.inference_state["solutions"]

        # setters
        self.inference_args["epsilon"] = eps

        solutions_accepted_theta_0 = []
        solutions_accepted_dist_0 = []
        solutions_accepted_mask = []
        solutions_accepted_mapping = []
        for i in range(len(solutions)):

            ind = solutions_obtained_mapping[i]
            res = solutions[i]
            if res.fun < eps:
                solutions_accepted_mask.append(True)
                solutions_accepted_theta_0.append(res.x)
                solutions_accepted_dist_0.append(res.fun)
                solutions_accepted_mapping.append(ind)
            else:
                solutions_accepted_mask.append(False)

        # update inference state
        self.inference_state["solutions_accepted_theta_0"] = solutions_accepted_theta_0
        self.inference_state["solutions_accepted_dist_0"] = solutions_accepted_dist_0
        self.inference_state["solutions_accepted_mask"] = solutions_accepted_mask
        self.inference_state["accepted_points"] = np.sum(solutions_accepted_mask)
        self.inference_state["solutions_accepted_mapping"] = solutions_accepted_mapping

        # update method state
        self.method_state["_has_filtered_solutions"] = True

        print("epsilon             : %.3f" % eps)
        print("NOF accepted points : %d " % np.sum(solutions_accepted_mask))

    def estimate_region(self):
        def dummy_estimation(dist, theta_0, eps):
            assert isinstance(theta_0, np.ndarray)
            assert theta_0.ndim == 1
            assert theta_0.shape[0] == dim

            # check that best point is in limit
            assert dist(theta_0) < eps

            step = 0.1
            BB = []
            for j in range(dim):
                BB.append([])

                # right side
                for i in range(1, 101):
                    point = theta_0.copy()
                    point[j] += i*step
                    if dist(point) > eps:
                        v_right = (i-1)*step
                        break
                    if i == 100:
                        v_right = (i-1)*step

                for i in range(1, 101):
                    point = theta_0.copy()
                    point[j] -= i*step
                    if dist(point) > eps:
                        v_left = - (i-1)*step
                        break
                    if i == 100:
                        v_left = - (i-1)*step

                BB[j].append(theta_0[j] + v_left)
                BB[j].append(theta_0[j] + v_right)
            return BB

        # getters
        dim = self.inference_state["flat_theta_dim"]

        deterministic_funcs = self.inference_state["deterministic_funcs"]
        theta0 = self.inference_state["solutions_accepted_theta_0"]
        eps = self.inference_args["epsilon"]
        solutions_accepted_mapping = self.inference_state["solutions_accepted_mapping"]
        regions = []

        for i in range(len(solutions_accepted_mapping)):
            ind = solutions_accepted_mapping[i]
            func = deterministic_funcs[ind]
            th = theta0[i]

            cur_BB = dummy_estimation(func, th, eps)
            regions.append(cur_BB)

        # update inference state
        self.inference_state["regions"] = np.array(regions)

        # update method state
        self.method_state["_has_estimated_proposals"] = True

    def visualize(self, i):
        mapping = self.inference_state["solutions_accepted_mapping"]
        func = self.inference_state["deterministic_funcs"][mapping[i]]
        BB = self.inference_state["regions"]
        eps = self.inference_args["epsilon"]
        theta0 = self.inference_state["solutions_accepted_theta_0"]

        plt.figure()
        plt.title("u[i] = %d" % self.inference_state["u"][mapping[i]])
        x = np.linspace(-15, 15, 100)
        y = [float(func(np.array([theta]))) for theta in x]
        plt.plot(x, y, 'r--')
        plt.ylim(0, 5)
        plt.axvspan(BB[i][0][0], BB[i][0][1])
        plt.axvline(theta0[i], color="y")
        plt.axhline(eps, color="g")
        plt.show(block=False)

    def approx_Z(self):
        nof_points = 1000
        Z = 0
        for theta in np.linspace(-2.5, 2.5, nof_points):
            Z += self.unnormalized_posterior(np.array([theta])) * 5 / nof_points
        self.inference_state["Z"] = Z

    def unnormalized_posterior(self, theta):
        assert isinstance(theta, np.ndarray)
        BB = self.inference_state["regions"]
        assert isinstance(BB, np.ndarray)

        jj = 0
        for i in range(BB.shape[0]):
            inside = True
            vol = 1
            for j in range(BB.shape[1]):
                vol *= BB[i][j][1] - BB[i][j][0]
                if (theta < BB[i][j][0]) or (theta > BB[i][j][1]):
                    inside = False
                    break

            if inside:
                jj += 1 / vol
        return jj * self.prior(theta)

    def posterior(self, theta):
        # assert isinstance(self.BB, np.ndarray)
        return self.unnormalized_posterior(theta) / self.inference_state["Z"]


    def compute_expectation(self, h, N2, seed=Noneo):
        # assert all stages have been executed
        assert self.method_state["_has_estimated_proposals"]

        def combine_N1_N2_dims(tensor):
            tmp = []
            for i in range(tensor.shape[1]):
                tmp.append(tensor[:, i, :].flatten())
            flat = np.stack(tmp, axis=1)
            return flat


        # setters
        self.inference_args["N2"] = N2

        # getters
        N1 = self.inference_args["N1"]
        BB = self.inference_state["regions"]
        dim = self.inference_state["flat_theta_dim"]
        dist_funcs = self.inference_state["deterministic_funcs"]
        epsilon = self.inference_args["epsilon"]
        nof_accepted_points = self.inference_state["accepted_points"]
        mapping = self.inference_state["solutions_accepted_mapping"]

        assert isinstance(BB, np.ndarray)
        assert BB.ndim == 3
        assert BB.shape[0] == nof_accepted_points
        assert BB.shape[1] == dim
        assert BB.shape[2] == 2

        # draw samples from q(theta_ij)
        loc = np.expand_dims(BB[:, :, 0], -1)
        scale = np.expand_dims(BB[:, :, 1] - BB[:, :, 0], -1)
        loc = loc.repeat(N2, axis=-1)
        scale = scale.repeat(N2, axis=-1)

        # dummy check for very small scale
        tt = 0.01
        loc[scale == 0] -= tt
        scale[scale == 0] += 2*tt
        
        # define proposal distributions q_i(theta)
        q_dist = ss.uniform(loc=loc, scale=scale)

        # draw samples from q_i(theta): N1xDxN2
        theta_samples = q_dist.rvs(random_state=seed) # theta: N1xDxN2

        # flatten N1, N2 dimensions to (N1*N2)xD
        theta1 = combine_N1_N2_dims(theta_samples)

        assert theta1.shape[0] == nof_accepted_points * N2
        assert theta1.shape[1] == dim

        # compute expectacion
        ptheta = self.prior(theta1)

        # (ii) q(theta_ij)
        qtheta = q_dist.pdf(theta_samples)
        qtheta = combine_N1_N2_dims(qtheta)

        # (iii) indicator(theta_ij) TODO add multivariate case
        tmp = []
        for i in range(nof_accepted_points):
            tmp.append([])
            theta_j = np.transpose(theta_samples[i], (1, 0))
            theta_j = np.squeeze(theta_j)
            for j in range(theta_j.shape[0]):
                tmp[i].append( dist_funcs[mapping[i]](theta_j[j:j+1]) < epsilon )
        indicator = np.array(tmp, dtype=float)
        indicator = combine_N1_N2_dims(np.expand_dims(indicator, axis=1))

        # (iv) h(theta_ij)
        htheta = h(theta1)

        # compute expected value
        down = ptheta/qtheta*indicator
        up = down*htheta

        return np.sum(up)/np.sum(down)

    def sample(self, prior, N1, *args, **kwargs):
        bar = kwargs.pop('bar', True)
        self.infer(N1, *args, bar=bar, **kwargs)

    def update(self, batch, batch_index):
        super(ROMC, self).update(batch, batch_index)

        if self.state["samples"] is None:
            self.state["samples"] = None

    def set_objective(self, N1):
        # init state
        self.state["samples"] = []

        n_samples = N1
        n_batches = ceil(N1 / self.batch_size)

        self.objective = {"n_samples": n_samples, "n_batches": n_batches}

        # reset the inference
        self.batches.reset()

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : Sample

        """
        if self.state['samples'] is None:
            raise ValueError('Nothing to extract')

        # Take out the correct number of samples
        # outputs = dict()
        # for k, v in self.state['samples'].items():
        #     outputs[k] = v[:self.objective['n_samples']]
        outputs = {'theta': 0}

        return Sample(outputs=outputs, **self._extract_result_kwargs())
