"""This module contains implementation of bolfire."""

__all__ = ['BOLFIRE']

import logging

import numpy as np

import elfi.methods.mcmc as mcmc
from elfi.classifiers.classifier import Classifier, LogisticRegression
from elfi.loader import get_sub_seed
from elfi.methods.bo.acquisition import LCBSC, AcquisitionBase
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.inference.parameter_inference import ParameterInference
from elfi.methods.posteriors import BOLFIREPosterior
from elfi.methods.results import BOLFIRESample
from elfi.methods.utils import arr2d_to_batch, batch_to_arr2d
from elfi.model.elfi_model import ElfiModel, Summary
from elfi.model.extensions import ModelPrior

logger = logging.getLogger(__name__)


class BOLFIRE(ParameterInference):
    """Bayesian Optimization and Classification in Likelihood-Free Inference (BOLFIRE)."""

    def __init__(self,
                 model,
                 n_training_data=10,
                 marginal=None,
                 seed_marginal=None,
                 classifier=None,
                 bounds=None,
                 n_initial_evidence=0,
                 acq_noise_var=0,
                 exploration_rate=10,
                 update_interval=1,
                 target_model=None,
                 acquisition_method=None,
                 *args, **kwargs):
        """Initialize the BOLFIRE method.

        Parameters
        ----------
        model: ElfiModel
            Elfi graph used by the algorithm.
        n_training_data: int, optional
            Size of training data.
        marginal: np.ndnarray, optional
            Marginal data.
        seed_marginal: int, optional
            Seed for marginal data generation.
        classifier: str, optional
            Classifier to be used. Default LogisticRegression.
        bounds: dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters: dict('parameter_name': (lower, upper), ... ). Not used if
            custom target_model is given.
        n_initial_evidence: int, optional
            Number of initial evidence.
        acq_noise_var: float or np.ndarray, optional
            Variance(s) of the noise added in the default LCBSC acquisition method.
            If an array, should be 1d specifying the variance for each dimension.
        exploration_rate: float, optional
            Exploration rate of the acquisition method.
        update_interval : int, optional
            How often to update the GP hyperparameters of the target_model.
        target_model: GPyRegression, optional
            A surrogate model to be used.
        acquisition_method: Acquisition, optional
            Method of acquiring evidence points. Default LCBSC.

        """
        # Resolve model and initialize
        model = self._resolve_model(model)
        super(BOLFIRE, self).__init__(model, output_names=None, *args, **kwargs)

        # Initialize attributes
        self.n_training_data = self._resolve_n_training_data(n_training_data)
        self.summary_names = self._get_summary_names(self.model)
        self.marginal = self._resolve_marginal(marginal, seed_marginal)
        self.classifier = self._resolve_classifier(classifier)
        self.observed = self._get_observed_summary_values(self.model, self.summary_names)
        self.prior = ModelPrior(self.model)

        # TODO: write resolvers for the attributes below
        self.bounds = bounds
        self.acq_noise_var = acq_noise_var
        self.exploration_rate = exploration_rate
        self.update_interval = update_interval

        # Initialize GP regression
        self.target_model = self._resolve_target_model(target_model)

        # Initialize BO
        self.n_initial_evidence = self._resolve_n_initial_evidence(n_initial_evidence)
        self.acquisition_method = self._resolve_acquisition_method(acquisition_method)

        # Initialize state dictionary
        self.state['n_evidence'] = 0
        self.state['last_GP_update'] = self.n_initial_evidence

        # Initialize classifier attributes list
        self.classifier_attributes = []

    @property
    def n_evidence(self):
        """Return the number of acquired evidence points."""
        return self.state['n_evidence']

    def set_objective(self, n_evidence):
        """Set an objective for inference. You can continue BO by giving a larger n_evidence.

        Parameters
        ----------
        n_evidence: int
            Number of total evidence for the GP fitting. This includes any initial evidence.

        """
        if n_evidence < self.n_evidence:
            logger.warning('Requesting less evidence than there already exists.')
        self.objective['n_sim'] = n_evidence

    def extract_result(self):
        """Extract the results from the current state."""
        return BOLFIREPosterior(self.parameter_names,
                                self.target_model,
                                self.prior,
                                self.classifier_attributes)

    def update(self, batch, batch_index):
        """Update the GP regression model of the target node with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int
            Index of batch.

        """
        super(BOLFIRE, self).update(batch, batch_index)

        # Predict log-ratio
        likelihood = self._generate_likelihood(self._get_parameter_values(batch))
        X, y = self._generate_training_data(likelihood, self.marginal)
        negative_log_ratio_value = -1 * self.predict_log_ratio(X, y, self.observed)

        # Update classifier attributes list
        self.classifier_attributes += [self.classifier.attributes]

        # BO part
        self.state['n_evidence'] += self.batch_size
        parameter_values = batch_to_arr2d(batch, self.parameter_names)
        optimize = self._should_optimize()
        self.target_model.update(parameter_values, negative_log_ratio_value, optimize)
        if optimize:
            self.state['last_GP_update'] = self.target_model.n_evidence

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        Parameters
        ----------
        batch_index: int

        Returns
        -------
        batch: dict

        """
        t = batch_index - self.n_initial_evidence
        if t < 0:  # Sample parameter values from the model priors
            return

        # Acquire parameter values from the acquisition function
        acquisition = self.acquisition_method.acquire(self.batch_size, t)
        return arr2d_to_batch(acquisition, self.parameter_names)

    def predict_log_ratio(self, X, y, X_obs):
        """Predict the log-ratio, i.e, logarithm of likelihood / marginal.

        Parameters
        ----------
        X: np.ndarray
            Training data features.
        y: np.ndarray
            Training data labels.
        X_obs: np.ndarray
            Observed data.

        Returns
        -------
        np.ndarray

        """
        self.classifier.fit(X, y)
        return self.classifier.predict_log_likelihood_ratio(X_obs)

    def fit(self, n_evidence, bar=True):
        """Fit the surrogate model.

        That is, generate a regression model for the negative posterior value given the parameters.
        Currently only GP regression are supported as surrogate models.

        Parameters
        ----------
        n_evidence: int
            Number of evidence for fitting.
        bar: bool, optional
            Flag to show or hide the progress bar during fit.

        Returns
        -------
        BOLFIREPosterior

        """
        logger.info('BOLFIRE: Fitting the surrogate model...')
        if isinstance(n_evidence, int) and n_evidence > 0:
            return self.infer(n_evidence, bar=bar)
        raise TypeError('n_evidence must be a positive integer.')

    def sample(self,
               n_samples,
               warmup=None,
               n_chains=4,
               initials=None,
               algorithm='nuts',
               sigma_proposals=None,
               n_evidence=None,
               *args, **kwargs):
        """Sample from the posterior distribution of BOLFIRE.

        Sampling is performed with an MCMC sampler.

        Parameters
        ----------
        n_samples: int
            Number of requested samples from the posterior for each chain. This includes warmup,
            and note that the effective sample size is usually considerably smaller.
        warmup: int, optional
            Length of warmup sequence in MCMC sampling.
        n_chains: int, optional
            Number of independent chains.
        initials: np.ndarray (n_chains, n_params), optional
            Initial values for the sampled parameters for each chain.
        algorithm: str, optional
            Sampling algorithm to use.
        sigma_proposals: np.ndarray
            Standard deviations for Gaussian proposals of each parameter for Metropolis-Hastings.
        n_evidence: int, optional
            If the surrogate model is not fitted yet, specify the amount of evidence.

        Returns
        -------
        BOLFIRESample

        """
        # Fit posterior in case not done
        if self.state['n_batches'] == 0:
            self.fit(n_evidence)

        # Check algorithm
        if algorithm not in ['nuts', 'metropolis']:
            raise ValueError('The given algorithm is not supported.')

        # Check standard deviations of Gaussian proposals when using Metropolis-Hastings
        if algorithm == 'metropolis':
            if sigma_proposals is None:
                raise ValueError('Gaussian proposal standard deviations have '
                                 'to be provided for Metropolis-sampling.')
            elif sigma_proposals.shape[0] != self.target_model.input_dim:
                raise ValueError('The length of Gaussian proposal standard '
                                 'deviations must be n_params.')

        posterior = self.extract_result()
        warmup = warmup or n_samples // 2

        # Unless given, select the evidence points with best likelihood ratio
        if initials is not None:
            if np.asarray(initials).shape != (n_chains, self.target_model.input_dim):
                raise ValueError('The shape of initials must be (n_chains, n_params).')
        else:
            inds = np.argsort(self.target_model.Y[:, 0])
            initials = np.asarray(self.target_model.X[inds])

        # Enable caching for default RBF kernel
        self.target_model.is_sampling = True

        tasks_ids = []
        ii_initial = 0
        for ii in range(n_chains):
            seed = get_sub_seed(self.seed, ii)
            # Discard bad initialization points
            while np.isinf(posterior.logpdf(initials[ii_initial])):
                ii_initial += 1
                if ii_initial == len(inds):
                    raise ValueError('BOLFIRE.sample: Cannot find enough acceptable '
                                     'initialization points!')

            if algorithm == 'nuts':
                tasks_ids.append(
                    self.client.apply(mcmc.nuts,
                                      n_samples,
                                      initials[ii_initial],
                                      posterior.logpdf,
                                      posterior.gradient_logpdf,
                                      n_adapt=warmup,
                                      seed=seed,
                                      **kwargs))

            elif algorithm == 'metropolis':
                tasks_ids.append(
                    self.client.apply(mcmc.metropolis,
                                      n_samples,
                                      initials[ii_initial],
                                      posterior.logpdf,
                                      sigma_proposals,
                                      warmup,
                                      seed=seed,
                                      **kwargs))

            ii_initial += 1

        # Get results from completed tasks or run sampling (client-specific)
        chains = []
        for id in tasks_ids:
            chains.append(self.client.get_result(id))

        chains = np.asarray(chains)

        logger.info(f'{n_chains} chains of {n_samples} iterations acquired. '
                    'Effective sample size and Rhat for each parameter:')
        for ii, node in enumerate(self.parameter_names):
            logger.info(f'{node} {mcmc.eff_sample_size(chains[:, :, ii])} '
                        f'{mcmc.gelman_rubin_statistic(chains[:, :, ii])}')

        self.target_model.is_sampling = False

        return BOLFIRESample(method_name='BOLFIRE',
                             chains=chains,
                             parameter_names=self.parameter_names,
                             warmup=warmup,
                             n_sim=self.state['n_sim'],
                             seed=self.seed,
                             *args, **kwargs)

    def _resolve_model(self, model):
        """Resolve a given elfi model."""
        if not isinstance(model, ElfiModel):
            raise ValueError('model must be an ElfiModel.')
        if len(self._get_summary_names(model)) == 0:
            raise NotImplementedError('model must have at least one Summary node.')
        return model

    def _resolve_n_training_data(self, n_training_data):
        """Resolve the size of training data to be used."""
        if isinstance(n_training_data, int) and n_training_data > 0:
            return n_training_data
        raise TypeError('n_training_data must be a positive int.')

    def _get_summary_names(self, model):
        """Return the names of summary statistics."""
        return [node for node in model.nodes if isinstance(model[node], Summary)
                and not node.startswith('_')]

    def _resolve_marginal(self, marginal, seed_marginal=None):
        """Resolve marginal data."""
        if marginal is None:
            marginal = self._generate_marginal(seed_marginal)
            x, y = marginal.shape
            logger.info(f'New marginal data ({x} x {y}) are generated.')
            return marginal
        if isinstance(marginal, np.ndarray) and len(marginal.shape) == 2:
            return marginal
        raise TypeError('marginal must be 2d numpy array.')

    def _generate_marginal(self, seed_marginal=None):
        """Generate marginal data."""
        batch = self.model.generate(self.n_training_data,
                                    outputs=self.summary_names,
                                    seed=seed_marginal)
        return np.column_stack([batch[summary_name] for summary_name in self.summary_names])

    def _generate_likelihood(self, parameter_values):
        """Generate likelihood data."""
        batch = self.model.generate(self.n_training_data,
                                    outputs=self.summary_names,
                                    with_values=parameter_values)
        return np.column_stack([batch[summary_name] for summary_name in self.summary_names])

    def _generate_training_data(self, likelihood, marginal):
        """Generate training data."""
        X = np.vstack((likelihood, marginal))
        y = np.concatenate((np.ones(likelihood.shape[0]), -1 * np.ones(marginal.shape[0])))
        return X, y

    def _resolve_classifier(self, classifier):
        """Resolve classifier."""
        if classifier is None:
            return LogisticRegression()
        if isinstance(classifier, Classifier):
            return classifier
        raise ValueError('classifier must be an instance of Classifier.')

    def _get_observed_summary_values(self, model, summary_names):
        """Return observed values for summary statistics."""
        return np.column_stack([model[summary_name].observed for summary_name in summary_names])

    def _get_parameter_values(self, batch):
        """Return parameter values from a given batch."""
        return {parameter_name: float(batch[parameter_name]) for parameter_name
                in self.model.parameter_names}

    def _resolve_n_initial_evidence(self, n_initial_evidence):
        """Resolve number of initial evidence."""
        if isinstance(n_initial_evidence, int) and n_initial_evidence >= 0:
            return n_initial_evidence
        raise ValueError('n_initial_evidence must be a non-negative integer.')

    def _resolve_target_model(self, target_model):
        """Resolve target model."""
        if target_model is None:
            return GPyRegression(self.model.parameter_names, self.bounds)
        if isinstance(target_model, GPyRegression):
            return target_model
        raise TypeError('target_model must be an instance of GPyRegression.')

    def _resolve_acquisition_method(self, acquisition_method):
        """Resolve acquisition method."""
        if acquisition_method is None:
            return LCBSC(model=self.target_model,
                         prior=self.prior,
                         noise_var=self.acq_noise_var,
                         exploration_rate=self.exploration_rate,
                         seed=self.seed,
                         include_prior=True)
        if isinstance(acquisition_method, AcquisitionBase):
            return acquisition_method
        raise TypeError('acquisition_method must be an instance of AcquisitionBase.')

    def _should_optimize(self):
        """Check whether GP hyperparameters should be optimized."""
        current = self.target_model.n_evidence + self.batch_size
        next_update = self.state['last_GP_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update
