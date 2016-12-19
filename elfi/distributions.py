# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
import numpy.random as npr
from functools import partial

from . import core


# TODO: combine with `simulator_wrapper` and perhaps with `summary_wrapper`?
def random_wrapper(operation, input_dict):
    """Provides operation a RandomState object for generating random quantities.

    Parameters
    ----------
    operation: callable(*parent_data, batch_size, random_state)
        parent_data : numpy array
        batch_size : number of simulations to perform
        random_state : RandomState object
    input_dict: dict
        ELFI input_dict for transformations

    Notes
    -----
    It is crucial to use the provided RandomState object for generating the random
    quantities when running the simulator. This ensures that results are reproducible and
    inference will be valid.

    If the simulator is implemented in another language, one should extract the state
    of the random_state and use it in generating the random numbers.

    """
    prng = npr.RandomState(0)
    prng.set_state(input_dict['random_state'])
    batch_size = input_dict["n"]
    data = operation(*input_dict['data'], batch_size=batch_size, random_state=prng)
    return core.to_output_dict(input_dict, data=data, random_state=prng.get_state())


class Distribution:
    """Abstract class for an ELFI compatible random distribution.

    Note that the class signature is a subset of that of `scipy.rv_continuous`
    """

    def __init__(self, name=None):
        """

        Parameters
        ----------
        name : name of the distribution
        """
        self.name = name or self.__class__.__name__

    def rvs(self, *params, size=(1,), random_state):
        """Random variates

        Parameters
        ----------
        param1, param2, ... : array_like
            Parameter(s) of the distribution
        size : int or tuple of ints, optional
        random_state : RandomState

        Returns
        -------
        rvs : ndarray
            Random variates of given size.
        """
        raise NotImplementedError

    def pdf(self, x, *params, **kwargs):
        """Probability density function at x

        Parameters
        ----------
        x : array_like
           points where to evaluate the pdf
        param1, param2, ... : array_like
           parameters of the model

        Returns
        -------
        pdf : ndarray
           Probability density function evaluated at x
        """
        raise NotImplementedError

    def logpdf(self, x, *params, **kwargs):
        """Log of the probability density function at x.

        Parameters
        ----------
        x : array_like
            points where to evaluate the pdf
        param1, param2, ... : array_like
            parameters of the model
        kwargs

        Returns
        -------
        pdf : ndarray
           Log of the probability density function evaluated at x
        """
        raise NotImplementedError


# TODO: this might be needed for rv_discrete instances in the future?
class ScipyDistribution(Distribution):

    # Convert some common names to scipy equivalents
    ALIASES = {'normal': 'norm',
               'exponential': 'expon',
               'unif': 'uniform',
               'bin': 'binom',
               'binomial': 'binom'}

    def __init__(self, distribution):
        if isinstance(distribution, str):
            distribution = self.__class__.from_str(distribution)
        elif not isinstance(distribution, (ss.rv_continuous, ss.rv_discrete)):
            raise ValueError("Unknown distribution type {}".format(distribution))

        self.ss_distribution = distribution

        name = distribution.name
        super(ScipyDistribution, self).__init__(name=name)

    def rvs(self, *params, size=1, random_state=None):
        return self.ss_distribution.rvs(*params, size=size, random_state=random_state)

    def pdf(self, x, *params, **kwargs):
        """Probability density function at x of the given RV.
        """
        if self.is_discrete:
            return self.ss_distribution.pmf(x, *params, **kwargs)
        else:
            return self.ss_distribution.pdf(x, *params, **kwargs)

    def logpdf(self, x, *params, **kwargs):
        """Log probability density function at x of the given RV.
        """
        if self.is_discrete:
            return self.ss_distribution.logpmf(x, *params, **kwargs)
        else:
            return self.ss_distribution.logpdf(x, *params, **kwargs)

    def cdf(self, x, *params, **kwargs):
        """Cumulative scipy_distribution function of the given RV.
        """
        return self.ss_distribution.cdf(x, *params, **kwargs)

    @property
    def is_discrete(self):
        return isinstance(self.ss_distribution, ss.rv_discrete)

    @classmethod
    def from_str(cls, name):
        name = name.lower()
        name = cls.ALIASES.get(name, name)
        return getattr(ss, name)


def rvs_operation(*params, batch_size=1, random_state, distribution, size=(1,)):
    size = (batch_size,) + size
    return distribution.rvs(*params, size=size, random_state=random_state)


class RandomVariable(core.RandomStateMixin, core.Operation):
    """

    Parameters
    ----------
    distribution : string or Distribution
        string is interpreted as an equivalent scipy distribution
    size : tuple or int
        Size of the RV output

    Examples
    --------
    RandomVariable('tau', scipy.stats.norm, 5, size=(2,3))
    """

    operation_wrapper = random_wrapper

    def _prepare_operation(self, distribution, size=(1,), **kwargs):
        if isinstance(distribution, str):
            distribution = ScipyDistribution.from_str(distribution)
        if not hasattr(distribution, 'rvs'):
            raise ValueError("Distribution {} must implement rvs method".format(distribution))

        if not isinstance(size, tuple):
            size = (size,)

        self.distribution = distribution
        return partial(rvs_operation, distribution=distribution, size=size)

    def __str__(self):
        d = self.distribution
        if hasattr(d, 'name'):
            name = d.name
        elif isinstance(d, type):
            name = self.distribution.__name__
        else:
            name = self.distribution.__class__.__name__

        return super(RandomVariable, self).__str__()[0:-1] + \
               ", '{}')".format(name)


class Prior(RandomVariable):
    def __init__(self, name, distribution="uniform", *args, **kwargs):
        super(Prior, self).__init__(name, distribution, *args, **kwargs)


class Model(core.ObservedMixin, RandomVariable):
    def __init__(self, *args, observed=None, size=None, **kwargs):
        if observed is None:
            raise ValueError('Observed cannot be None')
        if size is None:
            size = observed.shape
        super(Model, self).__init__(*args, observed=observed, size=size, **kwargs)


# TODO: Fixme
class SMC_Distribution():
    """Distribution that samples near previous values of parameters by sampling
    Gaussian distributions centered at previous values.

    Used in SMC ABC as priors for subsequent particle populations.
    """

    def rvs(current_params, weighted_sd, weights, random_state, size=1):
        """Random value source

        Parameters
        ----------
        current_params : 2D np.ndarray
            Expected values for samples. Shape should match weights.
        weighted_sd : float
            Weighted standard deviation to use for the Gaussian.
        weights : 2D np.ndarray
            The probability of selecting each current_params. Shape should match current_params.
        random_state : np.random.RandomState
        size : tuple

        Returns
        -------
        params : np.ndarray
            shape == size
        """
        a = np.arange(current_params.shape[0])
        p = weights[:,0]
        selections = random_state.choice(a=a, size=size, p=p)
        selections = selections[:,0]
        params = current_params[selections]
        noise = ss.norm.rvs(scale=weighted_sd, size=size, random_state=random_state)
        params += noise
        return params

    def pdf(params, current_params, weighted_sd, weights):
        """Probability density function, which is here that of a Gaussian.
        """
        return ss.norm.pdf(params, current_params, weighted_sd)
