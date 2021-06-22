"""Example implementation of the ARCH(1) model."""

import logging
from itertools import combinations

import numpy as np

import elfi

logger = logging.getLogger(__name__)


def get_model(n_obs=100, true_params=None, seed_obs=None, n_lags=5):
    """Return a complete ARCH(1) model.

    Parameters
    ----------
    n_obs: int
        Observation length of the ARCH(1) process.
    true_params: list, optinal
        Parameters with which the observed data are generated.
    seed_obs: int, optional
        Seed for the observed data generation.
    n_lags: int, optional
        Number of lags in summary statistics.

    Returns
    -------
    elfi.ElfiModel

    """
    if true_params is None:
        true_params = [0.3, 0.7]
        logger.info(f'true_params were not given. Now using [t1, t2] = {true_params}.')

    # elfi model
    m = elfi.ElfiModel()

    # priors
    t1 = elfi.Prior('uniform', -1, 2, model=m)
    t2 = elfi.Prior('uniform',  0, 1, model=m)
    priors = [t1, t2]

    # observations
    y_obs = arch(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))

    # simulator
    Y = elfi.Simulator(arch, *priors, observed=y_obs)

    # summary statistics
    elfi.Summary(sample_mean, Y, name='MU', model=m)
    elfi.Summary(sample_variance, Y, name='VAR', model=m)
    for i in range(1, n_lags + 1):
        elfi.Summary(autocorr, Y, i, name=f'AC_{i}', model=m)
    for i, j in combinations(range(1, n_lags + 1), 2):
        elfi.Summary(pairwise_autocorr, Y, i, j, name=f'PW_{i}_{j}', model=m)

    return m


def arch(t1, t2, n_obs=100, batch_size=1, random_state=None):
    """Generate a sequence of samples from the ARCH(1) model.

    Parameters
    ----------
    t1: float
        Mean process parameter in the ARCH(1) process.
    t2: float
        Variance process parameter in the ARCH(1) process.
    n_obs: int, optional
        Observation length of the ARCH(1) process.
    batch_size: int, optional
        Number of simulations.
    random_state: np.random.RandomState, optional

    Returns
    -------
    np.ndarray

    """
    random_state = random_state or np.random
    y = np.zeros((batch_size, n_obs + 1))
    e = E(t2, n_obs, batch_size, random_state)
    for i in range(1, n_obs + 1):
        y[:, i] = t1 * y[:, i - 1] + e[:, i]
    return y[:, 1:]


def E(t2, n_obs=100, batch_size=1, random_state=None):
    """Variance process function in the ARCH(1) model.

    Parameters
    ----------
    t2: float
        Variance process parameter in the ARCH(1) process.
    n_obs: int, optional
        Observation length of the ARCH(1) process.
    batch_size: int, optional
        Number of simulations.
    random_state: np.random.RandomState

    Returns
    -------
    np.ndarray

    """
    random_state = random_state or np.random
    xi = random_state.normal(size=(batch_size, n_obs + 1))
    e = np.zeros((batch_size, n_obs + 1))
    e[:, 0] = random_state.normal(size=batch_size)
    for i in range(1, n_obs + 1):
        e[:, i] = xi[:, i] * np.sqrt(0.2 + t2 * np.power(e[:, i - 1], 2))
    return e


def sample_mean(x):
    """Calculate the sample mean.

    Parameters
    ----------
    x: np.ndarray
        Simulated/observed data.

    Returns
    -------
    np.ndarray

    """
    return np.mean(x, axis=1)


def sample_variance(x):
    """Calculate the sample variance.

    Parameters
    ----------
    x: np.ndarray
        Simulated/observed data.

    Returns
    -------
    np.ndarray

    """
    return np.var(x, axis=1, ddof=1)


def autocorr(x, lag=1):
    """Calculate the autocorrelation.

    Parameters
    ----------
    x: np.ndarray
        Simulated/observed data.
    lag: int, optional
        Lag in autocorrelation.

    Returns
    -------
    np.ndarray

    """
    n = x.shape[1]
    x_mu = np.mean(x, axis=1)
    x_std = np.std(x, axis=1, ddof=1)
    sc_x = ((x.T - x_mu) / x_std).T
    C = np.sum(sc_x[:, lag:] * sc_x[:, :-lag], axis=1) / (n - lag)
    return C


def pairwise_autocorr(x, lag_i=1, lag_j=1):
    """Calculate the pairwise autocorrelation.

    Parameters
    x: np.ndarray
        Simulated/observed data.
    lag_i: int, optional
        Lag in autocorrelation.
    lag_j: int, optinal
        Lag in autocorrelation.

    Returns
    -------
    np.ndarray

    """
    ac_i = autocorr(x, lag_i)
    ac_j = autocorr(x, lag_j)
    return ac_i * ac_j
