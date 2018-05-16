"""Example implementation of the Lorenz forecast model.
References
----------
- Ritabrata Dutta, Jukka Corander, Samuel Kaski, and Michael U. Gutmann.
  Likelihood-free inference by ratio estimation
"""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi
from elfi.examples.ma2 import autocov


def lorenz_ode(y, params, batch_size=1):
    """
    Generate samples from the stochastic Lorenz model.

    Parameters
    ----------
    y : numpy.ndarray of dimension px1
        The value of timeseries where we evaluate the ODE.
    params : list
        The list of parameters needed to evaluate function. In this case it is
        list of two elements - eta and theta.
    batch_size : int, optional

    Returns
    -------
    dY_dt : np.array
        ODE for further application.
    """

    dY_dt = np.zeros(shape=(batch_size, y.shape[1]))

    eta = params[0]
    theta1 = params[1]
    theta2 = params[2]
    theta = np.array([[theta1, theta2]])
    F = params[3]

    y_1 = np.ones(shape=y.shape)

    y_k = np.array([y_1, pow(y, 1)])

    g = np.sum(y_k.T * theta, axis=1).T

    dY_dt[:, 0] = (-y[:, -2] * y[:, -1] + y[:, -1] * y[:, 1] - y[:, 0] +
                   F - g[:, 0] + eta[:, 0])
    dY_dt[:, 1] = (-y[:, -1] * y[:, 0] + y[:, 0] * y[:, 2] - y[:, 1] + F -
                   g[:, 1] + eta[:, 1])

    dY_dt[:, 2:-1] = (-y[:, -3] * y[:, 1:-2] + y[:, 1:-2] * y[:, 3:] -
                      y[:, 2:-1] + F - g[:, 2:-1] + eta[:, 2:-1])

    dY_dt[:, -1] = (-y[:, -3] * y[:, -2] + y[:, -2] * y[:, 0] - y[:, -1] + F -
                    g[:, -1] + eta[:, -1])

    return dY_dt


def runge_kutta_ode_solver(ode, timespan, timeseries_initial, params,
                           batch_size=1):
    """
    4th order Runge-Kutta ODE solver. For more description see section 6.5 at:
    Carnahan, B., Luther, H. A., and Wilkes, J. O. (1969).
    Applied Numerical Methods. Wiley, New York.

    Parameters
    ----------
    ode : function
        Ordinary differential equation function
    timespan : numpy.ndarray
        Contains the time points where the ode needs to be
        solved. The first time point corresponds to the initial value
    timeseries_initial : np.ndarray of dimension px1
        Initial value of the time-series, corresponds to the first value of
        timespan
    params : list of parameters
        The parameters needed to evaluate the ode, i.e. eta and theta
    batch_size : int, optional

    Returns
    -------
    np.ndarray
        Timeseries initiated at timeseries_init and satisfying ode solved by
        this solver.
    """

    time_diff = timespan

    k1 = time_diff * ode(timeseries_initial, params, batch_size)

    k2 = time_diff * ode(timeseries_initial + k1 / 2, params, batch_size)

    k3 = time_diff * ode(timeseries_initial + k2 / 2, params, batch_size)

    k4 = time_diff * ode(timeseries_initial + k3, params, batch_size)

    timeseries_initial = timeseries_initial + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return timeseries_initial


def forecast_lorenz(theta1=None, theta2=None, F=10.,
                    phi=0.4, dim=40, n_timestep=160, batch_size=1,
                    initial_state=None, random_state=None):
    """
    The forecast Lorenz model.
    Wilks, D. S. (2005). Effects of stochastic parametrizations in the
    Lorenz ’96 system. Quarterly Journal of the Royal Meteorological Society,
    131(606), 389–407.

    Parameters
    ----------
    theta1, theta2: list or numpy.ndarray, optional
        Closure parameters. If the parameter is omitted, sampled
        from the prior.
    phi : float, optional
    initial_state: numpy.ndarray, optional
        Initial state value of the time-series. The default value is None,
        which assumes a previously computed value from a full Lorenz model as
        the Initial value.
    F : float
        Force term. The default value is 10.0.

    Returns
    -------
    np.ndarray
        Timeseries initiated at timeseries_arr and satisfying ode.
    """

    if not initial_state:
        initial_state = np.zeros(shape=(batch_size, dim))

    y = initial_state

    timestep = 4 / n_timestep

    random_state = random_state or np.random.RandomState(batch_size)

    e = random_state.normal(0, 1, (batch_size, initial_state.shape[1]))

    eta = np.sqrt(1 - pow(phi, 2)) * e

    for i in range(n_timestep):
        params = (eta, theta1, theta2, F)
        y = runge_kutta_ode_solver(ode=lorenz_ode,
                                   timespan=timestep,
                                   timeseries_initial=y,
                                   params=params,
                                   batch_size=batch_size)

        eta = phi * eta + e * np.sqrt(1 - pow(phi, 2))

    return y


def get_model(true_params=None, seed_obs=None, initial_state=None, dim=40,
              F=10.):
    """Return a complete Lorenz model in inference task.
    This is a simplified example that achieves reasonable predictions.
    For more extensive treatment and description using, see:
    Hakkarainen, J., Ilin, A., Solonen, A., Laine, M., Haario, H., Tamminen,
    J., Oja, E., and Järvinen, H. (2012). On closure parameter estimation in
    chaotic systems. Nonlinear Processes in Geophysics, 19(1), 127–143.

    Parameters
    ----------
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.
    initial_state : ndarray

    Returns
    -------
    m : elfi.ElfiModel
    """

    simulator = partial(forecast_lorenz, initial_state=initial_state,
                        F=F, dim=dim)

    if not true_params:
        true_params = [2.1, .1]

    m = elfi.ElfiModel()

    sim_fn = elfi.tools.vectorize(simulator)

    y_obs = sim_fn(*true_params,
                   random_state=np.random.RandomState(seed_obs))
    sumstats = []

    elfi.Prior(ss.uniform, 0.5, 3., model=m, name='theta1')
    elfi.Prior(ss.uniform, 0, 0.3, model=m, name='theta2')
    elfi.Simulator(sim_fn, m['theta1'], m['theta2'], observed=y_obs,
                   name='Lorenz')
    sumstats.append(
        elfi.Summary(partial(np.mean, axis=1), m['Lorenz'], name='Mean'))
    sumstats.append(
        elfi.Summary(partial(np.var, axis=1), m['Lorenz'], name='Var'))
    sumstats.append(
        elfi.Summary(autocov, m['Lorenz'], name='Autocov'))

    elfi.Discrepancy(cost_function, *sumstats, name='d')

    return m


def cost_function(*simulated, observed):
    """Define cost function as in Hakkarainen et al. (2012).

    Parameters
    ----------
    observed : tuple of np.arrays
    simulated : np.arrays

    Returns
    -------
    c : ndarray
        The calculated cost function
    """
    simulated = np.column_stack(simulated)
    observed = np.column_stack(observed)

    return np.sum((simulated - observed) ** 2. / observed, axis=1)
