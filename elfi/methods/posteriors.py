"""The module contains implementations of approximate posteriors."""

import logging
from collections import OrderedDict
from multiprocessing import Pool
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from elfi.methods.bo.utils import minimize
from elfi.methods.utils import NDimBoundingBox
from elfi.model.extensions import ModelPrior
from elfi.visualization.visualization import ProgressBar

logger = logging.getLogger(__name__)


# TODO: separate the likelihood to its own class
class BolfiPosterior:
    r"""Container for the approximate posterior in the BOLFI framework.

    Here the likelihood is defined as

    L \propto F((h - \mu) / \sigma)

    where F is the cdf of N(0,1), h is a threshold, and \mu and \sigma are the mean and (noisy)
    standard deviation of the Gaussian process.

    Note that when using a log discrepancy, h should become log(h).

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

    """

    def __init__(self, model, threshold=None, prior=None, n_inits=10, max_opt_iters=1000, seed=0):
        """Initialize a BOLFI posterior.

        Parameters
        ----------
        model : elfi.bo.gpy_regression.GPyRegression
            Instance of the surrogate model
        threshold : float, optional
            The threshold value used in the calculation of the posterior, see the BOLFI paper
            for details. By default, the minimum value of discrepancy estimate mean is used.
        prior : ScipyLikeDistribution, optional
            By default uniform distribution within model bounds.
        n_inits : int, optional
            Number of initialization points in internal optimization.
        max_opt_iters : int, optional
            Maximum number of iterations performed in internal optimization.
        seed : int, optional

        """
        super(BolfiPosterior, self).__init__()
        self.threshold = threshold
        self.model = model
        self.random_state = np.random.RandomState(seed)
        self.n_inits = n_inits
        self.max_opt_iters = max_opt_iters

        self.prior = prior
        self.dim = self.model.input_dim

        if self.threshold is None:
            # TODO: the evidence could be used for a good guess for starting locations
            minloc, minval = minimize(
                self.model.predict_mean,
                self.model.bounds,
                grad=self.model.predictive_gradient_mean,
                prior=self.prior,
                n_start_points=self.n_inits,
                maxiter=self.max_opt_iters,
                random_state=self.random_state)
            self.threshold = minval
            logger.info("Using optimized minimum value (%.4f) of the GP discrepancy mean "
                        "function as a threshold" % (self.threshold))

    def rvs(self, size=None, random_state=None):
        """Sample the posterior.

        Currently unimplemented. Please use a sampler to sample from the posterior.
        """
        raise NotImplementedError('Currently not implemented. Please use a sampler to '
                                  'sample from the posterior.')

    def logpdf(self, x):
        """Return the unnormalized log-posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        float

        """
        return self._unnormalized_loglikelihood(x) + self.prior.logpdf(x)

    def pdf(self, x):
        """Return the unnormalized posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        float

        """
        return np.exp(self.logpdf(x))

    def gradient_logpdf(self, x):
        """Return the gradient of the unnormalized log-posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        np.array

        """
        grads = self._gradient_unnormalized_loglikelihood(x) + \
            self.prior.gradient_logpdf(x)

        # nan grads are result from -inf logpdf
        # return np.where(np.isnan(grads), 0, grads)[0]
        return grads

    def _unnormalized_loglikelihood(self, x):
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        logpdf = -np.ones(len(x)) * np.inf

        logi = self._within_bounds(x)
        x = x[logi, :]
        if len(x) == 0:
            if ndim == 0 or (ndim == 1 and self.dim > 1):
                logpdf = logpdf[0]
            return logpdf

        mean, var = self.model.predict(x)
        logpdf[logi] = ss.norm.logcdf(
            self.threshold, mean, np.sqrt(var)).squeeze()

        if ndim == 0 or (ndim == 1 and self.dim > 1):
            logpdf = logpdf[0]

        return logpdf

    def _gradient_unnormalized_loglikelihood(self, x):
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        grad = np.zeros_like(x)

        logi = self._within_bounds(x)
        x = x[logi, :]
        if len(x) == 0:
            if ndim == 0 or (ndim == 1 and self.dim > 1):
                grad = grad[0]
            return grad

        mean, var = self.model.predict(x)
        std = np.sqrt(var)

        grad_mean, grad_var = self.model.predictive_gradients(x)

        factor = -grad_mean * std - \
            (self.threshold - mean) * 0.5 * grad_var / std
        factor = factor / var
        term = (self.threshold - mean) / std
        pdf = ss.norm.pdf(term)
        cdf = ss.norm.cdf(term)

        grad[logi, :] = factor * pdf / cdf

        if ndim == 0 or (ndim == 1 and self.dim > 1):
            grad = grad[0]

        return grad

    # TODO: check if these are used
    def _unnormalized_likelihood(self, x):
        return np.exp(self._unnormalized_loglikelihood(x))

    def _neg_unnormalized_loglikelihood(self, x):
        return -1 * self._unnormalized_loglikelihood(x)

    def _gradient_neg_unnormalized_loglikelihood(self, x):
        return -1 * self._gradient_unnormalized_loglikelihood(x)

    def _neg_unnormalized_logposterior(self, x):
        return -1 * self.logpdf(x)

    def _gradient_neg_unnormalized_logposterior(self, x):
        return -1 * self.gradient_logpdf(x)

    def _within_bounds(self, x):
        x = x.reshape((-1, self.dim))
        logical = np.ones(len(x), dtype=bool)
        for i in range(self.dim):
            logical *= (x[:, i] >= self.model.bounds[i][0])
            logical *= (x[:, i] <= self.model.bounds[i][1])
        return logical

    def plot(self, logpdf=False):
        """Plot the posterior pdf.

        Currently only supports 1 and 2 dimensional cases.

        Parameters
        ----------
        logpdf : bool
            Whether to plot logpdf instead of pdf.

        """
        if logpdf:
            fun = self.logpdf
        else:
            fun = self.pdf

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')

            if len(self.model.bounds) == 1:
                mn = self.model.bounds[0][0]
                mx = self.model.bounds[0][1]
                dx = (mx - mn) / 200.0
                x = np.arange(mn, mx, dx)
                pd = np.zeros(len(x))
                for i in range(len(x)):
                    pd[i] = fun([x[i]])
                plt.figure()
                plt.plot(x, pd)
                plt.xlim(mn, mx)
                plt.ylim(min(pd) * 1.05, max(pd) * 1.05)
                plt.show()

            elif len(self.model.bounds) == 2:
                x, y = np.meshgrid(
                    np.linspace(*self.model.bounds[0]), np.linspace(*self.model.bounds[1]))
                z = (np.vectorize(lambda a, b: fun(np.array([a, b]))))(x, y)
                plt.contour(x, y, z)
                plt.show()

            else:
                raise NotImplementedError("Currently unsupported for dim > 2")


class BOLFIREPosterior:
    """Results from BOLFIRE inference method."""

    def __init__(self,
                 parameter_names,
                 model,
                 prior,
                 classifier_attributes,
                 n_opt_inits=10,
                 max_opt_iters=1000,
                 *args, **kwargs):
        """Initialize BOLFIREPosterior.

        Parameters
        ----------
        parameter_names: list
            Names of the parameter nodes.
        model: elfi.bo.gpy_regression.GPyRegression
            Instance of the surrogate model
        prior: elfi.methods.utils.ModelPrior
            Joint prior of the elfi model.
        classifier_attributes: list
            Classifier's attributes on each inference round.
        n_opt_inits: int, optional
            Number of initialization points in internal optimization.
        max_opt_iters: int, optional
            Maximum number of iterations performed in internal optimization.

        """
        self._parameter_names = parameter_names
        self._model = model
        self._prior = prior
        self._n_opt_inits = n_opt_inits
        self._max_opt_iters = max_opt_iters
        self._classifier_attributes = classifier_attributes

        # compute map estimates
        self._map_estimates = self._compute_map_estimates()

    @property
    def map_estimates(self):
        """Return the maximum a posterior estimate for each parameter.

        Returns
        -------
        OrderedDict

        """
        return OrderedDict([(parameter_name, self._map_estimates[i]) for i, parameter_name
                            in enumerate(self._parameter_names)])

    @property
    def classifier_attributes(self):
        """Return the classifier's attributes."""
        return self._classifier_attributes

    @property
    def surrogate_model_attributes(self):
        """Return the surrogate model's (GP regression) attributes."""
        return {
            'parameters': self._model._gp.param_array.tolist(),
            'X': self._model.X.tolist(),
            'Y': self._model.Y.tolist()
        }

    def pdf(self, x):
        """Return the unnormalized posterior at x.

        Parameters
        ----------
        x: np.ndarray

        Returns
        -------
        np.ndarray

        """
        return np.exp(self.logpdf(x))

    def _negative_pdf(self, x):
        """Return the negative unnormalized posterior at x."""
        return -1 * self.pdf(x)

    def logpdf(self, x):
        """Return the unnormalized log-posterior at x.

        Parameters
        ----------
        x: np.ndarray

        Returns
        -------
        np.ndarray

        """
        return self._prior.logpdf(x).reshape(-1, 1) - self._model.predict_mean(x)

    def gradient_pdf(self, x):
        """Return the gradient of the unnormalized posterior pdf at x.

        Parameters
        ----------
        x: np.ndarray

        Returns
        -------
        np.ndarray

        """
        return np.exp(self.logpdf(x)) * self.gradient_logpdf(x)

    def _negative_gradient_pdf(self, x):
        """Return the negative gradient of the unnormalized posterior pdf at x."""
        return -1 * self.gradient_pdf(x)

    def gradient_logpdf(self, x):
        """Return the gradient of unnormalized log-posterior pdf at x.

        Parameters
        ----------
        x: np.ndarray

        Returns
        -------
        np.ndarray

        """
        return self._prior.gradient_logpdf(x).reshape(1, -1) \
            - self._model.predictive_gradient_mean(x)

    def _compute_map_estimates(self):
        """Return the maximum a posterior estimate for each parameter."""
        minimum_location, _ = minimize(
            fun=self._negative_pdf,
            bounds=self._model.bounds,
            grad=self._negative_gradient_pdf,
            prior=self._prior,
            n_start_points=self._n_opt_inits,
            maxiter=self._max_opt_iters
        )
        return minimum_location


class RomcPosterior:
    r"""
    Approximation of the posterior distribution as defined by the ROMC method.

    References
    ----------
    Ikonomov, B., & Gutmann, M. U. (2019). Robust Optimisation Monte Carlo.
    http://arxiv.org/abs/1904.00670

    """

    def __init__(self,
                 regions: List[NDimBoundingBox],
                 objectives: List[Callable],
                 nuisance: List[int],
                 objectives_unique: List[Callable],
                 prior: ModelPrior,
                 left_lim,
                 right_lim,
                 eps_filter,
                 eps_region,
                 eps_cutoff,
                 parallelize=False):
        """Class constructor.

        Parameters
        ----------
        regions: List[NDimBoundingBox]
            List of the n-dimensional regions
        objectives: List[Callable]
            all the objective functions, equal len with regions.  if an objective function
            produces more than one region, this list repeats the same objective as many times
            as need in symmetry with the regions list.
        nuisance: List[int]
            the seeds used for defining the objectives
        objectives_unique: List[Callable]
            all unique objective functions
        prior: ModelPrior
            the prior distribution
        left_lim: np.ndarray
            left limit
        right_lim: np.ndarray
            right limit
        eps_region: float
            the threshold defining the acceptance region

        """
        self.regions = regions
        self.funcs = objectives
        self.nuisance = nuisance
        self.funcs_unique = objectives_unique
        self.prior = prior
        self.eps_filter = eps_filter
        self.eps_region = eps_region
        self.eps_cutoff = eps_cutoff
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.dim = prior.dim
        self.parallelize = parallelize
        self.partition = None

        self.progress_bar = ProgressBar(prefix='Progress', suffix='Complete',
                                        decimals=1, length=50, fill='=')

    def _pdf_unnorm_single_point(self, theta: np.ndarray) -> float:
        """Evaluate the unnormalised pdf, at a single input point.

        Parameters
        ----------
        theta: (D,)

        Returns
        -------
        evaluation of the unnormalized pdf

        """
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 1

        prior = self.prior
        pr = float(prior.pdf(np.expand_dims(theta, 0)))

        indicator_sum = self._sum_over_indicators(theta)
        # indicator_sum = self._sum_over_regions_indicators(theta)

        val = pr * indicator_sum
        return val

    def _sum_over_indicators(self, theta: np.ndarray) -> int:
        """Evaluate g_i(theta) for all i and count how many obey g_i(theta) <= eps.

        Parameters
        ----------
        theta: np.ndarray, shape: (D,)
          The input point to be evaluated

        """
        funcs = self.funcs_unique
        eps = self.eps_cutoff
        nof_inside = 0
        for i in range(len(funcs)):
            func = funcs[i]
            if func(theta) <= eps:
                nof_inside += 1
        return nof_inside

    def _sum_over_regions(self, theta: np.ndarray) -> int:
        """Count how many n-dimensional regions contain theta.

        Parameters
        ----------
        theta: np.ndarray, shape: (D,)
          The input point to be evaluated

        """
        regions = self.regions

        nof_inside = 0
        for i in range(len(regions)):
            reg = regions[i]
            if reg.contains(theta):
                nof_inside += 1
        return nof_inside

    def _sum_over_regions_indicators(self, theta: np.ndarray) -> int:
        """Count how many n-dimensional regions contain theta.

        Parameters
        ----------
        theta: np.ndarray, shape: (D,)
          The input point to be evaluated

        """
        regions = self.regions
        funcs = self.funcs
        eps = self.eps_cutoff

        nof_inside = 0
        for i in range(len(regions)):
            reg = regions[i]
            func = funcs[i]
            if reg.contains(theta) and (func(theta) <= eps):
                nof_inside += 1
        return nof_inside

    def _worker_eval_unnorm(self, args):
        unnorm, theta = args
        return unnorm(theta)

    def _worker_sample(self, args):
        region, n2 = args
        return region.sample(n2)

    def _worker_compute_weight(self, args):
        i, theta, region, prior, func, eps, n2 = args
        distances = []
        w = []
        for j in range(n2):
            cur_theta = theta[j]
            q = region.pdf(cur_theta)
            if q == 0.0:
                logger.warning("Zero q")
            pr = float(prior.pdf(np.expand_dims(cur_theta, 0)))
            dist = func(cur_theta)
            distances.append(dist)
            ind = dist < eps

            # compute
            if q > 0:
                res = ind * pr / q
            else:
                res = 0

            w.append(res)
        return w, distances

    def pdf_unnorm_batched(self, theta: np.ndarray):
        """Compute the value of the unnormalized posterior in a batched fashion.

        The operation is NOT vectorized.

        Parameters
        ----------
        theta: np.ndarray (BS, D)

        Returns
        -------
        np.array: (BS,)

        """
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 2
        batch_size = theta.shape[0]

        # iterate over all points
        if self.parallelize is False:
            pdf_eval = []
            for i in range(batch_size):
                pdf_eval.append(self._pdf_unnorm_single_point(theta[i]))
        else:
            pool = Pool()
            args = ((self._pdf_unnorm_single_point,
                     theta[i]) for i in range(len(theta)))
            pdf_eval = pool.map(self._worker_eval_unnorm, args)
            pool.close()
            pool.join()
        return np.array(pdf_eval)

    def reset_eps_cutoff(self, eps_cutoff):
        """Reset the threshold for the indicator function.

        Parameters
        ----------
        eps_cutoff : float
            the new threshold

        """
        self.eps_cutoff = eps_cutoff
        self.partition = None

    def _approximate_partition(self, nof_points: int = 30):
        """Approximate Z, computing the integral as a sum.

        Parameters
        ----------
        nof_points: int
            nof points to use in each dimension

        """
        assert 0 <= self.dim <= 2, "Approximate partition implemented only for 1D, 2D case."
        dim = self.dim
        left_lim = self.left_lim
        right_lim = self.right_lim

        partition = 0
        vol_per_point = np.prod((right_lim - left_lim) / nof_points)

        if dim == 1:
            theta = []
            for i in np.linspace(left_lim[0], right_lim[0], nof_points):
                theta.append([i])
            theta = np.array(theta)
            partition = np.sum(self.pdf_unnorm_batched(theta) * vol_per_point)
        elif dim == 2:
            theta = []
            for i in np.linspace(left_lim[0], right_lim[0], nof_points):
                for j in np.linspace(left_lim[1], right_lim[1], nof_points):
                    theta.append([i, j])
            theta = np.array(theta)
            partition = np.sum(self.pdf_unnorm_batched(theta) * vol_per_point)
        else:
            logger.error("ERROR: Approximate partition is not implemented for D > 2")

        # update inference state
        self.partition = partition
        return partition

    def pdf(self, theta):
        """Evaluate the pdf at theta. Theta is defined in a batched fashion.

        Parameters
        ----------
        theta: np.ndarray, shape: (BS,D)
            the input points

        Returns
        -------
        np.ndarray, shape(BS,)
            the pdf evaluation

        """
        assert theta.ndim == 2
        assert theta.shape[1] == self.dim
        assert self.dim <= 2, "PDF can be computed up to 2 dimensional problems."

        if self.partition is not None:
            partition = self.partition
        else:
            partition = self._approximate_partition()
            self.partition = partition

        return self.pdf_unnorm_batched(theta) / partition
        # pdf_eval = []
        # for i in range(theta.shape[0]):
        #     pdf_eval.append(self.pdf_unnorm_batched(
        #         theta[i:i + 1]) / partition)
        # return self.pdf_unnorm_batched(theta[i:i + 1]) / partition
        #
        # return np.array(pdf_eval)

    def sample(self, n2: int, seed=None) -> (np.ndarray, np.ndarray):
        """Sample n2 points from each region of the posterior.

        Parameters
        ----------
        n2: int
            number of points per region
        seed: int
            seed of the sampling procedure

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            the samples, the weights and the distances

        """
        regions = self.regions
        funcs = self.funcs
        nof_regions = len(regions)
        prior = self.prior
        eps = self.eps_cutoff

        # loop over all regions and sample
        if self.parallelize is False:
            theta = []
            for i in range(nof_regions):
                theta.append(regions[i].sample(n2, seed))
            theta = np.array(theta)
        else:
            pool = Pool()
            args = ((region, n2) for region in regions)
            result = pool.map(self._worker_sample, args)
            pool.close()
            pool.join()
            theta = np.array(result)

        # compute weight - o(n1xn2) complexity
        if self.parallelize is False:
            w = []
            distances = []
            self.progress_bar.reinit_progressbar(reinit_msg="Sampling posterior regions")
            for i in range(nof_regions):
                w.append([])
                # indicator_region = self.regions[i].contains
                for j in range(n2):
                    self.progress_bar.update_progressbar(i * n2 + j + 1, nof_regions * n2)
                    cur_theta = theta[i, j]
                    q = regions[i].pdf(cur_theta)
                    if q == 0.0:
                        logger.warning("Zero q")
                    # (ii) p
                    pr = float(prior.pdf(np.expand_dims(cur_theta, 0)))

                    # (iii) indicator
                    # # ind = indicator_region(cur_theta)
                    # # if not ind:
                    # #     logger.warning("Negative indicator")
                    dist = funcs[i](cur_theta)
                    distances.append(dist)
                    ind = dist < eps

                    # compute
                    if q > 0:
                        res = ind * pr / q
                    else:
                        res = 0

                    w[i].append(res)
            w = np.array(w)
            distances = np.array(distances)
        else:
            pool = Pool()
            args = ((i, theta[i], regions[i], prior, funcs[i], eps, n2)
                    for i in range(nof_regions))
            result = pool.map(self._worker_compute_weight, args)
            pool.close()
            pool.join()
            w = np.array([result[i][0] for i in range(len(result))])
            distances = np.array([result[i][1]
                                  for i in range(len(result))]).flatten()

        return theta, w, distances

    def compute_expectation(self, h, theta, w):
        """Compute the expectation h, based on the weighted samples.

        Parameters
        ----------
        h: Callable
        theta: np.ndarray, shape: (nof_samples, D)
            the samples
        w: np.ndarray, shape: (nof_samples,)
            the weight of the samples

        Returns
        -------
        the return type of h

        """
        h_theta = h(theta)

        numer = np.sum(h_theta * w)
        denom = np.sum(w)
        return numer / denom

    def visualize_region(self, i, samples, savefig):
        """Plot the i-th n-dimensional bounding box region.

        Parameters
        ----------
        i: int
          the index of the region
        samples: np.ndarray
          the samples drawn from this region
        savefig: Union[str, None]
          the path for saving the plot or None

        """
        assert i < len(self.funcs)
        dim = self.dim
        func = self.funcs[i]
        region = self.regions[i]

        if dim == 1:
            plt.figure()
            plt.title("Optimisation problem %d (seed = %d)." %
                      (i, self.nuisance[i]))

            # plot sampled points
            if samples is not None:
                x = samples[i, :, 0]
                plt.plot(x, np.zeros_like(x), "bo", label="samples")

            x = np.linspace(region.center +
                            region.limits[0, 0] -
                            0.2, region.center +
                            region.limits[0, 1] +
                            0.2, 30)
            y = [func(np.atleast_1d(theta)) for theta in x]
            plt.plot(x, y, 'r--', label="distance")
            plt.plot(region.center, 0, 'ro', label="center")
            plt.xlabel("theta")
            plt.ylabel("distance")
            plt.axvspan(region.center +
                        region.limits[0, 0], region.center +
                        region.limits[0, 1], label="acceptance region")
            plt.axhline(region.eps_region, color="g", label="eps")
            plt.legend()
            if savefig:
                plt.savefig(savefig, bbox_inches='tight')
            plt.show(block=False)
        else:
            plt.figure()
            plt.title("Optimisation problem %d (seed = %d)." %
                      (i, self.nuisance[i]))

            max_offset = np.sqrt(
                2 * (np.max(np.abs(region.limits)) ** 2)) + 0.2
            x = np.linspace(
                region.center[0] - max_offset, region.center[0] + max_offset, 30)
            y = np.linspace(
                region.center[1] - max_offset, region.center[1] + max_offset, 30)
            X, Y = np.meshgrid(x, y)

            Z = []
            for k, ii in enumerate(x):
                Z.append([])
                for kk, jj in enumerate(y):
                    Z[k].append(func(np.array([X[k, kk], Y[k, kk]])))
            Z = np.array(Z)
            plt.contourf(X, Y, Z, 100, cmap="RdGy")
            plt.plot(region.center[0], region.center[1], "ro")

            # plot sampled points
            if samples is not None:
                plt.plot(samples[i, :, 0], samples[i, :, 1],
                         "bo", label="samples")

            # plot eigenectors
            x = region.center
            x1 = region.center + region.rotation[:, 0] * region.limits[0][0]
            plt.plot([x[0], x1[0]], [x[1], x1[1]], "y-o",
                     label="-v1, f(-v1)=%.2f" % (func(x1)))
            x3 = region.center + region.rotation[:, 0] * region.limits[0][1]
            plt.plot([x[0], x3[0]], [x[1], x3[1]], "g-o",
                     label="v1, f(v1)=%.2f" % (func(x3)))

            x2 = region.center + region.rotation[:, 1] * region.limits[1][0]
            plt.plot([x[0], x2[0]], [x[1], x2[1]], "k-o",
                     label="-v2, f(-v2)=%.2f" % (func(x2)))
            x4 = region.center + region.rotation[:, 1] * region.limits[1][1]
            plt.plot([x[0], x4[0]], [x[1], x4[1]], "c-o",
                     label="v2, f(v2)=%.2f" % (func(x3)))

            # plot boundaries
            def plot_side(x, x1, x2):
                tmp = x + (x1 - x) + (x2 - x)
                plt.plot([x1[0], tmp[0], x2[0]], [x1[1], tmp[1], x2[1]], "r-o")

            plot_side(x, x1, x2)
            plot_side(x, x2, x3)
            plot_side(x, x3, x4)
            plot_side(x, x4, x1)

            plt.xlabel("theta 1")
            plt.ylabel("theta 2")

            plt.legend()
            plt.colorbar()
            if savefig:
                plt.savefig(savefig, bbox_inches='tight')
            plt.show(block=False)
