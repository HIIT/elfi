import numpy as np
import scipy.stats as ss

import elfi
from elfi.methods.utils import weighted_var, GMDistribution, normalize_weights, ModelPrior
from elfi.methods.bo.utils import stochastic_optimization, minimize


def test_stochastic_optimization():
    fun = lambda x : x**2
    bounds = ((-1, 1),)
    its = int(1e3)
    polish=True
    loc, val = stochastic_optimization(fun, bounds, its, polish)
    assert abs(loc - 0.0) < 1e-5
    assert abs(val - 0.0) < 1e-5


def test_minimize():
    fun = lambda x : x[0]**2 + (x[1]-1)**4
    grad = lambda x : np.array([2*x[0], 4*(x[1]-1)**3])
    bounds = ((-2, 2), (-2, 3))
    loc, val = minimize(fun, grad, bounds)
    assert np.isclose(val, 0, atol=0.01)
    assert np.allclose(loc, np.array([0, 1]), atol=0.02)


def test_weighted_var():
    # 1d case
    std = .3
    x = np.random.RandomState(12345).normal(-2, std, size=1000)
    w = np.array([1] * len(x))
    assert (weighted_var(x, w) - std) < .1

    # 2d case
    cov = [[.5, 0], [0, 3.2]]
    x = np.random.RandomState(12345).multivariate_normal([1,2], cov, size=1000)
    w = np.array([1] * len(x))
    assert np.linalg.norm(weighted_var(x, w) - np.diag(cov)) < .1


class TestGMDistribution:

    def test_pdf(self):
        # 1d case
        x = [1, 2, -1]
        means = [0, 2]
        weights = normalize_weights([.4, .1])
        d = GMDistribution.pdf(x, means, weights=weights)
        d_true = weights[0]*ss.norm.pdf(x, loc=means[0]) + weights[1]*ss.norm.pdf(x, loc=means[1])
        assert np.allclose(d, d_true)

        # 2d case
        x = [[1, 2, -1], [0,0,2]]
        means = [[0,0,0], [-1,-.2, .1]]
        d = GMDistribution.pdf(x, means, weights=weights)
        d_true = weights[0]*ss.multivariate_normal.pdf(x, mean=means[0]) + \
                 weights[1]*ss.multivariate_normal.pdf(x, mean=means[1])
        assert np.allclose(d, d_true)

    def test_rvs(self):
        means = [[1000, 3], [-1000, -3]]
        weights = [.3, .7]
        N = 10000
        random = np.random.RandomState(12042017)
        rvs = GMDistribution.rvs(means, weights=weights, size=N, random_state=random)
        rvs = rvs[rvs[:,0] < 0, :]

        # Test correct proportion of samples near the second mode
        assert np.abs(len(rvs)/N - .7) < .01

        # Test that the mean of the second mode is correct
        assert np.abs(np.mean(rvs[:,1]) + 3) < .1


class TestModelPrior:

    def test_pdf(self, ma2):
        prior = ModelPrior(ma2)
        rv = prior.rvs(size=10)
        assert np.allclose(prior.pdf(rv), np.exp(prior.logpdf(rv)))

    def test_gradient_logpdf(self, ma2):
        prior = ModelPrior(ma2)
        rv = prior.rvs(size=10)
        grads = prior.gradient_logpdf(rv)
        assert grads.shape == rv.shape
        assert np.allclose(grads, 0)

    def test_numerical_grad_logpdf(self):
        # Test gradient with a normal distribution
        loc = 2.2
        scale = 1.1
        x = np.random.rand()
        analytical_grad_logpdf = -(x - loc) / scale ** 2
        prior_node = elfi.Prior('normal', loc, scale)
        num_grad = ModelPrior(prior_node.model).gradient_logpdf(x)
        assert np.isclose(num_grad, analytical_grad_logpdf, atol=0.01)

