"""MCMC sampling methods."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# TODO: combine ESS and Rhat?, consider transforming parameters to allowed
# region to increase acceptance ratio


def eff_sample_size(chains):
    """Calculate the effective sample size for 1 or more chains.

    See:

    Gelman, Carlin, Stern, Dunson, Vehtari, Rubin: Bayesian Data Analysis, 2013.

    Stan modeling language user's guide and reference manual, v. 2.14.0.

    Parameters
    ----------
    chains : np.array of shape (N,) or (M, N)
        Samples of a parameter from an MCMC algorithm. No burn-in subtracted here!

    Returns
    -------
    ess : float

    """
    chains = np.atleast_2d(chains)
    n_chains, n_samples = chains.shape
    means = np.mean(chains, axis=1)
    variances = np.var(chains, ddof=1, axis=1)

    var_between = 0 if n_chains == 1 else n_samples * np.var(means, ddof=1)
    var_within = np.mean(variances)
    var_pooled = ((n_samples - 1.) * var_within + var_between) / n_samples

    # autocovariances for lags 1..n_samples
    # https://en.wikipedia.org/wiki/Autocorrelation#Estimation
    n_padded = int(2**np.ceil(1 + np.log2(n_samples)))
    freqs = np.fft.rfft(chains - means[:, None], n_padded)
    autocov = np.fft.irfft(np.abs(freqs)**2)[:, :n_samples].real
    autocov = autocov / np.arange(n_samples, 0, -1)

    estimator_sum = 0.
    lag = 1
    while lag < n_samples:
        # estimate multi-chain autocorrelation using variogram
        temp = 1. - (var_within - np.mean(autocov[:, lag])) / var_pooled

        # only use the first non-negative autocorrelations to avoid noise
        if temp >= 0:
            estimator_sum += temp
            lag += 1
        else:
            break

    ess = n_chains * n_samples / (1. + 2. * estimator_sum)

    return ess


def gelman_rubin_statistic(chains):
    r"""Calculate the Gelman--Rubin convergence statistic.

    Also known as the potential scale reduction factor, or \hat{R}.
    Uses the split version, as in Stan.

    See:

    Gelman, Carlin, Stern, Dunson, Vehtari, Rubin: Bayesian Data Analysis, 2013.

    Gelman, A. and D. B. Rubin: Inference from iterative simulation using
    multiple sequences (with discussion). Statistical Science, 7:457-511, 1992.

    Stan modeling language user's guide and reference manual, v. 2.14.0.

    Parameters
    ----------
    chains : np.array of shape (M, N)
        Samples of a parameter from an MCMC algorithm, 1 row per chain. No burn-in subtracted here!

    Returns
    -------
    psrf : float
        Should be below 1.1 to support convergence, or at least below 1.2 for all parameters.

    """
    chains = np.atleast_2d(chains)
    n_chains, n_samples = chains.shape

    # split chains in the middle
    n_chains *= 2
    n_samples //= 2  # drop 1 if odd
    chains = chains[:, :2 * n_samples].reshape((n_chains, n_samples))

    means = np.mean(chains, axis=1)
    variances = np.var(chains, ddof=1, axis=1)

    var_between = n_samples * np.var(means, ddof=1)
    var_within = np.mean(variances)

    var_pooled = ((n_samples - 1.) * var_within + var_between) / n_samples

    # potential scale reduction factor, should be close to 1
    psrf = np.sqrt(var_pooled / var_within)

    return psrf


def nuts(n_iter,
         params0,
         target,
         grad_target,
         n_adapt=None,
         target_prob=0.6,
         max_depth=5,
         seed=0,
         info_freq=100,
         max_retry_inits=20,
         stepsize=None):
    r"""Sample the target using the NUTS algorithm.

    No-U-Turn Sampler, an improved version of the Hamiltonian (Markov Chain) Monte Carlo sampler.

    Based on Algorithm 6 in
    Hoffman & Gelman, JMLR 15, 1593-1623, 2014.

    Parameters
    ----------
    n_iter : int
        The number of iterations, including n_adapt and possible other warmup iterations.
    params0 : np.array
        Initial values for sampled parameters.
    target : function
        The target's log density to sample (possibly unnormalized).
    grad_target : function
        The gradient of target.
    n_adapt : int, optional
        The number of automatic adjustments to stepsize. Defaults to n_iter/2.
    target_prob : float, optional
        Desired average acceptance probability. (Parameter \delta in the original paper.)
    max_depth : int, optional
        Maximum recursion depth.
    seed : int, optional
        Seed for pseudo-random number generator.
    info_freq : int, optional
        How often to log progress to loglevel INFO.
    max_retry_inits : int, optional
        How many times to retry finding initial stepsize (if stepped outside allowed region).
    stepsize : float, optional
        Initial stepsize (will be still adapted). Defaults to finding by trial and error.

    Returns
    -------
    samples : np.array
        Samples from the MCMC algorithm, including those during adaptation.

    """
    random_state = np.random.RandomState(seed)
    n_adapt = n_adapt if n_adapt is not None else n_iter // 2

    logger.info("NUTS: Performing {} iterations with {} adaptation steps.".format(n_iter, n_adapt))

    target0 = target(params0)
    if np.isinf(target0):
        raise ValueError("NUTS: Bad initialization point {}, logpdf -> -inf.".format(params0))

    # ********************************
    # Find reasonable initial stepsize
    # ********************************
    if stepsize is None:
        grad0 = grad_target(params0)
        logger.debug("NUTS: Trying to find initial stepsize from point {} with gradient {}.".
                     format(params0, grad0))
        init_tries = 0
        while init_tries < max_retry_inits:  # might step into region unallowed by priors
            stepsize = np.exp(-init_tries)
            init_tries += 1
            momentum0 = random_state.randn(*params0.shape)

            # leapfrog
            momentum1 = momentum0 + 0.5 * stepsize * grad0
            params1 = params0 + stepsize * momentum1
            momentum1 += 0.5 * stepsize * grad_target(params1)

            joint0 = target0 - 0.5 * np.inner(momentum0, momentum0)
            joint1 = target(params1) - 0.5 * np.inner(momentum1, momentum1)

            if np.isfinite(joint1):
                break
            else:
                if init_tries == max_retry_inits:
                    raise ValueError(
                        "NUTS: Cannot find acceptable stepsize starting from point {}. All "
                        "trials ended in region with 0 probability.".format(params0))
                # logger.debug("momentum0 {}, momentum1 {}, params1 {}, joint0 {}, joint1 {}"
                #              .format(momentum0, momentum1, params1, joint0, joint1))
                logger.debug("NUTS: Problem finding acceptable stepsize, now {}. Retrying {}/{}."
                             .format(stepsize, init_tries, max_retry_inits))

        plusminus = 1 if np.exp(joint1 - joint0) > 0.5 else -1
        factor = 2. if plusminus == 1 else 0.5
        while factor * np.exp(plusminus * (joint1 - joint0)) > 1.:
            stepsize *= factor
            if stepsize == 0. or stepsize > 1e7:  # bounds as in STAN
                raise SystemExit("NUTS: Found invalid stepsize {} starting from point {}."
                                 .format(stepsize, params0))

            # leapfrog
            momentum1 = momentum0 + 0.5 * stepsize * grad0
            params1 = params0 + stepsize * momentum1
            momentum1 += 0.5 * stepsize * grad_target(params1)

            joint1 = target(params1) - 0.5 * np.inner(momentum1, momentum1)

    logger.debug("NUTS: Set initial stepsize {}.".format(stepsize))

    # Some parameters from the NUTS paper, used for adapting the stepsize
    target_stepsize = np.log(10. * stepsize)
    log_avg_stepsize = 0.
    accept_ratio = 0.  # tends to target_prob
    shrinkage = 0.05  # controls shrinkage accept_ratio to target_prob
    ii_offset = 10.  # stabilizes initialization
    discount = -0.75  # reduce weight of past

    # ********
    # Sampling
    # ********
    samples = np.empty((n_iter + 1, ) + params0.shape)
    samples[0, :] = params0
    n_diverged = 0  # counter for proposals whose error diverged
    n_outside = 0  # counter for proposals outside priors (pdf=0)
    n_total = 0  # total number of proposals

    for ii in range(1, n_iter + 1):
        momentum0 = random_state.randn(*params0.shape)
        samples_prev = samples[ii - 1, :]
        log_joint0 = target(samples_prev) - 0.5 * np.inner(momentum0, momentum0)
        log_slicevar = log_joint0 - random_state.exponential()
        samples[ii, :] = samples_prev
        params_left = samples_prev
        params_right = samples_prev
        momentum_left = momentum0
        momentum_right = momentum0
        depth = 0
        n_ok = 1
        all_ok = True  # criteria for no U-turn, diverging error

        while all_ok and depth <= max_depth:
            direction = 1 if random_state.rand() < 0.5 else -1
            if direction == -1:
                params_left, momentum_left, _, _, params1, n_sub, sub_ok, mh_ratio, n_steps, \
                    is_div, is_out = _build_tree_nuts(
                        params_left, momentum_left, log_slicevar, -stepsize, depth, log_joint0,
                        target, grad_target, random_state)
            else:
                _, _, params_right, momentum_right, params1, n_sub, sub_ok, mh_ratio, n_steps, \
                    is_div, is_out = _build_tree_nuts(
                        params_right, momentum_right, log_slicevar, stepsize, depth, log_joint0,
                        target, grad_target, random_state)

            if sub_ok == 1:
                if random_state.rand() < float(n_sub) / n_ok:
                    samples[ii, :] = params1  # accept proposal
            n_ok += n_sub
            if not is_out:  # params1 outside allowed region; don't count this as diverging error
                n_diverged += is_div
            n_outside += is_out
            n_total += n_steps
            all_ok = sub_ok and \
                (np.inner(params_right - params_left, momentum_left) >= 0) and \
                (np.inner(params_right - params_left, momentum_right) >= 0)
            depth += 1
            if depth > max_depth:
                logger.debug("NUTS: Maximum recursion depth {} exceeded.".format(max_depth))

        # adjust stepsize according to target acceptance ratio
        if ii <= n_adapt:
            accept_ratio = (1. - 1. / (ii + ii_offset)) * accept_ratio \
                + (target_prob - float(mh_ratio) / n_steps) / (ii + ii_offset)
            log_stepsize = target_stepsize - np.sqrt(ii) / shrinkage * accept_ratio
            log_avg_stepsize = ii ** discount * log_stepsize + \
                (1. - ii ** discount) * log_avg_stepsize
            stepsize = np.exp(log_stepsize)

        elif ii == n_adapt + 1:  # adaptation/warmup finished
            stepsize = np.exp(log_avg_stepsize)  # final stepsize
            n_diverged = 0
            n_outside = 0
            n_total = 0
            logger.info("NUTS: Adaptation/warmup finished. Sampling...")
            logger.debug("NUTS: Set final stepsize {}.".format(stepsize))

        if ii % info_freq == 0 and ii < n_iter:
            logger.info("NUTS: Iterations performed: {}/{}...".format(ii, n_iter))

    info_str = "NUTS: Acceptance ratio: {:.3f}".format(float(n_iter - n_adapt) / n_total)
    if n_outside > 0:
        info_str += ". After warmup {} proposals were outside of the region allowed by priors " \
                    "and rejected, decreasing acceptance ratio.".format(n_outside)
    logger.info(info_str)

    if n_diverged > 0:
        logger.warning("NUTS: Diverged proposals after warmup (i.e. n_adapt={} steps): {}".format(
            n_adapt, n_diverged))

    return samples[1:, :]


def _build_tree_nuts(params, momentum, log_slicevar, step, depth, log_joint0, target, grad_target,
                     random_state):
    """Recursively build a balanced binary tree needed by NUTS.

    Based on Algorithm 6 in
    Hoffman & Gelman, JMLR 15, 1351-1381, 2014.

    """
    # Base case: one leapfrog step
    if depth == 0:
        momentum1 = momentum + 0.5 * step * grad_target(params)
        params1 = params + step * momentum1
        momentum1 = momentum1 + 0.5 * step * grad_target(params1)

        log_joint = target(params1) - 0.5 * np.inner(momentum1, momentum1)
        n_ok = float(log_slicevar <= log_joint)
        sub_ok = log_slicevar < (1000. + log_joint)  # check for diverging error
        is_out = False
        if not sub_ok:
            if np.isinf(target(params1)):  # logpdf(params1) = -inf i.e. pdf(params1) = 0
                is_out = True
            else:
                logger.debug(
                    "NUTS: Diverging error: log_joint={}, params={}, params1={}, momentum={}, "
                    "momentum1={}.".format(log_joint, params, params1, momentum, momentum1))
            mh_ratio = 0.  # reject
        else:
            mh_ratio = min(1., np.exp(log_joint - log_joint0))

        return params1, momentum1, params1, momentum1, params1, n_ok, sub_ok, mh_ratio, 1., \
            not sub_ok, is_out

    else:
        # Recursion to build subtrees, doubling size
        params_left, momentum_left, params_right, momentum_right, params1, n_sub, sub_ok, \
            mh_ratio, n_steps, is_div, is_out = _build_tree_nuts(
                params, momentum, log_slicevar, step, depth - 1, log_joint0, target,
                grad_target, random_state)

        if sub_ok:  # recurse further
            if step < 0:
                params_left, momentum_left, _, _, params2, n_sub2, sub_ok, mh_ratio2, n_steps2, \
                    is_div, is_out = _build_tree_nuts(
                        params_left, momentum_left, log_slicevar,
                        step, depth - 1, log_joint0, target, grad_target, random_state)
            else:
                _, _, params_right, momentum_right, params2, n_sub2, sub_ok, mh_ratio2, n_steps2, \
                    is_div, is_out = _build_tree_nuts(
                        params_right, momentum_right, log_slicevar,
                        step, depth - 1, log_joint0, target, grad_target, random_state)

            if n_sub2 > 0:
                if float(n_sub2) / (n_sub + n_sub2) > random_state.rand():
                    params1 = params2  # accept move
            mh_ratio += mh_ratio2
            n_steps += n_steps2
            sub_ok = sub_ok and \
                (np.inner(params_right - params_left, momentum_left) >= 0) and \
                (np.inner(params_right - params_left, momentum_right) >= 0)
            n_sub += n_sub2

        return params_left, momentum_left, params_right, momentum_right, params1, n_sub, sub_ok, \
            mh_ratio, n_steps, is_div, is_out


def metropolis(n_samples, params0, target, sigma_proposals, warmup=0, seed=0):
    """Sample the target with a Metropolis Markov Chain Monte Carlo using Gaussian proposals.

    Parameters
    ----------
    n_samples : int
        The number of requested samples.
    params0 : np.array
        Initial values for each sampled parameter.
    target : function
        The target log density to sample (possibly unnormalized).
    sigma_proposals : np.array
        Standard deviations for Gaussian proposals of each parameter.
    warmup : int
        Number of warmup samples.
    seed : int, optional
        Seed for pseudo-random number generator.

    Returns
    -------
    samples : np.array

    """
    random_state = np.random.RandomState(seed)

    samples = np.empty((n_samples + warmup + 1, ) + params0.shape)
    samples[0, :] = params0
    target_current = target(params0)
    if np.isinf(target_current):
        raise ValueError(
            "Metropolis: Bad initialization point {},logpdf -> -inf.".format(params0))

    n_accepted = 0

    for ii in range(1, n_samples + warmup + 1):
        samples[ii, :] = samples[ii - 1, :] + sigma_proposals * random_state.randn(*params0.shape)
        target_prev = target_current
        target_current = target(samples[ii, :])
        if ((np.exp(target_current - target_prev) < random_state.rand())
           or np.isinf(target_current)
           or np.isnan(target_current)):  # reject proposal
            samples[ii, :] = samples[ii - 1, :]
            target_current = target_prev
        else:
            n_accepted += 1

    logger.info(
        "{}: Total acceptance ratio: {:.3f}".format(__name__,
                                                    float(n_accepted) / (n_samples + warmup)))

    return samples[(1 + warmup):, :]
