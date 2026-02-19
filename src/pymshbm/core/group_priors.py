"""Estimate group priors via hierarchical Bayesian EM.

Ports CBIG_MSHBM_estimate_group_priors.m with three nested EM loops:
    inter-subject -> intra-subject -> session-level vMF.
"""

import logging

import numpy as np

from pymshbm.math.vmf import ad, cdln, inv_ad, inv_ad_batch, vmf_log_probability
from pymshbm.types import MSHBMParams

logger = logging.getLogger(__name__)


def estimate_group_priors(
    data: np.ndarray,
    g_mu: np.ndarray,
    settings: dict,
) -> MSHBMParams:
    """Estimate group priors from training data.

    Args:
        data: (N, D, S, T) normalized FC profiles.
        g_mu: (D, L) group-level cluster centroids.
        settings: Dict with keys: num_sub, num_session, num_clusters,
                  dim, ini_concentration, epsilon, conv_th, max_iter.

    Returns:
        MSHBMParams with estimated group priors.
    """
    N, D, S, T = data.shape
    logger.info("  EM estimation: N=%d vertices, D=%d features, "
                "S=%d subjects, T=%d sessions, L=%d clusters",
                N, D, S, T, settings["num_clusters"])

    params = initialize_params(data, g_mu, settings)

    cost_inter = 0.0
    record = []

    for iteration in range(1, settings["max_iter"] + 1):
        params.iter_inter = iteration

        # Reset intra-subject params each outer iteration
        L = settings["num_clusters"]
        S = settings["num_sub"]
        params.sigma = np.full(L, settings["ini_concentration"], dtype=np.float64)
        params.s_psi = np.tile(g_mu[:, :, np.newaxis], (1, 1, S))

        # Inner EM: session-level clustering + intra-subject
        logger.debug("  Outer iter %d/%d: session-level vMF clustering",
                     iteration, settings["max_iter"])
        params = vmf_clustering_subject_session(params, settings, data)
        logger.debug("  Outer iter %d/%d: intra-subject variability",
                     iteration, settings["max_iter"])
        params = intra_subject_var(params, settings)

        # Inter-subject variability
        logger.debug("  Outer iter %d/%d: inter-subject variability",
                     iteration, settings["max_iter"])
        params = inter_subject_var(params, settings)

        # Compute cost
        update_cost = _compute_inter_cost(params, settings, data)
        record.append(float(update_cost))

        if iteration > 1 and abs(cost_inter) > 0:
            rel_change = abs((update_cost - cost_inter) / cost_inter)
            logger.info("  Outer iter %d/%d: cost=%.4f  rel_change=%.2e",
                        iteration, settings["max_iter"],
                        update_cost, rel_change)
            if rel_change <= settings["conv_th"]:
                logger.info("  Converged at outer iteration %d", iteration)
                break
        else:
            logger.info("  Outer iter %d/%d: cost=%.4f",
                        iteration, settings["max_iter"], update_cost)
        cost_inter = update_cost

    params.record = record
    return params


def initialize_params(
    data: np.ndarray,
    g_mu: np.ndarray,
    settings: dict,
) -> MSHBMParams:
    """Initialize all parameters from group centroids and data.

    Args:
        data: (N, D, S, T) normalized FC profiles.
        g_mu: (D, L) group centroids.
        settings: Problem settings dict.

    Returns:
        MSHBMParams with initial values.
    """
    N, D, S, T = data.shape
    L = settings["num_clusters"]
    c0 = settings["ini_concentration"]
    dim = settings["dim"]

    mu = g_mu.copy()
    epsil = np.full(L, c0, dtype=np.float64)
    sigma = np.full(L, c0, dtype=np.float64)
    kappa = np.full(L, c0, dtype=np.float64)
    s_psi = np.tile(g_mu[:, :, np.newaxis], (1, 1, S))
    s_t_nu = np.tile(g_mu[:, :, np.newaxis, np.newaxis], (1, 1, T, S))

    # Initialize s_lambda via vMF log likelihood
    s_lambda = _compute_initial_s_lambda(data, s_t_nu, kappa, dim, L)

    # Initialize theta
    theta = _compute_theta(s_lambda)

    return MSHBMParams(
        mu=mu,
        epsil=epsil,
        sigma=sigma,
        theta=theta,
        kappa=kappa,
        s_psi=s_psi,
        s_t_nu=s_t_nu,
        s_lambda=s_lambda,
        iter_inter=0,
        record=[],
    )


def vmf_clustering_subject_session(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
) -> MSHBMParams:
    """Inter-region level EM: update kappa, s_t_nu, s_lambda, theta.

    Args:
        params: Current parameters.
        settings: Problem settings.
        data: (N, D, S, T) FC profiles.

    Returns:
        Updated MSHBMParams.
    """
    N, D, S, T = data.shape
    L = settings["num_clusters"]
    dim = settings["dim"]
    epsilon = settings["epsilon"]
    c0 = settings["ini_concentration"]

    cost = np.zeros(S)

    for iter_em in range(1, 102):
        # M-step: update kappa and s_t_nu
        # weighted_data depends only on s_lambda (constant within M-step),
        # so compute it once before the M-step inner loop.
        weighted_data = _compute_weighted_data(data, params.s_lambda, S, T)
        _m_step(params, settings, data, epsilon, c0,
                weighted_data=weighted_data)

        # E-step: update s_lambda and theta; cache log-likelihoods
        log_vmf_cache = _e_step(params, settings, data)

        # Check convergence (reuse cached log-likelihoods)
        update_cost = _compute_em_cost(params, settings, data,
                                       log_vmf_cache=log_vmf_cache)
        with np.errstate(divide="ignore", invalid="ignore"):
            converged = np.where(np.abs(cost) > 0, np.abs((update_cost - cost) / cost) <= epsilon, False)
        if np.all(converged) and iter_em > 1:
            logger.debug("    Inner EM converged at iteration %d", iter_em)
            break
        if iter_em > 100:
            logger.debug("    Inner EM reached max iterations (100)")
            break
        cost = update_cost

    logger.debug("    Inner EM: %d iterations, kappa=%.1f",
                 iter_em, float(params.kappa[0]))
    return params


def _compute_weighted_data(
    data: np.ndarray,
    s_lambda: np.ndarray,
    S: int,
    T: int,
) -> np.ndarray:
    """Compute weighted_data[d,l,s,t] = sum_n data[n,d,s,t] * s_lambda[n,l,s].

    Uses explicit BLAS matmuls per (s,t) slice for guaranteed GEMM dispatch.
    Result shape: (D, L, S, T).
    """
    N, D = data.shape[0], data.shape[1]
    L = s_lambda.shape[1]
    result = np.empty((D, L, S, T), dtype=np.float64)
    for s in range(S):
        sl = s_lambda[:, :, s]  # (N, L)
        for t in range(T):
            # (D, N) @ (N, L) -> (D, L)
            result[:, :, s, t] = data[:, :, s, t].T @ sl
    return result


def _m_step(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
    epsilon: float,
    c0: float,
    weighted_data: np.ndarray | None = None,
) -> None:
    """M-step: update kappa and s_t_nu.

    Args:
        weighted_data: Precomputed (D, L, S, T) from _compute_weighted_data.
            When provided, avoids redundant recomputation within the inner loop.
    """
    N, D, S, T = data.shape
    L = settings["num_clusters"]
    dim = settings["dim"]

    if weighted_data is None:
        weighted_data = _compute_weighted_data(data, params.s_lambda, S, T)

    for _ in range(50):  # Max M-step iterations
        # Update kappa — accumulate directly without (N,L,S,T) intermediate
        kappa_num = 0.0
        for s in range(S):
            sl = params.s_lambda[:, :, s]  # (N, L)
            for t in range(T):
                dot_st = data[:, :, s, t] @ params.s_t_nu[:, :, t, s]  # (N, L)
                kappa_num += np.nansum(sl * dot_st)
        kappa_den = np.nansum(params.s_lambda)

        if kappa_den > 0:
            rbar = kappa_num / kappa_den
            kappa_new = inv_ad(dim, min(max(rbar, 1e-10), 1 - 1e-10))
            kappa_new = max(kappa_new, c0)
            if np.isinf(kappa_new):
                kappa_new = params.kappa[0]
            params.kappa = np.full(L, kappa_new)

        # Update s_t_nu using precomputed weighted_data
        # lambda_X[d,l,t,s] = kappa * weighted_data[d,l,s,t] + sigma * s_psi[d,l,s]
        lambda_X = (
            params.kappa[np.newaxis, :, np.newaxis, np.newaxis]
            * weighted_data.transpose(0, 1, 3, 2)  # (D, L, T, S)
            + params.sigma[np.newaxis, :, np.newaxis, np.newaxis]
            * params.s_psi[:, :, np.newaxis, :]  # (D, L, 1, S) broadcast to T
        )
        norms = np.linalg.norm(lambda_X, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        nu_new = lambda_X / norms

        # Convergence check: cosine similarity between old and new
        cos_sim = np.sum(nu_new * params.s_t_nu, axis=0)  # (L, T, S)
        cos_sim = np.where(np.isnan(cos_sim), 1.0, cos_sim)
        all_converged = np.all(1 - cos_sim < epsilon)

        params.s_t_nu = nu_new

        if all_converged:
            break


def _e_step(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
) -> list[np.ndarray]:
    """E-step: update s_lambda and theta.

    Returns:
        List of per-subject (N, L) accumulated log-vmf arrays for cost reuse.
    """
    N, D, S, T = data.shape
    L = settings["num_clusters"]

    # Precompute log normalizing constant (kappa is uniform across L)
    dim = settings["dim"]
    log_c = cdln(params.kappa, dim)

    log_vmf_cache = []
    for s in range(S):
        log_vmf_total = np.zeros((N, L))
        for t in range(T):
            X = data[:, :, s, t]  # (N, D)
            nu = params.s_t_nu[:, :, t, s]  # (D, L)
            log_vmf = vmf_log_probability(X, nu, params.kappa, log_c=log_c)
            log_vmf_total += np.nan_to_num(log_vmf, nan=0.0)

        log_vmf_cache.append(log_vmf_total)

        # Add log theta prior
        log_theta = np.log(np.maximum(params.theta, np.finfo(float).tiny))
        log_posterior = log_vmf_total + log_theta

        # Softmax for numerical stability
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        s_lambda = np.exp(log_posterior)
        row_sums = s_lambda.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        s_lambda /= row_sums

        # Mask zero rows (medial wall)
        zero_mask = np.all(log_vmf_total == 0, axis=1)
        s_lambda[zero_mask] = 0.0

        params.s_lambda[:, :, s] = s_lambda

    params.theta = _compute_theta(params.s_lambda)
    return log_vmf_cache


def intra_subject_var(
    params: MSHBMParams,
    settings: dict,
) -> MSHBMParams:
    """Update s_psi and sigma (intra-subject variability level)."""
    S = settings["num_sub"]
    T = settings["num_session"]
    L = settings["num_clusters"]
    dim = settings["dim"]
    epsilon = settings["epsilon"]

    for intra_iter in range(1, 51):
        # Update s_psi — vectorized over S
        # s_t_nu: (D, L, T, S), sum over T -> (D, L, S)
        accum = (
            params.sigma[np.newaxis, :, np.newaxis] * params.s_t_nu.sum(axis=2)
            + params.epsil[np.newaxis, :, np.newaxis] * params.mu[:, :, np.newaxis]
        )
        norms = np.linalg.norm(accum, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        s_psi_new = accum / norms

        # Convergence check — vectorized
        cos_sim = np.sum(s_psi_new * params.s_psi, axis=0)  # (L, S)
        all_converged = np.all(1 - cos_sim < epsilon)
        params.s_psi = s_psi_new

        # Update sigma — vectorized with einsum + batch inv_ad
        # rbar[l] = mean over (s,t) of dot(s_psi[:,l,s], s_t_nu[:,l,t,s])
        rbar_all = np.einsum("dls,dlts->l", params.s_psi, params.s_t_nu) / (S * T)
        rbar_clamped = np.clip(rbar_all, 1e-10, 1 - 1e-10)
        sigma_new = inv_ad_batch(dim, rbar_clamped)

        sigma_converged = (
            np.mean(np.abs(params.sigma - sigma_new)
                    / np.maximum(params.sigma, 1e-10)) < epsilon
        )
        if all_converged and sigma_converged:
            params.sigma = sigma_new
            break
        params.sigma = sigma_new

    logger.debug("    Intra-subject: %d iterations, "
                 "sigma=[%s]", intra_iter,
                 ", ".join(f"{s:.1f}" for s in params.sigma))
    return params


def inter_subject_var(
    params: MSHBMParams,
    settings: dict,
) -> MSHBMParams:
    """Update mu and epsil (inter-subject variability level)."""
    S = settings["num_sub"]
    L = settings["num_clusters"]
    dim = settings["dim"]
    c0 = settings["ini_concentration"]

    # Update mu
    mu_update = params.s_psi.sum(axis=2)  # (D, L)
    norms = np.linalg.norm(mu_update, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    params.mu = mu_update / norms

    # Update epsil — vectorized with einsum + batch inv_ad
    rbar_all = np.einsum("dls,dl->l", params.s_psi, params.mu) / S
    rbar_clamped = np.clip(rbar_all, 1e-10, 1 - 1e-10)
    epsil_new = inv_ad_batch(dim, rbar_clamped)
    epsil_new = np.maximum(epsil_new, c0)
    # Keep old values where result is inf
    inf_mask = np.isinf(epsil_new)
    epsil_new[inf_mask] = params.epsil[inf_mask]
    params.epsil = epsil_new

    return params


def _compute_initial_s_lambda(
    data: np.ndarray,
    s_t_nu: np.ndarray,
    kappa: np.ndarray,
    dim: int,
    L: int,
) -> np.ndarray:
    """Compute initial s_lambda from vMF log likelihoods."""
    N, D, S, T = data.shape
    s_lambda = np.zeros((N, L, S))
    log_c = cdln(kappa, dim)

    for s in range(S):
        log_vmf_total = np.zeros((N, L))
        for t in range(T):
            X = data[:, :, s, t]
            nu = s_t_nu[:, :, t, s]
            lv = vmf_log_probability(X, nu, kappa, log_c=log_c)
            log_vmf_total += np.nan_to_num(lv, nan=0.0)

        log_vmf_total -= log_vmf_total.max(axis=1, keepdims=True)
        sl = np.exp(log_vmf_total)
        row_sums = sl.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sl /= row_sums

        zero_mask = np.all(log_vmf_total == 0, axis=1)
        sl[zero_mask] = 0.0
        s_lambda[:, :, s] = sl

    return s_lambda


def _compute_theta(s_lambda: np.ndarray) -> np.ndarray:
    """Compute theta as mean of s_lambda across subjects, handling zeros."""
    nonzero_count = np.sum(s_lambda != 0, axis=2)
    theta_sum = s_lambda.sum(axis=2)
    theta = np.zeros_like(theta_sum)
    mask = nonzero_count > 0
    theta[mask] = theta_sum[mask] / nonzero_count[mask]
    return theta


def _compute_em_cost(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
    log_vmf_cache: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Compute per-subject EM cost.

    Args:
        log_vmf_cache: Optional precomputed per-subject (N, L) log-vmf arrays
            from _e_step. When provided, skips redundant vmf_log_probability
            recomputation.
    """
    N, D, S, T = data.shape
    costs = np.zeros(S)
    log_theta = np.log(np.maximum(params.theta, np.finfo(float).tiny))
    log_c = None if log_vmf_cache is not None else cdln(params.kappa, settings["dim"])

    for s in range(S):
        if log_vmf_cache is not None:
            log_vmf_total = log_vmf_cache[s]
        else:
            log_vmf_total = np.zeros((N, settings["num_clusters"]))
            for t in range(T):
                X = data[:, :, s, t]
                nu = params.s_t_nu[:, :, t, s]
                lv = vmf_log_probability(X, nu, params.kappa, log_c=log_c)
                log_vmf_total += np.nan_to_num(lv, nan=0.0)

        sl = params.s_lambda[:, :, s]
        log_sl = np.log(np.maximum(sl, np.finfo(float).tiny))

        costs[s] = np.nansum(sl * log_vmf_total) + np.nansum(sl * log_theta) - np.nansum(sl * log_sl)
    return costs


def _compute_inter_cost(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
) -> float:
    """Compute total inter-subject cost."""
    return float(np.sum(_compute_em_cost(params, settings, data)))
