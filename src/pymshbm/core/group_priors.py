"""Estimate group priors via hierarchical Bayesian EM.

Ports CBIG_MSHBM_estimate_group_priors.m with three nested EM loops:
    inter-subject -> intra-subject -> session-level vMF.
"""

import numpy as np

from pymshbm.math.vmf import ad, cdln, inv_ad, vmf_log_probability
from pymshbm.types import MSHBMParams


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
        params = vmf_clustering_subject_session(params, settings, data)
        params = intra_subject_var(params, settings)

        # Inter-subject variability
        params = inter_subject_var(params, settings)

        # Compute cost
        update_cost = _compute_inter_cost(params, settings, data)
        record.append(float(update_cost))

        if iteration > 1 and abs(cost_inter) > 0:
            if abs((update_cost - cost_inter) / cost_inter) <= settings["conv_th"]:
                break
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
        _m_step(params, settings, data, epsilon, c0)

        # E-step: update s_lambda and theta; cache log-likelihoods
        log_vmf_cache = _e_step(params, settings, data)

        # Check convergence (reuse cached log-likelihoods)
        update_cost = _compute_em_cost(params, settings, data,
                                       log_vmf_cache=log_vmf_cache)
        with np.errstate(divide="ignore", invalid="ignore"):
            converged = np.where(np.abs(cost) > 0, np.abs((update_cost - cost) / cost) <= epsilon, False)
        if np.all(converged) and iter_em > 1:
            break
        if iter_em > 100:
            break
        cost = update_cost

    return params


def _m_step(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
    epsilon: float,
    c0: float,
) -> None:
    """M-step: update kappa and s_t_nu."""
    N, D, S, T = data.shape
    L = settings["num_clusters"]
    dim = settings["dim"]

    # s_t_nu is (D, L, T, S) but einsum needs matching index for S
    # s_lambda is (N, L, S), data is (N, D, S, T)
    for _ in range(50):  # Max M-step iterations
        # Update kappa — vectorized over S, T
        # dot products: sum_n,d data[n,d,s,t] * s_t_nu[d,l,t,s] * s_lambda[n,l,s]
        # First compute data @ s_t_nu per (s,t): einsum for dot product
        dots = np.einsum("ndst,dlts->nlst", data, params.s_t_nu)  # (N, L, S, T)
        kappa_num = np.nansum(params.s_lambda[:, :, :, np.newaxis] * dots)
        kappa_den = np.nansum(params.s_lambda)

        if kappa_den > 0:
            rbar = kappa_num / kappa_den
            kappa_new = inv_ad(dim, min(max(rbar, 1e-10), 1 - 1e-10))
            kappa_new = max(kappa_new, c0)
            if np.isinf(kappa_new):
                kappa_new = params.kappa[0]
            params.kappa = np.full(L, kappa_new)

        # Update s_t_nu — vectorized over S, T
        # weighted_data[d,l,s,t] = sum_n data[n,d,s,t] * s_lambda[n,l,s]
        weighted_data = np.einsum("ndst,nls->dlst", data, params.s_lambda)
        # lambda_X[d,l,t,s] = kappa[l] * weighted_data[d,l,s,t] + sigma[l] * s_psi[d,l,s]
        # Note: need to transpose (s,t) -> (t,s) to match s_t_nu layout (D,L,T,S)
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

    for _ in range(50):
        # Update s_psi — vectorized over S
        # s_t_nu: (D, L, T, S), sum over T → (D, L, S)
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

        # Update sigma — vectorized with einsum
        # rbar[l] = mean over (s,t) of dot(s_psi[:,l,s], s_t_nu[:,l,t,s])
        rbar_all = np.einsum("dls,dlts->l", params.s_psi, params.s_t_nu) / (S * T)
        sigma_new = np.zeros(L)
        for l_idx in range(L):
            rb = min(max(float(rbar_all[l_idx]), 1e-10), 1 - 1e-10)
            sigma_new[l_idx] = inv_ad(dim, rb)

        sigma_converged = (
            np.mean(np.abs(params.sigma - sigma_new)
                    / np.maximum(params.sigma, 1e-10)) < epsilon
        )
        if all_converged and sigma_converged:
            params.sigma = sigma_new
            break
        params.sigma = sigma_new

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

    # Update epsil — vectorized with einsum
    rbar_all = np.einsum("dls,dl->l", params.s_psi, params.mu) / S
    for l_idx in range(L):
        rb = min(max(float(rbar_all[l_idx]), 1e-10), 1 - 1e-10)
        epsil_new = inv_ad(dim, rb)
        epsil_new = max(epsil_new, c0)
        if np.isinf(epsil_new):
            epsil_new = params.epsil[l_idx]
        params.epsil[l_idx] = epsil_new

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
    log_c = None
    for s in range(S):
        if log_vmf_cache is not None:
            log_vmf_total = log_vmf_cache[s]
        else:
            if log_c is None:
                log_c = cdln(params.kappa, settings["dim"])
            log_vmf_total = np.zeros((N, settings["num_clusters"]))
            for t in range(T):
                X = data[:, :, s, t]
                nu = params.s_t_nu[:, :, t, s]
                lv = vmf_log_probability(X, nu, params.kappa, log_c=log_c)
                log_vmf_total += np.nan_to_num(lv, nan=0.0)

        sl = params.s_lambda[:, :, s]
        log_theta = np.log(np.maximum(params.theta, np.finfo(float).tiny))
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
