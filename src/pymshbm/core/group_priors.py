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

        # E-step: update s_lambda and theta
        _e_step(params, settings, data)

        # Check convergence
        update_cost = _compute_em_cost(params, settings, data)
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

    for _ in range(50):  # Max M-step iterations
        # Update kappa
        kappa_num = 0.0
        kappa_den = 0.0
        for s in range(S):
            for t in range(T):
                X = data[:, :, s, t]  # (N, D)
                nu = params.s_t_nu[:, :, t, s]  # (D, L)
                sl = params.s_lambda[:, :, s]  # (N, L)
                dot = X @ nu  # (N, L)
                kappa_num += np.nansum(sl * dot)
            kappa_den += np.nansum(params.s_lambda[:, :, s])

        if kappa_den > 0:
            rbar = kappa_num / kappa_den
            kappa_new = inv_ad(dim, min(max(rbar, 1e-10), 1 - 1e-10))
            kappa_new = max(kappa_new, c0)
            if np.isinf(kappa_new):
                kappa_new = params.kappa[0]
            params.kappa = np.full(L, kappa_new)

        # Update s_t_nu
        all_converged = True
        for s in range(S):
            for t in range(T):
                X = data[:, :, s, t]  # (N, D)
                sl = params.s_lambda[:, :, s]  # (N, L)
                lambda_X = params.kappa[np.newaxis, :] * (X.T @ sl) + \
                    params.sigma[np.newaxis, :] * params.s_psi[:, :, s]
                norms = np.linalg.norm(lambda_X, axis=0, keepdims=True)
                norms[norms == 0] = 1.0
                nu_new = lambda_X / norms

                old_nu = params.s_t_nu[:, :, t, s]
                check = np.sum(nu_new * old_nu, axis=0)
                check = np.where(np.isnan(check), 1.0, check)
                if np.any(1 - check >= epsilon):
                    all_converged = False

                params.s_t_nu[:, :, t, s] = nu_new

        if all_converged:
            break


def _e_step(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
) -> None:
    """E-step: update s_lambda and theta."""
    N, D, S, T = data.shape
    L = settings["num_clusters"]
    dim = settings["dim"]

    for s in range(S):
        log_vmf_total = np.zeros((N, L))
        for t in range(T):
            X = data[:, :, s, t]  # (N, D)
            nu = params.s_t_nu[:, :, t, s]  # (D, L)
            log_vmf = vmf_log_probability(X, nu, params.kappa)
            log_vmf_total += np.nan_to_num(log_vmf, nan=0.0)

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
        # Update s_psi
        s_psi_new = np.zeros_like(params.s_psi)
        for s in range(S):
            accum = np.zeros_like(params.s_psi[:, :, s])
            for t in range(T):
                accum += params.sigma[np.newaxis, :] * params.s_t_nu[:, :, t, s]
            accum += params.epsil[np.newaxis, :] * params.mu
            norms = np.linalg.norm(accum, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            s_psi_new[:, :, s] = accum / norms

        # Check convergence
        all_converged = True
        for s in range(S):
            check = np.sum(s_psi_new[:, :, s] * params.s_psi[:, :, s], axis=0)
            if np.any(1 - check >= epsilon):
                all_converged = False
        params.s_psi = s_psi_new

        # Update sigma
        sigma_new = np.zeros(L)
        for l_idx in range(L):
            accum = 0.0
            count = 0
            for s in range(S):
                for t in range(T):
                    val = np.sum(params.s_psi[:, l_idx, s] * params.s_t_nu[:, l_idx, t, s])
                    if not np.isnan(val):
                        accum += val
                        count += 1
            if count > 0:
                rbar = accum / count
                rbar = min(max(rbar, 1e-10), 1 - 1e-10)
                sigma_new[l_idx] = inv_ad(dim, rbar)
            else:
                sigma_new[l_idx] = params.sigma[l_idx]

        if all_converged and np.mean(np.abs(params.sigma - sigma_new) / np.maximum(params.sigma, 1e-10)) < epsilon:
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

    # Update epsil
    for l_idx in range(L):
        rbar = 0.0
        for s in range(S):
            rbar += np.sum(params.s_psi[:, l_idx, s] * params.mu[:, l_idx])
        rbar /= S
        rbar = min(max(rbar, 1e-10), 1 - 1e-10)
        epsil_new = inv_ad(dim, rbar)
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

    for s in range(S):
        log_vmf_total = np.zeros((N, L))
        for t in range(T):
            X = data[:, :, s, t]
            nu = s_t_nu[:, :, t, s]
            lv = vmf_log_probability(X, nu, kappa)
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
) -> np.ndarray:
    """Compute per-subject EM cost."""
    N, D, S, T = data.shape
    costs = np.zeros(S)
    for s in range(S):
        log_vmf_total = np.zeros((N, settings["num_clusters"]))
        for t in range(T):
            X = data[:, :, s, t]
            nu = params.s_t_nu[:, :, t, s]
            lv = vmf_log_probability(X, nu, params.kappa)
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
