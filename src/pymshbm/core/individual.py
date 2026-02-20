"""Generate individual parcellations using group priors + MRF smoothness."""

import numpy as np

from pymshbm.math.mrf import build_sparse_adjacency, v_lambda_product
from pymshbm.math.vmf import cdln, inv_ad_batch, vmf_log_probability
from pymshbm.types import MSHBMParams


def generate_individual_parcellation(
    group_priors: MSHBMParams,
    data: np.ndarray,
    neighborhood: np.ndarray,
    w: float = 100.0,
    c: float = 50.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Generate individual parcellation for a single subject.

    Args:
        group_priors: Estimated group priors (mu, epsil, sigma, theta, kappa).
        data: (N, D, 1, T) normalized FC profiles for one subject.
        neighborhood: (N_valid, max_nb) neighbor indices, -1 = invalid.
        w: Weight of spatial prior theta.
        c: Weight of MRF smoothness prior.
        max_iter: Maximum EM iterations.

    Returns:
        (N,) integer labels (1-indexed, 0 = medial wall).
    """
    N, D, S, T = data.shape
    L = group_priors.mu.shape[1]
    dim = D - 1

    if dim < 1200:
        ini_concentration = 500
    elif dim < 1800:
        ini_concentration = 650
    else:
        raise ValueError(f"Dimension {dim} too high")

    # Initialize from group priors (read-only params don't need copy)
    mu = group_priors.mu
    epsil = group_priors.epsil
    sigma = group_priors.sigma
    theta = group_priors.theta
    kappa = np.full(L, ini_concentration, dtype=np.float64)
    s_psi = np.tile(mu[:, :, np.newaxis], (1, 1, 1))
    s_t_nu = np.tile(mu[:, :, np.newaxis, np.newaxis], (1, 1, T, 1))

    # Initialize s_lambda with precomputed log_c
    log_c = cdln(kappa, dim)
    s_lambda = _init_s_lambda(data, s_t_nu, kappa, dim, L, log_c=log_c)

    # Identify valid (non-medial-wall) vertices
    valid_mask = s_lambda[:, :, 0].sum(axis=1) != 0

    # Build V_same and V_diff
    v_same = np.zeros_like(neighborhood, dtype=np.float64)
    v_diff = np.ones_like(neighborhood, dtype=np.float64)

    # Build sparse adjacency once for MRF fast-path
    adj_sparse = build_sparse_adjacency(neighborhood)

    cost = 0.0
    for iteration in range(1, max_iter + 1):
        # Reset kappa and s_t_nu each iteration
        kappa = np.full(L, ini_concentration, dtype=np.float64)
        s_t_nu = np.tile(mu[:, :, np.newaxis, np.newaxis], (1, 1, T, 1))

        # Session-level clustering
        s_lambda, kappa, s_t_nu, log_vmf_cached = _vmf_em(
            data, s_lambda, kappa, s_t_nu, s_psi, sigma, theta,
            neighborhood, v_same, v_diff, w, c, dim, L, ini_concentration,
            valid_mask, adj_sparse=adj_sparse,
        )

        # Intra-subject update (single iteration for individual parcellation)
        s_psi = _update_s_psi(s_t_nu, sigma, epsil, mu)

        # Convergence check (reuse cached log_vmf from _vmf_em)
        update_cost = _compute_cost(
            data, s_lambda, s_t_nu, s_psi, sigma, mu, epsil,
            kappa, theta, dim, log_vmf_cached=log_vmf_cached,
        )
        if iteration > 1 and abs(cost) > 0:
            if abs((update_cost - cost) / cost) <= 1e-4:
                break
        if iteration >= max_iter:
            break
        cost = update_cost

    # Generate labels from s_lambda
    labels = np.zeros(N, dtype=np.int32)
    active = s_lambda[:, :, 0].sum(axis=1) != 0
    if np.any(active):
        labels[active] = np.argmax(s_lambda[active, :, 0], axis=1) + 1

    return labels


def build_neighborhood(
    adjacency: dict[int, list[int]],
    num_vertices: int,
) -> np.ndarray:
    """Build neighborhood array from adjacency dict.

    Args:
        adjacency: Dict mapping vertex index -> list of neighbor indices.
        num_vertices: Total number of vertices.

    Returns:
        (num_vertices, max_neighbors) int64 array, -1 for invalid.
    """
    max_nb = max(len(v) for v in adjacency.values())
    neighborhood = np.full((num_vertices, max_nb), -1, dtype=np.int64)
    for i, neighbors in adjacency.items():
        for j, nb in enumerate(neighbors):
            if nb != i:
                neighborhood[i, j] = nb
    return neighborhood


def _init_s_lambda(data, s_t_nu, kappa, dim, L, log_c=None):
    """Initialize s_lambda via vMF log likelihood."""
    N, D, S, T = data.shape
    if log_c is None:
        log_c = cdln(kappa, dim)
    s_lambda = np.zeros((N, L, S))
    for s in range(S):
        log_vmf_total = _compute_log_vmf_vectorized(
            data, s_t_nu, kappa, s, log_c,
        )

        log_vmf_total -= log_vmf_total.max(axis=1, keepdims=True)
        sl = np.exp(log_vmf_total)
        row_sums = sl.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        sl /= row_sums

        zero_mask = np.all(log_vmf_total == 0, axis=1)
        sl[zero_mask] = 0.0
        s_lambda[:, :, s] = sl
    return s_lambda


def _compute_log_vmf_vectorized(data, s_t_nu, kappa, s, log_c):
    """Compute sum of vMF log-likelihoods across timepoints using einsum.

    Args:
        data: (N, D, S, T) data array.
        s_t_nu: (D, L, T, S) mean directions.
        kappa: (L,) concentrations.
        s: Session index.
        log_c: (L,) precomputed log normalizing constants.

    Returns:
        (N, L) total log-likelihood summed over T.
    """
    T = data.shape[3]
    data_s = data[:, :, s, :]  # (N, D, T)
    nu_s = s_t_nu[:, :, :, s]  # (D, L, T)
    # Compute dot products for all timepoints at once
    dots = np.einsum('ndt,dlt->ntl', data_s, nu_s)  # (N, T, L)
    dot_sum = dots.sum(axis=1)  # (N, L)
    log_vmf_total = T * log_c[np.newaxis, :] + kappa[np.newaxis, :] * dot_sum
    return np.nan_to_num(log_vmf_total, nan=0.0)


def _vmf_em(data, s_lambda, kappa, s_t_nu, s_psi, sigma, theta,
            neighborhood, v_same, v_diff, w, c, dim, L, ini_concentration,
            valid_mask, adj_sparse=None):
    """Inner EM loop with MRF smoothness."""
    N, D, S, T = data.shape
    epsilon = 1e-4
    tiny = np.finfo(float).tiny

    # Hoist constant log_theta before E-step loop (Opt 4)
    log_theta = np.log(np.maximum(theta, tiny))

    cost = np.zeros(S)
    log_vmf_cached = {}
    for iter_em in range(1, 102):
        # M-step: update kappa and s_t_nu
        for _ in range(50):
            kappa_num = 0.0
            kappa_den = 0.0
            for s in range(S):
                for t in range(T):
                    X = data[:, :, s, t]
                    nu = s_t_nu[:, :, t, s]
                    sl = s_lambda[:, :, s]
                    kappa_num += np.nansum(sl * (X @ nu))
                kappa_den += np.nansum(s_lambda[:, :, s])

            if kappa_den > 0:
                rbar = kappa_num / (kappa_den * T)
                rbar = min(max(rbar, 1e-10), 1 - 1e-10)
                # Opt 7: use inv_ad_batch instead of scalar inv_ad
                kappa_new = float(inv_ad_batch(dim, np.array([rbar]))[0])
                kappa_new = max(kappa_new, ini_concentration)
                if np.isinf(kappa_new):
                    kappa_new = kappa[0]
                kappa = np.full(L, kappa_new)

            converged = True
            for s in range(S):
                for t in range(T):
                    X = data[:, :, s, t]
                    sl = s_lambda[:, :, s]
                    lambda_X = kappa[np.newaxis, :] * (X.T @ sl) + \
                        sigma[np.newaxis, :] * s_psi[:, :, s]
                    norms = np.linalg.norm(lambda_X, axis=0, keepdims=True)
                    norms[norms == 0] = 1.0
                    nu_new = lambda_X / norms
                    check = np.sum(nu_new * s_t_nu[:, :, t, s], axis=0)
                    if np.any(1 - np.nan_to_num(check, nan=1.0) >= epsilon):
                        converged = False
                    s_t_nu[:, :, t, s] = nu_new

            if converged:
                break

        # Opt 2: Precompute log_c once after M-step
        log_c = cdln(kappa, dim)

        # Opt 1: Cache log_vmf after M-step (reuse across all 100 E-step iters)
        for s in range(S):
            log_vmf_cached[s] = _compute_log_vmf_vectorized(
                data, s_t_nu, kappa, s, log_c,
            )

        # E-step with MRF
        for _ in range(100):
            for s in range(S):
                # Reuse cached log_vmf (Opt 1)
                log_vmf_total = log_vmf_cached[s]

                # MRF term (Opt 6: sparse fast-path)
                V_temp = np.zeros((N, L))
                if c > 0:
                    V_temp = v_lambda_product(
                        neighborhood, v_same, v_diff, s_lambda[:, :, s],
                        adjacency_matrix=adj_sparse,
                    )

                log_posterior = log_vmf_total + w * log_theta - 2 * c * V_temp

                log_posterior -= log_posterior.max(axis=1, keepdims=True)
                sl_new = np.exp(log_posterior)
                row_sums = sl_new.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                sl_new /= row_sums
                sl_new[~valid_mask] = 0.0

                change = np.mean(np.abs(sl_new - s_lambda[:, :, s]))
                s_lambda[:, :, s] = sl_new

            if change <= epsilon:
                break

        # Cost convergence (reuse cached log_vmf, Opt 1)
        update_cost = np.zeros(S)
        for s in range(S):
            log_vmf_total = log_vmf_cached[s]
            sl = s_lambda[:, :, s]
            log_sl = np.log(np.maximum(sl, tiny))
            update_cost[s] = np.nansum(sl * log_vmf_total) + \
                np.nansum(sl * w * log_theta) - np.nansum(sl * log_sl)

        with np.errstate(divide="ignore", invalid="ignore"):
            converged_subs = np.where(
                np.abs(cost) > 0,
                np.abs((update_cost - cost) / cost) <= epsilon,
                False,
            )
        if np.all(converged_subs) and iter_em > 1:
            break
        if iter_em > 100:
            break
        cost = update_cost

    return s_lambda, kappa, s_t_nu, log_vmf_cached


def _update_s_psi(s_t_nu, sigma, epsil, mu):
    """Single-iteration s_psi update for individual parcellation."""
    D, L, T, S = s_t_nu.shape
    s_psi = np.zeros((D, L, S))
    for s in range(S):
        accum = np.zeros((D, L))
        for t in range(T):
            accum += sigma[np.newaxis, :] * s_t_nu[:, :, t, s]
        accum += epsil[np.newaxis, :] * mu
        norms = np.linalg.norm(accum, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        s_psi[:, :, s] = accum / norms
    return s_psi


def _compute_cost(data, s_lambda, s_t_nu, s_psi, sigma, mu, epsil,
                  kappa, theta, dim, log_vmf_cached=None):
    """Compute total cost for convergence checking."""
    N, D, S, T = data.shape
    tiny = np.finfo(float).tiny
    log_theta = np.log(np.maximum(theta, tiny))
    total = 0.0
    for s in range(S):
        if log_vmf_cached is not None and s in log_vmf_cached:
            log_vmf_total = log_vmf_cached[s]
        else:
            log_c = cdln(kappa, dim)
            log_vmf_total = _compute_log_vmf_vectorized(
                data, s_t_nu, kappa, s, log_c,
            )
        sl = s_lambda[:, :, s]
        log_sl = np.log(np.maximum(sl, tiny))
        total += np.nansum(sl * log_vmf_total) + \
            np.nansum(sl * log_theta) - np.nansum(sl * log_sl)
    return total
