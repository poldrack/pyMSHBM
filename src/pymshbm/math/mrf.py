"""MRF V*lambda product (replaces compiled MEX function)."""

import numpy as np


def v_lambda_product(
    neighborhood: np.ndarray,
    v_same: np.ndarray,
    v_diff: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    """Compute V*lambda product for MRF smoothness prior.

    For each vertex m, cluster k:
        V_lam[m, k] = sum over neighbors n of:
            sum over clusters zj:
                if zj == k: v_same[m, n] * lam[neighbor, zj]
                else:       v_diff[m, n] * lam[neighbor, zj]

    Args:
        neighborhood: (N, max_neighbors) vertex neighbor indices; -1 = invalid.
        v_same: (N, max_neighbors) cost for same-label neighbors.
        v_diff: (N, max_neighbors) cost for different-label neighbors.
        lam: (N, K) posterior probability matrix.

    Returns:
        (N, K) weighted MRF potential.
    """
    N, K = lam.shape
    max_nb = neighborhood.shape[1]
    v_lam = np.zeros((N, K), dtype=np.float64)

    valid = neighborhood >= 0  # (N, max_nb)

    for n_idx in range(max_nb):
        nb = neighborhood[:, n_idx]  # (N,)
        mask = valid[:, n_idx]  # (N,)
        if not np.any(mask):
            continue

        nb_valid = nb[mask]
        lam_nb = lam[nb_valid, :]  # (n_valid, K)
        lam_nb_sum = lam_nb.sum(axis=1, keepdims=True)  # (n_valid, 1)

        vs = v_same[mask, n_idx]  # (n_valid,)
        vd = v_diff[mask, n_idx]  # (n_valid,)

        # For each cluster k:
        # contribution = v_same * lam_nb[k] + v_diff * sum_{j!=k} lam_nb[j]
        #              = v_same * lam_nb[k] + v_diff * (lam_nb_sum - lam_nb[k])
        #              = (v_same - v_diff) * lam_nb[k] + v_diff * lam_nb_sum
        diff_coeff = (vs - vd)[:, np.newaxis]  # (n_valid, 1)
        sum_coeff = vd[:, np.newaxis]  # (n_valid, 1)

        v_lam[mask, :] += diff_coeff * lam_nb + sum_coeff * lam_nb_sum

    return v_lam
