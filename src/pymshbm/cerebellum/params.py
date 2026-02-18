"""Generate MSHBM parameters from labeled profiles.

Ports CBIG_IndCBM_generate_MSHBM_params.m.
"""

import numpy as np

from pymshbm.math.correlation import normalize_series


def generate_mshbm_params(
    profiles: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    """Compute cluster centers from labeled FC profiles.

    Args:
        profiles: (N, D) FC profiles.
        labels: (N,) integer labels (1-indexed, 0 = ignore).
        num_clusters: Number of clusters K.

    Returns:
        (D, K) cluster centroids (unit-normalized).
    """
    D = profiles.shape[1]
    centroids = np.zeros((D, num_clusters), dtype=np.float64)

    for k in range(1, num_clusters + 1):
        mask = labels == k
        if mask.sum() > 0:
            centroid = profiles[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            centroids[:, k - 1] = centroid

    return centroids
