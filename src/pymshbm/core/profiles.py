"""FC profile generation, averaging, and initial parameter estimation."""

from pathlib import Path

import numpy as np
import scipy.io as sio

from pymshbm.math.clustering import vmf_series_clustering
from pymshbm.math.correlation import pearson_corr, stable_atanh


def generate_profiles(
    targ_data: np.ndarray,
    seed_data: np.ndarray,
    censor: np.ndarray | None = None,
) -> np.ndarray:
    """Compute FC profiles between target and seed vertices.

    Args:
        targ_data: (T, N_targ) time series for target vertices.
        seed_data: (T, N_seed) time series for seed regions.
        censor: Optional (T,) boolean mask. True = keep, False = exclude.

    Returns:
        (N_targ, N_seed) Fisher-Z transformed correlation profile.
    """
    if censor is not None:
        targ_data = targ_data[censor]
        seed_data = seed_data[censor]
    r = pearson_corr(targ_data, seed_data)
    return stable_atanh(r)


def avg_profiles(profiles: list[np.ndarray]) -> np.ndarray:
    """Average a list of profile matrices element-wise.

    Args:
        profiles: List of (N, D) profile arrays.

    Returns:
        (N, D) averaged profile.
    """
    return np.mean(np.stack(profiles, axis=0), axis=0)


def generate_ini_params(
    avg_profile: np.ndarray,
    num_clusters: int,
    num_init: int = 1000,
    out_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate initialization parameters via vMF clustering on averaged profiles.

    Args:
        avg_profile: (N, D) averaged FC profile matrix (rows = vertices).
        num_clusters: Number of networks K.
        num_init: Number of random initializations.
        out_dir: Optional directory to save group.mat.

    Returns:
        labels: (N,) cluster labels (1-indexed for MATLAB compat, 0 for masked).
        centroids: (D, K) cluster centroids.
    """
    data = avg_profile.T  # (D, N) for clustering
    norms = np.linalg.norm(data, axis=0, keepdims=True)
    mask = norms.ravel() > 0
    data_valid = data[:, mask]
    data_valid /= np.linalg.norm(data_valid, axis=0, keepdims=True)

    labels_valid, centroids = vmf_series_clustering(
        data_valid, num_clusters, num_init=num_init,
    )

    labels = np.zeros(avg_profile.shape[0], dtype=np.int32)
    labels[mask] = labels_valid + 1  # 1-indexed

    if out_dir is not None:
        out_dir = Path(out_dir)
        group_dir = out_dir / "group"
        group_dir.mkdir(parents=True, exist_ok=True)
        n_half = avg_profile.shape[0] // 2
        sio.savemat(
            str(group_dir / "group.mat"),
            {
                "lh_labels": labels[:n_half].reshape(-1, 1),
                "rh_labels": labels[n_half:].reshape(-1, 1),
                "labels": labels.reshape(-1, 1),
            },
        )

    return labels, centroids
