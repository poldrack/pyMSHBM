"""Parameter validation via homogeneity analysis."""

import numpy as np


def parameters_validation(
    labels: np.ndarray,
    data: np.ndarray,
) -> float:
    """Compute parcellation homogeneity.

    Measures how well vertices within each parcel share similar FC profiles.

    Args:
        labels: (N,) integer labels (0 = medial wall).
        data: (N, D) FC profiles.

    Returns:
        Mean within-parcel correlation (higher = better).
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return 0.0

    total_corr = 0.0
    count = 0

    for label in unique_labels:
        mask = labels == label
        parcel_data = data[mask]
        if parcel_data.shape[0] < 2:
            continue
        # Mean profile
        mean_profile = parcel_data.mean(axis=0)
        norm_mean = np.linalg.norm(mean_profile)
        if norm_mean == 0:
            continue
        mean_profile /= norm_mean
        # Average correlation with mean
        norms = np.linalg.norm(parcel_data, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = parcel_data / norms
        corrs = normed @ mean_profile
        total_corr += corrs.sum()
        count += len(corrs)

    return float(total_corr / count) if count > 0 else 0.0
