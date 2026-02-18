"""Compute volume-to-surface functional connectivity.

Ports CBIG_IndCBM_compute_vol2surf_fc.m.
"""

import numpy as np

from pymshbm.math.correlation import pearson_corr, stable_atanh


def compute_vol2surf_fc(
    vol_data: np.ndarray | list[np.ndarray],
    surf_data: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """Compute FC between cerebellar voxels and cortical vertices.

    Computes Pearson correlation then Fisher-Z transform, averaging
    across runs if multiple are provided.

    Args:
        vol_data: (T, M) array or list of (T, M) arrays for cerebellar voxels.
        surf_data: (T, N) array or list of (T, N) arrays for cortical vertices.

    Returns:
        (M, N) Fisher-Z transformed functional connectivity matrix.
    """
    if isinstance(vol_data, np.ndarray):
        vol_data = [vol_data]
        surf_data = [surf_data]

    num_runs = len(vol_data)
    fc_sum = None

    for vol, surf in zip(vol_data, surf_data):
        r = pearson_corr(vol, surf)  # (M, N)
        z = stable_atanh(r)
        if fc_sum is None:
            fc_sum = z
        else:
            fc_sum += z

    return fc_sum / num_runs
