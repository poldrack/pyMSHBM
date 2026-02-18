"""Winner-take-all cerebellum parcellation.

Ports CBIG_IndCBM_wta.m.
"""

import numpy as np


def winner_take_all(
    surf_labels: np.ndarray,
    vol2surf_fc: np.ndarray,
    top_x: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign cerebellar voxel labels via winner-take-all.

    For each cerebellar voxel, find the top_x most correlated cortical
    vertices and assign the most frequent label among them.

    Args:
        surf_labels: (N,) cortical parcellation labels (0 = medial wall).
        vol2surf_fc: (M, N) functional connectivity matrix.
        top_x: Number of top correlated vertices to consider.

    Returns:
        assign: (M,) labels for each cerebellar voxel.
        confidence: (M,) confidence scores.
    """
    num_voxels, num_vertices = vol2surf_fc.shape

    # Remove medial wall
    surf_mask = surf_labels != 0
    surf_labels_valid = surf_labels[surf_mask]

    assign = np.zeros(num_voxels, dtype=np.float64)
    confidence = np.zeros(num_voxels, dtype=np.float64)

    for i in range(num_voxels):
        corr = vol2surf_fc[i, :]
        corr_valid = corr[surf_mask]

        if np.all(np.isnan(corr_valid)):
            assign[i] = np.nan
            confidence[i] = np.nan
            continue

        corr_valid = np.where(np.isnan(corr_valid), -np.inf, corr_valid)
        top_idx = np.argsort(corr_valid)[::-1][:top_x]
        top_labels = surf_labels_valid[top_idx]

        unique_labels, counts = np.unique(top_labels, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]

        assign[i] = unique_labels[sorted_idx[0]]

        if len(unique_labels) < 2:
            confidence[i] = 1.0
        else:
            n_first = counts[sorted_idx[0]]
            n_second = counts[sorted_idx[1]]
            confidence[i] = 1.0 - n_second / n_first

    return assign, confidence
