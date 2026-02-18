"""Cerebellum parcellation orchestrator.

Ports CBIG_IndCBM_cerebellum_parcellation.m.
"""

import numpy as np

from pymshbm.cerebellum.wta import winner_take_all


def cerebellum_parcellation(
    surf_labels: np.ndarray,
    vol2surf_fc: np.ndarray,
    top_x: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate cerebellar parcellation using winner-take-all.

    Args:
        surf_labels: (N,) cortical parcellation labels (0 = medial wall).
        vol2surf_fc: (M, N) FC matrix or path to .mat file.
        top_x: Number of top correlated vertices to consider.

    Returns:
        labels: (M,) cerebellar parcellation labels.
        confidence: (M,) confidence scores.
    """
    return winner_take_all(surf_labels, vol2surf_fc, top_x=top_x)
