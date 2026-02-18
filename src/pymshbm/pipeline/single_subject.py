"""Simplified single-subject parcellation wrapper.

Ports CBIG_MSHBM_parcellation_single_subject.m.
"""

from pathlib import Path

import numpy as np

from pymshbm.core.individual import generate_individual_parcellation
from pymshbm.io.mat_interop import load_params_final
from pymshbm.types import MSHBMParams


def parcellation_single_subject(
    data: np.ndarray,
    group_priors: MSHBMParams | str | Path,
    neighborhood: np.ndarray,
    w: float = 200.0,
    c: float = 50.0,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate individual parcellation for a single subject.

    Args:
        data: (N, D, 1, T) normalized FC profiles.
        group_priors: MSHBMParams or path to Params_Final.mat.
        neighborhood: (N, max_nb) neighbor indices.
        w: Weight of spatial prior.
        c: Weight of MRF smoothness prior.
        max_iter: Maximum iterations.

    Returns:
        (lh_labels, rh_labels) hemisphere label arrays.
    """
    if isinstance(group_priors, (str, Path)):
        group_priors = load_params_final(group_priors)

    labels = generate_individual_parcellation(
        group_priors=group_priors,
        data=data,
        neighborhood=neighborhood,
        w=w,
        c=c,
        max_iter=max_iter,
    )

    N = len(labels)
    vertex_num = N // 2
    lh_labels = labels[:vertex_num]
    rh_labels = labels[vertex_num:]
    return lh_labels, rh_labels
