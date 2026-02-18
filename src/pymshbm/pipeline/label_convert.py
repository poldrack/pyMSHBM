"""Label to CIFTI conversion utilities.

Ports label2cifti.m.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio


def label2cifti(
    parcellation_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert label .mat files to a dict of subject labels.

    Args:
        parcellation_dir: Directory containing Ind_parcellation_MSHBM_*.mat files.
        output_dir: Optional output directory for converted files.

    Returns:
        Dict mapping filename -> (lh_labels, rh_labels).
    """
    parcellation_dir = Path(parcellation_dir)
    results = {}

    for mat_file in sorted(parcellation_dir.glob("Ind_parcellation_MSHBM_*.mat")):
        data = sio.loadmat(str(mat_file))
        lh = data["lh_labels"].ravel()
        rh = data["rh_labels"].ravel()
        results[mat_file.stem] = (lh, rh)

    return results
