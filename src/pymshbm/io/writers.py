"""Write neuroimaging output files."""

from pathlib import Path

import nibabel as nib
import numpy as np


def write_nifti_labels(
    path: str | Path,
    labels: np.ndarray,
    affine: np.ndarray,
) -> None:
    """Write a label array to a NIfTI file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(labels.astype(np.int32), affine)
    nib.save(img, str(path))
