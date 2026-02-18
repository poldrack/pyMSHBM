"""GIFTI surface file creation.

Ports CBIG_IndCBM_create_surf_gifti.m.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def create_surf_gifti(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str | Path,
) -> None:
    """Create a GIFTI surface file from vertices and faces.

    Args:
        vertices: (V, 3) vertex coordinates.
        faces: (F, 3) triangle face indices.
        output_path: Output .surf.gii path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coord_array = nib.gifti.GiftiDataArray(
        data=vertices.astype(np.float32),
        intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    face_array = nib.gifti.GiftiDataArray(
        data=faces.astype(np.int32),
        intent="NIFTI_INTENT_TRIANGLE",
        datatype="NIFTI_TYPE_INT32",
    )
    img = nib.gifti.GiftiImage(darrays=[coord_array, face_array])
    nib.save(img, str(output_path))
