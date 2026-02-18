"""Read fMRI data from NIfTI, CIFTI, and .mat files."""

from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io as sio

from pymshbm.types import DataBundle


def read_fmri(path: str | Path) -> DataBundle:
    """Read fMRI data, dispatching on file extension.

    Supports:
      - .mat files containing a `profile_mat` variable
      - .nii / .nii.gz NIfTI files
      - .dtseries.nii CIFTI files

    Returns a DataBundle with series shaped (num_vertices_or_voxels, num_timepoints_or_ROIs).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    name = path.name

    if name.endswith(".mat"):
        return _read_mat(path)
    elif name.endswith(".dtseries.nii"):
        return _read_cifti(path)
    elif name.endswith(".nii.gz") or name.endswith(".nii"):
        return _read_nifti(path)
    else:
        raise ValueError(f"Unsupported file extension: {name}")


def _read_mat(path: Path) -> DataBundle:
    """Read FC profile matrix from .mat file."""
    raw = sio.loadmat(str(path))
    vol = np.asarray(raw["profile_mat"], dtype=np.float32)
    return DataBundle(series=vol)


def _read_nifti(path: Path) -> DataBundle:
    """Read NIfTI file and reshape to (num_voxels, num_timepoints)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    vol_size = data.shape
    spatial = int(np.prod(vol_size[:3]))
    rest = int(np.prod(vol_size)) // spatial
    vol = data.reshape(spatial, rest)
    return DataBundle(series=vol)


def _read_cifti(path: Path) -> DataBundle:
    """Read CIFTI dtseries file."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return DataBundle(series=data)
