"""Tests for data writers."""

import numpy as np
import pytest

from pymshbm.io.writers import write_nifti_labels


def test_write_nifti_labels(tmp_path):
    """Write label array to a NIfTI file and read it back."""
    import nibabel as nib
    labels = np.array([1, 2, 3, 0, 2], dtype=np.int32)
    affine = np.eye(4)
    path = tmp_path / "labels.nii.gz"
    write_nifti_labels(path, labels, affine)
    img = nib.load(str(path))
    loaded = np.asarray(img.dataobj).ravel()
    np.testing.assert_array_equal(loaded, labels)


def test_write_nifti_labels_creates_parent_dir(tmp_path):
    """Create parent directories if they don't exist."""
    import nibabel as nib
    labels = np.array([1, 0], dtype=np.int32)
    path = tmp_path / "sub" / "dir" / "labels.nii.gz"
    write_nifti_labels(path, labels, np.eye(4))
    assert path.exists()
