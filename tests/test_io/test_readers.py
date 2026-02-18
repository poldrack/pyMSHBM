"""Tests for fMRI readers."""

import numpy as np
import pytest
import scipy.io as sio

from pymshbm.io.readers import read_fmri
from pymshbm.types import DataBundle


def test_read_fmri_mat(tmp_path):
    """Read an fMRI profile from a .mat file containing profile_mat."""
    profile_mat = np.random.default_rng(0).standard_normal((100, 20)).astype(np.float32)
    path = tmp_path / "profile.mat"
    sio.savemat(str(path), {"profile_mat": profile_mat})
    bundle = read_fmri(path)
    assert isinstance(bundle, DataBundle)
    np.testing.assert_array_almost_equal(bundle.series, profile_mat)
    assert bundle.num_vertices == 20
    assert bundle.num_timepoints == 100


def test_read_fmri_nifti(tmp_path):
    """Read fMRI data from a NIfTI file."""
    import nibabel as nib
    data = np.random.default_rng(0).standard_normal((10, 1, 1, 5)).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "test.nii.gz"
    nib.save(img, str(path))
    bundle = read_fmri(path)
    assert isinstance(bundle, DataBundle)
    assert bundle.series.shape == (10, 5)


def test_read_fmri_missing_file(tmp_path):
    """Raise FileNotFoundError for missing input."""
    with pytest.raises(FileNotFoundError):
        read_fmri(tmp_path / "nonexistent.nii.gz")


def test_read_fmri_unsupported_extension(tmp_path):
    """Raise ValueError for unsupported file extension."""
    path = tmp_path / "data.xyz"
    path.write_text("dummy")
    with pytest.raises(ValueError):
        read_fmri(path)
