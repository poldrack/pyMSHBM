"""Tests for mat_interop module."""

import numpy as np
import pytest
import scipy.io as sio

from pymshbm.io.mat_interop import load_mat, save_mat, load_params_final
from pymshbm.types import FileFormat, MSHBMParams


def test_save_load_mat_v5_roundtrip(tmp_path):
    """Save and load a dict of arrays in MATLAB v5 format."""
    data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.eye(3)}
    path = tmp_path / "test.mat"
    save_mat(path, data, fmt=FileFormat.MAT_V5)
    loaded = load_mat(path)
    np.testing.assert_array_equal(loaded["x"].ravel(), data["x"])
    np.testing.assert_array_equal(loaded["y"], data["y"])


def test_save_load_mat_v73_roundtrip(tmp_path):
    """Save and load a dict of arrays in MATLAB v7.3 (HDF5) format."""
    data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.eye(3)}
    path = tmp_path / "test.mat"
    save_mat(path, data, fmt=FileFormat.MAT_V73)
    loaded = load_mat(path)
    np.testing.assert_array_almost_equal(loaded["x"].ravel(), data["x"])
    np.testing.assert_array_almost_equal(loaded["y"], data["y"])


def test_save_load_npz_roundtrip(tmp_path):
    """Save and load a dict of arrays in .npz format."""
    data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.eye(3)}
    path = tmp_path / "test.npz"
    save_mat(path, data, fmt=FileFormat.NPZ)
    loaded = load_mat(path)
    np.testing.assert_array_equal(loaded["x"], data["x"])
    np.testing.assert_array_equal(loaded["y"], data["y"])


def test_load_mat_auto_detect_v5(tmp_path):
    """Auto-detect v5 format from file contents."""
    data = {"val": np.array([42.0])}
    path = tmp_path / "test.mat"
    sio.savemat(str(path), data)
    loaded = load_mat(path)
    assert loaded["val"].ravel()[0] == pytest.approx(42.0)


def test_load_mat_auto_detect_npz(tmp_path):
    """Auto-detect .npz format from extension."""
    data = {"val": np.array([42.0])}
    path = tmp_path / "test.npz"
    np.savez(str(path), **data)
    loaded = load_mat(path)
    assert loaded["val"].ravel()[0] == pytest.approx(42.0)


def test_load_params_final_from_example():
    """Load Params_Final.mat from the MATLAB example data."""
    import os
    mat_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "matlab_src", "Kong2019_MSHBM", "examples",
        "input", "priors", "Params_Final.mat"
    )
    if not os.path.exists(mat_path):
        pytest.skip("Example Params_Final.mat not found")
    params = load_params_final(mat_path)
    assert isinstance(params, MSHBMParams)
    assert params.mu.ndim == 2
    assert params.epsil.ndim == 1
    assert params.sigma.ndim == 1
    assert params.kappa.ndim == 1
    assert params.theta.ndim == 2
    assert params.mu.shape[1] == params.theta.shape[1]


def test_load_params_final_missing_file(tmp_path):
    """Raise FileNotFoundError for missing Params_Final.mat."""
    with pytest.raises(FileNotFoundError):
        load_params_final(tmp_path / "nonexistent.mat")


def test_save_mat_unsupported_format(tmp_path):
    """Raise ValueError for unsupported format."""
    with pytest.raises(ValueError):
        save_mat(tmp_path / "test.xyz", {"x": np.array([1])}, fmt=FileFormat.AUTO)
