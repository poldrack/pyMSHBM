"""Tests for volume-to-surface functional connectivity."""

import numpy as np
import pytest

from pymshbm.cerebellum.vol2surf_fc import compute_vol2surf_fc


def test_basic_shape():
    """Output should be (M, N) where M=cerebellar voxels, N=cortical vertices."""
    rng = np.random.default_rng(42)
    M, N, T = 20, 30, 100
    vol_data = rng.standard_normal((T, M))
    surf_data = rng.standard_normal((T, N))
    fc = compute_vol2surf_fc(vol_data, surf_data)
    assert fc.shape == (M, N)


def test_fisher_z_range():
    """FC values should be Fisher-Z transformed (finite, reasonable range)."""
    rng = np.random.default_rng(42)
    vol = rng.standard_normal((200, 10))
    surf = rng.standard_normal((200, 15))
    fc = compute_vol2surf_fc(vol, surf)
    assert np.all(np.isfinite(fc))


def test_multi_run_averaging():
    """Multiple runs should be averaged."""
    rng = np.random.default_rng(42)
    T, M, N = 100, 5, 8
    runs_vol = [rng.standard_normal((T, M)) for _ in range(3)]
    runs_surf = [rng.standard_normal((T, N)) for _ in range(3)]
    fc = compute_vol2surf_fc(runs_vol, runs_surf)
    assert fc.shape == (M, N)
    assert np.all(np.isfinite(fc))


def test_single_run_equals_direct():
    """Single run should match direct computation."""
    rng = np.random.default_rng(42)
    vol = rng.standard_normal((100, 5))
    surf = rng.standard_normal((100, 8))
    fc_single = compute_vol2surf_fc(vol, surf)
    fc_list = compute_vol2surf_fc([vol], [surf])
    np.testing.assert_array_almost_equal(fc_single, fc_list)
