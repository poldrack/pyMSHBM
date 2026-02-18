"""Tests for profile generation functions."""

import numpy as np
import pytest
import scipy.io as sio

from pymshbm.core.profiles import generate_profiles, avg_profiles, generate_ini_params


def test_generate_profiles_shape(tmp_path):
    """generate_profiles should produce a profile matrix of correct shape."""
    rng = np.random.default_rng(42)
    n_targ = 100  # target vertices
    n_seed = 20   # seed vertices
    n_timepoints = 50
    # Simulate target and seed time series
    targ_data = rng.standard_normal((n_timepoints, n_targ)).astype(np.float32)
    seed_data = rng.standard_normal((n_timepoints, n_seed)).astype(np.float32)
    profile = generate_profiles(targ_data, seed_data)
    assert profile.shape == (n_targ, n_seed)


def test_generate_profiles_correlation_range(tmp_path):
    """Profile values should be Fisher-Z transformed correlations."""
    rng = np.random.default_rng(42)
    targ = rng.standard_normal((200, 50)).astype(np.float32)
    seed = rng.standard_normal((200, 10)).astype(np.float32)
    profile = generate_profiles(targ, seed)
    # Fisher-Z transform of r in [-1,1] gives values typically in [-3, 3]
    assert np.all(np.isfinite(profile))


def test_generate_profiles_with_censor(tmp_path):
    """Censor mask should exclude timepoints from correlation."""
    rng = np.random.default_rng(42)
    n_tp = 100
    targ = rng.standard_normal((n_tp, 30)).astype(np.float32)
    seed = rng.standard_normal((n_tp, 10)).astype(np.float32)
    censor = np.ones(n_tp, dtype=bool)
    censor[50:] = False  # censor second half
    profile_full = generate_profiles(targ, seed)
    profile_censored = generate_profiles(targ, seed, censor=censor)
    # Different censoring should produce different profiles
    assert not np.allclose(profile_full, profile_censored)


def test_avg_profiles():
    """avg_profiles should compute element-wise mean across a list of profiles."""
    rng = np.random.default_rng(42)
    p1 = rng.standard_normal((50, 20))
    p2 = rng.standard_normal((50, 20))
    p3 = rng.standard_normal((50, 20))
    result = avg_profiles([p1, p2, p3])
    expected = (p1 + p2 + p3) / 3
    np.testing.assert_array_almost_equal(result, expected)


def test_avg_profiles_single():
    """Averaging a single profile should return that profile."""
    rng = np.random.default_rng(42)
    p = rng.standard_normal((50, 20))
    result = avg_profiles([p])
    np.testing.assert_array_almost_equal(result, p)


def test_generate_ini_params_shape():
    """generate_ini_params should return labels and centroids."""
    rng = np.random.default_rng(42)
    N, D, K = 100, 20, 5
    avg_profile = rng.standard_normal((N, D))
    norms = np.linalg.norm(avg_profile, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    avg_profile /= norms
    labels, centroids = generate_ini_params(avg_profile, K, num_init=3)
    assert labels.shape == (N,)
    assert centroids.shape == (D, K)
    # Labels are 1-indexed (MATLAB convention), 0 = medial wall
    assert set(labels).issubset(set(range(K + 1)))


def test_generate_ini_params_saves_group_mat(tmp_path):
    """generate_ini_params should save group.mat when out_dir is provided."""
    rng = np.random.default_rng(42)
    N, D, K = 80, 15, 3
    avg_profile = rng.standard_normal((N, D))
    norms = np.linalg.norm(avg_profile, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    avg_profile /= norms
    labels, centroids = generate_ini_params(avg_profile, K, num_init=2, out_dir=tmp_path)
    group_file = tmp_path / "group" / "group.mat"
    assert group_file.exists()
    data = sio.loadmat(str(group_file))
    assert "lh_labels" in data or "labels" in data
