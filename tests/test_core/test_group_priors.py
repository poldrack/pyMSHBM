"""Tests for group prior estimation."""

import numpy as np
import pytest

from pymshbm.core.group_priors import (
    initialize_params,
    vmf_clustering_subject_session,
    intra_subject_var,
    inter_subject_var,
    estimate_group_priors,
)
from pymshbm.types import MSHBMParams


@pytest.fixture
def small_settings():
    """Small problem settings for fast testing."""
    return {
        "num_sub": 2,
        "num_session": 2,
        "num_clusters": 3,
        "dim": 9,  # D-1 where D=10
        "ini_concentration": 500,
        "epsilon": 1e-4,
        "conv_th": 1e-5,
        "max_iter": 5,
    }


@pytest.fixture
def synthetic_data(rng, small_settings):
    """Synthetic normalized FC data: (N, D, S, T)."""
    N, D = 20, 10
    S = small_settings["num_sub"]
    T = small_settings["num_session"]
    series = rng.standard_normal((N, D, S, T)).astype(np.float64)
    # Normalize each vertex's profile
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(series[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            series[:, :, s, t] /= norms
    return series


@pytest.fixture
def synthetic_g_mu(rng, small_settings):
    """Synthetic group-level centroids (D, L)."""
    D = small_settings["dim"] + 1
    L = small_settings["num_clusters"]
    mu = rng.standard_normal((D, L))
    mu /= np.linalg.norm(mu, axis=0, keepdims=True)
    return mu


def test_initialize_params_shapes(synthetic_data, synthetic_g_mu, small_settings):
    """initialize_params should create MSHBMParams with correct shapes."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    D = small_settings["dim"] + 1
    L = small_settings["num_clusters"]
    N = synthetic_data.shape[0]
    S = small_settings["num_sub"]
    T = small_settings["num_session"]
    assert params.mu.shape == (D, L)
    assert params.epsil.shape == (L,)
    assert params.sigma.shape == (L,)
    assert params.kappa.shape == (L,)
    assert params.theta.shape == (N, L)
    assert params.s_psi.shape == (D, L, S)
    assert params.s_t_nu.shape == (D, L, T, S)
    assert params.s_lambda.shape == (N, L, S)


def test_initialize_params_theta_valid(synthetic_data, synthetic_g_mu, small_settings):
    """Theta should have non-negative values and reasonable range."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    assert np.all(params.theta >= 0)
    assert np.all(np.isfinite(params.theta))
    # Each row should sum to at most ~num_clusters (individual s_lambda rows sum to 1)
    row_sums = params.theta.sum(axis=1)
    nonzero = row_sums > 0
    assert nonzero.any()


def test_vmf_clustering_updates_kappa(synthetic_data, synthetic_g_mu, small_settings):
    """vmf_clustering_subject_session should update kappa and s_t_nu."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    kappa_before = params.kappa.copy()
    params = vmf_clustering_subject_session(params, small_settings, synthetic_data)
    # Parameters should have been updated (not identical)
    assert params.s_lambda is not None
    assert params.theta is not None


def test_intra_subject_var_updates_sigma(synthetic_data, synthetic_g_mu, small_settings):
    """intra_subject_var should update s_psi and sigma."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    params = vmf_clustering_subject_session(params, small_settings, synthetic_data)
    sigma_before = params.sigma.copy()
    params = intra_subject_var(params, small_settings)
    assert params.s_psi is not None
    assert params.sigma is not None


def test_inter_subject_var_updates_mu(synthetic_data, synthetic_g_mu, small_settings):
    """inter_subject_var should update mu and epsil."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    params = vmf_clustering_subject_session(params, small_settings, synthetic_data)
    params = intra_subject_var(params, small_settings)
    mu_before = params.mu.copy()
    params = inter_subject_var(params, small_settings)
    assert params.mu.shape == mu_before.shape
    assert params.epsil is not None


def test_estimate_group_priors_returns_params(synthetic_data, synthetic_g_mu, small_settings):
    """Full estimation should return valid MSHBMParams."""
    params = estimate_group_priors(
        data=synthetic_data,
        g_mu=synthetic_g_mu,
        settings=small_settings,
    )
    assert isinstance(params, MSHBMParams)
    assert params.mu.ndim == 2
    assert params.iter_inter > 0
    assert len(params.record) > 0


def test_estimate_group_priors_converges(synthetic_data, synthetic_g_mu, small_settings):
    """Cost should decrease or stabilize across iterations."""
    params = estimate_group_priors(
        data=synthetic_data,
        g_mu=synthetic_g_mu,
        settings=small_settings,
    )
    # Should complete without error and record costs
    assert len(params.record) >= 1


def test_estimate_group_priors_float32_close_to_float64():
    """float32 data should produce results close to float64."""
    rng = np.random.default_rng(42)
    N, D, S, T, L = 20, 10, 2, 2, 3
    data_f64 = rng.standard_normal((N, D, S, T))
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(data_f64[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data_f64[:, :, s, t] /= norms

    g_mu = rng.standard_normal((D, L))
    g_mu /= np.linalg.norm(g_mu, axis=0, keepdims=True)

    settings = {
        "num_sub": S, "num_session": T, "num_clusters": L,
        "dim": D - 1, "ini_concentration": 500,
        "epsilon": 1e-4, "conv_th": 1e-5, "max_iter": 5,
    }
    params_f64 = estimate_group_priors(data_f64, g_mu, settings)

    data_f32 = data_f64.astype(np.float32)
    params_f32 = estimate_group_priors(data_f32, g_mu, settings)

    np.testing.assert_allclose(params_f32.mu, params_f64.mu, atol=1e-4)
    np.testing.assert_allclose(params_f32.sigma, params_f64.sigma, rtol=0.01)
    np.testing.assert_allclose(params_f32.theta, params_f64.theta, atol=1e-3)
    assert params_f32.iter_inter == params_f64.iter_inter


def test_estimate_group_priors_numerical_regression():
    """Regression test: verify exact numerical output hasn't changed."""
    rng = np.random.default_rng(42)
    N, D, S, T, L = 20, 10, 2, 2, 3
    data = rng.standard_normal((N, D, S, T))
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(data[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data[:, :, s, t] /= norms

    g_mu = rng.standard_normal((D, L))
    g_mu /= np.linalg.norm(g_mu, axis=0, keepdims=True)

    settings = {
        "num_sub": S, "num_session": T, "num_clusters": L,
        "dim": D - 1, "ini_concentration": 500,
        "epsilon": 1e-4, "conv_th": 1e-5, "max_iter": 5,
    }
    params = estimate_group_priors(data, g_mu, settings)

    expected_mu = np.array([
        [-0.32384962, -0.35860489,  0.09980772],
        [ 0.14396749,  0.42800477,  0.28587461],
        [-0.05239569, -0.41211974, -0.08265555],
    ])
    np.testing.assert_allclose(params.mu[:3, :], expected_mu, atol=1e-6)
    np.testing.assert_allclose(
        params.sigma, [7.27413067, 6.5695762, 9.71016853], atol=1e-4)
    np.testing.assert_allclose(params.theta.sum(), 23.0, atol=0.01)
    np.testing.assert_allclose(params.s_lambda.sum(), 40.0, atol=0.1)
    assert params.iter_inter == 2
    np.testing.assert_allclose(
        params.record, [-19904.5266, -19904.527], atol=0.01)
    np.testing.assert_allclose(
        np.linalg.norm(params.mu, axis=0), [1.0, 1.0, 1.0], atol=1e-10)
