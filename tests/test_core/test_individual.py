"""Tests for individual parcellation generation."""

import numpy as np
import pytest

from pymshbm.core.individual import (
    generate_individual_parcellation,
    build_neighborhood,
    _init_s_lambda,
    _vmf_em,
    _compute_cost,
)
from pymshbm.math.vmf import cdln, vmf_log_probability
from pymshbm.types import MSHBMParams


@pytest.fixture
def small_group_priors(rng):
    """Small synthetic group priors for testing."""
    D, L, N = 10, 3, 40
    mu = rng.standard_normal((D, L))
    mu /= np.linalg.norm(mu, axis=0, keepdims=True)
    theta = rng.dirichlet(np.ones(L), size=N)
    return MSHBMParams(
        mu=mu,
        epsil=np.full(L, 500.0),
        sigma=np.full(L, 500.0),
        theta=theta,
        kappa=np.full(L, 500.0),
    )


@pytest.fixture
def small_test_data(rng):
    """Synthetic test data (N, D, 1, T)."""
    N, D, T = 40, 10, 2
    data = rng.standard_normal((N, D, 1, T))
    for t in range(T):
        norms = np.linalg.norm(data[:, :, 0, t], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        data[:, :, 0, t] /= norms
    return data


@pytest.fixture
def chain_neighborhood():
    """Simple chain neighborhood for 40 vertices (20 lh + 20 rh)."""
    N = 40
    max_nb = 2
    neighborhood = np.full((N, max_nb), -1, dtype=np.int64)
    for i in range(N):
        if i > 0 and (i != 20):  # Don't cross hemispheres
            neighborhood[i, 0] = i - 1
        if i < N - 1 and (i != 19):
            neighborhood[i, 1] = i + 1
    return neighborhood


def test_generate_individual_parcellation_shape(
    small_group_priors, small_test_data, chain_neighborhood,
):
    """Output should be (N,) labels with values in [0, L]."""
    N = small_test_data.shape[0]
    L = small_group_priors.mu.shape[1]
    labels = generate_individual_parcellation(
        group_priors=small_group_priors,
        data=small_test_data,
        neighborhood=chain_neighborhood,
        w=10.0,
        c=5.0,
        max_iter=3,
    )
    assert labels.shape == (N,)
    assert np.all(labels >= 0)
    assert np.all(labels <= L)


def test_generate_individual_parcellation_nonzero(
    small_group_priors, small_test_data, chain_neighborhood,
):
    """Most labels should be non-zero (not medial wall)."""
    labels = generate_individual_parcellation(
        group_priors=small_group_priors,
        data=small_test_data,
        neighborhood=chain_neighborhood,
        w=10.0,
        c=5.0,
        max_iter=3,
    )
    assert np.sum(labels > 0) > 0


def test_generate_individual_parcellation_with_medial_wall(rng):
    """MRF should handle medial wall vertices without IndexError.

    When some vertices are zeroed out (medial wall), the neighborhood matrix
    still references full-surface indices. The MRF term must use full-surface
    s_lambda so neighbor lookups don't go out of bounds.
    """
    N, D, L, T = 50, 10, 3, 2

    # Group priors
    mu = rng.standard_normal((D, L))
    mu /= np.linalg.norm(mu, axis=0, keepdims=True)
    theta = rng.dirichlet(np.ones(L), size=N)
    priors = MSHBMParams(
        mu=mu,
        epsil=np.full(L, 500.0),
        sigma=np.full(L, 500.0),
        theta=theta,
        kappa=np.full(L, 500.0),
    )

    # Data with medial wall: vertices 10-19 are zeroed out
    data = rng.standard_normal((N, D, 1, T))
    for t in range(T):
        norms = np.linalg.norm(data[:, :, 0, t], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        data[:, :, 0, t] /= norms
    medial_wall = np.arange(10, 20)
    data[medial_wall, :, :, :] = 0.0

    # Full-surface neighborhood — indices span 0..N-1
    neighborhood = np.full((N, 2), -1, dtype=np.int64)
    for i in range(N):
        if i > 0:
            neighborhood[i, 0] = i - 1
        if i < N - 1:
            neighborhood[i, 1] = i + 1

    labels = generate_individual_parcellation(
        group_priors=priors,
        data=data,
        neighborhood=neighborhood,
        w=10.0,
        c=5.0,
        max_iter=3,
    )

    assert labels.shape == (N,)
    # Medial wall vertices should get label 0
    assert np.all(labels[medial_wall] == 0)
    # Non-medial-wall vertices should have valid labels
    assert np.all(labels >= 0)
    assert np.all(labels <= L)


def test_generate_individual_parcellation_deterministic(
    small_group_priors, small_test_data, chain_neighborhood,
):
    """Same inputs should produce identical labels across two runs."""
    kwargs = dict(
        group_priors=small_group_priors,
        data=small_test_data,
        neighborhood=chain_neighborhood,
        w=10.0,
        c=5.0,
        max_iter=3,
    )
    labels1 = generate_individual_parcellation(**kwargs)
    labels2 = generate_individual_parcellation(**kwargs)
    np.testing.assert_array_equal(labels1, labels2)


def test_compute_log_vmf_cached_matches_loop(rng):
    """Precomputed log_c + einsum should match the naive per-timepoint loop."""
    N, D, L, T = 30, 10, 3, 4
    data = rng.standard_normal((N, D, 1, T))
    for t in range(T):
        norms = np.linalg.norm(data[:, :, 0, t], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        data[:, :, 0, t] /= norms

    nu = rng.standard_normal((D, L))
    nu /= np.linalg.norm(nu, axis=0, keepdims=True)
    kappa = np.full(L, 500.0)
    dim = D - 1

    # Naive loop (without precomputed log_c)
    log_vmf_loop = np.zeros((N, L))
    for t in range(T):
        X = data[:, :, 0, t]
        lv = vmf_log_probability(X, nu, kappa)
        log_vmf_loop += np.nan_to_num(lv, nan=0.0)

    # Precomputed log_c + loop
    log_c = cdln(kappa, dim)
    log_vmf_cached = np.zeros((N, L))
    for t in range(T):
        X = data[:, :, 0, t]
        lv = vmf_log_probability(X, nu, kappa, log_c=log_c)
        log_vmf_cached += np.nan_to_num(lv, nan=0.0)

    np.testing.assert_allclose(log_vmf_cached, log_vmf_loop, atol=1e-5)


def test_vectorized_vmf_across_sessions_matches_loop(rng):
    """Einsum-based vectorized vMF computation should match per-timepoint loop."""
    N, D, L, T = 30, 10, 3, 4
    data = rng.standard_normal((N, D, 1, T))
    for t in range(T):
        norms = np.linalg.norm(data[:, :, 0, t], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        data[:, :, 0, t] /= norms

    nu = rng.standard_normal((D, L))
    nu /= np.linalg.norm(nu, axis=0, keepdims=True)
    kappa = np.full(L, 500.0)
    dim = D - 1
    log_c = cdln(kappa, dim)

    # Loop-based
    log_vmf_loop = np.zeros((N, L))
    for t in range(T):
        X = data[:, :, 0, t]
        dot_product = X @ nu
        log_vmf_loop += log_c[np.newaxis, :] + kappa[np.newaxis, :] * dot_product

    # Einsum-based: data_s is (N, D, T), nu_s is (D, L)
    data_s = data[:, :, 0, :]  # (N, D, T)
    dots = np.einsum('ndt,dl->ntl', data_s, nu)  # (N, T, L)
    log_vmf_einsum = T * log_c[np.newaxis, :] + kappa[np.newaxis, :] * dots.sum(axis=1)

    np.testing.assert_allclose(log_vmf_einsum, log_vmf_loop, atol=1e-5)


def test_build_neighborhood_shape():
    """build_neighborhood should return (N, max_nb) int array."""
    N = 20
    # Simulate a simple adjacency list
    adjacency = {i: [max(0, i-1), min(N-1, i+1)] for i in range(N)}
    nb = build_neighborhood(adjacency, N)
    assert nb.shape[0] == N
    assert nb.shape[1] >= 1
    assert nb.dtype == np.int64
