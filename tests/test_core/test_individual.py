"""Tests for individual parcellation generation."""

import numpy as np
import pytest

from pymshbm.core.individual import (
    generate_individual_parcellation,
    build_neighborhood,
)
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


def test_build_neighborhood_shape():
    """build_neighborhood should return (N, max_nb) int array."""
    N = 20
    # Simulate a simple adjacency list
    adjacency = {i: [max(0, i-1), min(N-1, i+1)] for i in range(N)}
    nb = build_neighborhood(adjacency, N)
    assert nb.shape[0] == N
    assert nb.shape[1] >= 1
    assert nb.dtype == np.int64
