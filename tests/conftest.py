"""Shared fixtures for pymshbm tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_series(rng):
    """Synthetic fMRI time series (T=100 timepoints, N=50 vertices)."""
    return rng.standard_normal((100, 50)).astype(np.float64)


@pytest.fixture
def synthetic_profiles(rng):
    """Synthetic FC profiles (D=20 features, N=50 vertices)."""
    profiles = rng.standard_normal((20, 50)).astype(np.float64)
    norms = np.linalg.norm(profiles, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return profiles / norms


@pytest.fixture
def synthetic_mu(rng):
    """Synthetic group-level connectivity profiles (D=20, L=5)."""
    mu = rng.standard_normal((20, 5)).astype(np.float64)
    norms = np.linalg.norm(mu, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return mu / norms


@pytest.fixture
def num_clusters():
    """Default number of clusters for tests."""
    return 5


@pytest.fixture
def num_vertices():
    """Default number of vertices for tests."""
    return 50


@pytest.fixture
def num_features():
    """Default feature dimensionality for tests."""
    return 20


@pytest.fixture
def tmp_project_dir(tmp_path):
    """Temporary project directory with standard subdirectories."""
    for subdir in ["profiles", "priors", "estimate", "ind_parcellation"]:
        (tmp_path / subdir).mkdir()
    return tmp_path
