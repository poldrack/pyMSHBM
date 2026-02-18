"""Tests for correlation utilities."""

import numpy as np
import pytest

from pymshbm.math.correlation import pearson_corr, stable_atanh, normalize_series


def test_pearson_corr_identity():
    """Correlation of X with itself should be identity."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    result = pearson_corr(X, X)
    np.testing.assert_array_almost_equal(np.diag(result), np.ones(5))


def test_pearson_corr_shape():
    """Output shape should be (M, N) for inputs (T, M) and (T, N)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    Y = rng.standard_normal((100, 3))
    result = pearson_corr(X, Y)
    assert result.shape == (5, 3)


def test_pearson_corr_range():
    """Correlation values should be in [-1, 1]."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    Y = rng.standard_normal((50, 8))
    result = pearson_corr(X, Y)
    assert np.all(result >= -1 - 1e-10)
    assert np.all(result <= 1 + 1e-10)


def test_pearson_corr_known():
    """Test with perfectly correlated signals."""
    X = np.array([[1, 2, 3, 4, 5]], dtype=float).T
    Y = np.array([[2, 4, 6, 8, 10]], dtype=float).T
    result = pearson_corr(X, Y)
    assert result[0, 0] == pytest.approx(1.0, abs=1e-10)


def test_stable_atanh_within_bounds():
    """stable_atanh should be finite for r in [-1, 1]."""
    r = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    result = stable_atanh(r)
    assert np.all(np.isfinite(result))


def test_stable_atanh_matches_arctanh_interior():
    """For interior values, stable_atanh should match np.arctanh."""
    r = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
    np.testing.assert_array_almost_equal(stable_atanh(r), np.arctanh(r))


def test_normalize_series_unit_norm():
    """Non-zero rows should have unit L2 norm after normalization."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 50))
    result = normalize_series(data)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(10))


def test_normalize_series_zero_mean():
    """Rows should be zero-mean after normalization."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 50)) + 5.0
    result = normalize_series(data)
    means = result.mean(axis=1)
    np.testing.assert_array_almost_equal(means, np.zeros(10), decimal=10)


def test_normalize_series_zero_rows():
    """All-zero rows should remain zero."""
    data = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    result = normalize_series(data)
    np.testing.assert_array_equal(result[0], np.zeros(3))
    assert np.linalg.norm(result[1]) == pytest.approx(1.0)
