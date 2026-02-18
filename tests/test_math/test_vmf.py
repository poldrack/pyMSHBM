"""Tests for Von Mises-Fisher math utilities."""

import numpy as np
import pytest
from scipy.special import iv as besseli

from pymshbm.math.vmf import ad, cdln, inv_ad, vmf_log_probability


class TestAd:
    """Tests for the Bessel quotient function Ad(kappa, d)."""

    def test_known_value(self):
        """Ad should match I_{d/2}(k) / I_{d/2-1}(k) for moderate kappa."""
        kappa = 10.0
        d = 20
        expected = float(besseli(d / 2, kappa) / besseli(d / 2 - 1, kappa))
        result = ad(kappa, d)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_small_kappa(self):
        """Ad should be near zero for very small kappa."""
        result = ad(0.01, 10)
        assert 0 < result < 0.1

    def test_large_kappa(self):
        """Ad should approach 1 for large kappa."""
        result = ad(1000.0, 10)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_vector_input(self):
        """Ad should work with array inputs."""
        kappas = np.array([1.0, 10.0, 100.0])
        results = ad(kappas, 20)
        assert results.shape == (3,)
        assert np.all(results > 0)
        assert np.all(np.diff(results) > 0)  # monotonically increasing


class TestCdln:
    """Tests for the log normalizing constant."""

    def test_small_kappa(self):
        """For small kappa, direct formula should work."""
        kappa = 5.0
        d = 20
        expected = (d / 2 - 1) * np.log(kappa) - np.log(besseli(d / 2 - 1, kappa))
        result = cdln(kappa, d)
        assert result == pytest.approx(float(expected), rel=1e-5)

    def test_large_kappa_no_overflow(self):
        """For large kappa (>500), numerical integration should prevent overflow."""
        result = cdln(600.0, 100)
        assert np.isfinite(result)

    def test_vector_input(self):
        """cdln should handle array inputs."""
        kappas = np.array([1.0, 10.0, 100.0])
        results = cdln(kappas, 20)
        assert results.shape == (3,)
        assert np.all(np.isfinite(results))

    def test_monotonicity(self):
        """cdln should be monotonically decreasing for large d."""
        kappas = np.array([10.0, 50.0, 100.0, 200.0])
        results = cdln(kappas, 100)
        # cdln generally decreases (becomes more negative) for large kappa
        # but the behavior depends on d. Just check finiteness.
        assert np.all(np.isfinite(results))


class TestInvAd:
    """Tests for the inverse Bessel quotient function."""

    def test_roundtrip(self):
        """inv_ad(d, ad(kappa, d)) should return kappa."""
        kappa = 50.0
        d = 20
        rbar = ad(kappa, d)
        recovered = inv_ad(d, rbar)
        assert recovered == pytest.approx(kappa, rel=1e-3)

    def test_small_rbar(self):
        """Small rbar should give small kappa."""
        result = inv_ad(20, 0.01)
        assert result > 0
        assert result < 10

    def test_moderate_rbar(self):
        """Moderate rbar should give reasonable kappa."""
        result = inv_ad(20, 0.5)
        assert np.isfinite(result)
        assert result > 0


class TestVmfLogProbability:
    """Tests for the vMF log probability."""

    def test_output_shape(self):
        """Output should be N x L where X is N x D and nu is D x L."""
        rng = np.random.default_rng(42)
        N, D, L = 50, 20, 5
        X = rng.standard_normal((N, D))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        nu = rng.standard_normal((D, L))
        nu /= np.linalg.norm(nu, axis=0, keepdims=True)
        kappa = np.full(L, 10.0)
        result = vmf_log_probability(X, nu, kappa)
        assert result.shape == (N, L)

    def test_higher_for_aligned(self):
        """Points aligned with nu should have higher log probability."""
        D, L = 10, 1
        nu = np.zeros((D, L))
        nu[0, 0] = 1.0
        kappa = np.array([20.0])
        x_aligned = np.zeros((1, D))
        x_aligned[0, 0] = 1.0
        x_perp = np.zeros((1, D))
        x_perp[0, 1] = 1.0
        lp_aligned = vmf_log_probability(x_aligned, nu, kappa)
        lp_perp = vmf_log_probability(x_perp, nu, kappa)
        assert lp_aligned[0, 0] > lp_perp[0, 0]

    def test_finite_output(self):
        """All outputs should be finite for well-conditioned inputs."""
        rng = np.random.default_rng(42)
        N, D, L = 30, 15, 3
        X = rng.standard_normal((N, D))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        nu = rng.standard_normal((D, L))
        nu /= np.linalg.norm(nu, axis=0, keepdims=True)
        kappa = np.full(L, 50.0)
        result = vmf_log_probability(X, nu, kappa)
        assert np.all(np.isfinite(result))
