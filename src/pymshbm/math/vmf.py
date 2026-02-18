"""Von Mises-Fisher distribution utilities."""

import numpy as np
from scipy.special import iv as besseli
from scipy.optimize import brentq


def ad(kappa: float | np.ndarray, d: int) -> float | np.ndarray:
    """Bessel quotient A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa).

    Uses scaled Bessel functions for large kappa to avoid overflow.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    # Use ive (exponentially scaled) to avoid overflow for large kappa
    from scipy.special import ive
    num = ive(d / 2, kappa)
    den = ive(d / 2 - 1, kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / den
    # For very large kappa, Ad -> 1
    result = np.where(np.isnan(result) | np.isinf(result), 1.0, result)
    return result


def cdln(kappa: float | np.ndarray, d: int) -> float | np.ndarray:
    """Log normalizing constant of the vMF distribution.

    Uses numerical integration for large kappa to avoid Bessel overflow.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    scalar_input = kappa.ndim == 0
    kappa = np.atleast_1d(kappa)

    if d < 1200:
        k0 = 500
    elif d < 1800:
        k0 = 650
    else:
        raise ValueError(f"Dimension {d} too high, need to specify k0")

    out = (d / 2 - 1) * np.log(kappa) - np.log(besseli(d / 2 - 1, kappa))

    mask_overflow = kappa > k0
    if np.any(mask_overflow):
        fk0 = (d / 2 - 1) * np.log(k0) - np.log(besseli(d / 2 - 1, k0))
        kof = kappa[mask_overflow]
        n_grids = 1000
        ofintv = (kof - k0) / n_grids
        tempcnt = np.arange(1, n_grids + 1) - 0.5
        ks = k0 + np.outer(ofintv, tempcnt)
        half_d_minus_1 = 0.5 * (d - 1)
        ratio = half_d_minus_1 / ks
        adsum = np.sum(1.0 / (ratio + np.sqrt(1 + ratio**2)), axis=1)
        out[mask_overflow] = fk0 - ofintv * adsum

    out = out.astype(np.float32)
    if scalar_input:
        return float(out[0])
    return out


def inv_ad(d: int, rbar: float) -> float:
    """Inverse of the Bessel quotient: find kappa such that A_d(kappa) = rbar."""
    rbar = float(rbar)
    # Initial approximation (Banerjee et al. 2005)
    kappa_init = (d - 1) * rbar / (1 - rbar**2) + d / (d - 1) * rbar

    # Check if initial approximation is valid for root finding
    try:
        i_val = besseli(d / 2 - 1, kappa_init)
        if np.isinf(i_val) or np.isnan(i_val) or i_val == 0:
            return kappa_init - d / (d - 1) * rbar / 2
    except (OverflowError, FloatingPointError):
        return kappa_init - d / (d - 1) * rbar / 2

    try:
        upper = max(kappa_init * 5, 10000)
        result = brentq(lambda k: ad(k, d) - rbar, 1e-6, upper, xtol=1e-12)
        return result
    except (ValueError, RuntimeError):
        return kappa_init - d / (d - 1) * rbar / 2


def vmf_log_probability(
    X: np.ndarray,
    nu: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    """Log probability under vMF distribution.

    Args:
        X: Data matrix (N x D), each row a unit vector.
        nu: Mean directions (D x L), each column a unit vector.
        kappa: Concentration parameters (L,).

    Returns:
        N x L matrix of log probabilities.
    """
    d = X.shape[1] - 1
    log_c = cdln(kappa, d)  # (L,)
    dot_product = X @ nu  # (N, L)
    return log_c[np.newaxis, :] + kappa[np.newaxis, :] * dot_product
