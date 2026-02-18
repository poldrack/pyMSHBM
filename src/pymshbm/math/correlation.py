"""Correlation and normalization utilities."""

import numpy as np


def pearson_corr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation between columns of X and Y.

    Args:
        X: (T, M) array.
        Y: (T, N) array.

    Returns:
        (M, N) correlation matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    X_std = np.sqrt((X_centered**2).sum(axis=0, keepdims=True))
    Y_std = np.sqrt((Y_centered**2).sum(axis=0, keepdims=True))
    X_std[X_std == 0] = 1.0
    Y_std[Y_std == 0] = 1.0
    return (X_centered / X_std).T @ (Y_centered / Y_std)


def stable_atanh(r: np.ndarray) -> np.ndarray:
    """Fisher Z-transform with clipping to avoid infinities at +/- 1."""
    eps = np.finfo(np.float64).eps
    r = np.asarray(r, dtype=np.float64)
    return np.arctanh(np.clip(r, -1 + eps, 1 - eps))


def normalize_series(data: np.ndarray) -> np.ndarray:
    """Mean-center rows and L2-normalize non-zero rows.

    Args:
        data: (N, T) array where rows are series to normalize.

    Returns:
        (N, T) normalized array.
    """
    data = np.asarray(data, dtype=np.float64).copy()
    data -= data.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    nonzero = norms.ravel() > 0
    data[nonzero] /= norms[nonzero]
    return data
