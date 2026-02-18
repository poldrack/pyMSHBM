"""Von Mises-Fisher series clustering."""

import numpy as np


def vmf_series_clustering(
    data: np.ndarray,
    num_clusters: int,
    num_init: int = 10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster unit-norm data using iterative vMF clustering.

    Args:
        data: (D, N) array of unit-norm column vectors.
        num_clusters: Number of clusters K.
        num_init: Number of random initializations.
        max_iter: Maximum iterations per initialization.

    Returns:
        labels: (N,) integer cluster assignments.
        centroids: (D, K) unit-norm cluster centroids.
    """
    D, N = data.shape
    best_cost = -np.inf
    best_labels = None
    best_centroids = None

    for _ in range(num_init):
        labels, centroids, cost = _vmf_cluster_once(data, num_clusters, max_iter)
        if cost > best_cost:
            best_cost = cost
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids


def _vmf_cluster_once(
    data: np.ndarray,
    num_clusters: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Single run of vMF clustering."""
    D, N = data.shape
    K = num_clusters

    # Random initialization: pick K random data points as centroids
    rng = np.random.default_rng()
    idx = rng.choice(N, size=K, replace=False)
    centroids = data[:, idx].copy()  # (D, K)
    _normalize_centroids(centroids)

    labels = np.zeros(N, dtype=np.int64)

    for _ in range(max_iter):
        # Assignment step: cosine similarity = dot product for unit vectors
        similarities = centroids.T @ data  # (K, N)
        new_labels = np.argmax(similarities, axis=0)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update step
        centroids = _update_centroids(data, labels, K, D)

    cost = _compute_cost(data, centroids, labels)
    return labels, centroids, cost


def _normalize_centroids(centroids: np.ndarray) -> None:
    """Normalize centroid columns to unit norm in-place."""
    norms = np.linalg.norm(centroids, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    centroids /= norms


def _update_centroids(
    data: np.ndarray,
    labels: np.ndarray,
    K: int,
    D: int,
) -> np.ndarray:
    """Compute new centroids as normalized mean of assigned points."""
    centroids = np.zeros((D, K), dtype=np.float64)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            centroids[:, k] = data[:, mask].mean(axis=1)
    _normalize_centroids(centroids)
    return centroids


def _compute_cost(
    data: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute total cosine similarity (higher is better)."""
    total = 0.0
    for k in range(centroids.shape[1]):
        mask = labels == k
        if mask.sum() > 0:
            total += float(centroids[:, k] @ data[:, mask].sum(axis=1))
    return total
