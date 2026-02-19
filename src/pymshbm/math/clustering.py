"""Von Mises-Fisher series clustering via scikit-learn KMeans.

For unit-norm data, Euclidean KMeans is equivalent to spherical (cosine)
k-means because ||x - c||^2 = 2 - 2*(x . c) for unit vectors.  Centroids
are re-normalized to the unit sphere after fitting.
"""

import numpy as np
from sklearn.cluster import KMeans


def vmf_series_clustering(
    data: np.ndarray,
    num_clusters: int,
    num_init: int = 10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster unit-norm data using spherical k-means (via sklearn KMeans).

    Args:
        data: (D, N) array of unit-norm column vectors.
        num_clusters: Number of clusters K.
        num_init: Number of k-means++ initializations.
        max_iter: Maximum iterations per initialization.

    Returns:
        labels: (N,) integer cluster assignments.
        centroids: (D, K) unit-norm cluster centroids.
    """
    km = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=num_init,
        max_iter=max_iter,
    )
    # sklearn expects (N, D)
    km.fit(data.T)

    labels = km.labels_
    # km.cluster_centers_ is (K, D) — transpose to (D, K) and normalize
    centroids = km.cluster_centers_.T.copy()
    norms = np.linalg.norm(centroids, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    centroids /= norms

    return labels, centroids
