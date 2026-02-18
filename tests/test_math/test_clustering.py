"""Tests for vMF series clustering."""

import numpy as np
import pytest

from pymshbm.math.clustering import vmf_series_clustering


def test_output_shape():
    """Output labels should be (N,) and centroids (D, K)."""
    rng = np.random.default_rng(42)
    N, D, K = 50, 10, 3
    data = rng.standard_normal((D, N))
    data /= np.linalg.norm(data, axis=0, keepdims=True)
    labels, centroids = vmf_series_clustering(data, K, num_init=2)
    assert labels.shape == (N,)
    assert centroids.shape == (D, K)
    assert set(labels).issubset(set(range(K)))


def test_well_separated_clusters():
    """Clearly separated clusters should be recovered."""
    rng = np.random.default_rng(42)
    D, K = 10, 3
    points_per_cluster = 30
    centers = np.eye(D)[:K].T  # D x K
    data_list = []
    true_labels = []
    for k in range(K):
        pts = centers[:, k:k+1] + 0.1 * rng.standard_normal((D, points_per_cluster))
        pts /= np.linalg.norm(pts, axis=0, keepdims=True)
        data_list.append(pts)
        true_labels.extend([k] * points_per_cluster)
    data = np.hstack(data_list)
    true_labels = np.array(true_labels)

    labels, centroids = vmf_series_clustering(data, K, num_init=5)
    # Check that clusters are internally consistent (purity)
    # Since label indices may differ, check via purity
    purity = 0
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            counts = [np.sum(true_labels[mask] == j) for j in range(K)]
            purity += max(counts)
    purity /= len(true_labels)
    assert purity > 0.9


def test_single_cluster():
    """With K=1, all points should be in one cluster."""
    rng = np.random.default_rng(42)
    D, N = 5, 20
    data = rng.standard_normal((D, N))
    data /= np.linalg.norm(data, axis=0, keepdims=True)
    labels, centroids = vmf_series_clustering(data, 1, num_init=1)
    assert np.all(labels == 0)
    assert centroids.shape == (D, 1)


def test_centroids_unit_norm():
    """Cluster centroids should be approximately unit norm."""
    rng = np.random.default_rng(42)
    D, N, K = 8, 40, 3
    data = rng.standard_normal((D, N))
    data /= np.linalg.norm(data, axis=0, keepdims=True)
    _, centroids = vmf_series_clustering(data, K, num_init=3)
    norms = np.linalg.norm(centroids, axis=0)
    np.testing.assert_array_almost_equal(norms, np.ones(K))
