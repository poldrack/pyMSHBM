"""Tests for MRF V*lambda product."""

import numpy as np
import pytest

from pymshbm.math.mrf import build_sparse_adjacency, v_lambda_product


def test_basic_shape():
    """Output shape should be (N, K) matching lambda."""
    N, K = 10, 3
    # 2 neighbors per vertex
    neighborhood = np.zeros((N, 2), dtype=np.int64)
    for i in range(N):
        neighborhood[i, 0] = max(0, i - 1)
        neighborhood[i, 1] = min(N - 1, i + 1)
    v_same = np.zeros((N, 2))
    v_diff = np.ones((N, 2))
    lam = np.ones((N, K)) / K
    result = v_lambda_product(neighborhood, v_same, v_diff, lam)
    assert result.shape == (N, K)


def test_uniform_lambda():
    """With uniform lambda, V_lam should reflect v_diff weights."""
    N, K = 5, 3
    neighborhood = np.array([
        [1, 2],
        [0, 2],
        [0, 1],
        [2, 4],
        [2, 3],
    ])
    v_same = np.zeros((N, 2))
    v_diff = np.ones((N, 2))
    lam = np.ones((N, K)) / K
    result = v_lambda_product(neighborhood, v_same, v_diff, lam)
    # With uniform lambda, each neighbor contributes:
    # v_same * (1/K) + v_diff * (K-1)/K = 0 + (K-1)/K per neighbor
    # = 2 * (K-1)/K for 2 neighbors per vertex
    expected_per_vertex = 2 * (K - 1) / K
    np.testing.assert_array_almost_equal(result, expected_per_vertex)


def test_deterministic_lambda():
    """With deterministic lambda (one-hot), matched neighbors should get v_same."""
    N, K = 4, 2
    # chain: 0-1-2-3
    neighborhood = np.array([
        [1, -1],
        [0, 2],
        [1, 3],
        [2, -1],
    ])
    v_same = np.zeros_like(neighborhood, dtype=float)
    v_diff = np.ones_like(neighborhood, dtype=float)
    # All vertices in cluster 0
    lam = np.zeros((N, K))
    lam[:, 0] = 1.0
    # Mark invalid neighbors
    neighborhood[0, 1] = -1
    neighborhood[3, 1] = -1
    result = v_lambda_product(neighborhood, v_same, v_diff, lam)
    # All neighbors have same label: v_same contribution
    # v_same=0, so result should be 0 for cluster 0
    assert result[1, 0] == pytest.approx(0.0)
    # For cluster 1 (which no vertex belongs to),
    # each valid neighbor contributes v_diff=1
    assert result[1, 1] == pytest.approx(2.0)


def test_invalid_neighbors_ignored():
    """Neighbors with index -1 should be ignored."""
    N, K = 3, 2
    neighborhood = np.array([
        [1, -1],
        [0, 2],
        [1, -1],
    ])
    v_same = np.zeros((N, 2))
    v_diff = np.ones((N, 2))
    lam = np.ones((N, K)) / K
    result = v_lambda_product(neighborhood, v_same, v_diff, lam)
    # Vertex 0 has 1 valid neighbor, vertex 1 has 2, vertex 2 has 1
    assert result[0, 0] < result[1, 0]


def test_build_sparse_adjacency_shape():
    """build_sparse_adjacency should return CSR matrix with correct shape."""
    N, max_nb = 10, 3
    neighborhood = np.full((N, max_nb), -1, dtype=np.int64)
    # Simple chain: each vertex connects to next
    for i in range(N - 1):
        neighborhood[i, 0] = i + 1
        neighborhood[i + 1, 1] = i
    adj = build_sparse_adjacency(neighborhood)
    assert adj.shape == (N, N)
    # Should have nnz equal to number of valid entries
    valid_count = int(np.sum(neighborhood >= 0))
    assert adj.nnz == valid_count


def test_build_sparse_adjacency_invalid_neighbors():
    """-1 entries should be excluded from the sparse matrix."""
    N = 5
    neighborhood = np.array([
        [1, -1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, -1],
    ], dtype=np.int64)
    adj = build_sparse_adjacency(neighborhood)
    # Total valid entries: 8 (vertices 0,4 have 1 each; vertices 1,2,3 have 2 each)
    assert adj.nnz == 8
    # Row 0 should have exactly 1 nonzero entry (column 1)
    assert adj[0, 1] == 1.0
    assert adj[0, 0] == 0.0


def test_sparse_v_lambda_product_matches_original():
    """Sparse path (v_same=0, v_diff=1) should match the loop-based path."""
    N, K = 20, 4
    rng = np.random.default_rng(42)
    neighborhood = np.full((N, 3), -1, dtype=np.int64)
    for i in range(N):
        if i > 0:
            neighborhood[i, 0] = i - 1
        if i < N - 1:
            neighborhood[i, 1] = i + 1
        if i + 2 < N:
            neighborhood[i, 2] = i + 2
    v_same = np.zeros_like(neighborhood, dtype=np.float64)
    v_diff = np.ones_like(neighborhood, dtype=np.float64)
    lam = rng.dirichlet(np.ones(K), size=N)

    adj = build_sparse_adjacency(neighborhood)

    result_loop = v_lambda_product(neighborhood, v_same, v_diff, lam)
    result_sparse = v_lambda_product(
        neighborhood, v_same, v_diff, lam, adjacency_matrix=adj,
    )
    np.testing.assert_allclose(result_sparse, result_loop, atol=1e-12)


def test_sparse_v_lambda_product_general_weights():
    """With non-trivial v_same/v_diff, sparse fast-path should NOT be used;
    loop path should still be correct."""
    N, K = 10, 3
    rng = np.random.default_rng(99)
    neighborhood = np.full((N, 2), -1, dtype=np.int64)
    for i in range(N - 1):
        neighborhood[i, 0] = i + 1
        neighborhood[i + 1, 1] = i
    v_same = rng.random((N, 2))
    v_diff = rng.random((N, 2))
    lam = rng.dirichlet(np.ones(K), size=N)

    adj = build_sparse_adjacency(neighborhood)

    # With non-zero v_same, should fall back to loop path
    result_no_sparse = v_lambda_product(neighborhood, v_same, v_diff, lam)
    result_with_sparse = v_lambda_product(
        neighborhood, v_same, v_diff, lam, adjacency_matrix=adj,
    )
    np.testing.assert_allclose(result_with_sparse, result_no_sparse, atol=1e-12)
