"""Tests for extract_mshbm_result."""

import numpy as np
import pytest
import scipy.io as sio

from pymshbm.core.extract_results import extract_mshbm_result
from pymshbm.types import MSHBMParams


def test_extract_basic():
    """Extract labels from s_lambda via argmax."""
    N, L, S = 20, 3, 2
    rng = np.random.default_rng(42)
    s_lambda = rng.dirichlet(np.ones(L), size=(N, S)).transpose(0, 2, 1)
    # s_lambda shape: (N, L, S)
    # Swap to (N, L, S) by reshaping
    s_lambda_correct = np.zeros((N, L, S))
    for s in range(S):
        s_lambda_correct[:, :, s] = rng.dirichlet(np.ones(L), size=N)
    params = MSHBMParams(
        mu=np.zeros((5, L)),
        epsil=np.zeros(L),
        sigma=np.zeros(L),
        theta=np.zeros((N, L)),
        kappa=np.zeros(L),
        s_lambda=s_lambda_correct,
    )
    results = extract_mshbm_result(params)
    assert len(results) == S
    for lh, rh in results:
        assert lh.shape == (N // 2,)
        assert rh.shape == (N // 2,)
        # Labels should be 1-indexed (non-zero)
        assert np.all(lh >= 0)
        assert np.all(rh >= 0)
        assert np.all(lh <= L)
        assert np.all(rh <= L)


def test_extract_medial_wall():
    """Vertices with all-zero s_lambda should get label 0."""
    N, L, S = 10, 2, 1
    s_lambda = np.zeros((N, L, S))
    s_lambda[2:8, :, 0] = np.array([
        [0.8, 0.2], [0.3, 0.7], [0.6, 0.4],
        [0.1, 0.9], [0.5, 0.5], [0.4, 0.6],
    ])
    params = MSHBMParams(
        mu=np.zeros((3, L)),
        epsil=np.zeros(L),
        sigma=np.zeros(L),
        theta=np.zeros((N, L)),
        kappa=np.zeros(L),
        s_lambda=s_lambda,
    )
    results = extract_mshbm_result(params)
    lh, rh = results[0]
    # First vertex and last vertex in each hemisphere should be 0 (medial wall)
    assert lh[0] == 0
    assert lh[1] == 0
    assert rh[-1] == 0


def test_extract_saves_mat(tmp_path):
    """extract_mshbm_result should save .mat files when out_dir provided."""
    N, L, S = 20, 3, 2
    rng = np.random.default_rng(42)
    s_lambda = np.zeros((N, L, S))
    for s in range(S):
        s_lambda[:, :, s] = rng.dirichlet(np.ones(L), size=N)
    params = MSHBMParams(
        mu=np.zeros((5, L)),
        epsil=np.zeros(L),
        sigma=np.zeros(L),
        theta=np.zeros((N, L)),
        kappa=np.zeros(L),
        s_lambda=s_lambda,
    )
    subject_ids = ["sub01", "sub02"]
    results = extract_mshbm_result(params, out_dir=tmp_path, subject_ids=subject_ids)
    for i, sid in enumerate(subject_ids):
        mat_file = tmp_path / f"Ind_parcellation_MSHBM_sub{i+1}_{sid}.mat"
        assert mat_file.exists()
        data = sio.loadmat(str(mat_file))
        assert "lh_labels" in data
        assert "rh_labels" in data
