"""Tests for extract_mshbm_result_sub (plug-in replacement for CBIG_IndCBM_extract_MSHBM_result_SUB.m)."""

import numpy as np
import pytest
import scipy.io as sio

from pymshbm.core.extract_results import extract_mshbm_result_sub


def _make_params_final(
    tmp_path, N=20, L=3, S=2, *, include_colors=False, seed=42
):
    """Create a project directory with Params_Final.mat and optionally group.mat."""
    rng = np.random.default_rng(seed)

    s_lambda = np.zeros((N, L, S))
    for s in range(S):
        s_lambda[:, :, s] = rng.dirichlet(np.ones(L), size=N)

    params_dict = {
        "mu": rng.standard_normal((5, L)),
        "epsil": rng.random((1, L)),
        "sigma": rng.random((1, L)),
        "theta": rng.random((N, L)),
        "kappa": rng.random((1, L)),
        "s_lambda": s_lambda,
        "s_psi": np.empty((0,)),
        "s_t_nu": np.empty((0,)),
        "iter_inter": np.array([[10]]),
        "Record": rng.random((1, 10)),
    }

    priors_dir = tmp_path / "priors"
    priors_dir.mkdir(parents=True)
    sio.savemat(str(priors_dir / "Params_Final.mat"), {"Params": params_dict})

    if include_colors:
        group_dir = tmp_path / "group"
        group_dir.mkdir(parents=True)
        colors = rng.integers(0, 256, size=(L, 3)).astype(np.uint8)
        sio.savemat(str(group_dir / "group.mat"), {"colors": colors})

    return s_lambda


def test_basic_extraction(tmp_path):
    """Extracts parcellations and saves one .mat per subject."""
    N, L, S = 20, 3, 2
    s_lambda = _make_params_final(tmp_path, N=N, L=L, S=S)
    subject_ids = ["SUBJ_A", "SUBJ_B"]

    extract_mshbm_result_sub(str(tmp_path), subject_ids)

    out_dir = tmp_path / "ind_parcellation"
    assert out_dir.exists()

    for i, sid in enumerate(subject_ids):
        mat_file = out_dir / f"Ind_parcellation_MSHBM_sub{i + 1}_{sid}.mat"
        assert mat_file.exists(), f"Missing output file for {sid}"

        data = sio.loadmat(str(mat_file))
        assert "lh_labels" in data
        assert "rh_labels" in data
        assert "num_clusters" in data

        lh = data["lh_labels"].ravel()
        rh = data["rh_labels"].ravel()
        assert lh.shape == (N // 2,)
        assert rh.shape == (N // 2,)
        assert int(data["num_clusters"].ravel()[0]) == L


def test_labels_match_argmax(tmp_path):
    """Labels should be argmax of s_lambda (1-indexed)."""
    N, L, S = 20, 3, 1
    s_lambda = _make_params_final(tmp_path, N=N, L=L, S=S)
    subject_ids = ["TEST"]

    extract_mshbm_result_sub(str(tmp_path), subject_ids)

    mat_file = tmp_path / "ind_parcellation" / "Ind_parcellation_MSHBM_sub1_TEST.mat"
    data = sio.loadmat(str(mat_file))
    lh = data["lh_labels"].ravel()
    rh = data["rh_labels"].ravel()
    labels = np.concatenate([lh, rh])

    expected = np.argmax(s_lambda[:, :, 0], axis=1) + 1
    np.testing.assert_array_equal(labels, expected)


def test_medial_wall_labeled_zero(tmp_path):
    """Vertices with all-zero s_lambda rows get label 0."""
    N, L, S = 10, 2, 1

    # Build s_lambda with zero rows at vertices 0, 1, 8, 9
    rng = np.random.default_rng(99)
    s_lambda = np.zeros((N, L, S))
    s_lambda[2:8, :, 0] = rng.dirichlet(np.ones(L), size=6)

    priors_dir = tmp_path / "priors"
    priors_dir.mkdir(parents=True)
    params_dict = {
        "mu": np.zeros((3, L)),
        "epsil": np.ones((1, L)),
        "sigma": np.ones((1, L)),
        "theta": np.zeros((N, L)),
        "kappa": np.ones((1, L)),
        "s_lambda": s_lambda,
        "s_psi": np.empty((0,)),
        "s_t_nu": np.empty((0,)),
        "iter_inter": np.array([[1]]),
        "Record": np.array([[0.0]]),
    }
    sio.savemat(str(priors_dir / "Params_Final.mat"), {"Params": params_dict})

    extract_mshbm_result_sub(str(tmp_path), ["S1"])

    mat_file = tmp_path / "ind_parcellation" / "Ind_parcellation_MSHBM_sub1_S1.mat"
    data = sio.loadmat(str(mat_file))
    lh = data["lh_labels"].ravel()
    rh = data["rh_labels"].ravel()

    # Vertices 0,1 are in lh → should be 0
    assert lh[0] == 0
    assert lh[1] == 0
    # Vertices 8,9 are in rh (indices 3,4) → should be 0
    assert rh[3] == 0
    assert rh[4] == 0
    # Vertices 2-7 should be nonzero
    assert np.all(np.concatenate([lh[2:], rh[:3]]) > 0)


def test_includes_colors_from_group(tmp_path):
    """When group.mat has a colors field, it should be included in output."""
    N, L, S = 20, 3, 1
    _make_params_final(tmp_path, N=N, L=L, S=S, include_colors=True)

    extract_mshbm_result_sub(str(tmp_path), ["COL"])

    mat_file = tmp_path / "ind_parcellation" / "Ind_parcellation_MSHBM_sub1_COL.mat"
    data = sio.loadmat(str(mat_file))
    assert "colors" in data
    assert data["colors"].shape == (L, 3)


def test_no_colors_without_group_file(tmp_path):
    """When group.mat does not exist, output should not have colors."""
    _make_params_final(tmp_path, S=1, include_colors=False)

    extract_mshbm_result_sub(str(tmp_path), ["NC"])

    mat_file = tmp_path / "ind_parcellation" / "Ind_parcellation_MSHBM_sub1_NC.mat"
    data = sio.loadmat(str(mat_file))
    assert "colors" not in data


def test_no_colors_field_in_group(tmp_path):
    """When group.mat exists but has no colors field, output should not have colors."""
    _make_params_final(tmp_path, S=1, include_colors=False)

    # Create group.mat without colors
    group_dir = tmp_path / "group"
    group_dir.mkdir(parents=True)
    sio.savemat(str(group_dir / "group.mat"), {"lh_labels": np.array([1, 2])})

    extract_mshbm_result_sub(str(tmp_path), ["NCF"])

    mat_file = tmp_path / "ind_parcellation" / "Ind_parcellation_MSHBM_sub1_NCF.mat"
    data = sio.loadmat(str(mat_file))
    assert "colors" not in data


def test_subject_count_mismatch_raises(tmp_path):
    """Passing wrong number of subject IDs should raise ValueError."""
    _make_params_final(tmp_path, S=2)

    with pytest.raises(ValueError, match="subject"):
        extract_mshbm_result_sub(str(tmp_path), ["only_one"])


def test_empty_s_lambda_raises(tmp_path):
    """Empty s_lambda should raise ValueError."""
    priors_dir = tmp_path / "priors"
    priors_dir.mkdir(parents=True)
    params_dict = {
        "mu": np.zeros((3, 2)),
        "epsil": np.ones((1, 2)),
        "sigma": np.ones((1, 2)),
        "theta": np.zeros((10, 2)),
        "kappa": np.ones((1, 2)),
        "s_lambda": np.empty((0,)),
        "s_psi": np.empty((0,)),
        "s_t_nu": np.empty((0,)),
        "iter_inter": np.array([[1]]),
        "Record": np.array([[0.0]]),
    }
    sio.savemat(str(priors_dir / "Params_Final.mat"), {"Params": params_dict})

    with pytest.raises(ValueError, match="s_lambda is empty"):
        extract_mshbm_result_sub(str(tmp_path), ["X"])


def test_missing_params_file_raises(tmp_path):
    """Missing Params_Final.mat should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_mshbm_result_sub(str(tmp_path), ["X"])


def test_creates_output_dir(tmp_path):
    """Output directory should be created if it doesn't exist."""
    _make_params_final(tmp_path, S=1)

    # Verify ind_parcellation doesn't exist yet
    assert not (tmp_path / "ind_parcellation").exists()

    extract_mshbm_result_sub(str(tmp_path), ["NEW"])

    assert (tmp_path / "ind_parcellation").exists()


def test_returns_results_list(tmp_path):
    """Function should return list of (lh_labels, rh_labels) tuples."""
    N, L, S = 20, 3, 2
    _make_params_final(tmp_path, N=N, L=L, S=S)

    results = extract_mshbm_result_sub(str(tmp_path), ["A", "B"])

    assert len(results) == S
    for lh, rh in results:
        assert lh.shape == (N // 2,)
        assert rh.shape == (N // 2,)
        assert np.all(lh >= 0)
        assert np.all(rh >= 0)
        assert np.all(lh <= L)
        assert np.all(rh <= L)


def test_custom_priors_file(tmp_path):
    """priors_file parameter should override default Params_Final.mat location."""
    N, L, S = 20, 3, 1
    rng = np.random.default_rng(42)

    s_lambda = np.zeros((N, L, S))
    for s in range(S):
        s_lambda[:, :, s] = rng.dirichlet(np.ones(L), size=N)

    params_dict = {
        "mu": rng.standard_normal((5, L)),
        "epsil": rng.random((1, L)),
        "sigma": rng.random((1, L)),
        "theta": rng.random((N, L)),
        "kappa": rng.random((1, L)),
        "s_lambda": s_lambda,
        "s_psi": np.empty((0,)),
        "s_t_nu": np.empty((0,)),
        "iter_inter": np.array([[1]]),
        "Record": rng.random((1, 1)),
    }

    custom_file = tmp_path / "my_custom_priors.mat"
    sio.savemat(str(custom_file), {"Params": params_dict})

    results = extract_mshbm_result_sub(str(tmp_path), ["SUB1"], priors_file=str(custom_file))

    mat_file = tmp_path / "ind_parcellation" / "Ind_parcellation_MSHBM_sub1_SUB1.mat"
    assert mat_file.exists()
    assert len(results) == 1


def test_flat_mat_without_params_wrapper(tmp_path):
    """Handle .mat files where s_lambda is at top level (no Params struct)."""
    N, L, S = 20, 3, 1
    rng = np.random.default_rng(42)

    s_lambda = np.zeros((N, L, S))
    for s in range(S):
        s_lambda[:, :, s] = rng.dirichlet(np.ones(L), size=N)

    flat_file = tmp_path / "flat_priors.mat"
    sio.savemat(str(flat_file), {"s_lambda": s_lambda})

    results = extract_mshbm_result_sub(str(tmp_path), ["F1"], priors_file=str(flat_file))

    mat_file = tmp_path / "ind_parcellation" / "Ind_parcellation_MSHBM_sub1_F1.mat"
    assert mat_file.exists()

    data = sio.loadmat(str(mat_file))
    lh = data["lh_labels"].ravel()
    rh = data["rh_labels"].ravel()
    labels = np.concatenate([lh, rh])
    expected = np.argmax(s_lambda[:, :, 0], axis=1) + 1
    np.testing.assert_array_equal(labels, expected)


def test_mat_missing_s_lambda_raises(tmp_path):
    """A .mat file with no s_lambda anywhere should give a clear error."""
    bad_file = tmp_path / "no_slambda.mat"
    sio.savemat(str(bad_file), {"lh_labels": np.array([1, 2])})

    with pytest.raises(ValueError, match="s_lambda"):
        extract_mshbm_result_sub(str(tmp_path), ["X"], priors_file=str(bad_file))
