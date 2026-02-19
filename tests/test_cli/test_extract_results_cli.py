"""Tests for the extract-mshbm-result CLI entrypoint."""

import subprocess
import sys

import numpy as np
import pytest
import scipy.io as sio


def _make_project(tmp_path, N=20, L=3, S=2, *, include_colors=False, seed=42):
    """Create a project directory with Params_Final.mat."""
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


def test_cli_produces_output_files(tmp_path):
    """CLI should create ind_parcellation .mat files."""
    N, L, S = 20, 3, 2
    _make_project(tmp_path, N=N, L=L, S=S)

    result = subprocess.run(
        [sys.executable, "-m", "pymshbm.cli.extract_results",
         str(tmp_path), "SUBJ_A", "SUBJ_B"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    for i, sid in enumerate(["SUBJ_A", "SUBJ_B"]):
        mat_file = tmp_path / "ind_parcellation" / f"Ind_parcellation_MSHBM_sub{i+1}_{sid}.mat"
        assert mat_file.exists()
        data = sio.loadmat(str(mat_file))
        assert "lh_labels" in data
        assert "rh_labels" in data
        assert "num_clusters" in data
        assert data["lh_labels"].ravel().shape == (N // 2,)


def test_cli_missing_project_dir(tmp_path):
    """CLI should exit non-zero when project dir is missing."""
    result = subprocess.run(
        [sys.executable, "-m", "pymshbm.cli.extract_results",
         str(tmp_path / "nonexistent"), "A"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_cli_no_subject_ids(tmp_path):
    """CLI should exit non-zero when no subject IDs are given."""
    result = subprocess.run(
        [sys.executable, "-m", "pymshbm.cli.extract_results",
         str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_cli_subject_count_mismatch(tmp_path):
    """CLI should exit non-zero when subject count doesn't match s_lambda."""
    _make_project(tmp_path, S=2)

    result = subprocess.run(
        [sys.executable, "-m", "pymshbm.cli.extract_results",
         str(tmp_path), "only_one"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
