"""Tests for the MSHBM wrapper CLI entrypoint."""

import subprocess
import sys
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest


def _setup_wrapper_data(tmp_path, n_vertices=20, n_timepoints=30, n_seeds=5,
                        n_subs=2, n_sess=2, seed=42):
    """Create synthetic data for a full wrapper CLI run."""
    rng = np.random.default_rng(seed)
    data_dir = tmp_path / "data"

    sub_ids = [f"sub{i:03d}" for i in range(1, n_subs + 1)]
    for sub_id in sub_ids:
        sub_dir = data_dir / sub_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        for sess in range(1, n_sess + 1):
            for hemi in ("lh", "rh"):
                fname = f"{hemi}_sess{sess}_nat_resid_bpss_fsaverage6_sm6.nii.gz"
                data = rng.standard_normal((n_vertices, 1, 1, n_timepoints))
                img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
                nib.save(img, str(sub_dir / fname))

    # CSV
    csv_file = tmp_path / "subs.csv"
    lines = ["subject_id,data_dir\n"]
    for sub_id in sub_ids:
        lines.append(f"{sub_id},{data_dir}/\n")
    csv_file.write_text("".join(lines))

    # Seed labels
    seed_labels = np.repeat(np.arange(1, n_seeds + 1),
                             n_vertices // n_seeds).astype(np.int32)
    lh_npy = tmp_path / "seed_labels_lh.npy"
    rh_npy = tmp_path / "seed_labels_rh.npy"
    np.save(str(lh_npy), seed_labels)
    np.save(str(rh_npy), seed_labels)

    return csv_file, lh_npy, rh_npy


def test_cli_runs_successfully(tmp_path):
    """CLI should complete without errors on valid input."""
    csv_file, lh_npy, rh_npy = _setup_wrapper_data(tmp_path)
    output_dir = tmp_path / "output"

    result = subprocess.run(
        [sys.executable, "-m", "pymshbm.cli.wrapper",
         str(csv_file), str(output_dir),
         "--seed-labels-lh", str(lh_npy),
         "--seed-labels-rh", str(rh_npy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # Verify some output was created
    output_dirs = list(output_dir.iterdir())
    assert len(output_dirs) > 0


def test_cli_missing_seed_labels_and_no_freesurfer(tmp_path):
    """CLI should fail when no seed labels and no FreeSurfer dir available."""
    csv_file = tmp_path / "subs.csv"
    csv_file.write_text("subject_id,data_dir\nsub001,/data/\n")

    # Clear env vars that would allow auto-computation
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("FREESURFER_HOME", "SUBJECTS_DIR")}

    result = subprocess.run(
        [sys.executable, "-m", "pymshbm.cli.wrapper",
         str(csv_file), str(tmp_path / "output")],
        capture_output=True, text=True,
        env=env,
    )
    assert result.returncode != 0


def test_cli_missing_sub_list(tmp_path):
    """CLI should fail when sub_list file doesn't exist."""
    lh_npy = tmp_path / "seed_labels_lh.npy"
    rh_npy = tmp_path / "seed_labels_rh.npy"
    np.save(str(lh_npy), np.array([1, 2, 3]))
    np.save(str(rh_npy), np.array([1, 2, 3]))

    result = subprocess.run(
        [sys.executable, "-m", "pymshbm.cli.wrapper",
         str(tmp_path / "nonexistent.csv"), str(tmp_path / "output"),
         "--seed-labels-lh", str(lh_npy),
         "--seed-labels-rh", str(rh_npy)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# test_cli_auto_seed_labels
# ---------------------------------------------------------------------------

def _make_sphere_surface(subjects_dir, mesh, hemi, n_vertices, seed=42):
    """Create a synthetic sphere.reg surface in FreeSurfer layout."""
    surf_dir = subjects_dir / mesh / "surf"
    surf_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    coords = rng.standard_normal((n_vertices, 3)).astype(np.float32)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / norms * 100.0
    n_faces = max(1, n_vertices - 2)
    faces = np.zeros((n_faces, 3), dtype=np.int32)
    for i in range(n_faces):
        faces[i] = [0, i + 1, min(i + 2, n_vertices - 1)]
    fs.write_geometry(str(surf_dir / f"{hemi}.sphere.reg"), coords, faces)


def test_cli_auto_seed_labels_with_freesurfer_dir(tmp_path):
    """CLI should auto-compute seed labels when --freesurfer-dir is given."""
    n_vertices, n_timepoints, n_seeds = 20, 30, 5
    n_seed_vertices = 5

    # Set up FreeSurfer surfaces
    fs_dir = tmp_path / "freesurfer"
    for hemi in ("lh", "rh"):
        _make_sphere_surface(fs_dir, "fsaverage3", hemi, n_seed_vertices,
                             seed=42)
        _make_sphere_surface(fs_dir, "fsaverage6", hemi, n_vertices,
                             seed=99)

    # Set up subject data (without seed label npy files)
    csv_file, _, _ = _setup_wrapper_data(tmp_path, n_vertices=n_vertices,
                                         n_timepoints=n_timepoints,
                                         n_seeds=n_seeds, n_subs=1, n_sess=1)
    output_dir = tmp_path / "output_auto"

    result = subprocess.run(
        [sys.executable, "-m", "pymshbm.cli.wrapper",
         str(csv_file), str(output_dir),
         "--freesurfer-dir", str(fs_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    # Verify output was created
    output_dirs = list(output_dir.iterdir())
    assert len(output_dirs) > 0
