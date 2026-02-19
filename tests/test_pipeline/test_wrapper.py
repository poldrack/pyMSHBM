"""Tests for the MSHBM wrapper pipeline."""

from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest

from pymshbm.pipeline.wrapper import (
    average_profiles_nifti,
    compute_seed_timeseries,
    create_profile_lists,
    discover_fmri_files,
    generate_and_save_profile,
    parse_sub_list,
    run_wrapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_nifti(path: Path, data: np.ndarray) -> None:
    """Write a small NIfTI file with identity affine.

    If data is 2D (N, T), reshapes to (N, 1, 1, T) so that _read_nifti
    correctly interprets the first 3 dims as spatial.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data.reshape(data.shape[0], 1, 1, data.shape[1])
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, str(path))


def _make_fmri_files(data_dir: Path, subject_id: str, n_sess: int,
                     n_vertices: int, n_timepoints: int,
                     rng: np.random.Generator) -> None:
    """Create synthetic lh/rh fMRI NIfTI files for a subject."""
    sub_dir = data_dir / subject_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    for sess in range(1, n_sess + 1):
        for hemi in ("lh", "rh"):
            fname = f"{hemi}_sess{sess}_nat_resid_bpss_fsaverage6_sm6.nii.gz"
            data = rng.standard_normal((n_vertices, n_timepoints))
            _write_nifti(sub_dir / fname, data)


# ---------------------------------------------------------------------------
# test_parse_sub_list
# ---------------------------------------------------------------------------

def test_parse_sub_list_two_subjects(tmp_path):
    csv_file = tmp_path / "subs.csv"
    csv_file.write_text(
        "subject_id,data_dir\n"
        "sub001,/data/project/\n"
        "sub002,/data/project/\n"
    )
    result = parse_sub_list(csv_file)
    assert len(result) == 2
    assert result[0] == ("sub001", Path("/data/project/"))
    assert result[1] == ("sub002", Path("/data/project/"))


def test_parse_sub_list_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_sub_list(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# test_discover_fmri_files
# ---------------------------------------------------------------------------

def test_discover_fmri_files_finds_matching(tmp_path):
    rng = np.random.default_rng(42)
    _make_fmri_files(tmp_path, "sub001", n_sess=2, n_vertices=10,
                     n_timepoints=5, rng=rng)
    lh_files, rh_files = discover_fmri_files(tmp_path, "sub001")
    assert len(lh_files) == 2
    assert len(rh_files) == 2
    assert all("lh" in f.name for f in lh_files)
    assert all("rh" in f.name for f in rh_files)
    # Sorted order
    assert lh_files[0].name < lh_files[1].name


def test_discover_fmri_files_empty_dir(tmp_path):
    sub_dir = tmp_path / "sub_empty"
    sub_dir.mkdir()
    lh, rh = discover_fmri_files(tmp_path, "sub_empty")
    assert lh == []
    assert rh == []


def test_discover_fmri_files_custom_pattern(tmp_path):
    rng = np.random.default_rng(42)
    sub_dir = tmp_path / "sub001"
    sub_dir.mkdir()
    # Create files with a different pattern
    data = rng.standard_normal((10, 5))
    _write_nifti(sub_dir / "lh_custom_run1.nii.gz", data)
    _write_nifti(sub_dir / "rh_custom_run1.nii.gz", data)
    lh, rh = discover_fmri_files(tmp_path, "sub001",
                                  file_pattern="*custom*.nii.gz")
    assert len(lh) == 1
    assert len(rh) == 1


# ---------------------------------------------------------------------------
# test_compute_seed_timeseries
# ---------------------------------------------------------------------------

def test_compute_seed_timeseries_shape():
    rng = np.random.default_rng(42)
    n_vertices, n_timepoints, n_seeds = 20, 30, 5
    fmri_data = rng.standard_normal((n_vertices, n_timepoints))
    # Assign each vertex to a seed (1-indexed, 0=unassigned)
    seed_labels = np.array([1, 2, 3, 4, 5] * 4, dtype=np.int32)
    result = compute_seed_timeseries(fmri_data, seed_labels, n_seeds)
    assert result.shape == (n_timepoints, n_seeds)


def test_compute_seed_timeseries_correct_average():
    """Verify that seed time series is the mean of vertices in each ROI."""
    n_vertices, n_timepoints = 6, 4
    fmri_data = np.arange(n_vertices * n_timepoints, dtype=np.float64).reshape(
        n_vertices, n_timepoints
    )
    # 3 seeds: vertices 0,1 -> seed 1; vertices 2,3 -> seed 2; vertices 4,5 -> seed 3
    seed_labels = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
    result = compute_seed_timeseries(fmri_data, seed_labels, 3)
    # Seed 1 = mean of rows 0 and 1
    expected_seed1 = (fmri_data[0] + fmri_data[1]) / 2
    np.testing.assert_allclose(result[:, 0], expected_seed1)


def test_compute_seed_timeseries_skips_label_zero():
    """Vertices with label 0 should not contribute to any seed."""
    n_vertices, n_timepoints = 4, 3
    fmri_data = np.ones((n_vertices, n_timepoints), dtype=np.float64)
    fmri_data[0, :] = 999.0  # This vertex has label 0 — should be ignored
    seed_labels = np.array([0, 1, 1, 2], dtype=np.int32)
    result = compute_seed_timeseries(fmri_data, seed_labels, 2)
    # Seed 1 = mean of vertices 1,2 = 1.0
    np.testing.assert_allclose(result[:, 0], 1.0)
    # Seed 2 = vertex 3 = 1.0
    np.testing.assert_allclose(result[:, 1], 1.0)


# ---------------------------------------------------------------------------
# test_generate_and_save_profile
# ---------------------------------------------------------------------------

def test_generate_and_save_profile_creates_nifti(tmp_path):
    rng = np.random.default_rng(42)
    n_vertices, n_timepoints, n_seeds = 20, 30, 5

    # Create fMRI files
    lh_data = rng.standard_normal((n_vertices, n_timepoints))
    rh_data = rng.standard_normal((n_vertices, n_timepoints))
    lh_path = tmp_path / "lh_fmri.nii.gz"
    rh_path = tmp_path / "rh_fmri.nii.gz"
    _write_nifti(lh_path, lh_data)
    _write_nifti(rh_path, rh_data)

    # Seed labels: assign vertices to seeds
    seed_labels_lh = np.repeat(np.arange(1, n_seeds + 1), n_vertices // n_seeds)
    seed_labels_rh = np.repeat(np.arange(1, n_seeds + 1), n_vertices // n_seeds)

    out_dir = tmp_path / "profiles"
    generate_and_save_profile(
        lh_fmri_path=lh_path,
        rh_fmri_path=rh_path,
        seed_labels_lh=seed_labels_lh,
        seed_labels_rh=seed_labels_rh,
        out_dir=out_dir,
        sub_idx=1,
        sess_idx=1,
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
    )

    # Check output files exist
    lh_profile = (out_dir / "sub1" / "sess1" /
                  "lh.sub1_sess1_fsaverage6_roifsaverage3.surf2surf_profile.nii.gz")
    rh_profile = (out_dir / "sub1" / "sess1" /
                  "rh.sub1_sess1_fsaverage6_roifsaverage3.surf2surf_profile.nii.gz")
    assert lh_profile.exists()
    assert rh_profile.exists()

    # Load and check shape: (N_targ, N_seed) where N_seed = 2 * n_seeds (lh + rh)
    lh_img = nib.load(str(lh_profile))
    lh_profile_data = np.asarray(lh_img.dataobj)
    total_seeds = 2 * n_seeds
    assert lh_profile_data.shape[0] == n_vertices
    assert lh_profile_data.shape[-1] == total_seeds


# ---------------------------------------------------------------------------
# test_create_profile_lists
# ---------------------------------------------------------------------------

def test_create_profile_lists_content(tmp_path):
    """Verify profile list files contain correct paths."""
    n_subs, n_sess = 2, 3
    profile_dir = tmp_path / "profiles"
    seed_mesh, targ_mesh = "fsaverage3", "fsaverage6"

    # Create dummy profile files
    for sub in range(1, n_subs + 1):
        for sess in range(1, n_sess + 1):
            for hemi in ("lh", "rh"):
                p = (profile_dir / f"sub{sub}" / f"sess{sess}" /
                     f"{hemi}.sub{sub}_sess{sess}_{targ_mesh}_roi{seed_mesh}.surf2surf_profile.nii.gz")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("dummy")

    sessions_per_sub = [n_sess, n_sess]
    out_dir = tmp_path / "profile_lists"
    create_profile_lists(profile_dir, n_subs, sessions_per_sub, out_dir,
                         seed_mesh, targ_mesh)

    # Check that session list files exist
    for sess in range(1, n_sess + 1):
        lh_list = out_dir / f"lh_sess{sess}.txt"
        rh_list = out_dir / f"rh_sess{sess}.txt"
        assert lh_list.exists(), f"Missing {lh_list}"
        assert rh_list.exists(), f"Missing {rh_list}"

        # Each file should have n_subs lines
        lines = lh_list.read_text().strip().splitlines()
        assert len(lines) == n_subs


def test_create_profile_lists_unequal_sessions(tmp_path):
    """Subjects with fewer sessions should get NONE entries."""
    profile_dir = tmp_path / "profiles"
    seed_mesh, targ_mesh = "fsaverage3", "fsaverage6"

    # Sub1 has 2 sessions, sub2 has 1 session
    for sub, n_sess in [(1, 2), (2, 1)]:
        for sess in range(1, n_sess + 1):
            for hemi in ("lh", "rh"):
                p = (profile_dir / f"sub{sub}" / f"sess{sess}" /
                     f"{hemi}.sub{sub}_sess{sess}_{targ_mesh}_roi{seed_mesh}.surf2surf_profile.nii.gz")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("dummy")

    sessions_per_sub = [2, 1]
    out_dir = tmp_path / "profile_lists"
    create_profile_lists(profile_dir, 2, sessions_per_sub, out_dir,
                         seed_mesh, targ_mesh)

    # Session 2 list should have NONE for sub2
    lh_sess2 = out_dir / "lh_sess2.txt"
    lines = lh_sess2.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[1] == "NONE"


# ---------------------------------------------------------------------------
# test_average_profiles_nifti
# ---------------------------------------------------------------------------

def test_average_profiles_correct_values(tmp_path):
    """Verify element-wise averaging across subjects/sessions."""
    rng = np.random.default_rng(42)
    n_vertices, n_seeds = 10, 4
    n_subs, max_sess = 2, 2
    profile_dir = tmp_path / "profiles"
    seed_mesh, targ_mesh = "fsaverage3", "fsaverage6"

    all_profiles = []
    for sub in range(1, n_subs + 1):
        for sess in range(1, max_sess + 1):
            for hemi in ("lh", "rh"):
                data = rng.standard_normal((n_vertices, 1, 1, n_seeds))
                p = (profile_dir / f"sub{sub}" / f"sess{sess}" /
                     f"{hemi}.sub{sub}_sess{sess}_{targ_mesh}_roi{seed_mesh}.surf2surf_profile.nii.gz")
                _write_nifti(p, data)
                all_profiles.append((hemi, data.reshape(n_vertices, n_seeds)))

    out_dir = tmp_path / "avg"
    sessions_per_sub = [max_sess, max_sess]
    average_profiles_nifti(profile_dir, n_subs, sessions_per_sub, out_dir,
                           seed_mesh, targ_mesh)

    # Check output exists
    avg_lh = out_dir / f"lh_{targ_mesh}_roi{seed_mesh}_avg_profile.nii.gz"
    assert avg_lh.exists()

    # Verify averaging: collect all lh profiles and compute expected average
    lh_profiles = [data for hemi, data in all_profiles if hemi == "lh"]
    expected = np.mean(np.stack(lh_profiles), axis=0)
    avg_img = nib.load(str(avg_lh))
    avg_data = np.asarray(avg_img.dataobj).reshape(n_vertices, n_seeds)
    np.testing.assert_allclose(avg_data, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# test_run_wrapper_end_to_end
# ---------------------------------------------------------------------------

def test_run_wrapper_end_to_end(tmp_path):
    """Full pipeline with 2 subjects, 2 sessions each."""
    rng = np.random.default_rng(42)
    n_vertices, n_timepoints, n_seeds = 20, 30, 5

    # Set up subject data
    data_dir = tmp_path / "data"
    for sub_id in ("sub001", "sub002"):
        _make_fmri_files(data_dir, sub_id, n_sess=2,
                         n_vertices=n_vertices, n_timepoints=n_timepoints,
                         rng=rng)

    # Write CSV
    csv_file = tmp_path / "subs.csv"
    csv_file.write_text(
        "subject_id,data_dir\n"
        f"sub001,{data_dir}/\n"
        f"sub002,{data_dir}/\n"
    )

    # Seed labels
    seed_labels_lh = np.repeat(np.arange(1, n_seeds + 1),
                                n_vertices // n_seeds).astype(np.int32)
    seed_labels_rh = np.repeat(np.arange(1, n_seeds + 1),
                                n_vertices // n_seeds).astype(np.int32)

    output_dir = tmp_path / "output"
    result_dir = run_wrapper(
        sub_list=csv_file,
        output_dir=output_dir,
        seed_labels_lh=seed_labels_lh,
        seed_labels_rh=seed_labels_rh,
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
    )

    assert result_dir.exists()

    # Check directory structure
    params_dir = result_dir / "Params_training"
    gen_dir = params_dir / "generate_profiles_and_ini_params"
    assert gen_dir.exists()

    # fMRI list files
    fmri_list_dir = gen_dir / "data_list" / "fMRI_list"
    assert (fmri_list_dir / "lh_sub1_sess1.txt").exists()
    assert (fmri_list_dir / "rh_sub2_sess2.txt").exists()

    # Profile files
    profiles_dir = gen_dir / "profiles"
    assert (profiles_dir / "sub1" / "sess1").is_dir()

    # Profile list files
    test_set = (params_dir / "generate_individual_parcellations" /
                "profile_list" / "test_set")
    assert (test_set / "lh_sess1.txt").exists()

    # Training set (copy)
    training_set = (params_dir / "estimate_group_priors" /
                    "profile_list" / "training_set")
    assert (training_set / "lh_sess1.txt").exists()

    # Average profiles
    avg_dir = profiles_dir / "avg_profile"
    assert (avg_dir / "lh_fsaverage6_roifsaverage3_avg_profile.nii.gz").exists()
    assert (avg_dir / "rh_fsaverage6_roifsaverage3_avg_profile.nii.gz").exists()


# ---------------------------------------------------------------------------
# test_run_wrapper_auto_seed_labels
# ---------------------------------------------------------------------------

def _make_sphere_surface(subjects_dir: Path, mesh: str, hemi: str,
                         n_vertices: int, seed: int = 42) -> None:
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


def test_run_wrapper_auto_seed_labels(tmp_path):
    """Pipeline should auto-compute seed labels from FreeSurfer surfaces."""
    rng = np.random.default_rng(42)
    n_vertices, n_timepoints = 20, 30
    n_seed_vertices = 5

    # Set up FreeSurfer-like subjects dir with sphere surfaces
    fs_dir = tmp_path / "freesurfer"
    for hemi in ("lh", "rh"):
        _make_sphere_surface(fs_dir, "fsaverage3", hemi, n_seed_vertices,
                             seed=42)
        _make_sphere_surface(fs_dir, "fsaverage6", hemi, n_vertices,
                             seed=99)

    # Set up subject data
    data_dir = tmp_path / "data"
    _make_fmri_files(data_dir, "sub001", n_sess=1,
                     n_vertices=n_vertices, n_timepoints=n_timepoints,
                     rng=rng)

    csv_file = tmp_path / "subs.csv"
    csv_file.write_text(
        "subject_id,data_dir\n"
        f"sub001,{data_dir}/\n"
    )

    output_dir = tmp_path / "output"
    result_dir = run_wrapper(
        sub_list=csv_file,
        output_dir=output_dir,
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
        freesurfer_dir=fs_dir,
    )

    assert result_dir.exists()
    # Verify profiles were created (proves seed labels were computed)
    params_dir = result_dir / "Params_training"
    profiles_dir = (params_dir / "generate_profiles_and_ini_params" /
                    "profiles")
    assert (profiles_dir / "sub1" / "sess1").is_dir()
