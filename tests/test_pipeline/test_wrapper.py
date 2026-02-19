"""Tests for the MSHBM wrapper pipeline."""

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pytest

from pymshbm.pipeline.wrapper import (
    _compute_initial_centroids,
    _load_profiles_tensor,
    _profile_filename,
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


# ---------------------------------------------------------------------------
# test_run_wrapper_no_fmri_files_raises
# ---------------------------------------------------------------------------

def test_run_wrapper_no_fmri_files_raises(tmp_path):
    """Pipeline should raise ValueError when no fMRI files match."""
    n_seeds = 5
    n_vertices = 20

    # Create subject dirs but no fMRI files inside them
    data_dir = tmp_path / "data"
    (data_dir / "sub001").mkdir(parents=True)

    csv_file = tmp_path / "subs.csv"
    csv_file.write_text(
        "subject_id,data_dir\n"
        f"sub001,{data_dir}/\n"
    )

    seed_labels = np.repeat(np.arange(1, n_seeds + 1),
                             n_vertices // n_seeds).astype(np.int32)

    with pytest.raises(ValueError, match="No fMRI files found"):
        run_wrapper(
            sub_list=csv_file,
            output_dir=tmp_path / "output",
            seed_labels_lh=seed_labels,
            seed_labels_rh=seed_labels,
        )


# ---------------------------------------------------------------------------
# test_run_wrapper_skips_existing_profiles / overwrite_fc
# ---------------------------------------------------------------------------

def _setup_wrapper_with_precomputed_profiles(tmp_path, rng):
    """Set up wrapper data and pre-create profile files with known mtime."""
    n_vertices, n_timepoints, n_seeds = 20, 30, 5
    data_dir = tmp_path / "data"
    _make_fmri_files(data_dir, "sub001", n_sess=1,
                     n_vertices=n_vertices, n_timepoints=n_timepoints,
                     rng=rng)

    csv_file = tmp_path / "subs.csv"
    csv_file.write_text(
        "subject_id,data_dir\n"
        f"sub001,{data_dir}/\n"
    )

    seed_labels = np.repeat(np.arange(1, n_seeds + 1),
                             n_vertices // n_seeds).astype(np.int32)

    # Run wrapper once to generate profiles
    output_dir = tmp_path / "output"
    run_wrapper(
        sub_list=csv_file,
        output_dir=output_dir,
        seed_labels_lh=seed_labels,
        seed_labels_rh=seed_labels,
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
    )
    return csv_file, output_dir, seed_labels, n_vertices


def test_run_wrapper_skips_existing_profiles(tmp_path):
    """Step 2 should skip profile generation when files already exist."""
    rng = np.random.default_rng(42)
    csv_file, output_dir, seed_labels, n_vertices = \
        _setup_wrapper_with_precomputed_profiles(tmp_path, rng)

    # Find the generated profile and record its mtime
    result_dir = list(output_dir.iterdir())[0]
    profiles_dir = (result_dir / "Params_training" /
                    "generate_profiles_and_ini_params" / "profiles")
    lh_fname = _profile_filename("lh", 1, 1, "fsaverage6", "fsaverage3")
    profile_path = profiles_dir / "sub1" / "sess1" / lh_fname
    assert profile_path.exists()
    original_mtime = profile_path.stat().st_mtime

    # Run again — profiles should NOT be regenerated
    import time
    time.sleep(0.05)  # ensure mtime would differ if file were rewritten
    run_wrapper(
        sub_list=csv_file,
        output_dir=output_dir,
        seed_labels_lh=seed_labels,
        seed_labels_rh=seed_labels,
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
    )
    assert profile_path.stat().st_mtime == original_mtime


def test_run_wrapper_overwrite_fc_regenerates_profiles(tmp_path):
    """With overwrite_fc=True, profiles should be regenerated."""
    rng = np.random.default_rng(42)
    csv_file, output_dir, seed_labels, n_vertices = \
        _setup_wrapper_with_precomputed_profiles(tmp_path, rng)

    result_dir = list(output_dir.iterdir())[0]
    profiles_dir = (result_dir / "Params_training" /
                    "generate_profiles_and_ini_params" / "profiles")
    lh_fname = _profile_filename("lh", 1, 1, "fsaverage6", "fsaverage3")
    profile_path = profiles_dir / "sub1" / "sess1" / lh_fname
    original_mtime = profile_path.stat().st_mtime

    import time
    time.sleep(0.05)
    run_wrapper(
        sub_list=csv_file,
        output_dir=output_dir,
        seed_labels_lh=seed_labels,
        seed_labels_rh=seed_labels,
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
        overwrite_fc=True,
    )
    assert profile_path.stat().st_mtime > original_mtime


# ---------------------------------------------------------------------------
# Helpers for _load_profiles_tensor / _compute_initial_centroids tests
# ---------------------------------------------------------------------------

def _create_profile_niftis(
    profile_dir: Path,
    num_subs: int,
    sessions_per_sub: list[int],
    n_targ: int,
    n_seed: int,
    targ_mesh: str,
    seed_mesh: str,
    rng: np.random.Generator,
) -> None:
    """Create synthetic profile NIfTI files matching wrapper layout."""
    for sub in range(1, num_subs + 1):
        for sess in range(1, sessions_per_sub[sub - 1] + 1):
            for hemi in ("lh", "rh"):
                fname = (f"{hemi}.sub{sub}_sess{sess}_{targ_mesh}"
                         f"_roi{seed_mesh}.surf2surf_profile.nii.gz")
                data = rng.standard_normal((n_targ, 1, 1, n_seed))
                p = profile_dir / f"sub{sub}" / f"sess{sess}" / fname
                _write_nifti(p, data)


# ---------------------------------------------------------------------------
# test__load_profiles_tensor
# ---------------------------------------------------------------------------

def test_load_profiles_tensor_shape(tmp_path):
    """Output shape should be (2*N_targ, D, S, max_T)."""
    rng = np.random.default_rng(42)
    n_targ, n_seed = 10, 6
    sessions_per_sub = [2, 2]
    num_subs = 2
    profile_dir = tmp_path / "profiles"

    _create_profile_niftis(
        profile_dir, num_subs, sessions_per_sub,
        n_targ, n_seed, "fsaverage6", "fsaverage3", rng,
    )

    tensor = _load_profiles_tensor(
        profile_dir, num_subs, sessions_per_sub,
        seed_mesh="fsaverage3", targ_mesh="fsaverage6",
    )
    # N = 2 * n_targ (lh + rh), D = n_seed, S = num_subs, T = max sessions
    assert tensor.shape == (2 * n_targ, n_seed, num_subs, max(sessions_per_sub))


def test_load_profiles_tensor_row_normalized(tmp_path):
    """Non-zero rows should be unit vectors."""
    rng = np.random.default_rng(42)
    n_targ, n_seed = 10, 6
    sessions_per_sub = [1]
    profile_dir = tmp_path / "profiles"

    _create_profile_niftis(
        profile_dir, 1, sessions_per_sub,
        n_targ, n_seed, "fsaverage6", "fsaverage3", rng,
    )

    tensor = _load_profiles_tensor(
        profile_dir, 1, sessions_per_sub,
        seed_mesh="fsaverage3", targ_mesh="fsaverage6",
    )
    # Check that non-zero rows have unit norm
    for s in range(tensor.shape[2]):
        for t in range(tensor.shape[3]):
            col = tensor[:, :, s, t]
            norms = np.linalg.norm(col, axis=1)
            nonzero = norms > 0
            np.testing.assert_allclose(norms[nonzero], 1.0, atol=1e-6)


def test_load_profiles_tensor_unequal_sessions(tmp_path):
    """Missing sessions should be zero-padded."""
    rng = np.random.default_rng(42)
    n_targ, n_seed = 10, 6
    sessions_per_sub = [2, 1]  # sub2 has only 1 session
    profile_dir = tmp_path / "profiles"

    _create_profile_niftis(
        profile_dir, 2, sessions_per_sub,
        n_targ, n_seed, "fsaverage6", "fsaverage3", rng,
    )

    tensor = _load_profiles_tensor(
        profile_dir, 2, sessions_per_sub,
        seed_mesh="fsaverage3", targ_mesh="fsaverage6",
    )
    assert tensor.shape == (2 * n_targ, n_seed, 2, 2)
    # Sub2 session 2 should be all zeros
    np.testing.assert_array_equal(tensor[:, :, 1, 1], 0.0)
    # Sub2 session 1 should NOT be all zeros
    assert np.any(tensor[:, :, 1, 0] != 0)


# ---------------------------------------------------------------------------
# test__compute_initial_centroids
# ---------------------------------------------------------------------------

def test_compute_initial_centroids_shape(tmp_path):
    """Centroids should be (D, K) with unit-norm columns."""
    rng = np.random.default_rng(42)
    n_targ, n_seed = 20, 8
    num_clusters = 3
    avg_dir = tmp_path / "avg_profile"

    # Create lh and rh averaged profile NIfTIs
    for hemi in ("lh", "rh"):
        data = rng.standard_normal((n_targ, 1, 1, n_seed)).astype(np.float32)
        fname = f"{hemi}_fsaverage6_roifsaverage3_avg_profile.nii.gz"
        _write_nifti(avg_dir / fname, data)

    centroids = _compute_initial_centroids(
        avg_dir, num_clusters,
        targ_mesh="fsaverage6", seed_mesh="fsaverage3",
    )
    assert centroids.shape == (n_seed, num_clusters)
    # Check columns are unit-norm
    norms = np.linalg.norm(centroids, axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# test_run_wrapper with num_clusters (training + CIFTI)
# ---------------------------------------------------------------------------

def _make_mock_params(n_vertices, num_clusters, num_subs):
    """Build a mock MSHBMParams with valid s_lambda for extraction."""
    from pymshbm.types import MSHBMParams
    N = 2 * n_vertices
    D = 10
    L = num_clusters
    S = num_subs
    rng = np.random.default_rng(99)
    s_lambda = rng.random((N, L, S))
    # Normalize to look like posterior probs
    s_lambda /= s_lambda.sum(axis=1, keepdims=True)
    return MSHBMParams(
        mu=rng.random((D, L)),
        epsil=rng.random(L),
        sigma=rng.random(L),
        theta=rng.random((N, L)),
        kappa=rng.random(L),
        s_lambda=s_lambda,
        s_psi=rng.random((D, L, S)),
        s_t_nu=rng.random((D, L, 1, S)),
        iter_inter=5,
        record=[1.0, 2.0],
    )


def _mock_parcellation(data, group_priors, neighborhood, w=200.0, c=50.0,
                       max_iter=50):
    """Mock parcellation_single_subject that returns random labels."""
    n_vertices = data.shape[0] // 2
    rng = np.random.default_rng(42)
    lh = rng.integers(0, 4, size=n_vertices).astype(np.int32)
    rh = rng.integers(0, 4, size=n_vertices).astype(np.int32)
    return lh, rh


def _setup_num_clusters_run(tmp_path, rng, n_vertices=20, n_timepoints=30,
                            n_seeds=5, num_clusters=3, n_subs=2, n_sess=2):
    """Set up data and mocks for a num_clusters wrapper run."""
    data_dir = tmp_path / "data"
    sub_ids = [f"sub{i:03d}" for i in range(1, n_subs + 1)]
    for sub_id in sub_ids:
        _make_fmri_files(data_dir, sub_id, n_sess=n_sess,
                         n_vertices=n_vertices, n_timepoints=n_timepoints,
                         rng=rng)

    csv_file = tmp_path / "subs.csv"
    lines = ["subject_id,data_dir\n"]
    for sub_id in sub_ids:
        lines.append(f"{sub_id},{data_dir}/\n")
    csv_file.write_text("".join(lines))

    seed_labels = np.repeat(np.arange(1, n_seeds + 1),
                             n_vertices // n_seeds).astype(np.int32)

    mock_params = _make_mock_params(n_vertices, num_clusters, n_subs)
    mock_nb = np.full((2 * n_vertices, 3), -1, dtype=np.int64)

    return csv_file, seed_labels, mock_params, mock_nb


def test_run_wrapper_with_num_clusters(tmp_path):
    """End-to-end pipeline with training + CIFTI output."""
    rng = np.random.default_rng(42)
    num_clusters = 3
    csv_file, seed_labels, mock_params, mock_nb = _setup_num_clusters_run(
        tmp_path, rng, num_clusters=num_clusters,
    )

    with patch("pymshbm.pipeline.wrapper.params_training",
               return_value=mock_params), \
         patch("pymshbm.pipeline.wrapper.parcellation_single_subject",
               side_effect=_mock_parcellation), \
         patch("pymshbm.pipeline.wrapper.load_surface_neighborhood",
               return_value=mock_nb):
        result_dir = run_wrapper(
            sub_list=csv_file,
            output_dir=tmp_path / "output",
            seed_labels_lh=seed_labels,
            seed_labels_rh=seed_labels,
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            num_clusters=num_clusters,
        )

    # Check that CIFTI parcellations exist
    cifti_dir = result_dir / "cifti_parcellations"
    assert cifti_dir.exists()
    for sub_id in ("sub001", "sub002"):
        dlabel = cifti_dir / f"{sub_id}.dlabel.nii"
        assert dlabel.exists(), f"Missing {dlabel}"
        img = nib.load(str(dlabel))
        assert isinstance(img, nib.Cifti2Image)


def test_run_wrapper_without_num_clusters_skips_training(tmp_path):
    """When num_clusters is not provided, no training artifacts should exist."""
    rng = np.random.default_rng(42)
    n_vertices, n_timepoints, n_seeds = 20, 30, 5

    data_dir = tmp_path / "data"
    _make_fmri_files(data_dir, "sub001", n_sess=1,
                     n_vertices=n_vertices, n_timepoints=n_timepoints,
                     rng=rng)

    csv_file = tmp_path / "subs.csv"
    csv_file.write_text(
        "subject_id,data_dir\n"
        f"sub001,{data_dir}/\n"
    )

    seed_labels = np.repeat(np.arange(1, n_seeds + 1),
                             n_vertices // n_seeds).astype(np.int32)

    result_dir = run_wrapper(
        sub_list=csv_file,
        output_dir=tmp_path / "output",
        seed_labels_lh=seed_labels,
        seed_labels_rh=seed_labels,
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
    )

    # No CIFTI dir should exist
    assert not (result_dir / "cifti_parcellations").exists()
    # No Params_Final.mat
    assert not (result_dir / "priors" / "Params_Final.mat").exists()


# ---------------------------------------------------------------------------
# test_run_wrapper centroid caching (--overwrite-kmeans)
# ---------------------------------------------------------------------------

def test_run_wrapper_saves_centroids(tmp_path):
    """Step 7 should save centroids to a .npy file."""
    rng = np.random.default_rng(42)
    num_clusters = 3
    csv_file, seed_labels, mock_params, mock_nb = _setup_num_clusters_run(
        tmp_path, rng, num_clusters=num_clusters,
    )

    with patch("pymshbm.pipeline.wrapper.params_training",
               return_value=mock_params), \
         patch("pymshbm.pipeline.wrapper.parcellation_single_subject",
               side_effect=_mock_parcellation), \
         patch("pymshbm.pipeline.wrapper.load_surface_neighborhood",
               return_value=mock_nb):
        result_dir = run_wrapper(
            sub_list=csv_file,
            output_dir=tmp_path / "output",
            seed_labels_lh=seed_labels,
            seed_labels_rh=seed_labels,
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            num_clusters=num_clusters,
        )

    # Centroids file should exist in avg_profile dir
    profiles_dir = (result_dir / "Params_training" /
                    "generate_profiles_and_ini_params" / "profiles")
    centroids_path = profiles_dir / "avg_profile" / f"g_mu_K{num_clusters}.npy"
    assert centroids_path.exists()
    centroids = np.load(str(centroids_path))
    n_seeds = 2 * 5  # lh + rh seeds (5 each)
    assert centroids.shape == (n_seeds, num_clusters)


def test_run_wrapper_reuses_existing_centroids(tmp_path):
    """Step 7 should reuse saved centroids instead of recomputing."""
    import time
    rng = np.random.default_rng(42)
    num_clusters = 3
    csv_file, seed_labels, mock_params, mock_nb = _setup_num_clusters_run(
        tmp_path, rng, num_clusters=num_clusters,
    )

    # First run to generate centroids
    with patch("pymshbm.pipeline.wrapper.params_training",
               return_value=mock_params), \
         patch("pymshbm.pipeline.wrapper.parcellation_single_subject",
               side_effect=_mock_parcellation), \
         patch("pymshbm.pipeline.wrapper.load_surface_neighborhood",
               return_value=mock_nb):
        result_dir = run_wrapper(
            sub_list=csv_file,
            output_dir=tmp_path / "output",
            seed_labels_lh=seed_labels,
            seed_labels_rh=seed_labels,
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            num_clusters=num_clusters,
        )

    profiles_dir = (result_dir / "Params_training" /
                    "generate_profiles_and_ini_params" / "profiles")
    centroids_path = profiles_dir / "avg_profile" / f"g_mu_K{num_clusters}.npy"
    original_mtime = centroids_path.stat().st_mtime

    time.sleep(0.05)

    # Second run — centroids should NOT be recomputed
    with patch("pymshbm.pipeline.wrapper.params_training",
               return_value=mock_params), \
         patch("pymshbm.pipeline.wrapper.parcellation_single_subject",
               side_effect=_mock_parcellation), \
         patch("pymshbm.pipeline.wrapper.load_surface_neighborhood",
               return_value=mock_nb):
        run_wrapper(
            sub_list=csv_file,
            output_dir=tmp_path / "output",
            seed_labels_lh=seed_labels,
            seed_labels_rh=seed_labels,
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            num_clusters=num_clusters,
        )

    assert centroids_path.stat().st_mtime == original_mtime


def test_run_wrapper_overwrite_kmeans_recomputes(tmp_path):
    """With overwrite_kmeans=True, centroids should be recomputed."""
    import time
    rng = np.random.default_rng(42)
    num_clusters = 3
    csv_file, seed_labels, mock_params, mock_nb = _setup_num_clusters_run(
        tmp_path, rng, num_clusters=num_clusters,
    )

    # First run
    with patch("pymshbm.pipeline.wrapper.params_training",
               return_value=mock_params), \
         patch("pymshbm.pipeline.wrapper.parcellation_single_subject",
               side_effect=_mock_parcellation), \
         patch("pymshbm.pipeline.wrapper.load_surface_neighborhood",
               return_value=mock_nb):
        result_dir = run_wrapper(
            sub_list=csv_file,
            output_dir=tmp_path / "output",
            seed_labels_lh=seed_labels,
            seed_labels_rh=seed_labels,
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            num_clusters=num_clusters,
        )

    profiles_dir = (result_dir / "Params_training" /
                    "generate_profiles_and_ini_params" / "profiles")
    centroids_path = profiles_dir / "avg_profile" / f"g_mu_K{num_clusters}.npy"
    original_mtime = centroids_path.stat().st_mtime

    time.sleep(0.05)

    # Second run with overwrite_kmeans=True
    with patch("pymshbm.pipeline.wrapper.params_training",
               return_value=mock_params), \
         patch("pymshbm.pipeline.wrapper.parcellation_single_subject",
               side_effect=_mock_parcellation), \
         patch("pymshbm.pipeline.wrapper.load_surface_neighborhood",
               return_value=mock_nb):
        run_wrapper(
            sub_list=csv_file,
            output_dir=tmp_path / "output",
            seed_labels_lh=seed_labels,
            seed_labels_rh=seed_labels,
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            num_clusters=num_clusters,
            overwrite_kmeans=True,
        )

    assert centroids_path.stat().st_mtime > original_mtime
