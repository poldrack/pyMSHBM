"""MSHBM wrapper pipeline — data preparation for training.

Plug-in replacement for MSHBM_wrapper.m.
"""

import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from pymshbm.core.profiles import generate_profiles
from pymshbm.io.freesurfer import compute_seed_labels
from pymshbm.io.profile_lists import write_profile_list
from pymshbm.io.readers import read_fmri


def parse_sub_list(csv_path: str | Path) -> list[tuple[str, Path]]:
    """Parse subject list CSV, returning (subject_id, data_dir) pairs."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Subject list not found: {csv_path}")
    result = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append((row["subject_id"], Path(row["data_dir"])))
    return result


def discover_fmri_files(
    data_dir: str | Path,
    subject_id: str,
    file_pattern: str = "*nat_resid_bpss_fsaverage6_sm*.nii.gz",
) -> tuple[list[Path], list[Path]]:
    """Find lh/rh fMRI files matching pattern for a subject."""
    sub_dir = Path(data_dir) / subject_id
    lh_files = sorted(sub_dir.glob(f"lh{file_pattern}"))
    rh_files = sorted(sub_dir.glob(f"rh{file_pattern}"))
    return lh_files, rh_files


def compute_seed_timeseries(
    fmri_data: np.ndarray,
    seed_labels: np.ndarray,
    num_seeds: int,
) -> np.ndarray:
    """Average fMRI time series within each seed ROI.

    Args:
        fmri_data: (N_vertices, T) fMRI data.
        seed_labels: (N_vertices,) int labels, 1-indexed. 0 = unassigned.
        num_seeds: Number of seed ROIs.

    Returns:
        (T, num_seeds) averaged time series per seed.
    """
    n_timepoints = fmri_data.shape[1]
    result = np.zeros((n_timepoints, num_seeds), dtype=np.float64)
    for s in range(1, num_seeds + 1):
        mask = seed_labels == s
        if mask.any():
            result[:, s - 1] = fmri_data[mask].mean(axis=0)
    return result


def _save_profile_nifti(path: Path, data: np.ndarray) -> None:
    """Save a profile array as NIfTI with identity affine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store as (N_targ, 1, 1, N_seed) to match MATLAB layout
    img = nib.Nifti1Image(
        data.astype(np.float32).reshape(data.shape[0], 1, 1, data.shape[1]),
        np.eye(4),
    )
    nib.save(img, str(path))


def _profile_filename(hemi: str, sub_idx: int, sess_idx: int,
                      targ_mesh: str, seed_mesh: str) -> str:
    """Generate profile filename matching MATLAB convention."""
    return (f"{hemi}.sub{sub_idx}_sess{sess_idx}_{targ_mesh}"
            f"_roi{seed_mesh}.surf2surf_profile.nii.gz")


def generate_and_save_profile(
    lh_fmri_path: Path,
    rh_fmri_path: Path,
    seed_labels_lh: np.ndarray,
    seed_labels_rh: np.ndarray,
    out_dir: Path,
    sub_idx: int,
    sess_idx: int,
    seed_mesh: str,
    targ_mesh: str,
) -> None:
    """Generate FC profile for one subject/session and save as NIfTI."""
    lh_bundle = read_fmri(lh_fmri_path)
    rh_bundle = read_fmri(rh_fmri_path)

    # series shape: (N_vertices, T)
    lh_data = lh_bundle.series
    rh_data = rh_bundle.series

    num_seeds_lh = int(seed_labels_lh.max())
    num_seeds_rh = int(seed_labels_rh.max())

    # Compute seed time series from both hemispheres
    lh_seed_ts = compute_seed_timeseries(lh_data, seed_labels_lh, num_seeds_lh)
    rh_seed_ts = compute_seed_timeseries(rh_data, seed_labels_rh, num_seeds_rh)
    # Concatenate seed time series: (T, num_seeds_lh + num_seeds_rh)
    seed_ts = np.hstack([lh_seed_ts, rh_seed_ts])

    # Generate profiles: target data needs to be (T, N_targ)
    lh_profile = generate_profiles(targ_data=lh_data.T, seed_data=seed_ts)
    rh_profile = generate_profiles(targ_data=rh_data.T, seed_data=seed_ts)

    # Save
    sess_dir = out_dir / f"sub{sub_idx}" / f"sess{sess_idx}"
    lh_fname = _profile_filename("lh", sub_idx, sess_idx, targ_mesh, seed_mesh)
    rh_fname = _profile_filename("rh", sub_idx, sess_idx, targ_mesh, seed_mesh)
    _save_profile_nifti(sess_dir / lh_fname, lh_profile)
    _save_profile_nifti(sess_dir / rh_fname, rh_profile)


def create_profile_lists(
    profile_dir: Path,
    num_subs: int,
    sessions_per_sub: list[int],
    out_dir: Path,
    seed_mesh: str,
    targ_mesh: str,
) -> None:
    """Create per-session text files listing profile paths.

    For each session, writes one file per hemisphere containing one line per
    subject (the path to that subject's profile). Subjects with fewer sessions
    get "NONE" entries.
    """
    max_sess = max(sessions_per_sub)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sess in range(1, max_sess + 1):
        for hemi in ("lh", "rh"):
            paths = []
            for sub in range(1, num_subs + 1):
                if sess <= sessions_per_sub[sub - 1]:
                    fname = _profile_filename(
                        hemi, sub, sess, targ_mesh, seed_mesh)
                    p = (profile_dir / f"sub{sub}" / f"sess{sess}" / fname)
                    paths.append(str(p))
                else:
                    paths.append("NONE")
            write_profile_list(out_dir / f"{hemi}_sess{sess}.txt", paths)


def average_profiles_nifti(
    profile_dir: Path,
    num_subs: int,
    sessions_per_sub: list[int],
    out_dir: Path,
    seed_mesh: str,
    targ_mesh: str,
) -> None:
    """Average profiles across all subjects/sessions and save as NIfTI."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for hemi in ("lh", "rh"):
        profiles = []
        for sub in range(1, num_subs + 1):
            for sess in range(1, sessions_per_sub[sub - 1] + 1):
                fname = _profile_filename(
                    hemi, sub, sess, targ_mesh, seed_mesh)
                p = profile_dir / f"sub{sub}" / f"sess{sess}" / fname
                if p.exists():
                    img = nib.load(str(p))
                    data = np.asarray(img.dataobj, dtype=np.float32)
                    # Flatten to (N_targ, N_seed)
                    n_targ = data.shape[0]
                    n_seed = data.shape[-1]
                    profiles.append(data.reshape(n_targ, n_seed))
        if profiles:
            avg = np.mean(np.stack(profiles), axis=0)
            avg_fname = f"{hemi}_{targ_mesh}_roi{seed_mesh}_avg_profile.nii.gz"
            _save_profile_nifti(out_dir / avg_fname, avg)


def _create_directory_structure(base_dir: Path) -> dict[str, Path]:
    """Create the MSHBM training directory structure."""
    dirs = {
        "fmri_list": (base_dir / "Params_training" /
                      "generate_profiles_and_ini_params" /
                      "data_list" / "fMRI_list"),
        "profiles": (base_dir / "Params_training" /
                     "generate_profiles_and_ini_params" / "profiles"),
        "test_set": (base_dir / "Params_training" /
                     "generate_individual_parcellations" /
                     "profile_list" / "test_set"),
        "training_set": (base_dir / "Params_training" /
                         "estimate_group_priors" /
                         "profile_list" / "training_set"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run_wrapper(
    sub_list: str | Path,
    output_dir: str | Path,
    seed_labels_lh: np.ndarray | None = None,
    seed_labels_rh: np.ndarray | None = None,
    file_pattern: str = "*nat_resid_bpss_fsaverage6_sm*.nii.gz",
    seed_mesh: str = "fsaverage3",
    targ_mesh: str = "fsaverage6",
    freesurfer_dir: str | Path | None = None,
) -> Path:
    """Run the full MSHBM wrapper pipeline."""
    output_dir = Path(output_dir)

    if seed_labels_lh is None:
        seed_labels_lh = compute_seed_labels(
            seed_mesh, targ_mesh, "lh", freesurfer_dir)
    if seed_labels_rh is None:
        seed_labels_rh = compute_seed_labels(
            seed_mesh, targ_mesh, "rh", freesurfer_dir)
    subjects = parse_sub_list(sub_list)

    # Build output dir name from first 3 chars of each subject ID
    sub_tag = "".join(s[0][:3] for s in subjects)
    main_dir = output_dir / f"Params_{sub_tag}"
    dirs = _create_directory_structure(main_dir)

    # Step 1: Discover fMRI files and write fMRI list text files
    sessions_per_sub = []
    for sub_num, (subject_id, data_dir) in enumerate(subjects, start=1):
        lh_files, rh_files = discover_fmri_files(
            data_dir, subject_id, file_pattern)
        sessions_per_sub.append(len(lh_files))

        for sess_num, (lh_f, rh_f) in enumerate(
                zip(lh_files, rh_files), start=1):
            for hemi, fpath in [("lh", lh_f), ("rh", rh_f)]:
                list_file = (dirs["fmri_list"] /
                             f"{hemi}_sub{sub_num}_sess{sess_num}.txt")
                list_file.write_text(str(fpath))

    # Step 2: Generate profiles
    for sub_num, (subject_id, data_dir) in enumerate(subjects, start=1):
        lh_files, rh_files = discover_fmri_files(
            data_dir, subject_id, file_pattern)
        for sess_num, (lh_f, rh_f) in enumerate(
                zip(lh_files, rh_files), start=1):
            generate_and_save_profile(
                lh_fmri_path=lh_f,
                rh_fmri_path=rh_f,
                seed_labels_lh=seed_labels_lh,
                seed_labels_rh=seed_labels_rh,
                out_dir=dirs["profiles"],
                sub_idx=sub_num,
                sess_idx=sess_num,
                seed_mesh=seed_mesh,
                targ_mesh=targ_mesh,
            )

    # Step 3: Create profile list files
    create_profile_lists(
        dirs["profiles"], len(subjects), sessions_per_sub,
        dirs["test_set"], seed_mesh, targ_mesh,
    )

    # Step 4: Copy profile lists to training set
    shutil.copytree(dirs["test_set"], dirs["training_set"],
                    dirs_exist_ok=True)

    # Step 5: Average profiles
    average_profiles_nifti(
        dirs["profiles"], len(subjects), sessions_per_sub,
        dirs["profiles"] / "avg_profile", seed_mesh, targ_mesh,
    )

    return main_dir
