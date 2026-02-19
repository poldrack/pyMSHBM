"""MSHBM wrapper pipeline — data preparation through training.

Plug-in replacement for MSHBM_wrapper.m.
"""

import csv
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from pymshbm.core.profiles import generate_ini_params, generate_profiles
from pymshbm.io.cifti import write_dlabel_cifti
from pymshbm.io.freesurfer import compute_seed_labels, load_surface_neighborhood
from pymshbm.io.profile_lists import write_profile_list
from pymshbm.io.readers import read_fmri
from pymshbm.pipeline.single_subject import parcellation_single_subject
from pymshbm.pipeline.training import params_training

logger = logging.getLogger(__name__)


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


def _profiles_exist(
    profile_dir: Path, sub_idx: int, sess_idx: int,
    targ_mesh: str, seed_mesh: str,
) -> bool:
    """Check whether both lh and rh profile files exist for a subject/session."""
    sess_dir = profile_dir / f"sub{sub_idx}" / f"sess{sess_idx}"
    lh = sess_dir / _profile_filename("lh", sub_idx, sess_idx, targ_mesh, seed_mesh)
    rh = sess_dir / _profile_filename("rh", sub_idx, sess_idx, targ_mesh, seed_mesh)
    return lh.exists() and rh.exists()


def _avg_profiles_exist(avg_dir: Path, targ_mesh: str, seed_mesh: str) -> bool:
    """Check whether both lh and rh averaged profile NIfTIs exist."""
    lh = avg_dir / f"lh_{targ_mesh}_roi{seed_mesh}_avg_profile.nii.gz"
    rh = avg_dir / f"rh_{targ_mesh}_roi{seed_mesh}_avg_profile.nii.gz"
    return lh.exists() and rh.exists()


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
        expected = 0
        for sub in range(1, num_subs + 1):
            for sess in range(1, sessions_per_sub[sub - 1] + 1):
                expected += 1
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
                else:
                    logger.warning("  %s profile missing: %s", hemi, p)
        if profiles:
            logger.info("  %s: averaging %d/%d profiles", hemi,
                        len(profiles), expected)
            avg = np.mean(np.stack(profiles), axis=0)
            avg_fname = f"{hemi}_{targ_mesh}_roi{seed_mesh}_avg_profile.nii.gz"
            _save_profile_nifti(out_dir / avg_fname, avg)
        else:
            logger.warning("  %s: no profile files found (expected %d) — "
                           "skipping average", hemi, expected)


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


def _load_profile_nifti(path: Path) -> np.ndarray:
    """Load a profile NIfTI and reshape to (N_targ, D)."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float64)
    n_targ = data.shape[0]
    n_seed = data.shape[-1]
    return data.reshape(n_targ, n_seed)


def _row_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length; zero rows stay zero."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _load_profiles_tensor(
    profile_dir: Path,
    num_subs: int,
    sessions_per_sub: list[int],
    seed_mesh: str,
    targ_mesh: str,
) -> np.ndarray:
    """Load all profiles into a (N, D, S, T) tensor for training.

    Args:
        profile_dir: Directory containing sub*/sess*/ profile NIfTIs.
        num_subs: Number of subjects.
        sessions_per_sub: Sessions per subject.
        seed_mesh: Seed mesh identifier.
        targ_mesh: Target mesh identifier.

    Returns:
        (N, D, S, max_T) float64, row-normalized. N = 2*N_targ (lh+rh).
    """
    max_sess = max(sessions_per_sub)

    # Probe shape from first available profile
    fname = _profile_filename("lh", 1, 1, targ_mesh, seed_mesh)
    first = _load_profile_nifti(profile_dir / "sub1" / "sess1" / fname)
    n_targ, n_seed = first.shape
    n_vertices = 2 * n_targ

    tensor = np.zeros((n_vertices, n_seed, num_subs, max_sess), dtype=np.float64)

    for sub in range(1, num_subs + 1):
        for sess in range(1, sessions_per_sub[sub - 1] + 1):
            lh_fname = _profile_filename("lh", sub, sess, targ_mesh, seed_mesh)
            rh_fname = _profile_filename("rh", sub, sess, targ_mesh, seed_mesh)
            lh = _load_profile_nifti(profile_dir / f"sub{sub}" / f"sess{sess}" / lh_fname)
            rh = _load_profile_nifti(profile_dir / f"sub{sub}" / f"sess{sess}" / rh_fname)
            stacked = np.vstack([lh, rh])
            tensor[:, :, sub - 1, sess - 1] = _row_normalize(stacked)

    return tensor


def _centroids_path(avg_profile_dir: Path, num_clusters: int) -> Path:
    """Return the path where cached k-means centroids are stored."""
    return avg_profile_dir / f"g_mu_K{num_clusters}.npy"


def _compute_initial_centroids(
    avg_profile_dir: Path,
    num_clusters: int,
    targ_mesh: str,
    seed_mesh: str,
    overwrite: bool = False,
) -> np.ndarray:
    """Compute initial group centroids via k-means on averaged profiles.

    Results are cached to disk and reused on subsequent runs unless
    *overwrite* is True.

    Args:
        avg_profile_dir: Directory containing lh/rh averaged profile NIfTIs.
        num_clusters: Number of clusters K.
        targ_mesh: Target mesh identifier.
        seed_mesh: Seed mesh identifier.
        overwrite: If True, recompute even when a cached file exists.

    Returns:
        (D, K) unit-normalized centroids.
    """
    cached = _centroids_path(avg_profile_dir, num_clusters)
    if not overwrite and cached.exists():
        logger.info("  Reusing cached centroids from %s", cached)
        return np.load(str(cached))

    lh_fname = f"lh_{targ_mesh}_roi{seed_mesh}_avg_profile.nii.gz"
    rh_fname = f"rh_{targ_mesh}_roi{seed_mesh}_avg_profile.nii.gz"
    lh = _load_profile_nifti(avg_profile_dir / lh_fname)
    rh = _load_profile_nifti(avg_profile_dir / rh_fname)
    stacked = np.vstack([lh, rh])
    _, centroids = generate_ini_params(stacked, num_clusters)
    np.save(str(cached), centroids)
    return centroids


def _write_parcellations_cifti(
    results: list[tuple[np.ndarray, np.ndarray]],
    subject_ids: list[str],
    num_vertices_lh: int,
    num_vertices_rh: int,
    output_dir: Path,
) -> None:
    """Write parcellation results as CIFTI dlabel files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for (lh_labels, rh_labels), sid in zip(results, subject_ids):
        out_path = output_dir / f"{sid}.dlabel.nii"
        write_dlabel_cifti(
            lh_labels, rh_labels, out_path,
            num_vertices_lh, num_vertices_rh,
        )


def run_wrapper(
    sub_list: str | Path,
    output_dir: str | Path,
    seed_labels_lh: np.ndarray | None = None,
    seed_labels_rh: np.ndarray | None = None,
    file_pattern: str = "*nat_resid_bpss_fsaverage6_sm*.nii.gz",
    seed_mesh: str = "fsaverage3",
    targ_mesh: str = "fsaverage6",
    freesurfer_dir: str | Path | None = None,
    num_clusters: int | None = None,
    max_iter: int = 50,
    mrf_weight: float = 50.0,
    spatial_weight: float = 200.0,
    overwrite_fc: bool = False,
    overwrite_kmeans: bool = False,
) -> Path:
    """Run the full MSHBM wrapper pipeline.

    Steps 1-5 prepare data (profiles, lists, averages).
    Steps 6-10 (conditional on num_clusters) run training, individual
    parcellation with MRF, and write CIFTI.
    """
    output_dir = Path(output_dir)
    total_steps = 10 if num_clusters else 5

    if seed_labels_lh is None:
        logger.info("Computing lh seed labels from %s -> %s sphere surfaces",
                     seed_mesh, targ_mesh)
        seed_labels_lh = compute_seed_labels(
            seed_mesh, targ_mesh, "lh", freesurfer_dir)
    if seed_labels_rh is None:
        logger.info("Computing rh seed labels from %s -> %s sphere surfaces",
                     seed_mesh, targ_mesh)
        seed_labels_rh = compute_seed_labels(
            seed_mesh, targ_mesh, "rh", freesurfer_dir)

    subjects = parse_sub_list(sub_list)
    logger.info("Loaded %d subjects from %s", len(subjects), sub_list)

    sub_tag = "".join(s[0][:3] for s in subjects)
    main_dir = output_dir / f"Params_{sub_tag}"
    dirs = _create_directory_structure(main_dir)
    logger.info("Output directory: %s", main_dir)

    # Step 1: Discover fMRI files and write fMRI list text files
    logger.info("Step 1/%d: Discovering fMRI files", total_steps)
    sessions_per_sub = []
    for sub_num, (subject_id, data_dir) in enumerate(subjects, start=1):
        lh_files, rh_files = discover_fmri_files(
            data_dir, subject_id, file_pattern)
        sessions_per_sub.append(len(lh_files))
        if len(lh_files) == 0:
            search_dir = Path(data_dir) / subject_id
            logger.warning("  %s: no fMRI files found in %s "
                           "matching lh/rh%s",
                           subject_id, search_dir, file_pattern)
        else:
            logger.info("  %s: found %d sessions", subject_id, len(lh_files))

        for sess_num, (lh_f, rh_f) in enumerate(
                zip(lh_files, rh_files), start=1):
            for hemi, fpath in [("lh", lh_f), ("rh", rh_f)]:
                list_file = (dirs["fmri_list"] /
                             f"{hemi}_sub{sub_num}_sess{sess_num}.txt")
                list_file.write_text(str(fpath))

    total_sessions = sum(sessions_per_sub)
    if total_sessions == 0:
        raise ValueError(
            f"No fMRI files found for any subject. Searched for files "
            f"matching lh/rh{file_pattern} in subject directories. "
            f"Check that data_dir paths in {sub_list} are correct and "
            f"files match the --file-pattern."
        )

    # Step 2: Generate profiles
    logger.info("Step 2/%d: Generating FC profiles", total_steps)
    for sub_num, (subject_id, data_dir) in enumerate(subjects, start=1):
        lh_files, rh_files = discover_fmri_files(
            data_dir, subject_id, file_pattern)
        for sess_num, (lh_f, rh_f) in enumerate(
                zip(lh_files, rh_files), start=1):
            if not overwrite_fc and _profiles_exist(
                    dirs["profiles"], sub_num, sess_num, targ_mesh, seed_mesh):
                logger.info("  %s session %d — reusing existing profiles",
                            subject_id, sess_num)
                continue
            logger.info("  %s session %d", subject_id, sess_num)
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
    logger.info("Step 3/%d: Creating profile list files", total_steps)
    create_profile_lists(
        dirs["profiles"], len(subjects), sessions_per_sub,
        dirs["test_set"], seed_mesh, targ_mesh,
    )

    # Step 4: Copy profile lists to training set
    logger.info("Step 4/%d: Copying profile lists to training set", total_steps)
    shutil.copytree(dirs["test_set"], dirs["training_set"],
                    dirs_exist_ok=True)

    # Step 5: Average profiles
    avg_dir = dirs["profiles"] / "avg_profile"
    if not overwrite_fc and _avg_profiles_exist(avg_dir, targ_mesh, seed_mesh):
        logger.info("Step 5/%d: Reusing existing averaged profiles",
                    total_steps)
    else:
        logger.info("Step 5/%d: Averaging profiles across subjects/sessions",
                    total_steps)
        average_profiles_nifti(
            dirs["profiles"], len(subjects), sessions_per_sub,
            avg_dir, seed_mesh, targ_mesh,
        )

    if num_clusters is not None:
        subject_ids = [s[0] for s in subjects]

        # Step 6: Load profiles tensor
        logger.info("Step 6/%d: Loading profiles into training tensor",
                    total_steps)
        data = _load_profiles_tensor(
            dirs["profiles"], len(subjects), sessions_per_sub,
            seed_mesh, targ_mesh,
        )

        # Step 7: Compute initial centroids
        logger.info("Step 7/%d: Computing initial centroids via k-means",
                    total_steps)
        g_mu = _compute_initial_centroids(
            avg_dir, num_clusters, targ_mesh, seed_mesh,
            overwrite=overwrite_kmeans,
        )

        # Step 8: Run training (group prior estimation)
        logger.info("Step 8/%d: Running group prior estimation", total_steps)
        params = params_training(
            data=data,
            g_mu=g_mu,
            num_clusters=num_clusters,
            max_iter=max_iter,
            save_all=True,
            output_dir=main_dir,
            subject_ids=subject_ids,
        )

        # Step 9: Individual parcellation with MRF
        logger.info("Step 9/%d: Running MRF-regularized individual parcellation "
                    "(c=%.1f, w=%.1f)", total_steps, mrf_weight, spatial_weight)
        neighborhood = load_surface_neighborhood(
            targ_mesh, freesurfer_dir,
        )
        results = []
        n_targ = data.shape[0] // 2
        for s, sid in enumerate(subject_ids):
            n_sess = sessions_per_sub[s]
            sub_data = data[:, :, s:s + 1, :n_sess]
            logger.info("  %s (%d sessions)", sid, n_sess)
            lh_labels, rh_labels = parcellation_single_subject(
                data=sub_data,
                group_priors=params,
                neighborhood=neighborhood,
                w=spatial_weight,
                c=mrf_weight,
                max_iter=max_iter,
            )
            results.append((lh_labels, rh_labels))

        # Step 10: Write CIFTI parcellations
        logger.info("Step 10/%d: Writing CIFTI parcellations", total_steps)
        _write_parcellations_cifti(
            results, subject_ids,
            num_vertices_lh=n_targ,
            num_vertices_rh=n_targ,
            output_dir=main_dir / "cifti_parcellations",
        )

    logger.info("Wrapper pipeline complete")
    return main_dir
