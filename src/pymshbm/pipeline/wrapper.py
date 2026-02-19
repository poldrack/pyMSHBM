"""MSHBM wrapper pipeline — data preparation through training.

Plug-in replacement for MSHBM_wrapper.m.
"""

import csv
import logging
import shutil
from pathlib import Path

import numpy as np
import zarr

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


def compute_profile(
    lh_fmri_path: Path,
    rh_fmri_path: Path,
    seed_labels_lh: np.ndarray,
    seed_labels_rh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FC profiles for one subject/session.

    Returns:
        (lh_profile, rh_profile) — each (N_targ, D) float arrays.
    """
    lh_bundle = read_fmri(lh_fmri_path)
    rh_bundle = read_fmri(rh_fmri_path)

    lh_data = lh_bundle.series
    rh_data = rh_bundle.series

    num_seeds_lh = int(seed_labels_lh.max())
    num_seeds_rh = int(seed_labels_rh.max())

    lh_seed_ts = compute_seed_timeseries(lh_data, seed_labels_lh, num_seeds_lh)
    rh_seed_ts = compute_seed_timeseries(rh_data, seed_labels_rh, num_seeds_rh)
    seed_ts = np.hstack([lh_seed_ts, rh_seed_ts])

    lh_profile = generate_profiles(targ_data=lh_data.T, seed_data=seed_ts)
    rh_profile = generate_profiles(targ_data=rh_data.T, seed_data=seed_ts)

    return lh_profile, rh_profile


def _profile_filename(hemi: str, sub_idx: int, sess_idx: int,
                      targ_mesh: str, seed_mesh: str) -> str:
    """Generate profile filename matching MATLAB convention."""
    return (f"{hemi}.sub{sub_idx}_sess{sess_idx}_{targ_mesh}"
            f"_roi{seed_mesh}.surf2surf_profile.nii.gz")


def create_profile_lists(
    profile_dir: Path,
    num_subs: int,
    sessions_per_sub: list[int],
    out_dir: Path,
    seed_mesh: str,
    targ_mesh: str,
) -> None:
    """Create per-session text files listing profile paths (MATLAB compat)."""
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


def _open_profiles_zarr(
    store_path: Path,
    n_vertices: int,
    n_seeds: int,
    num_subs: int,
    max_sessions: int,
    overwrite: bool = False,
) -> zarr.Array:
    """Open or create the profiles Zarr array.

    Shape: (N_vertices, D_seeds, S_subjects, T_max_sessions).
    Chunks: one chunk per (subject, session). fill_value=NaN for
    detecting unwritten chunks. Uses float32 to halve memory footprint.
    """
    mode = "w" if overwrite else "a"
    if not store_path.exists() or overwrite:
        return zarr.open_array(
            str(store_path), mode=mode,
            shape=(n_vertices, n_seeds, num_subs, max_sessions),
            chunks=(n_vertices, n_seeds, 1, 1),
            dtype="float32",
            fill_value=float("nan"),
        )
    return zarr.open_array(str(store_path), mode="r+")


def _profile_chunk_written(profiles: zarr.Array, sub: int, sess: int) -> bool:
    """Check if a profile chunk has been written (not NaN)."""
    return np.isfinite(profiles[0, 0, sub, sess])


def average_profiles(
    store_path: Path,
    num_subs: int,
    sessions_per_sub: list[int],
    avg_path: Path,
) -> None:
    """Average profiles from Zarr store and save as Zarr array.

    Computes element-wise mean across all valid subject/session slices.
    """
    profiles = zarr.open_array(str(store_path), mode="r")
    data = profiles[:]  # (N, D, S, max_T)

    # Collect all valid (sub, sess) slices and average
    slices = []
    for sub in range(num_subs):
        for sess in range(sessions_per_sub[sub]):
            slices.append(data[:, :, sub, sess])

    avg = np.mean(np.stack(slices), axis=0)  # (N, D)
    if avg_path.exists():
        shutil.rmtree(avg_path)
    zarr.save(str(avg_path), avg)


def _row_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length; zero rows stay zero."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _load_profiles_tensor(
    store_path: Path,
    num_subs: int,
    sessions_per_sub: list[int],
    cache_path: Path | None = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Load profiles from Zarr, row-normalize, and optionally cache.

    Returns:
        (N, D, S, max_T) float64, row-normalized. NaN → 0 for
        unwritten sessions.
    """
    if cache_path is not None and cache_path.exists() and not overwrite:
        logger.info("  Reusing cached normalized tensor from %s", cache_path)
        return zarr.open_array(str(cache_path), mode="r")[:]

    data = zarr.open_array(str(store_path), mode="r")[:]
    np.nan_to_num(data, nan=0.0, copy=False)

    # Vectorized row normalization across all (sub, session) slices
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data /= norms

    if cache_path is not None:
        if cache_path.exists():
            shutil.rmtree(cache_path)
        zarr.save(str(cache_path), data)

    return data


def _detect_valid_vertices(data: np.ndarray) -> np.ndarray:
    """Detect non-medial-wall vertices.

    A vertex is valid if it has a nonzero profile in at least one
    (subject, session) slice.

    Args:
        data: (N, D, S, T) row-normalized profiles (zeros = medial wall).

    Returns:
        Boolean mask of shape (N,), True for valid vertices.
    """
    # Sum of absolute values across (D, S, T) — zero only if all-zero
    return np.any(data != 0, axis=(1, 2, 3))


def _centroids_path(profiles_dir: Path, num_clusters: int) -> Path:
    """Return the path where cached k-means centroids are stored."""
    return profiles_dir / f"g_mu_K{num_clusters}.npy"


def _compute_initial_centroids(
    avg_path: Path,
    num_clusters: int,
    overwrite: bool = False,
    cache_dir: Path | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute initial group centroids via k-means on averaged profiles.

    Results are cached to disk and reused on subsequent runs unless
    *overwrite* is True.

    Args:
        avg_path: Path to averaged profiles Zarr array.
        num_clusters: Number of clusters K.
        overwrite: If True, recompute even when a cached file exists.
        cache_dir: Directory for centroid cache file. Defaults to
            avg_path's parent.
        valid_mask: Optional boolean mask (N,) to exclude medial wall
            vertices before k-means.

    Returns:
        (D, K) unit-normalized centroids.
    """
    if cache_dir is None:
        cache_dir = avg_path.parent
    cached = _centroids_path(cache_dir, num_clusters)
    if not overwrite and cached.exists():
        logger.info("  Reusing cached centroids from %s", cached)
        return np.load(str(cached))

    avg = zarr.open_array(str(avg_path), mode="r")[:]  # (N, D)
    if valid_mask is not None:
        avg = avg[valid_mask]
    _, centroids = generate_ini_params(avg, num_clusters)
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

    # Determine profile dimensions
    n_targ = len(seed_labels_lh)
    n_vertices = 2 * n_targ
    n_seeds = int(seed_labels_lh.max()) + int(seed_labels_rh.max())
    max_sess = max(sessions_per_sub)

    # Step 2: Generate profiles → Zarr store
    logger.info("Step 2/%d: Generating FC profiles", total_steps)
    store_path = dirs["profiles"] / "fc_profiles.zarr"
    profiles_zarr = _open_profiles_zarr(
        store_path, n_vertices, n_seeds, len(subjects), max_sess,
        overwrite=overwrite_fc,
    )

    for sub_num, (subject_id, data_dir) in enumerate(subjects, start=1):
        lh_files, rh_files = discover_fmri_files(
            data_dir, subject_id, file_pattern)
        for sess_num, (lh_f, rh_f) in enumerate(
                zip(lh_files, rh_files), start=1):
            sub_idx = sub_num - 1
            sess_idx = sess_num - 1
            if not overwrite_fc and _profile_chunk_written(
                    profiles_zarr, sub_idx, sess_idx):
                logger.info("  %s session %d — reusing existing profiles",
                            subject_id, sess_num)
                continue
            logger.info("  %s session %d", subject_id, sess_num)
            lh_profile, rh_profile = compute_profile(
                lh_fmri_path=lh_f,
                rh_fmri_path=rh_f,
                seed_labels_lh=seed_labels_lh,
                seed_labels_rh=seed_labels_rh,
            )
            stacked = np.vstack([lh_profile, rh_profile])
            profiles_zarr[:, :, sub_idx, sess_idx] = stacked

    # Step 3: Create profile list files (MATLAB compat)
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
    avg_path = dirs["profiles"] / "avg_profiles.zarr"
    if not overwrite_fc and avg_path.exists():
        logger.info("Step 5/%d: Reusing existing averaged profiles",
                    total_steps)
    else:
        logger.info("Step 5/%d: Averaging profiles across subjects/sessions",
                    total_steps)
        average_profiles(
            store_path, len(subjects), sessions_per_sub, avg_path,
        )

    if num_clusters is not None:
        subject_ids = [s[0] for s in subjects]

        # Step 6: Load profiles tensor + detect medial wall
        logger.info("Step 6/%d: Loading profiles into training tensor",
                    total_steps)
        tensor_cache = dirs["profiles"] / "normalized_tensor.zarr"
        data_full = _load_profiles_tensor(
            store_path, len(subjects), sessions_per_sub,
            cache_path=tensor_cache, overwrite=overwrite_fc,
        )
        valid_mask = _detect_valid_vertices(data_full)
        n_valid = int(valid_mask.sum())
        n_full = data_full.shape[0]
        logger.info("  %d/%d vertices are non-medial-wall (%.0f%% reduction)",
                    n_valid, n_full, (1 - n_valid / n_full) * 100)
        data_reduced = data_full[valid_mask]

        # Step 7: Compute initial centroids (on reduced vertices)
        logger.info("Step 7/%d: Computing initial centroids via k-means",
                    total_steps)
        g_mu = _compute_initial_centroids(
            avg_path, num_clusters,
            overwrite=overwrite_kmeans,
            cache_dir=dirs["profiles"],
            valid_mask=valid_mask,
        )

        # Step 8: Run training (group prior estimation) on reduced data
        logger.info("Step 8/%d: Running group prior estimation", total_steps)
        params = params_training(
            data=data_reduced,
            g_mu=g_mu,
            num_clusters=num_clusters,
            max_iter=max_iter,
            save_all=True,
            output_dir=main_dir,
            subject_ids=subject_ids,
        )

        # Expand theta and s_lambda back to full surface for step 9
        L = num_clusters
        theta_full = np.zeros((n_full, L), dtype=np.float64)
        theta_full[valid_mask] = params.theta
        params.theta = theta_full
        if params.s_lambda is not None:
            S = len(subject_ids)
            s_lambda_full = np.zeros((n_full, L, S), dtype=np.float64)
            s_lambda_full[valid_mask] = params.s_lambda
            params.s_lambda = s_lambda_full

        # Step 9: Individual parcellation with MRF (full surface)
        logger.info("Step 9/%d: Running MRF-regularized individual parcellation "
                    "(c=%.1f, w=%.1f)", total_steps, mrf_weight, spatial_weight)
        neighborhood = load_surface_neighborhood(
            targ_mesh, freesurfer_dir,
        )
        results = []
        n_targ = data_full.shape[0] // 2
        for s, sid in enumerate(subject_ids):
            n_sess = sessions_per_sub[s]
            sub_data = data_full[:, :, s:s + 1, :n_sess]
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
