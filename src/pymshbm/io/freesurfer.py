"""FreeSurfer sphere surface utilities for seed label computation."""

import os
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
from scipy.spatial import cKDTree


def _resolve_subjects_dir(freesurfer_dir: str | Path | None) -> Path:
    """Resolve the FreeSurfer subjects directory."""
    if freesurfer_dir is not None:
        p = Path(freesurfer_dir)
        if not p.exists():
            raise FileNotFoundError(
                f"FreeSurfer subjects directory not found: {p}")
        return p

    fs_home = os.environ.get("FREESURFER_HOME")
    if fs_home:
        return Path(fs_home) / "subjects"

    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if subjects_dir:
        return Path(subjects_dir)

    raise ValueError(
        "No FreeSurfer subjects directory found. "
        "Provide freesurfer_dir, or set $FREESURFER_HOME or $SUBJECTS_DIR."
    )


def compute_seed_labels(
    seed_mesh: str,
    targ_mesh: str,
    hemi: str,
    freesurfer_dir: str | Path | None = None,
) -> np.ndarray:
    """Map each target vertex to its nearest seed vertex on sphere surfaces.

    Args:
        seed_mesh: Seed mesh name (e.g. "fsaverage3").
        targ_mesh: Target mesh name (e.g. "fsaverage6").
        hemi: Hemisphere, "lh" or "rh".
        freesurfer_dir: FreeSurfer subjects directory. Falls back to
            $FREESURFER_HOME/subjects or $SUBJECTS_DIR.

    Returns:
        1-indexed int32 array of shape (N_targ,) mapping each target
        vertex to its nearest seed vertex.
    """
    subjects_dir = _resolve_subjects_dir(freesurfer_dir)

    seed_path = subjects_dir / seed_mesh / "surf" / f"{hemi}.sphere.reg"
    targ_path = subjects_dir / targ_mesh / "surf" / f"{hemi}.sphere.reg"

    if not seed_path.exists():
        raise FileNotFoundError(f"Seed sphere.reg not found: {seed_path}")
    if not targ_path.exists():
        raise FileNotFoundError(f"Target sphere.reg not found: {targ_path}")

    seed_coords, _ = fs.read_geometry(str(seed_path))
    targ_coords, _ = fs.read_geometry(str(targ_path))

    tree = cKDTree(seed_coords)
    _, nearest_indices = tree.query(targ_coords)

    return (nearest_indices + 1).astype(np.int32)
