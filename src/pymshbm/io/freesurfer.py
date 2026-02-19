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


def load_cortex_mask(
    targ_mesh: str,
    hemi: str,
    n_vertices: int,
    freesurfer_dir: str | Path | None = None,
) -> np.ndarray:
    """Load cortex label and return a boolean mask.

    Reads ``{targ_mesh}/label/{hemi}.cortex.label`` from the FreeSurfer
    subjects directory. Vertices listed in the label are cortex (True);
    all others are medial wall (False).

    Args:
        targ_mesh: Target mesh name (e.g. "fsaverage6").
        hemi: Hemisphere, "lh" or "rh".
        n_vertices: Total number of vertices on this hemisphere.
        freesurfer_dir: FreeSurfer subjects directory.

    Returns:
        Boolean array of shape (n_vertices,). True = cortex.

    Raises:
        FileNotFoundError: If the cortex.label file is missing.
    """
    subjects_dir = _resolve_subjects_dir(freesurfer_dir)
    label_path = subjects_dir / targ_mesh / "label" / f"{hemi}.cortex.label"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Cortex label not found: {label_path}. "
            f"Expected {hemi}.cortex.label in {subjects_dir / targ_mesh / 'label'}"
        )
    cortex_indices = fs.read_label(str(label_path))
    mask = np.zeros(n_vertices, dtype=bool)
    mask[cortex_indices] = True
    return mask


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


def load_surface_neighborhood(
    targ_mesh: str,
    freesurfer_dir: str | Path | None = None,
) -> np.ndarray:
    """Build combined lh+rh neighborhood matrix from surface faces.

    Args:
        targ_mesh: Target mesh name (e.g. "fsaverage6").
        freesurfer_dir: FreeSurfer subjects directory.

    Returns:
        (N_lh + N_rh, max_neighbors) int64 array, -1 = invalid.
    """
    subjects_dir = _resolve_subjects_dir(freesurfer_dir)

    lh_path = subjects_dir / targ_mesh / "surf" / "lh.sphere.reg"
    rh_path = subjects_dir / targ_mesh / "surf" / "rh.sphere.reg"
    if not lh_path.exists():
        raise FileNotFoundError(f"Surface not found: {lh_path}")
    if not rh_path.exists():
        raise FileNotFoundError(f"Surface not found: {rh_path}")

    lh_coords, lh_faces = fs.read_geometry(str(lh_path))
    rh_coords, rh_faces = fs.read_geometry(str(rh_path))
    n_lh = lh_coords.shape[0]
    n_rh = rh_coords.shape[0]

    lh_adj = _faces_to_adjacency(lh_faces, n_lh)
    rh_adj = _faces_to_adjacency(rh_faces, n_rh)

    # Offset rh indices by n_lh
    rh_adj_offset = {
        k + n_lh: {v + n_lh for v in neighbors}
        for k, neighbors in rh_adj.items()
    }

    # Combine and build matrix
    combined = {**lh_adj, **rh_adj_offset}
    return _adjacency_to_neighborhood(combined, n_lh + n_rh)


def _faces_to_adjacency(
    faces: np.ndarray, num_vertices: int,
) -> dict[int, set[int]]:
    """Build adjacency dict from triangle faces."""
    adj: dict[int, set[int]] = {i: set() for i in range(num_vertices)}
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[int(f[i])].add(int(f[j]))
    return adj


def _adjacency_to_neighborhood(
    adj: dict[int, set[int]], num_vertices: int,
) -> np.ndarray:
    """Convert adjacency dict to (N, max_nb) neighborhood array."""
    max_nb = max((len(s) for s in adj.values()), default=0)
    if max_nb == 0:
        return np.full((num_vertices, 1), -1, dtype=np.int64)
    neighborhood = np.full((num_vertices, max_nb), -1, dtype=np.int64)
    for i, neighbors in adj.items():
        for j, nb in enumerate(sorted(neighbors)):
            neighborhood[i, j] = nb
    return neighborhood
