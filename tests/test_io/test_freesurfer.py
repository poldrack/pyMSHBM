"""Tests for FreeSurfer sphere surface utilities."""

from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np
import pytest

from pymshbm.io.freesurfer import compute_seed_labels, load_surface_neighborhood


def _make_sphere_surface(subjects_dir: Path, mesh: str, hemi: str,
                         n_vertices: int) -> None:
    """Create a synthetic sphere.reg surface file in FreeSurfer layout."""
    surf_dir = subjects_dir / mesh / "surf"
    surf_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    coords = rng.standard_normal((n_vertices, 3)).astype(np.float32)
    # Normalize to unit sphere
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / norms * 100.0  # typical FreeSurfer sphere radius

    # Create triangular faces (minimal valid mesh)
    n_faces = max(1, n_vertices - 2)
    faces = np.zeros((n_faces, 3), dtype=np.int32)
    for i in range(n_faces):
        faces[i] = [0, i + 1, min(i + 2, n_vertices - 1)]

    fs.write_geometry(str(surf_dir / f"{hemi}.sphere.reg"), coords, faces)


# ---------------------------------------------------------------------------
# test_compute_seed_labels_shape
# ---------------------------------------------------------------------------

def test_compute_seed_labels_shape(tmp_path):
    """Output shape should match target mesh vertex count, with 1-indexed labels."""
    n_seed = 10
    n_targ = 100
    _make_sphere_surface(tmp_path, "fsaverage3", "lh", n_seed)
    _make_sphere_surface(tmp_path, "fsaverage6", "lh", n_targ)

    labels = compute_seed_labels(
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
        hemi="lh",
        freesurfer_dir=tmp_path,
    )

    assert labels.shape == (n_targ,)
    assert labels.dtype == np.int32
    # All labels should be 1-indexed (>= 1)
    assert labels.min() >= 1
    assert labels.max() <= n_seed


# ---------------------------------------------------------------------------
# test_compute_seed_labels_nearest_neighbor
# ---------------------------------------------------------------------------

def test_compute_seed_labels_nearest_neighbor(tmp_path):
    """Each target vertex should map to the closest seed vertex."""
    # Create known seed vertices on a unit sphere
    seed_coords = np.array([
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 100.0],
    ], dtype=np.float32)
    seed_faces = np.array([[0, 1, 2]], dtype=np.int32)

    # Create target vertices near each seed
    targ_coords = np.array([
        [99.0, 1.0, 0.0],   # nearest to seed 0 -> label 1
        [1.0, 99.0, 0.0],   # nearest to seed 1 -> label 2
        [0.0, 1.0, 99.0],   # nearest to seed 2 -> label 3
        [98.0, 2.0, 0.0],   # nearest to seed 0 -> label 1
    ], dtype=np.float32)
    targ_faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

    # Write surfaces
    seed_dir = tmp_path / "seed_mesh" / "surf"
    targ_dir = tmp_path / "targ_mesh" / "surf"
    seed_dir.mkdir(parents=True)
    targ_dir.mkdir(parents=True)
    fs.write_geometry(str(seed_dir / "lh.sphere.reg"), seed_coords, seed_faces)
    fs.write_geometry(str(targ_dir / "lh.sphere.reg"), targ_coords, targ_faces)

    labels = compute_seed_labels(
        seed_mesh="seed_mesh",
        targ_mesh="targ_mesh",
        hemi="lh",
        freesurfer_dir=tmp_path,
    )

    expected = np.array([1, 2, 3, 1], dtype=np.int32)
    np.testing.assert_array_equal(labels, expected)


# ---------------------------------------------------------------------------
# test_compute_seed_labels_rh
# ---------------------------------------------------------------------------

def test_compute_seed_labels_rh(tmp_path):
    """Should work for right hemisphere too."""
    n_seed = 10
    n_targ = 50
    _make_sphere_surface(tmp_path, "fsaverage3", "rh", n_seed)
    _make_sphere_surface(tmp_path, "fsaverage6", "rh", n_targ)

    labels = compute_seed_labels(
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
        hemi="rh",
        freesurfer_dir=tmp_path,
    )

    assert labels.shape == (n_targ,)
    assert labels.min() >= 1


# ---------------------------------------------------------------------------
# test_compute_seed_labels_missing_dir
# ---------------------------------------------------------------------------

def test_compute_seed_labels_missing_dir(tmp_path):
    """Should raise FileNotFoundError when FreeSurfer dir is not found."""
    with pytest.raises(FileNotFoundError, match="FreeSurfer"):
        compute_seed_labels(
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            hemi="lh",
            freesurfer_dir=tmp_path / "nonexistent",
        )


# ---------------------------------------------------------------------------
# test_compute_seed_labels_missing_surface
# ---------------------------------------------------------------------------

def test_compute_seed_labels_missing_surface(tmp_path):
    """Should raise FileNotFoundError when sphere.reg file is missing."""
    # Create only the seed surface, not the target
    _make_sphere_surface(tmp_path, "fsaverage3", "lh", 10)

    with pytest.raises(FileNotFoundError, match="sphere.reg"):
        compute_seed_labels(
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            hemi="lh",
            freesurfer_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# test_compute_seed_labels_env_fallback
# ---------------------------------------------------------------------------

def test_compute_seed_labels_env_freesurfer_home(tmp_path, monkeypatch):
    """Should fall back to $FREESURFER_HOME/subjects when no dir given."""
    subjects_dir = tmp_path / "subjects"
    monkeypatch.setenv("FREESURFER_HOME", str(tmp_path))

    _make_sphere_surface(subjects_dir, "fsaverage3", "lh", 10)
    _make_sphere_surface(subjects_dir, "fsaverage6", "lh", 50)

    labels = compute_seed_labels(
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
        hemi="lh",
    )

    assert labels.shape == (50,)


def test_compute_seed_labels_env_subjects_dir(tmp_path, monkeypatch):
    """Should fall back to $SUBJECTS_DIR when no dir or $FREESURFER_HOME."""
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    _make_sphere_surface(tmp_path, "fsaverage3", "lh", 10)
    _make_sphere_surface(tmp_path, "fsaverage6", "lh", 50)

    labels = compute_seed_labels(
        seed_mesh="fsaverage3",
        targ_mesh="fsaverage6",
        hemi="lh",
    )

    assert labels.shape == (50,)


def test_compute_seed_labels_no_dir_no_env(monkeypatch):
    """Should raise ValueError when no dir and no env vars set."""
    monkeypatch.delenv("FREESURFER_HOME", raising=False)
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)

    with pytest.raises(ValueError, match="FreeSurfer"):
        compute_seed_labels(
            seed_mesh="fsaverage3",
            targ_mesh="fsaverage6",
            hemi="lh",
        )


# ---------------------------------------------------------------------------
# test_load_surface_neighborhood
# ---------------------------------------------------------------------------

def test_load_surface_neighborhood_shape(tmp_path):
    """Combined lh+rh neighborhood should have N_lh + N_rh rows."""
    n_lh, n_rh = 20, 20
    _make_sphere_surface(tmp_path, "fsaverage6", "lh", n_lh)
    _make_sphere_surface(tmp_path, "fsaverage6", "rh", n_rh)

    nb = load_surface_neighborhood("fsaverage6", freesurfer_dir=tmp_path)
    assert nb.shape[0] == n_lh + n_rh
    assert nb.ndim == 2


def test_load_surface_neighborhood_valid_indices(tmp_path):
    """All non-padding entries should be valid vertex indices."""
    n_lh, n_rh = 20, 20
    _make_sphere_surface(tmp_path, "fsaverage6", "lh", n_lh)
    _make_sphere_surface(tmp_path, "fsaverage6", "rh", n_rh)

    nb = load_surface_neighborhood("fsaverage6", freesurfer_dir=tmp_path)
    valid = nb[nb >= 0]
    assert np.all(valid < n_lh + n_rh)


def test_load_surface_neighborhood_no_cross_hemi(tmp_path):
    """LH vertices should not have RH neighbors and vice versa."""
    n_lh, n_rh = 20, 20
    _make_sphere_surface(tmp_path, "fsaverage6", "lh", n_lh)
    _make_sphere_surface(tmp_path, "fsaverage6", "rh", n_rh)

    nb = load_surface_neighborhood("fsaverage6", freesurfer_dir=tmp_path)
    # LH vertices (rows 0..n_lh-1): neighbors should all be < n_lh or -1
    lh_nb = nb[:n_lh]
    valid_lh = lh_nb[lh_nb >= 0]
    assert np.all(valid_lh < n_lh)

    # RH vertices (rows n_lh..): neighbors should all be >= n_lh or -1
    rh_nb = nb[n_lh:]
    valid_rh = rh_nb[rh_nb >= 0]
    assert np.all(valid_rh >= n_lh)
