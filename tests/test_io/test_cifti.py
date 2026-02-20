"""Tests for CIFTI dlabel writing."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pymshbm.io.cifti import write_dlabel_cifti


@pytest.fixture()
def label_data():
    """Simple label arrays for lh/rh with 3 clusters."""
    rng = np.random.default_rng(42)
    lh = rng.integers(0, 4, size=10).astype(np.int32)  # 0=medial wall, 1-3
    rh = rng.integers(0, 4, size=10).astype(np.int32)
    return lh, rh


def test_write_dlabel_cifti_creates_file(tmp_path, label_data):
    lh, rh = label_data
    out = tmp_path / "parc.dlabel.nii"
    write_dlabel_cifti(lh, rh, out, num_vertices_lh=10, num_vertices_rh=10)
    assert out.exists()


def test_write_dlabel_cifti_loadable(tmp_path, label_data):
    lh, rh = label_data
    out = tmp_path / "parc.dlabel.nii"
    write_dlabel_cifti(lh, rh, out, num_vertices_lh=10, num_vertices_rh=10)
    img = nib.load(str(out))
    assert isinstance(img, nib.Cifti2Image)


def test_write_dlabel_cifti_correct_data(tmp_path, label_data):
    lh, rh = label_data
    out = tmp_path / "parc.dlabel.nii"
    write_dlabel_cifti(lh, rh, out, num_vertices_lh=10, num_vertices_rh=10)
    img = nib.load(str(out))
    data = np.asarray(img.dataobj).ravel()
    # Only cortex (non-medial-wall) vertices are in the data
    lh_cortex = lh[lh > 0]
    rh_cortex = rh[rh > 0]
    expected = np.concatenate([lh_cortex, rh_cortex]).astype(np.float32)
    np.testing.assert_array_equal(data, expected)


def test_write_dlabel_cifti_axes(tmp_path, label_data):
    lh, rh = label_data
    out = tmp_path / "parc.dlabel.nii"
    write_dlabel_cifti(lh, rh, out, num_vertices_lh=10, num_vertices_rh=10)
    img = nib.load(str(out))
    axes = [img.header.get_axis(i) for i in range(2)]
    axis_types = {type(a).__name__ for a in axes}
    assert "LabelAxis" in axis_types
    assert "BrainModelAxis" in axis_types


def test_write_dlabel_cifti_excludes_medial_wall(tmp_path):
    """Medial wall vertices (label 0) should not appear in BrainModelAxis."""
    lh = np.array([0, 1, 2, 0, 3], dtype=np.int32)
    rh = np.array([1, 0, 2, 3, 0], dtype=np.int32)
    out = tmp_path / "parc.dlabel.nii"
    write_dlabel_cifti(lh, rh, out, num_vertices_lh=5, num_vertices_rh=5)
    img = nib.load(str(out))
    bm_axis = img.header.get_axis(1)
    data = np.asarray(img.dataobj).ravel()

    # lh has 3 cortex vertices (indices 1,2,4), rh has 3 (indices 0,2,3)
    assert len(data) == 6

    # All data values should be > 0 (no medial wall in output)
    assert np.all(data > 0)

    # Verify vertex indices in BrainModel
    vertex_indices = list(bm_axis.vertex)
    assert vertex_indices == [1, 2, 4, 0, 2, 3]

    # num_vertices should still reflect full surface size
    structures = list(bm_axis.iter_structures())
    for name, _, bm in structures:
        assert bm.nvertices[name] == 5


def test_write_dlabel_cifti_label_table(tmp_path, label_data):
    lh, rh = label_data
    out = tmp_path / "parc.dlabel.nii"
    names = ["Network_A", "Network_B", "Network_C"]
    write_dlabel_cifti(
        lh, rh, out,
        num_vertices_lh=10, num_vertices_rh=10,
        label_names=names,
    )
    img = nib.load(str(out))
    label_axis = img.header.get_axis(0)
    label_dict = label_axis.label[0]
    # Key 0 should be the "???" / medial wall entry
    assert 0 in label_dict
    # Custom names should appear for keys 1-3
    for i, name in enumerate(names, start=1):
        assert i in label_dict
        assert label_dict[i][0] == name


def test_write_dlabel_cifti_custom_colors(tmp_path, label_data):
    lh, rh = label_data
    out = tmp_path / "parc.dlabel.nii"
    colors = np.array([
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
    ], dtype=np.uint8)
    write_dlabel_cifti(
        lh, rh, out,
        num_vertices_lh=10, num_vertices_rh=10,
        colors=colors,
    )
    img = nib.load(str(out))
    label_axis = img.header.get_axis(0)
    label_dict = label_axis.label[0]
    # Check RGBA for cluster 1 (index 1 in label_dict)
    rgba = label_dict[1][1]
    np.testing.assert_array_almost_equal(
        rgba, (1.0, 0.0, 0.0, 1.0), decimal=2,
    )
