"""Tests for winner-take-all cerebellum parcellation."""

import numpy as np
import pytest

from pymshbm.cerebellum.wta import winner_take_all


def test_basic_assignment():
    """WTA should assign most frequent label among top-X vertices."""
    # 5 cerebellar voxels, 10 cortical vertices
    rng = np.random.default_rng(42)
    surf_labels = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0])  # 0 = medial wall
    vol2surf_fc = rng.standard_normal((5, 10))
    # Make first voxel strongly correlated with label-1 vertices
    vol2surf_fc[0, 0:3] = 10.0
    assign, confidence = winner_take_all(surf_labels, vol2surf_fc, top_x=3)
    assert assign.shape == (5,)
    assert confidence.shape == (5,)
    assert assign[0] == 1  # Should be label 1


def test_medial_wall_excluded():
    """Medial wall vertices (label=0) should be excluded from consideration."""
    surf_labels = np.array([0, 0, 1, 2, 1])
    vol2surf_fc = np.array([[10, 10, 1, 1, 1]])  # high corr with medial wall
    assign, confidence = winner_take_all(surf_labels, vol2surf_fc, top_x=3)
    # Only non-medial-wall labels should appear
    assert assign[0] in [1, 2]


def test_confidence_range():
    """Confidence should be in [0, 1]."""
    rng = np.random.default_rng(42)
    surf_labels = np.array([1, 1, 2, 2, 3, 3])
    vol2surf_fc = rng.standard_normal((10, 6))
    assign, confidence = winner_take_all(surf_labels, vol2surf_fc, top_x=4)
    assert np.all(confidence >= 0)
    assert np.all(confidence <= 1)


def test_all_same_label_high_confidence():
    """If top-X all have same label, confidence should be 1."""
    surf_labels = np.array([1, 1, 1, 1, 1])
    vol2surf_fc = np.array([[5, 4, 3, 2, 1]])
    assign, confidence = winner_take_all(surf_labels, vol2surf_fc, top_x=3)
    assert assign[0] == 1
    assert confidence[0] == pytest.approx(1.0)


def test_nan_handling():
    """Voxels with all-NaN correlations should get NaN assignment."""
    surf_labels = np.array([1, 2, 3])
    vol2surf_fc = np.array([[np.nan, np.nan, np.nan]])
    assign, confidence = winner_take_all(surf_labels, vol2surf_fc, top_x=2)
    assert np.isnan(assign[0])
    assert np.isnan(confidence[0])
