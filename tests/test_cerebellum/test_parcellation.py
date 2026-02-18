"""Tests for cerebellum parcellation orchestrator."""

import numpy as np
import pytest

from pymshbm.cerebellum.parcellation import cerebellum_parcellation


def test_basic_parcellation():
    """Orchestrator should return labels for all cerebellar voxels."""
    rng = np.random.default_rng(42)
    M, N = 20, 30
    surf_labels = rng.integers(1, 4, size=N)
    vol2surf_fc = rng.standard_normal((M, N))
    labels, confidence = cerebellum_parcellation(surf_labels, vol2surf_fc, top_x=5)
    assert labels.shape == (M,)
    assert confidence.shape == (M,)
    assert np.all(labels > 0)


def test_confidence_computed():
    """Confidence should be non-negative."""
    rng = np.random.default_rng(42)
    surf_labels = rng.integers(1, 5, size=20)
    fc = rng.standard_normal((10, 20))
    labels, confidence = cerebellum_parcellation(surf_labels, fc, top_x=5)
    assert np.all(confidence >= 0)
    assert np.all(confidence <= 1)
