"""Data types for the MSHBM pipeline."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class FileFormat(Enum):
    """Supported file formats for saving/loading data."""

    MAT_V5 = "mat_v5"
    MAT_V73 = "mat_v73"
    NPZ = "npz"
    AUTO = "auto"


@dataclass
class MSHBMParams:
    """Parameters for the Multi-Scale Hierarchical Bayesian Model.

    Attributes:
        mu: Group-level connectivity profiles (D x L).
        epsil: Inter-subject variability concentrations (L,).
        sigma: Intra-subject variability concentrations (L,).
        theta: Spatial prior on network labels (N x L).
        kappa: Session-level concentrations (L,).
        s_psi: Subject-level connectivity profiles (D x L x S).
        s_t_nu: Session-level connectivity profiles (D x L x T x S).
        s_lambda: Posterior label assignments (N x L x S).
        iter_inter: Number of inter-subject iterations completed.
        record: Cost function value at each inter-subject iteration.
    """

    mu: NDArray[np.floating]
    epsil: NDArray[np.floating]
    sigma: NDArray[np.floating]
    theta: NDArray[np.floating]
    kappa: NDArray[np.floating]
    s_psi: NDArray[np.floating] | None = None
    s_t_nu: NDArray[np.floating] | None = None
    s_lambda: NDArray[np.floating] | None = None
    iter_inter: int = 0
    record: list[float] = field(default_factory=list)


@dataclass
class DataBundle:
    """Container for fMRI time series data.

    Attributes:
        series: The time series array (T x N) where T=timepoints, N=vertices.
        num_vertices: Number of vertices/voxels.
        num_timepoints: Number of timepoints.
    """

    series: NDArray[np.floating]

    @property
    def num_timepoints(self) -> int:
        return self.series.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.series.shape[1]
