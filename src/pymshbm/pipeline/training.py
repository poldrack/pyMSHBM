"""Full training pipeline wrapper.

Ports MSHBM_Params_Training.m — runs group prior estimation
and extracts individual parcellations.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

from pymshbm.core.extract_results import extract_mshbm_result
from pymshbm.core.group_priors import estimate_group_priors
from pymshbm.types import MSHBMParams


def params_training(
    data: np.ndarray,
    g_mu: np.ndarray,
    num_clusters: int,
    max_iter: int = 50,
    conv_th: float = 1e-5,
    output_dir: str | Path | None = None,
    subject_ids: list[str] | None = None,
    save_all: bool = False,
) -> MSHBMParams:
    """Full training pipeline: estimate group priors and extract parcellations.

    Args:
        data: (N, D, S, T) normalized FC profiles.
        g_mu: (D, L) group-level cluster centroids.
        num_clusters: Number of networks.
        max_iter: Maximum EM iterations.
        conv_th: Convergence threshold.
        output_dir: Optional directory for saving outputs.
        subject_ids: Optional list of subject ID strings.
        save_all: If True, keep s_lambda, s_psi, s_t_nu in output.

    Returns:
        MSHBMParams with estimated group priors.
    """
    N, D, S, T = data.shape
    dim = D - 1

    if dim < 1200:
        ini_concentration = 500
    elif dim < 1800:
        ini_concentration = 650
    else:
        raise ValueError(f"Dimension {dim} too high")

    settings = {
        "num_sub": S,
        "num_session": T,
        "num_clusters": num_clusters,
        "dim": dim,
        "ini_concentration": ini_concentration,
        "epsilon": 1e-4,
        "conv_th": conv_th,
        "max_iter": max_iter,
    }

    params = estimate_group_priors(data=data, g_mu=g_mu, settings=settings)

    if not save_all:
        params.s_psi = None
        params.s_t_nu = None
        params.s_lambda = None

    if output_dir is not None:
        output_dir = Path(output_dir)
        _save_params(params, output_dir)

        if save_all and params.s_lambda is not None:
            ind_dir = output_dir / "ind_parcellation"
            extract_mshbm_result(
                params,
                out_dir=ind_dir,
                subject_ids=subject_ids,
            )

    return params


def _save_params(params: MSHBMParams, output_dir: Path) -> None:
    """Save Params_Final.mat."""
    priors_dir = output_dir / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "mu": params.mu,
        "epsil": params.epsil.reshape(1, -1),
        "sigma": params.sigma.reshape(1, -1),
        "kappa": params.kappa.reshape(1, -1),
        "theta": params.theta,
        "iter_inter": np.array([[params.iter_inter]]),
        "Record": np.array([params.record]),
    }

    if params.s_psi is not None:
        save_dict["s_psi"] = params.s_psi
    if params.s_t_nu is not None:
        save_dict["s_t_nu"] = params.s_t_nu
    if params.s_lambda is not None:
        save_dict["s_lambda"] = params.s_lambda

    sio.savemat(str(priors_dir / "Params_Final.mat"), {"Params": save_dict})
