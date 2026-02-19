"""Extract individual parcellation results from MSHBM parameters.

Ports CBIG_IndCBM_extract_MSHBM_result_SUB.m.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

from pymshbm.types import MSHBMParams


def extract_mshbm_result(
    params: MSHBMParams,
    out_dir: str | Path | None = None,
    subject_ids: list[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract individual parcellations from s_lambda via argmax.

    Args:
        params: MSHBMParams with non-None s_lambda (N x L x S).
        out_dir: Optional output directory for .mat files.
        subject_ids: Optional list of subject ID strings for file naming.

    Returns:
        List of (lh_labels, rh_labels) tuples, one per subject.
    """
    if params.s_lambda is None:
        raise ValueError("s_lambda is empty. Use save_all flag when estimating group priors.")

    s_lambda = params.s_lambda
    N, L, S = s_lambda.shape
    vertex_num = N // 2

    results = []
    for i in range(S):
        sub = s_lambda[:, :, i]
        labels = np.argmax(sub, axis=1) + 1  # 1-indexed
        labels[sub.sum(axis=1) == 0] = 0  # Medial wall

        lh_labels = labels[:vertex_num]
        rh_labels = labels[vertex_num:]
        results.append((lh_labels, rh_labels))

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            sid = subject_ids[i] if subject_ids else str(i + 1)
            output_file = out_dir / f"Ind_parcellation_MSHBM_sub{i + 1}_{sid}.mat"
            save_dict = {
                "lh_labels": lh_labels.reshape(-1, 1),
                "rh_labels": rh_labels.reshape(-1, 1),
                "num_clusters": np.array(L),
            }
            sio.savemat(str(output_file), save_dict)

    return results


def extract_mshbm_result_sub(
    project_dir: str | Path,
    subject_ids: list[str],
    priors_file: str | Path | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract individual parcellations from a project directory.

    Plug-in replacement for CBIG_IndCBM_extract_MSHBM_result_SUB.m.
    Loads Params_Final.mat from project_dir/priors/, extracts per-subject
    parcellations via argmax on s_lambda, and saves .mat files to
    project_dir/ind_parcellation/.

    Args:
        project_dir: Project directory containing priors/Params_Final.mat.
        subject_ids: List of subject ID strings (one per subject in s_lambda).
        priors_file: Optional path to a custom priors file. If None, defaults to project_dir/priors/Params_Final.mat.
    Returns:
        List of (lh_labels, rh_labels) tuples, one per subject.
    """
    project_dir = Path(project_dir)
    params_path = Path(priors_file) if priors_file is not None else project_dir / "priors" / "Params_Final.mat"
    if not params_path.exists():
        raise FileNotFoundError(f"File not found: {params_path}")

    s_lambda = _load_s_lambda(params_path)

    N, L, S = s_lambda.shape

    if len(subject_ids) != S:
        raise ValueError(
            f"Number of subject IDs ({len(subject_ids)}) does not match "
            f"number of subjects in s_lambda ({S})."
        )

    colors = _load_group_colors(project_dir)

    output_dir = project_dir / "ind_parcellation"
    output_dir.mkdir(parents=True, exist_ok=True)

    vertex_num = N // 2
    results = []

    for i in range(S):
        sub = s_lambda[:, :, i]
        labels = np.argmax(sub, axis=1) + 1  # 1-indexed
        labels[sub.sum(axis=1) == 0] = 0  # Medial wall

        lh_labels = labels[:vertex_num]
        rh_labels = labels[vertex_num:]
        results.append((lh_labels, rh_labels))

        save_dict: dict[str, np.ndarray] = {
            "lh_labels": lh_labels.reshape(-1, 1),
            "rh_labels": rh_labels.reshape(-1, 1),
            "num_clusters": np.array(L),
        }
        if colors is not None:
            save_dict["colors"] = colors

        output_file = output_dir / (
            f"Ind_parcellation_MSHBM_sub{i + 1}_{subject_ids[i]}.mat"
        )
        sio.savemat(str(output_file), save_dict)

    return results


def _load_s_lambda(path: Path) -> np.ndarray:
    """Load s_lambda from a .mat file, handling multiple formats.

    Tries in order:
      1. Params struct wrapper: raw["Params"][0,0]["s_lambda"]
      2. Flat top-level key: raw["s_lambda"]
    """
    raw = sio.loadmat(str(path))
    keys = [k for k in raw if not k.startswith("_")]

    if "Params" in raw:
        p = raw["Params"]
        if hasattr(p.dtype, "names") and p.dtype.names is not None:
            s_lambda = np.asarray(p[0, 0]["s_lambda"], dtype=np.float64)
            if s_lambda.size == 0:
                raise ValueError(
                    "s_lambda is empty. Use save_all flag when estimating group priors."
                )
            return s_lambda

    if "s_lambda" in raw:
        s_lambda = np.asarray(raw["s_lambda"], dtype=np.float64)
        if s_lambda.size == 0:
            raise ValueError(
                "s_lambda is empty. Use save_all flag when estimating group priors."
            )
        return s_lambda

    raise ValueError(
        f"Cannot find s_lambda in {path.name}. "
        f"Expected a 'Params' struct or top-level 's_lambda' key. "
        f"Found keys: {keys}"
    )


def _load_group_colors(project_dir: Path) -> np.ndarray | None:
    """Load color table from project_dir/group/group.mat if present."""
    group_file = project_dir / "group" / "group.mat"
    if not group_file.exists():
        return None
    group = sio.loadmat(str(group_file))
    if "colors" in group:
        return np.asarray(group["colors"])
    return None
