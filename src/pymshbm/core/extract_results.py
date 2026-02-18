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
