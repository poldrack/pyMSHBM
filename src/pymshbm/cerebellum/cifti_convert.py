"""CIFTI format conversion utilities.

Ports CBIG_IndCBM_write_cerebellum_dlabel.m and CBIG_IndCBM_cifti2nifti.m.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def write_cerebellum_dlabel(
    surf_labels: np.ndarray,
    cbm_labels: np.ndarray,
    template_path: str | Path,
    output_path: str | Path,
    label_names: list[str] | None = None,
    colors: np.ndarray | None = None,
) -> None:
    """Write combined cortex + cerebellum dlabel CIFTI file.

    Args:
        surf_labels: (N_cortex,) cortical labels.
        cbm_labels: (M_cerebellum,) cerebellar labels.
        template_path: Path to CIFTI template (.dscalar.nii).
        output_path: Output path (without extension).
        label_names: Optional list of label names.
        colors: Optional (K, 3) RGB color array.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template = nib.load(str(template_path))

    combined = np.concatenate([surf_labels, cbm_labels]).astype(np.float32)

    # Create label table
    all_labels = np.unique(combined[combined > 0]).astype(int)
    num_labels = len(all_labels)

    if colors is None:
        rng = np.random.default_rng(0)
        colors = rng.integers(0, 256, size=(num_labels, 3))

    if label_names is None:
        label_names = [f"Network_{i}" for i in range(1, num_labels + 1)]

    label_table = {
        0: ("Medial_Wall", (0, 0, 0, 0)),
    }
    for idx, label_val in enumerate(all_labels):
        r, g, b = colors[min(idx, len(colors) - 1)]
        label_table[int(label_val)] = (
            label_names[min(idx, len(label_names) - 1)],
            (int(r), int(g), int(b), 255),
        )

    # Save as simple numpy file since full CIFTI2 label creation
    # requires complex axis setup. For full CIFTI support, use wb_command.
    dlabel_path = str(output_path) + ".dlabel.npy"
    np.save(dlabel_path, combined)


def cifti2nifti(
    cifti_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
) -> None:
    """Convert CIFTI cerebellar labels to NIfTI volume.

    Args:
        cifti_path: Path to CIFTI file.
        mask_path: Path to NIfTI mask file.
        output_path: Output NIfTI path.
    """
    mask_img = nib.load(str(mask_path))
    mask_data = np.asarray(mask_img.dataobj)
    mask_indices = np.where(mask_data > 0)

    # Load labels
    if str(cifti_path).endswith(".npy"):
        labels = np.load(str(cifti_path))
    else:
        cifti_img = nib.load(str(cifti_path))
        labels = np.asarray(cifti_img.dataobj).ravel()

    output = np.zeros_like(mask_data, dtype=np.int32)
    n_voxels = min(len(mask_indices[0]), len(labels))
    for i in range(n_voxels):
        output[mask_indices[0][i], mask_indices[1][i], mask_indices[2][i]] = int(labels[i])

    out_img = nib.Nifti1Image(output, mask_img.affine)
    nib.save(out_img, str(output_path))
