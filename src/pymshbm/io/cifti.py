"""Write parcellation labels as CIFTI dlabel files."""

from pathlib import Path

import nibabel as nib
import numpy as np


def write_dlabel_cifti(
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
    output_path: str | Path,
    num_vertices_lh: int,
    num_vertices_rh: int,
    label_names: list[str] | None = None,
    colors: np.ndarray | None = None,
) -> None:
    """Write parcellation labels as a CIFTI .dlabel.nii file.

    Args:
        lh_labels: (V_lh,) int labels, 0 = medial wall.
        rh_labels: (V_rh,) int labels.
        output_path: Path for the output .dlabel.nii file.
        num_vertices_lh: Total lh surface vertices.
        num_vertices_rh: Total rh surface vertices.
        label_names: Optional names for each cluster (length K).
        colors: Optional (K, 4) RGBA uint8 color table.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bm_lh = nib.cifti2.BrainModelAxis.from_surface(
        np.arange(len(lh_labels)), num_vertices_lh,
        "CIFTI_STRUCTURE_CORTEX_LEFT",
    )
    bm_rh = nib.cifti2.BrainModelAxis.from_surface(
        np.arange(len(rh_labels)), num_vertices_rh,
        "CIFTI_STRUCTURE_CORTEX_RIGHT",
    )
    bm_axis = bm_lh + bm_rh

    all_labels = np.concatenate([lh_labels, rh_labels])
    num_clusters = int(all_labels.max())

    label_dict = _build_label_dict(num_clusters, label_names, colors)
    label_axis = nib.cifti2.LabelAxis(["parcellation"], [label_dict])

    data = all_labels.reshape(1, -1).astype(np.float32)
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, bm_axis))
    img = nib.Cifti2Image(data, header)
    img.to_filename(str(output_path))


def _build_label_dict(
    num_clusters: int,
    label_names: list[str] | None,
    colors: np.ndarray | None,
) -> dict[int, tuple[str, tuple[float, float, float, float]]]:
    """Build the label dictionary for CIFTI LabelAxis."""
    label_dict: dict[int, tuple[str, tuple[float, float, float, float]]] = {
        0: ("???", (0.0, 0.0, 0.0, 0.0)),
    }
    for k in range(1, num_clusters + 1):
        name = label_names[k - 1] if label_names else f"Network_{k}"
        if colors is not None:
            rgba = tuple(float(c) / 255.0 for c in colors[k - 1])
        else:
            rgba = _default_color(k, num_clusters)
        label_dict[k] = (name, rgba)
    return label_dict


def _default_color(
    k: int, num_clusters: int,
) -> tuple[float, float, float, float]:
    """Generate a default color for cluster k using HSV rotation."""
    hue = (k - 1) / max(num_clusters, 1)
    r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
    return (r, g, b, 1.0)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV to RGB (all values 0-1)."""
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)
