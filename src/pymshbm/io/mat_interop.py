"""Load and save .mat (v5 and v7.3) and .npz files."""

from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

from pymshbm.types import FileFormat, MSHBMParams


def _is_hdf5(path: Path) -> bool:
    """Check if a file is HDF5 format by reading its magic bytes."""
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def load_mat(path: str | Path) -> dict[str, np.ndarray]:
    """Load data from a .mat or .npz file, auto-detecting format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".npz":
        data = dict(np.load(str(path)))
        return data

    if _is_hdf5(path):
        result = {}
        with h5py.File(str(path), "r") as f:
            for key in f.keys():
                if key.startswith("#"):
                    continue
                result[key] = np.array(f[key])
        return result

    raw = sio.loadmat(str(path))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def save_mat(
    path: str | Path,
    data: dict[str, np.ndarray],
    fmt: FileFormat = FileFormat.MAT_V5,
) -> None:
    """Save data to a .mat or .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == FileFormat.MAT_V5:
        sio.savemat(str(path), data)
    elif fmt == FileFormat.MAT_V73:
        with h5py.File(str(path), "w") as f:
            for key, val in data.items():
                f.create_dataset(key, data=np.asarray(val))
    elif fmt == FileFormat.NPZ:
        np.savez(str(path), **data)
    elif fmt == FileFormat.AUTO:
        if path.suffix == ".npz":
            np.savez(str(path), **data)
        elif path.suffix == ".mat":
            sio.savemat(str(path), data)
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {path.suffix}")
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load_params_final(path: str | Path) -> MSHBMParams:
    """Load a Params_Final.mat file and return an MSHBMParams dataclass."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = load_mat(path)
    params_struct = raw["Params"]

    if params_struct.dtype.names is not None:
        p = params_struct[0, 0]
        return _struct_to_params(p)

    raise ValueError(f"Unexpected Params format in {path}")


def _struct_to_params(p) -> MSHBMParams:
    """Convert a MATLAB struct (numpy structured array element) to MSHBMParams."""
    mu = np.asarray(p["mu"], dtype=np.float64)
    epsil = np.asarray(p["epsil"], dtype=np.float64).ravel()
    sigma = np.asarray(p["sigma"], dtype=np.float64).ravel()
    kappa = np.asarray(p["kappa"], dtype=np.float64).ravel()
    theta = np.asarray(p["theta"], dtype=np.float64)

    s_psi = _load_optional_array(p, "s_psi")
    s_t_nu = _load_optional_array(p, "s_t_nu")
    s_lambda = _load_optional_array(p, "s_lambda")

    iter_inter = int(np.asarray(p["iter_inter"]).ravel()[0])
    record = np.asarray(p["Record"], dtype=np.float64).ravel().tolist()

    return MSHBMParams(
        mu=mu,
        epsil=epsil,
        sigma=sigma,
        theta=theta,
        kappa=kappa,
        s_psi=s_psi,
        s_t_nu=s_t_nu,
        s_lambda=s_lambda,
        iter_inter=iter_inter,
        record=record,
    )


def _load_optional_array(p, name: str) -> np.ndarray | None:
    """Load an array field, returning None if it's empty."""
    arr = np.asarray(p[name])
    if arr.size == 0:
        return None
    return arr.astype(np.float64)
