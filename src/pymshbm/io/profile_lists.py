"""Parse and write profile list text files (one path per line)."""

from pathlib import Path


def read_profile_list(path: str | Path) -> list[str]:
    """Read a profile list file, returning non-empty lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def write_profile_list(path: str | Path, paths: list[str]) -> None:
    """Write a list of paths to a profile list file, one per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(paths) + "\n")
