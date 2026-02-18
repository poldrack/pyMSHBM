"""Tests for profile list file reading/writing."""

import pytest

from pymshbm.io.profile_lists import read_profile_list, write_profile_list


def test_read_profile_list(tmp_path):
    """Read a profile list text file (one path per line)."""
    paths = ["/data/sub01/ses1.nii.gz", "/data/sub01/ses2.nii.gz"]
    list_file = tmp_path / "profiles.txt"
    list_file.write_text("\n".join(paths) + "\n")
    result = read_profile_list(list_file)
    assert result == paths


def test_read_profile_list_skips_blank_lines(tmp_path):
    """Skip blank lines and whitespace-only lines."""
    list_file = tmp_path / "profiles.txt"
    list_file.write_text("/path/a.mat\n\n  \n/path/b.mat\n")
    result = read_profile_list(list_file)
    assert result == ["/path/a.mat", "/path/b.mat"]


def test_write_profile_list(tmp_path):
    """Write paths to a profile list text file."""
    paths = ["/data/sub01.mat", "/data/sub02.mat"]
    out_file = tmp_path / "list.txt"
    write_profile_list(out_file, paths)
    content = out_file.read_text()
    assert content == "/data/sub01.mat\n/data/sub02.mat\n"


def test_read_profile_list_missing_file(tmp_path):
    """Raise FileNotFoundError for missing list file."""
    with pytest.raises(FileNotFoundError):
        read_profile_list(tmp_path / "missing.txt")


def test_roundtrip(tmp_path):
    """Write then read back yields same paths."""
    paths = ["/a/b/c.nii.gz", "/d/e/f.mat"]
    out_file = tmp_path / "rt.txt"
    write_profile_list(out_file, paths)
    result = read_profile_list(out_file)
    assert result == paths
