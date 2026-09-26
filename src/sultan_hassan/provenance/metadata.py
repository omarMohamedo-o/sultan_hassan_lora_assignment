"""Provenance metadata tracking."""

from pathlib import Path

from sultan_hassan.provenance.hashes import compute_file_sha256


def get_provenance_info(file_path: Path | str) -> dict[str, str]:
    """Return provenance hash and status."""
    p = Path(file_path)
    sha = compute_file_sha256(p) if p.exists() else "missing"
    return {
        "filename": p.name,
        "sha256": sha,
        "source_type": "user_provided_video",
    }
