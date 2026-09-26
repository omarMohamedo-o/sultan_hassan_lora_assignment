"""Perceptual image hashing operations."""

from pathlib import Path

from PIL import Image

from sultan_hassan.provenance.hashes import (
    calculate_hamming_distance,
    compute_image_dhash,
    compute_image_phash,
)


def get_image_hash(
    image_path: Path | str,
    method: str = "phash",
    hash_size: int = 8,
) -> str:
    """Compute perceptual hash using specified algorithm (phash or dhash)."""
    with Image.open(image_path) as img:
        if method.lower() == "dhash":
            return compute_image_dhash(img, hash_size=hash_size)
        return compute_image_phash(img, hash_size=hash_size)


__all__ = [
    "get_image_hash",
    "calculate_hamming_distance",
    "compute_image_phash",
    "compute_image_dhash",
]
