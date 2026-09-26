"""Unit tests for SHA-256 and perceptual hashing."""

from pathlib import Path

from PIL import Image

from sultan_hassan.provenance.hashes import (
    calculate_hamming_distance,
    compute_file_sha256,
    compute_image_phash,
)


def test_sha256_computation(sample_image: Path) -> None:
    h1 = compute_file_sha256(sample_image)
    h2 = compute_file_sha256(sample_image)
    assert len(h1) == 64
    assert h1 == h2


def test_phash_and_hamming_distance(sample_image: Path, tmp_path: Path) -> None:
    # Compute hash of original image
    phash1 = compute_image_phash(sample_image)
    assert len(phash1) > 0

    # Slightly modified image copy (near-duplicate)
    copy_path = tmp_path / "copy.jpg"
    with Image.open(sample_image) as img:
        img = img.resize((1020, 1020)).resize((1024, 1024))
        img.save(copy_path, "JPEG")

    phash2 = compute_image_phash(copy_path)
    dist = calculate_hamming_distance(phash1, phash2)
    assert dist <= 6  # Highly similar frames have small distance
