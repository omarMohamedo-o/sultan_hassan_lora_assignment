"""Hashing utilities for data integrity, provenance, and perceptual deduplication."""

import hashlib
from pathlib import Path

import imagehash
from PIL import Image


def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hexadecimal hash of a file using streaming chunks.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte size of chunks read into memory.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    path = Path(file_path)
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_image_phash(image: Image.Image | Path | str, hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) of an image.

    Args:
        image: PIL Image or path to image file.
        hash_size: Size of the perceptual hash matrix.

    Returns:
        Hexadecimal pHash string.
    """
    if not isinstance(image, Image.Image):
        with Image.open(image) as img:
            return str(imagehash.phash(img, hash_size=hash_size))
    return str(imagehash.phash(image, hash_size=hash_size))


def compute_image_dhash(image: Image.Image | Path | str, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) of an image.

    Args:
        image: PIL Image or path to image file.
        hash_size: Size of the difference hash matrix.

    Returns:
        Hexadecimal dHash string.
    """
    if not isinstance(image, Image.Image):
        with Image.open(image) as img:
            return str(imagehash.dhash(img, hash_size=hash_size))
    return str(imagehash.dhash(image, hash_size=hash_size))


def calculate_hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate the Hamming distance between two hexadecimal perceptual hashes.

    Args:
        hash1: First hash string.
        hash2: Second hash string.

    Returns:
        Number of differing bits.
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return int(h1 - h2)
