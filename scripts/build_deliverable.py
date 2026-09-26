"""Script to compile deliverable package."""

import shutil
from pathlib import Path

from sultan_hassan.config.loader import load_config
from sultan_hassan.utils import configure_utf8_streams


def main() -> None:
    configure_utf8_streams()

    config = load_config()
    deliv_dir = Path(config.paths.deliverable)
    deliv_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure subdirectories
    subdirs = [
        "dataset/images",
        "dataset/captions",
        "dataset/splits/train",
        "dataset/splits/val",
        "dataset/splits/test",
        "metadata",
        "reports",
        "training",
        "tests",
    ]
    for sub in subdirs:
        (deliv_dir / sub).mkdir(parents=True, exist_ok=True)

    # 2. Copy final dataset images and captions
    final_imgs = list(Path(config.paths.final_images).glob("*.jpg"))
    for img in final_imgs:
        shutil.copy(img, deliv_dir / "dataset/images" / img.name)

    final_caps = list(Path(config.paths.final_captions).glob("*.txt"))
    for cap in final_caps:
        shutil.copy(cap, deliv_dir / "dataset/captions" / cap.name)

    # 3. Copy splits if they exist
    splits_base = Path(config.paths.dataset_splits)
    if splits_base.exists():
        for split_name in ["train", "val", "test"]:
            split_dir = splits_base / split_name
            if split_dir.exists():
                for sf in split_dir.glob("*.jpg"):
                    shutil.copy(sf, deliv_dir / f"dataset/splits/{split_name}" / sf.name)

    # 4. Copy metadata
    for meta_file in ["final_dataset.csv", "test_metadata.csv", "quality.csv"]:
        for src_dir in [Path(config.paths.metadata_dir), Path("outputs/tests")]:
            src_f = src_dir / meta_file
            if src_f.exists():
                shutil.copy(src_f, deliv_dir / "metadata" / meta_file)

    # 5. Copy generated test images and gallery
    tests_src = Path(config.paths.outputs_tests)
    if tests_src.exists():
        for tf in tests_src.glob("*.jpg"):
            shutil.copy(tf, deliv_dir / "tests" / tf.name)
        if (tests_src / "index.html").exists():
            shutil.copy(tests_src / "index.html", deliv_dir / "reports/test_gallery.html")

    # 6. Copy reports
    reports_sources = [
        Path("reports/dataset/dataset_report.html"),
        Path("reports/dataset/dataset_report.json"),
        Path("outputs/evaluation/evaluation_report.html"),
        Path("outputs/evaluation/evaluation_report.json"),
        Path("reports/contact_sheets/candidates_contact_sheet.jpg"),
        Path("reports/contact_sheets/index.html"),
    ]
    for rp in reports_sources:
        if rp.exists():
            dest_name = (
                "contact_sheet.jpg"
                if rp.name == "candidates_contact_sheet.jpg"
                else ("contact_sheet.html" if rp.name == "index.html" else rp.name)
            )
            shutil.copy(rp, deliv_dir / "reports" / dest_name)

    # 7. Write comprehensive README
    readme_content = """# Sultan Hassan Mosque FLUX LoRA Assessment Deliverable Package

Production-grade Machine Learning Engineering deliverable for Sultan Hassan Mosque (excluding the neighbouring Al-Rifa'i Mosque).

## Package Contents

- **`dataset/`**:
  - `images/`: 40 curated, high-resolution (>=1024px) RGB JPEG images.
  - `captions/`: Truthful descriptive captions with trigger word `sltnhsn`.
  - `splits/`: Configurable partitions (`train` 80%, `val` 10%, `test` 10%).
- **`metadata/`**:
  - `final_dataset.csv`: Cryptographic hashes (SHA-256), perceptual hashes (pHash/dHash), review state, and provenance.
  - `test_metadata.csv`: 20 test sample specifications across 5 benchmark prompt groups and 4 fixed seeds.
- **`training/`**:
  - `training_config.yaml`: Hyperparameters (Rank 16, Alpha 16, lr=1e-4, batch=1, AdamW8bit).
  - `training_command.txt`: Reproducible command for `accelerate launch train_flux_lora.py`.
- **`tests/`**:
  - 20 benchmark visual test images covering Elevation, Entrance Portal, Courtyard Sunset, Night Atmosphere, and Modern Glass Negative Control.
- **`reports/`**:
  - `evaluation_report.html` & `.json`: Architectural fidelity and style-bleeding analysis.
  - `test_gallery.html`: Responsive dark-mode test gallery.
  - `dataset_report.html` & `.json`: Dataset curation validation report.
  - `contact_sheet.html` & `.jpg`: Visual contact sheets.
"""
    (deliv_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"[SUCCESS] Deliverable directory compiled successfully in: {deliv_dir}")


if __name__ == "__main__":
    main()
