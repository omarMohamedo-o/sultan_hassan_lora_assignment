# Sultan Hassan Mosque FLUX LoRA Assessment Deliverable Package

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
