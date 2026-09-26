# Sultan Hassan Mosque-Madrasa SDXL LoRA

This repository contains the complete pipeline for curating, captioning, and training a high-quality Stable Diffusion XL (SDXL) LoRA for the historic Sultan Hassan Mosque-Madrasa in Cairo. 

It provides two distinct environments to execute the pipeline:
1. **Cloud Training (Google Colab)**: A lightweight, memory-efficient pipeline designed to run flawlessly on free T4 GPUs.
2. **Local Enterprise Training (Docker + MLflow)**: A robust local architecture using Docker Compose, MLflow tracking, and `uv` dependency management for users with dedicated high-VRAM hardware.

---

## 🚀 The Pipeline

### 1. Dataset Curation & Preprocessing
The dataset is carefully curated to encompass all iconic architectural elements of the mosque, including the exterior facades, the towering minarets, the mausoleum dome, the muqarnas entrance portal, and the interior iwans. 
- **Zero Contamination**: We strictly filter out any images containing the neighboring Al-Rifa'i Mosque to prevent architectural style bleeding.
- **Standardization**: All images are center-cropped and resized to precisely `1024x1024` pixels using Lanczos resampling.
- **Dynamic Captioning**: Instead of relying on a flat Dreambooth prompt, every image is paired with a highly detailed, distinct architectural description stored in a `metadata.jsonl` file.

### 2. Training Strategy
This pipeline utilizes the official HuggingFace `diffusers` text-to-image training script for maximum quality.
- **Trigger Word**: `sltnhsn` (automatically prepended to all captions).
- **Network Size**: `--rank=32` ensures the network has enough capacity to memorize complex Mamluk geometric patterns.
- **Optimization**: Uses `--mixed_precision="fp16"` and `xformers` memory-efficient attention.
- **Resilience**: Configured for `1500` maximum training steps with checkpoints saved every `100` steps to safely recover from cloud preemption.

---

## 🐳 Why Docker?
For local execution, this project relies heavily on **Docker Compose** (`docker-compose.train.yml`) and a custom `Dockerfile.train`.

- **Absolute Reproducibility**: By containerizing the environment, we lock down PyTorch 2.3.0, CUDA 12.1, and all dependencies. It eliminates the "it works on my machine" problem entirely.
- **System Isolation**: Massive deep learning libraries and CUDA toolkits are kept strictly inside the container, keeping your host operating system clean.
- **Integrated MLflow**: The Docker compose configuration automatically spins up an MLflow tracking server on `localhost:5000` alongside the training container. This seamlessly logs hyperparameters, loss curves, and artifact weights in real time to a local `mlruns.db` database.

---

## 💻 Usage Instructions

### Method A: Google Colab (Recommended for Free GPUs)
If you do not have a 16GB+ VRAM Nvidia GPU on your local machine, use this method.
1. Upload `notebooks/sultan_hassan_master.ipynb` to Google Colab.
2. Set the runtime to **T4 GPU**.
3. Upload your raw image zip (`data_final.zip`) when prompted.
4. Run all cells. The notebook automatically handles captioning, metadata generation, training, and testing.

### Method B: Local Docker Environment (Requires RTX 3090/4090)
1. Ensure Docker Desktop and WSL2 with Nvidia GPU passthrough are installed and configured.
2. Place your curated images in the `data/final/` directory.
3. Launch the automated training and monitoring pipeline:
   ```bash
   docker compose -f docker-compose.train.yml up --build
   ```
4. Navigate to `http://localhost:5000` to monitor your loss metrics live in the MLflow UI.
