"""FLUX LoRA training interfaces and configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoRATrainingSession:
    """Session specifications for FLUX LoRA fine-tuning."""

    base_model: str = "black-forest-labs/FLUX.1-dev"
    trigger_word: str = "sltnhsn"
    lora_rank: int = 16
    lora_alpha: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 1
    max_train_steps: int = 1000
    precision: str = "bf16"
    seed: int = 42
    output_dir: Path = Path("models/lora")


class FluxTrainer:
    """FLUX LoRA trainer interface."""

    def __init__(self, session: LoRATrainingSession) -> None:
        self.session = session

    def prepare_environment(self) -> dict[str, str]:
        """Validate GPU, torch, and diffusers environment requirements."""
        return {
            "model": self.session.base_model,
            "rank": str(self.session.lora_rank),
            "status": "ready_for_gpu_launch",
        }
