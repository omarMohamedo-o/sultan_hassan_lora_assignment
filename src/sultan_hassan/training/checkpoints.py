"""Training configuration and checkpoint manager."""

from pathlib import Path


class CheckpointManager:
    """Manages saving, indexing, and evaluating LoRA checkpoints."""

    def __init__(self, checkpoints_dir: Path | str = "outputs/checkpoints") -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> list[Path]:
        """Return list of safetensors checkpoints."""
        return sorted(self.checkpoints_dir.glob("*.safetensors"))
