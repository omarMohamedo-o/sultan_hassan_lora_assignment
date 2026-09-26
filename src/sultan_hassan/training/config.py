"""Training configuration schema."""

from pydantic import BaseModel, Field


class TrainingHyperparameters(BaseModel):
    """Explicit hyperparameters for reproducible FLUX training."""

    base_model: str = "black-forest-labs/FLUX.1-dev"
    trigger_word: str = "sltnhsn"
    lora_rank: int = Field(default=16, ge=4, le=64)
    lora_alpha: int = Field(default=16, ge=4, le=64)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    epochs: int = Field(default=10, ge=1)
    max_train_steps: int = Field(default=1000, ge=50)
    seed: int = 42
    precision: str = "bf16"
    checkpoint_frequency: int = 250
