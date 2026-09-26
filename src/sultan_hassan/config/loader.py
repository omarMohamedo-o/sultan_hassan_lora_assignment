"""Configuration loader for YAML configuration files with Pydantic validation."""

from pathlib import Path
from typing import Any

import yaml

from sultan_hassan.config.models import AppConfig
from sultan_hassan.domain.exceptions import ConfigurationError


def deep_merge(source: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge source dictionary into destination dictionary."""
    result = destination.copy()
    for key, value in source.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(value, result[key])
        else:
            result[key] = value
    return result


def load_config(
    config_path: Path | str = "config/config.yaml",
    overlay_path: Path | str | None = None,
) -> AppConfig:
    """Load, merge, and validate YAML configurations into AppConfig.

    Args:
        config_path: Primary configuration file path.
        overlay_path: Optional environment overlay (e.g., config/development.yaml).

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigurationError: If file not found, YAML syntax is invalid, or schema validation fails.
    """
    primary = Path(config_path)
    if not primary.exists():
        raise ConfigurationError(f"Primary configuration file not found at: {primary}")

    try:
        with open(primary, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        raise ConfigurationError(f"Failed to parse YAML from {primary}: {exc}") from exc

    if overlay_path:
        overlay = Path(overlay_path)
        if overlay.exists():
            try:
                with open(overlay, encoding="utf-8") as f:
                    overlay_data = yaml.safe_load(f) or {}
                    data = deep_merge(overlay_data, data)
            except Exception as exc:
                raise ConfigurationError(
                    f"Failed to parse overlay YAML from {overlay}: {exc}"
                ) from exc

    try:
        return AppConfig.model_validate(data)
    except Exception as exc:
        raise ConfigurationError(f"Configuration validation failed: {exc}") from exc
