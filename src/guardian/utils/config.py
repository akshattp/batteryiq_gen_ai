"""Configuration loading and validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

_DEFAULT_CONFIG_PATH = Path("config/config.yaml")
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load and parse the project configuration YAML file.

    Args:
        config_path: Path to the configuration YAML. Defaults to config/config.yaml.

    Returns:
        Dict containing all parsed configuration parameters.
    """
    global _CONFIG_CACHE
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        # Fallback to finding config.yaml relative to current file
        candidate = Path(__file__).resolve().parents[3] / "config" / "config.yaml"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Config file not found at {path} or {candidate}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    _CONFIG_CACHE = config
    return config


def get_config() -> Dict[str, Any]:
    """Retrieve the cached configuration, or load it if not loaded yet."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        return load_config()
    return _CONFIG_CACHE
