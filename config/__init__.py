"""Configuration loader."""
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config as a dict."""
    config_path = Path(path) if path else _CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
