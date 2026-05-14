from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

# Optional local overrides for NOVALM_* paths (see .env.example). Does not override existing env vars.
load_dotenv(ROOT / ".env")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_configs() -> dict[str, dict[str, Any]]:
    retrieval_path = CONFIG_DIR / "retrieval.yaml"
    retrieval = _read_yaml(retrieval_path) if retrieval_path.exists() else {}
    return {
        "model": _read_yaml(CONFIG_DIR / "model.yaml"),
        "train": _read_yaml(CONFIG_DIR / "train.yaml"),
        "infer": _read_yaml(CONFIG_DIR / "infer.yaml"),
        "data": _read_yaml(CONFIG_DIR / "data.yaml"),
        "retrieval": retrieval,
    }


def get_path_from_env_or_default(env_name: str, default_rel: str) -> Path:
    value = os.getenv(env_name)
    return Path(value) if value else ROOT / default_rel
