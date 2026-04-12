from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def save_pair_model_bundle(models_dir: Path, pair_id: str, bundle: dict[str, Any]) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{pair_id}.joblib"
    joblib.dump(bundle, path)
    return path
