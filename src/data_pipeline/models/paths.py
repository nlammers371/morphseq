"""Config-to-path resolution for model assets under a single models root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GroundingDinoPaths:
    repo_dir: Path
    config_path: Path
    weights_path: Path
    model_release: str
    model_name: str


@dataclass(frozen=True)
class Sam2Paths:
    models_root: Path
    config_path: Path
    checkpoint_path: Path
    model_release: str
    model_name: str


@dataclass(frozen=True)
class SegmentationAndTrackingModelPaths:
    models_root: Path
    groundingdino: GroundingDinoPaths
    sam2: Sam2Paths


def _as_str(d: Mapping[str, Any], key: str, default: str = "") -> str:
    val = d.get(key, default)
    if val is None:
        return default
    text = str(val).strip()
    return default if text == "" else text


def resolve_models_root(*, config: Mapping[str, Any], data_root: Path) -> Path:
    """
    Resolve the single models root for the pipeline.

    Config shape:
      segmentation_and_tracking:
        models_root: "models"  # relative to data_root unless absolute
    """
    raw = (config.get("segmentation_and_tracking") or {})
    models_root = Path(_as_str(raw, "models_root", "models"))
    return models_root if models_root.is_absolute() else (Path(data_root) / models_root)


def resolve_segmentation_and_tracking_model_paths(
    *,
    config: Mapping[str, Any],
    data_root: Path,
) -> SegmentationAndTrackingModelPaths:
    root = resolve_models_root(config=config, data_root=data_root)
    raw = (config.get("segmentation_and_tracking") or {})

    gd_repo = Path(_as_str(raw, "groundingdino_repo", "GroundingDINO"))
    gd_repo = gd_repo if gd_repo.is_absolute() else (root / gd_repo)

    gd_cfg = Path(_as_str(raw, "groundingdino_config", "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    gd_cfg = gd_cfg if gd_cfg.is_absolute() else (gd_repo / gd_cfg)

    gd_wts = Path(_as_str(raw, "groundingdino_weights", "weights/groundingdino_swint_ogc.pth"))
    gd_wts = gd_wts if gd_wts.is_absolute() else (gd_repo / gd_wts)
    if not gd_wts.exists():
        alt = gd_repo / "weights" / gd_wts.name
        if alt.exists():
            gd_wts = alt

    sam2_root = Path(_as_str(raw, "sam2_models_root", "sam2"))
    sam2_root = sam2_root if sam2_root.is_absolute() else (root / sam2_root)

    sam2_cfg = Path(_as_str(raw, "sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml"))
    # Relative resolution is handled by sam2 loader; keep as possibly-relative Path here.

    sam2_ckpt = Path(_as_str(raw, "sam2_checkpoint", "checkpoints/sam2.1_hiera_large.pt"))

    return SegmentationAndTrackingModelPaths(
        models_root=root,
        groundingdino=GroundingDinoPaths(
            repo_dir=gd_repo,
            config_path=gd_cfg,
            weights_path=gd_wts,
            model_release=_as_str(raw, "groundingdino_release", "unknown") or "unknown",
            model_name=_as_str(raw, "groundingdino_model_name", "SwinT_OGC") or "SwinT_OGC",
        ),
        sam2=Sam2Paths(
            models_root=sam2_root,
            config_path=sam2_cfg,
            checkpoint_path=sam2_ckpt,
            model_release=_as_str(raw, "sam2_release", "unknown") or "unknown",
            model_name=_as_str(raw, "sam2_model_name", "hiera_large") or "hiera_large",
        ),
    )

