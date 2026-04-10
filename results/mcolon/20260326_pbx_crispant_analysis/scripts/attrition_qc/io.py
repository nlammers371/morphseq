from __future__ import annotations

from pathlib import Path

from phenotypic_positioning.io import save_manifest


def ensure_output_dirs(results_dir: Path, figures_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)


__all__ = ["ensure_output_dirs", "save_manifest"]

