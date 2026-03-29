from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RERUN_ID = "rerun_20260329_no_yolk_removed"
EXPERIMENT_IDS = ["20260304", "20260306"]
EXPERIMENT_LABEL = "_".join(EXPERIMENT_IDS)


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "morphseq_playground").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate morphseq repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
BUILD06_DIR = REPO_ROOT / "morphseq_playground" / "metadata" / "build06_output"


def resolve_output_roots(
    *,
    results_subdir: str | None = None,
    figures_subdir: str | None = None,
) -> tuple[Path, Path]:
    results_name = results_subdir or RERUN_ID
    figures_name = figures_subdir or results_name
    return ANALYSIS_ROOT / "results" / results_name, ANALYSIS_ROOT / "figures" / figures_name


def resolve_bin_width_roots(
    *,
    bin_width: float,
    results_subdir: str | None = None,
    figures_subdir: str | None = None,
) -> tuple[Path, Path]:
    results_root, figures_root = resolve_output_roots(
        results_subdir=results_subdir,
        figures_subdir=figures_subdir,
    )
    stem = f"bin_width_{bin_width:.1f}hpf"
    return results_root / stem, figures_root / stem


def resolve_embedding_roots(
    *,
    results_subdir: str | None = None,
    figures_subdir: str | None = None,
) -> tuple[Path, Path]:
    results_root, figures_root = resolve_output_roots(
        results_subdir=results_subdir,
        figures_subdir=figures_subdir,
    )
    return results_root / "misclassification" / "embedding", figures_root / "misclassification" / "embedding"


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("rerun_id", RERUN_ID)
    payload.setdefault("experiment_ids", EXPERIMENT_IDS)
    payload.setdefault("experiment_label", EXPERIMENT_LABEL)
    payload.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
