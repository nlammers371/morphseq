#!/usr/bin/env python3
"""
Dry-run tester for ExperimentManager "run up to step" orchestration.

This script exercises ExperimentManager + Experiment needs/exists logic and
simulates running the pipeline up to a target step without executing heavy work.

Usage:
  python -m src.run_morphseq_pipeline.services.experiment_planner \
    --data-root /path/to/morphseq_playground \
    --exp 20250529_36hpf_ctrl_atf6 20250612_24hpf_wfs1_ctcf \
    --model-name 20241107_ds_sweep01_optimum

Target steps (in order): build01, sam2, build03, latents, build04, build06

Notes:
  - This is a read-only checker; it does not run the heavy steps. It prints
    what would run based on `needs_*` and file existence.
  - For an end-to-end dry-run that mirrors CLI output, prefer
    `python -m src.run_morphseq_pipeline.cli pipeline --action e2e --dry-run`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.build.pipeline_objects import ExperimentManager
from src.run_morphseq_pipeline.paths import get_well_metadata_xlsx
from src.run_morphseq_pipeline.run_experiment_manager import process_experiments_through_pipeline


# Processing order for planning (overall)
ORDER = ["build01", "sam2", "build03", "build04", "latents", "build06"]


def _exists(p: Path | None) -> bool:
    try:
        return bool(p and Path(p).exists())
    except Exception:
        return False

def _dir_nonempty(p: Path | None) -> bool:
    try:
        if not p or not Path(p).is_dir():
            return False
        it = Path(p).iterdir()
        next(it)
        return True
    except StopIteration:
        return False
    except Exception:
        return False


def summarize_exp(exp, model_name: str, root: Path) -> dict:
    """Collect quick status for an experiment, robust to missing bits."""
    def safe(callable_, fallback=None):
        try:
            return callable_()
        except Exception:
            return fallback

    qc_present, qc_total = safe(exp.qc_mask_status, (0, 5))
    b04_exists = safe(lambda: exp.build04_path.exists(), False)
    b06_exists = safe(lambda: exp.build06_path.exists(), False)
    try:
        well_meta_exists = get_well_metadata_xlsx(root, exp.date).exists()
    except Exception:
        well_meta_exists = False
    return {
        # Image generation: stitched FF and stitched Z only
        "ff": _exists(exp.stitch_ff_path),
        "ff_z": _exists(exp.stitch_z_path),
        "qc": (qc_present == qc_total and qc_total > 0),
        "qc_present": qc_present,
        "qc_total": qc_total,
        # SAM2 intermediates (explicit, all important)
        "sam2_csv": _exists(exp.sam2_csv_path),
        "sam2_masks_ok": _dir_nonempty(getattr(exp, "sam2_masks_dir", None)),
        "gdino_json": _exists(getattr(exp, "gdino_detections_path", None)),
        "seg_json": _exists(getattr(exp, "sam2_segmentations_path", None)),
        "mask_manifest": _exists(getattr(exp, "sam2_mask_export_manifest", None)),
        "exp_meta_json": _exists(getattr(exp, "sam2_experiment_metadata_json", None)),
        "in_df01": safe(exp.is_in_df01, False),
        "df02": b04_exists,
        "df03": b06_exists,
        "needs_build03": safe(lambda: exp.needs_build03, False),
        "has_latents": safe(lambda: exp.has_latents(model_name), False),
        "b03_path": exp.build03_path,
        "b04_path": exp.build04_path,
        "b06_path": exp.build06_path,
        "latents_path": exp.get_latent_path(model_name),
        "well_meta": well_meta_exists,
    }


def would_run_up_to(exp, manager: ExperimentManager, target: str, model_name: str) -> List[str]:
    """Return the list of steps that would be executed to reach `target` (excluding the target)."""
    steps: List[str] = []
    target_idx = ORDER.index(target)

    # Build01 (per-experiment)
    if target_idx > ORDER.index("build01"):
        need_b01 = not (exp.flags.get("ff", False) or _exists(exp.ff_path))
        if need_b01:
            steps.append("build01")

    # SAM2 (per-experiment)
    if target_idx > ORDER.index("sam2"):
        need_sam2 = False
        try:
            # Missing-any simple rule over intermediates
            csv_exists = getattr(exp, "sam2_csv_path", None)
            csv_ok = bool(csv_exists and csv_exists.exists())
            masks_dir = getattr(exp, "sam2_masks_dir", None)
            masks_ok = _dir_nonempty(masks_dir)
            gdino_ok = _exists(getattr(exp, "gdino_detections_path", None))
            seg_ok = _exists(getattr(exp, "sam2_segmentations_path", None))
            manifest_ok = _exists(getattr(exp, "sam2_mask_export_manifest", None))
            expmeta_ok = _exists(getattr(exp, "sam2_experiment_metadata_json", None))

            missing_any = not (csv_ok and masks_ok and gdino_ok and seg_ok and manifest_ok and expmeta_ok)

            # Also include stale CSV logic encapsulated in exp.needs_sam2
            need_sam2 = missing_any or bool(exp.needs_sam2)
        except Exception:
            need_sam2 = False
        if need_sam2:
            steps.append("sam2")

    # Build03/Build04/Latents (per-experiment) — enforce order: build03 -> build04 -> latents
    need_b03 = False
    if target_idx > ORDER.index("build03"):
        try:
            need_b03 = bool(exp.needs_build03)
        except Exception:
            need_b03 = False
        if need_b03:
            steps.append("build03")

    need_b04 = False
    if target_idx > ORDER.index("build04"):
        try:
            b03 = exp.build03_path
            b04 = exp.build04_path
            need_b04 = (not b04.exists()) or (b03.exists() and b03.stat().st_mtime > b04.stat().st_mtime)
        except Exception:
            need_b04 = False
        if need_b04:
            steps.append("build04")

    need_lat = False
    if target_idx > ORDER.index("latents"):
        try:
            need_lat = not exp.has_latents(model_name)
        except Exception:
            need_lat = False
        if need_lat:
            steps.append("latents")

    # Build06 (per-experiment freshness) — last
    if target_idx > ORDER.index("build06"):
        try:
            need_b06 = exp.needs_build06_per_experiment(model_name)
        except Exception:
            need_b06 = False
        if need_b06:
            steps.append("build06")

    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description="Test ExperimentManager up-to-step dry-run")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--exp", nargs="+", required=True)
    # No execution in this tool; always a dry-run plan to build06
    ap.add_argument("--model-name", default="20241107_ds_sweep01_optimum")
    args = ap.parse_args()

    root = Path(args.data_root)
    manager = ExperimentManager(root)

    print(f"Root: {root}")
    print("Dry-run plan (central manager)")
    print("Hint: Use CLI pipeline with --dry-run for the primary interface.")

    missing = [e for e in args.exp if e not in manager.experiments]
    if missing:
        print(f"❌ Missing experiments: {missing}")
        print(f"   Available: {list(manager.experiments.keys())[:5]}...")
        return 1

    # Delegate to central manager orchestration (prints a dry-run plan)
    ok = process_experiments_through_pipeline(manager, args.exp, dry_run=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
