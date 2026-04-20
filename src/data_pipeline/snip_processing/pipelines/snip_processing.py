from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.snip_processing.io import (
    SnipPaths,
    merged_output_dirs,
    per_well_output_dirs,
    pipeline_version,
    rel_to_root,
    stable_config_hash,
    validate_snip_manifest_df,
)
from data_pipeline.snip_processing.ops import (
    estimate_background_stats_full_frame,
    process_snip_row,
    utc_now_iso,
)


def run_snip_processing_well(
    *,
    output_root: Path,
    experiment_id: str,
    well_id: str,
    frame_contract_csv: Path,
    segmentation_tracking_csv: Path,
    pipeline_config: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Path]:
    output_root = Path(output_root)
    snip_cfg = (pipeline_config.get("snip_processing") or {})
    if not bool(snip_cfg.get("enabled", True)):
        raise ValueError("snip_processing.enabled is false; nothing to do.")

    mask_type = str(snip_cfg.get("mask_type", "embryo"))
    target_pixel_size_um = float(snip_cfg.get("target_pixel_size_um", 7.8))
    output_shape_hw = tuple(int(x) for x in (snip_cfg.get("output_shape_hw") or [576, 256]))
    blend_radius_um = float(snip_cfg.get("blend_radius_um", 75.0))
    save_raw_crops = bool(snip_cfg.get("save_raw_crops", True))
    overwrite = bool(snip_cfg.get("overwrite", False))
    skip_existing = bool(snip_cfg.get("skip_existing", True))
    yolk_enabled = bool((snip_cfg.get("yolk_mask") or {}).get("enabled", True))

    paths = per_well_output_dirs(output_root=output_root, experiment_id=experiment_id, well_id=well_id)

    tracking_df = pd.read_csv(segmentation_tracking_csv)
    if len(tracking_df) == 0:
        raise ValueError(f"Empty segmentation_tracking: {segmentation_tracking_csv}")
    if "mask_type" in tracking_df.columns:
        tracking_df = tracking_df[tracking_df["mask_type"].astype(str) == mask_type].copy()
    if len(tracking_df) == 0:
        raise ValueError(f"No rows for mask_type={mask_type!r} in: {segmentation_tracking_csv}")

    frame_df = pd.read_csv(frame_contract_csv, usecols=["image_id", "micrometers_per_pixel", "well_index", "well_id", "frame_index"])
    frame_df["image_id"] = frame_df["image_id"].astype(str)
    tracking_df["image_id"] = tracking_df["image_id"].astype(str)

    merged = tracking_df.merge(frame_df, on="image_id", how="left", suffixes=("", "_frame"))
    if merged["micrometers_per_pixel"].isna().any():
        bad = merged.loc[merged["micrometers_per_pixel"].isna(), ["image_id", "snip_id"]].head(10).to_dict(orient="records")
        raise ValueError(f"Missing micrometers_per_pixel after join with frame_contract for rows: {bad}")

    # Determine background stats.
    bg_cfg = (snip_cfg.get("background_stats") or {})
    bg_mode = str(bg_cfg.get("mode", "estimate"))
    if bg_mode == "fixed":
        fixed = (bg_cfg.get("fixed") or {})
        bg_mean = float(fixed.get("mean", 128.0))
        bg_std = float(fixed.get("std", 30.0))
        bg_def = str(bg_cfg.get("definition", "full_frame_outside_embryo"))
    else:
        est = (bg_cfg.get("estimate") or {})
        bg = estimate_background_stats_full_frame(
            rows=merged[["snip_id", "image_id", "source_image_path", "exported_mask_path"]].copy(),
            output_root=output_root,
            n_samples=int(est.get("n_samples", 100)),
            seed=int(est.get("seed", 309)),
        )
        bg_mean, bg_std, bg_def = float(bg.mean), float(bg.std), str(bg.definition)

    # Cache background stats for this per-well shard.
    try:
        (paths.artifacts_dir / "background_stats.json").write_text(
            json.dumps(
                {
                    "mean": bg_mean,
                    "std": bg_std,
                    "definition": bg_def,
                    "mode": bg_mode,
                },
                indent=2,
            )
            + "\n"
        )
    except Exception:
        pass

    pv = pipeline_version()
    cfg_hash = stable_config_hash(snip_cfg)
    ts = utc_now_iso()

    rows_out = []
    # Deterministic order.
    merged = merged.sort_values(["image_id", "snip_id"]).reset_index(drop=True)

    for _, row in merged.iterrows():
        snip_id = str(row["snip_id"])
        processed_path = paths.processed_dir / f"{snip_id}.jpg"

        if processed_path.exists() and (not overwrite) and skip_existing:
            # Still emit a manifest row, but do minimal validation.
            sz = int(processed_path.stat().st_size) if processed_path.exists() else 0
            rows_out.append(
                {
                    "snip_id": snip_id,
                    "mask_type": str(row.get("mask_type", mask_type)),
                    "experiment_id": str(experiment_id),
                    "well_id": str(row.get("well_id", well_id)),
                    "well_index": row.get("well_index"),
                    "image_id": str(row["image_id"]),
                    "embryo_id": str(row["embryo_id"]),
                    "frame_index": int(row["frame_index"]),
                    "source_image_path": str(row["source_image_path"]),
                    "exported_mask_path": str(row["exported_mask_path"]),
                    "yolk_mask_path": None,
                    "processed_snip_path": rel_to_root(processed_path, output_root=output_root),
                    "raw_crop_path": None,
                    "source_micrometers_per_pixel": float(row["micrometers_per_pixel"]),
                    "target_pixel_size_um": float(target_pixel_size_um),
                    "output_height_px": int(output_shape_hw[0]),
                    "output_width_px": int(output_shape_hw[1]),
                    "blend_radius_um": float(blend_radius_um),
                    "background_mean": float(bg_mean),
                    "background_std": float(bg_std),
                    "background_definition": str(bg_def),
                    "rotation_angle_rad": 0.0,
                    "rotation_angle_deg": 0.0,
                    "rotation_source": "embryo_only",
                    "pipeline_version": pv,
                    "snip_processing_config_hash": cfg_hash,
                    "processing_timestamp_utc": ts,
                    "processed_file_size_bytes": sz,
                    "raw_file_size_bytes": None,
                    "is_valid": bool(sz > 0),
                    "error_message": None if sz > 0 else "Missing/empty processed snip",
                }
            )
            continue

        out = process_snip_row(
            row=row,
            output_root=output_root,
            experiment_id=str(experiment_id),
            well_id=str(well_id),
            processed_dir=paths.processed_dir,
            raw_crops_dir=paths.raw_crops_dir,
            save_raw_crops=save_raw_crops,
            target_pixel_size_um=target_pixel_size_um,
            output_shape_hw=output_shape_hw,
            background_mean=bg_mean,
            background_std=bg_std,
            blend_radius_um=blend_radius_um,
            yolk_enabled=yolk_enabled,
        )
        out["pipeline_version"] = pv
        out["snip_processing_config_hash"] = cfg_hash
        out["processing_timestamp_utc"] = ts
        rows_out.append(out)

    manifest = pd.DataFrame(rows_out)
    validate_snip_manifest_df(manifest)
    manifest_path_pq = paths.contracts_dir / "snip_manifest.parquet"
    manifest_path_csv = paths.contracts_dir / "snip_manifest.csv"
    manifest.to_parquet(manifest_path_pq, index=False)
    manifest.to_csv(manifest_path_csv, index=False)

    # Per-well sentinel
    sentinel = paths.contracts_dir / ".snip_processing.validated"
    sentinel.write_text("validated\n")

    if verbose:
        ok = int(manifest["is_valid"].sum()) if "is_valid" in manifest.columns else 0
        print(f"[snip_processing] wrote {len(manifest)} rows ({ok} valid) to {paths.contracts_dir}")

    return {
        "per_well_root": paths.per_well_root,
        "manifest_parquet": manifest_path_pq,
        "manifest_csv": manifest_path_csv,
        "validated_flag": sentinel,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--well-id", required=True)
    p.add_argument("--frame-contract-csv", type=Path, required=True)
    p.add_argument("--segmentation-tracking-csv", type=Path, required=True)
    p.add_argument("--config-yaml", type=Path, required=True)
    p.add_argument("--verbose", default="false")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config_yaml).read_text()) or {}
    run_snip_processing_well(
        output_root=args.output_root,
        experiment_id=str(args.experiment),
        well_id=str(args.well_id),
        frame_contract_csv=args.frame_contract_csv,
        segmentation_tracking_csv=args.segmentation_tracking_csv,
        pipeline_config=cfg,
        verbose=str(args.verbose).strip().lower() in {"1", "true", "yes", "y", "on"},
    )


if __name__ == "__main__":
    main()
