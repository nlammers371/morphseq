"""Debug Keyence tile grouping, FF generation, and stitching for one frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import skimage.io as skio

from data_pipeline.metadata_ingest.stitched_index.materialize_stitched_images import (
    _infer_keyence_stack_lookup,
    _keyence_filter_size,
    _keyence_master_params_path,
    _keyence_orientation,
    _materialize_keyence_ff_tiles_with_log,
    _materialize_keyence_tile_projection,
    _resolve_device,
    _stitch_keyence_tile_projections,
)


def _stats(image: np.ndarray) -> dict[str, float | int | list[int]]:
    arr = image.astype(np.float32)
    return {
        "shape": [int(v) for v in image.shape],
        "dtype": str(image.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p99": float(np.percentile(arr, 99)),
    }


def _to_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _pad_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image
    pad_top = (target_h - h) // 2
    pad_bot = target_h - h - pad_top
    return np.pad(image, ((pad_top, pad_bot), (0, 0)), mode="constant")


def _tile_strip(images: Iterable[np.ndarray]) -> np.ndarray:
    items = list(images)
    if not items:
        return np.zeros((1, 1), dtype=np.uint8)
    target_h = max(int(img.shape[0]) for img in items)
    padded = [_pad_to_height(img, target_h) for img in items]
    return np.concatenate(padded, axis=1)


def build_debug_artifacts(
    raw_images_dir: Path,
    experiment: str,
    well_index: str,
    time_int: int,
    output_dir: Path,
    micrometers_per_pixel: float,
    ff_filter_res_um: float,
    device_preference: str,
) -> dict:
    lookup = _infer_keyence_stack_lookup(raw_images_dir)
    key = (well_index, int(time_int))
    tile_stacks = lookup.get(key)
    if not tile_stacks:
        available = sorted([k for k in lookup.keys() if k[0] == well_index])[:20]
        raise ValueError(f"No Keyence tiles found for {key}. Available for well: {available}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_ids = sorted(tile_stacks.keys())

    device = _resolve_device(device_preference)
    filter_size = _keyence_filter_size(
        micrometers_per_pixel=micrometers_per_pixel,
        ff_filter_res_um=ff_filter_res_um,
    )
    orientation = _keyence_orientation(experiment)
    master_params = _keyence_master_params_path(raw_images_dir, experiment)

    max_tiles = [_materialize_keyence_tile_projection(tile_stacks[tile_id]) for tile_id in tile_ids]
    log_tiles = _materialize_keyence_ff_tiles_with_log(tile_stacks, device=device, filter_size=filter_size)

    stitched_max_result = _stitch_keyence_tile_projections(
        tile_projections=max_tiles,
        orientation=orientation,
        master_params_path=master_params,
    )
    stitched_log_result = _stitch_keyence_tile_projections(
        tile_projections=log_tiles,
        orientation=orientation,
        master_params_path=master_params,
    )
    stitched_max = stitched_max_result.stitched
    stitched_log = stitched_log_result.stitched

    for tile_id, image in zip(tile_ids, max_tiles):
        skio.imsave(output_dir / f"tile_{tile_id:02d}_max.jpg", _to_u8(image), check_contrast=False)
    for tile_id, image in zip(tile_ids, log_tiles):
        skio.imsave(output_dir / f"tile_{tile_id:02d}_log.jpg", _to_u8(image), check_contrast=False)

    max_strip = _tile_strip([_to_u8(img) for img in max_tiles])
    log_strip = _tile_strip([_to_u8(img) for img in log_tiles])
    skio.imsave(output_dir / "tiles_max_strip.jpg", max_strip, check_contrast=False)
    skio.imsave(output_dir / "tiles_log_strip.jpg", log_strip, check_contrast=False)
    skio.imsave(output_dir / "stitched_max.jpg", _to_u8(stitched_max), check_contrast=False)
    skio.imsave(output_dir / "stitched_log.jpg", _to_u8(stitched_log), check_contrast=False)

    diff = np.abs(_to_u8(stitched_log).astype(np.int16) - _to_u8(stitched_max).astype(np.int16)).astype(np.uint8)
    skio.imsave(output_dir / "stitched_absdiff_log_vs_max.jpg", diff, check_contrast=False)

    report = {
        "experiment": experiment,
        "well_index": well_index,
        "time_int": int(time_int),
        "device_requested": device_preference,
        "device_used": device,
        "orientation": orientation,
        "master_params_path": str(master_params) if master_params else None,
        "filter_res_um": float(ff_filter_res_um),
        "filter_size": int(filter_size),
        "n_tiles": len(tile_ids),
        "tile_ids": [int(v) for v in tile_ids],
        "tile_depths": {str(t): int(len(tile_stacks[t])) for t in tile_ids},
        "tile_first_paths": {str(t): str(tile_stacks[t][0]) for t in tile_ids},
        "max_tile_stats": {str(t): _stats(img) for t, img in zip(tile_ids, max_tiles)},
        "log_tile_stats": {str(t): _stats(img) for t, img in zip(tile_ids, log_tiles)},
        "stitched_max_fallback_used": stitched_max_result.fallback_used,
        "stitched_log_fallback_used": stitched_log_result.fallback_used,
        "stitched_max_qc_passed": bool(stitched_max_result.qc.passed),
        "stitched_log_qc_passed": bool(stitched_log_result.qc.passed),
        "stitched_max_qc_reasons": list(stitched_max_result.qc.reasons),
        "stitched_log_qc_reasons": list(stitched_log_result.qc.reasons),
        "stitched_max_stats": _stats(stitched_max),
        "stitched_log_stats": _stats(stitched_log),
        "stitched_absdiff_stats": _stats(diff),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-images-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--well-index", required=True)
    parser.add_argument("--time-int", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--micrometers-per-pixel", type=float, default=1.0)
    parser.add_argument("--ff-filter-res-um", type=float, default=3.0)
    parser.add_argument("--device-preference", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_debug_artifacts(
        raw_images_dir=args.raw_images_dir,
        experiment=args.experiment,
        well_index=args.well_index,
        time_int=args.time_int,
        output_dir=args.output_dir,
        micrometers_per_pixel=args.micrometers_per_pixel,
        ff_filter_res_um=args.ff_filter_res_um,
        device_preference=args.device_preference,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "n_tiles": report["n_tiles"]}, indent=2))


if __name__ == "__main__":
    main()
