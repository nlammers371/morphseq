from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.snip_processing.run_per_well import SnipProcessingConfig, run_snip_processing_for_well


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmentation-tracking", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--output-shape", type=int, nargs=2, metavar=("H", "W"), required=True)
    ap.add_argument("--target-pixel-size-um", type=float, required=True)
    ap.add_argument("--save-raw-crops", type=int, default=1)
    ap.add_argument("--write-manifest-csv", type=int, default=1)
    ap.add_argument("--background-n-samples", type=int, default=100)
    ap.add_argument("--background-seed", type=int, default=309)
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    cfg = SnipProcessingConfig(
        output_shape=(int(args.output_shape[0]), int(args.output_shape[1])),
        target_pixel_size_um=float(args.target_pixel_size_um),
        save_raw_crops=bool(int(args.save_raw_crops)),
        write_manifest_csv=bool(int(args.write_manifest_csv)),
        background_n_samples=int(args.background_n_samples),
        background_seed=int(args.background_seed),
    )

    run_snip_processing_for_well(
        segmentation_tracking_path=args.segmentation_tracking,
        output_root=args.output_root,
        config=cfg,
        snip_processing_run_id=args.run_id,
    )


if __name__ == "__main__":
    main()

