from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.feature_extraction.core.fraction_alive import extract_fraction_alive_batch
from data_pipeline.feature_extraction.io.loaders import (
    load_auxiliary_masks_manifest,
    load_frame_contract,
    load_segmentation_tracking,
    merge_tracking_with_frame_contract,
)
from data_pipeline.feature_extraction.io.writers import write_feature_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmentation-tracking", type=Path, required=True)
    ap.add_argument("--frame-contract", type=Path, required=True)
    ap.add_argument("--auxiliary-masks-manifest", type=Path, default=None)
    ap.add_argument("--auxiliary-masks-dir", type=Path, default=None)
    ap.add_argument("--unet-masks-dir", type=Path, default=None)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    tracking_df = load_segmentation_tracking(args.segmentation_tracking)
    frame_df = load_frame_contract(args.frame_contract)
    merged = merge_tracking_with_frame_contract(tracking_df, frame_df)
    via_mask_lookup = None
    if args.auxiliary_masks_manifest is not None:
        auxiliary_masks_df = load_auxiliary_masks_manifest(args.auxiliary_masks_manifest)
        via_mask_lookup = {
            str(row["image_id"]): Path(str(row["via_mask_path"]))
            for _, row in auxiliary_masks_df.iterrows()
        }
    auxiliary_masks_dir = args.auxiliary_masks_dir or args.unet_masks_dir
    feature_df = extract_fraction_alive_batch(
        merged,
        via_mask_dir=auxiliary_masks_dir / "via" if auxiliary_masks_dir else None,
        via_mask_lookup=via_mask_lookup,
    )
    write_feature_table(feature_df, args.output_csv)


if __name__ == "__main__":
    main()
