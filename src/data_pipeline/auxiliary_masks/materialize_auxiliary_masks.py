"""
Auxiliary mask materialization harness.

This stage owns the auxiliary-mask output contract for now. Long term, the
implementation should move into the main `data_pipeline/segmentation` layer so
the pipeline stays fully self-contained, while this wrapper keeps the run
interface thin and explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from data_pipeline.auxiliary_masks.inference import run_auxiliary_mask_inference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-contract", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--output-manifest-csv", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--well-id", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_auxiliary_mask_inference(
        frame_contract=args.frame_contract,
        model_root=args.model_root,
        output_root=args.output_root,
        output_manifest_csv=args.output_manifest_csv,
        well_id=args.well_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
