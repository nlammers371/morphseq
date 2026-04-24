from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.segmentation_and_tracking.pipelines.segmentation_and_tracking import (
    run_segmentation_and_tracking,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--well-id", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pipeline-cfg-json", type=str, default="{}")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pipeline_cfg = json.loads(args.pipeline_cfg_json) if args.pipeline_cfg_json else {}
    run_segmentation_and_tracking(
        frame_contract_csv=args.frame_manifest,
        experiment_id=args.experiment_id,
        well_id=args.well_id,
        output_root=args.output_root,
        pipeline_config=pipeline_cfg,
        device=args.device,
        run_id=args.run_id,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
