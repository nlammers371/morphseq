#!/usr/bin/env python3
"""Batch creation of SAM2 evaluation videos.

The script accepts embryo or video IDs and produces MP4 videos with SAM2
segmentation overlays. It can be used as a command-line tool or imported as a
module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

# Add scripts directory to path so we can import utilities
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.parsing_utils import parse_entity_id  # noqa: E402
from utils.video_generation import VideoGenerator  # noqa: E402


def _gather_video_pairs(ids: Iterable[str]) -> Set[Tuple[str, str]]:
    """Resolve embryo or video IDs to unique (experiment_id, video_id) pairs."""
    pairs: Set[Tuple[str, str]] = set()
    for entity_id in ids:
        components = parse_entity_id(entity_id)
        experiment_id = components.get("experiment_id")
        video_id = components.get("video_id")
        if experiment_id and video_id:
            pairs.add((experiment_id, video_id))
        else:
            print(f"⚠️ Could not parse video from '{entity_id}', skipping")
    return pairs


def generate_videos(
    ids: Iterable[str],
    results_json: Path,
    output_dir: Path | None = None,
    suffix: str = "",
    show_bbox: bool = False,
    show_mask: bool = True,
    show_labels: bool = True,
) -> None:
    """Generate evaluation videos for the provided IDs."""
    output_dir = output_dir or Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    vg = VideoGenerator()
    video_pairs = _gather_video_pairs(ids)

    for experiment_id, video_id in sorted(video_pairs):
        out_name = f"{video_id}{suffix}.mp4"
        out_path = output_dir / out_name
        print(f"🎬 Generating {out_name}")
        vg.create_sam2_eval_video_from_results(
            results_json_path=results_json,
            experiment_id=experiment_id,
            video_id=video_id,
            output_video_path=out_path,
            show_bbox=show_bbox,
            show_mask=show_mask,
            show_metrics=show_labels,
            verbose=True,
        )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create SAM2 overlay videos from embryo or video IDs"
    )
    parser.add_argument("ids", nargs="+", help="Embryo or video IDs to process")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("data/segmentation/grounded_sam_segmentations.json"),
        help="Path to GroundedSAM annotations JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for output videos (default: current directory)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended before .mp4 in output filenames",
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Draw bounding boxes around embryos",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable mask overlays",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable embryo ID and metric labels",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    generate_videos(
        ids=args.ids,
        results_json=args.results_json,
        output_dir=args.output_dir,
        suffix=args.suffix,
        show_bbox=args.bbox,
        show_mask=not args.no_mask,
        show_labels=not args.no_labels,
    )


if __name__ == "__main__":
    main()
