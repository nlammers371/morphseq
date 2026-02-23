"""CLI for rendering segmentation evaluation videos from annotation JSON sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .results_adapter import list_videos
from .video_generator import VideoGenerator


def _derive_experiment_id(video_id: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    match = re.match(r"^(.+)_([A-H][0-9]{2})$", video_id)
    if match:
        return match.group(1)
    return video_id.rsplit("_", 1)[0] if "_" in video_id else video_id


def render_videos(
    *,
    results_json: Path,
    videos: list[str],
    experiment_id: str | None,
    output_dir: Path,
    output_single: Path | None,
    suffix: str,
    source_format: str,
    images_root: Path | None,
    show_bbox: bool,
    show_mask: bool,
    show_metrics: bool,
    show_qc: bool,
    show_labels: bool,
) -> int:
    vg = VideoGenerator()
    output_dir.mkdir(parents=True, exist_ok=True)

    multi = len(videos) > 1
    success = True

    for video_id in videos:
        exp_id = _derive_experiment_id(video_id, explicit=experiment_id)
        if output_single is not None and not multi:
            out_path = output_single
        else:
            out_path = output_dir / f"{video_id}{suffix}.mp4"

        print(f"Rendering {video_id} (exp={exp_id}) -> {out_path}")
        ok = vg.create_eval_video_from_results(
            results_json_path=results_json,
            experiment_id=exp_id,
            video_id=video_id,
            output_video_path=out_path,
            source_format=source_format,
            images_root=images_root,
            show_bbox=show_bbox,
            show_mask=show_mask,
            show_metrics=show_metrics,
            show_qc_flags=show_qc,
            show_labels=show_labels,
            verbose=True,
        )
        success = success and ok

    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Render segmentation evaluation videos")
    parser.add_argument("--json", required=True, help="Path to results JSON (GroundedSAM2 or COCO)")
    parser.add_argument(
        "--source-format",
        default="auto",
        choices=["auto", "grounded_sam2", "coco"],
        help="Annotation source format",
    )
    parser.add_argument("--exp", required=False, help="Experiment ID override")
    parser.add_argument("--video", action="append", help="Video ID; may be specified multiple times")
    parser.add_argument("--videos", help="Comma-separated list of video IDs")
    parser.add_argument("--all-in-exp", action="store_true", help="Render all videos under --exp")
    parser.add_argument("--images-root", help="Optional root directory for relative COCO image paths")
    parser.add_argument("--out", help="Single output mp4 path (single video only)")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--suffix", default="_eval", help="Suffix for generated MP4 names")
    parser.add_argument("--show-bbox", action="store_true", help="Show bounding boxes")
    parser.add_argument("--no-mask", action="store_true", help="Disable mask overlay")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metric text")
    parser.add_argument("--show-qc", action="store_true", help="Show qc flags (if available)")
    parser.add_argument("--no-labels", action="store_true", help="Disable object labels")
    args = parser.parse_args()

    results_json = Path(args.json)
    images_root = Path(args.images_root) if args.images_root else None

    videos: list[str] = []
    if args.video:
        videos.extend(args.video)
    if args.videos:
        videos.extend([v.strip() for v in args.videos.split(",") if v.strip()])

    if args.all_in_exp:
        if not args.exp:
            print("ERROR: --all-in-exp requires --exp")
            return 1
        videos = list_videos(
            results_json,
            source_format=args.source_format,
            experiment_id=args.exp,
        )

    if not videos:
        print("ERROR: no videos specified")
        return 1

    return render_videos(
        results_json=results_json,
        videos=videos,
        experiment_id=args.exp,
        output_dir=Path(args.out_dir),
        output_single=Path(args.out) if args.out else None,
        suffix=args.suffix,
        source_format=args.source_format,
        images_root=images_root,
        show_bbox=args.show_bbox,
        show_mask=not args.no_mask,
        show_metrics=not args.no_metrics,
        show_qc=args.show_qc,
        show_labels=not args.no_labels,
    )


if __name__ == "__main__":
    raise SystemExit(main())
