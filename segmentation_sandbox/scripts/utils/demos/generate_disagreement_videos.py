#!/usr/bin/env python3
"""
Generate tracking videos for the top‐disagreement videos between base and finetuned annotations.
Ranks videos by absolute difference in embryo count, and creates videos using sam2_video_generator.

Example:
    python generate_disagreement_videos.py \
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations.json" \
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json" \
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/visualization_ouput/20250716"
"""

# Make executable: chmod +x generate_disagreement_videos.py

import json
from pathlib import Path
import argparse
import cv2

# Import the video creation function
from scripts.utils.video_generation.sam2_video_generator import create_video, create_side_by_side_video

def load_counts_and_map(json_path):
    """
    Load annotation JSON and return:
      - counts: dict mapping video_id -> num_embryos
      - video_map: dict mapping video_id -> video_data dict
    """
    data = json.loads(Path(json_path).read_text())
    counts = {}
    video_map = {}
    for exp_data in data.get("experiments", {}).values():
        for vid, v in exp_data.get("videos", {}).items():
            # get count
            count = v.get("num_embryos", len(v.get("embryo_ids", [])))
            counts[vid] = count
            video_map[vid] = v
    return counts, video_map

def main():
    parser = argparse.ArgumentParser(
        description="Generate videos for top‐disagreement GSAM annotations"
    )
    parser.add_argument(
        "base_annotations",
        help="Path to base grounded_sam_annotations.json"
    )
    parser.add_argument(
        "finetuned_annotations",
        help="Path to finetuned grounded_sam_annotations_finetuned.json"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save disagreement videos"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top disagreements to process"
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="FPS for output videos"
    )
    args = parser.parse_args()

    base_counts, base_map = load_counts_and_map(args.base_annotations)
    ft_counts, ft_map     = load_counts_and_map(args.finetuned_annotations)

    # Compute absolute differences for videos present in both
    diffs = []
    for vid in set(base_counts) & set(ft_counts):
        d = abs(base_counts[vid] - ft_counts[vid])
        if d > 0:
            diffs.append((vid, d))
    if not diffs:
        print("No disagreements in embryo counts found.")
        return

    # Sort by descending difference
    diffs.sort(key=lambda x: x[1], reverse=True)
    top = diffs[: args.top_k]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid, diff in top:
        print(f"→ Generating video for {vid} (count diff = {diff})")
        base_video_data = base_map.get(vid)
        ft_video_data = ft_map.get(vid)
        if base_video_data is None or ft_video_data is None:
            print(f"   ❌ No video data for {vid}, skipping.")
            continue

        # Output paths
        base_out_path = out_dir / vid / "base.mp4"
        ft_out_path   = out_dir / vid / "ft.mp4"
        # Derive frames directories from the video output paths (matches create_video)
        base_frames_dir = Path(str(base_out_path).replace(".mp4", "")) / "frames"
        ft_frames_dir   = Path(str(ft_out_path).replace(".mp4", "")) / "frames"

        # Generate base video and export frames
        print(f"   Generating base video...")
        base_success = create_video(vid, base_video_data, str(base_out_path), fps=args.fps)
        # Frames are already exported by create_video

        # Generate finetuned video and export frames
        print(f"   Generating finetuned video...")
        ft_success = create_video(vid, ft_video_data, str(ft_out_path), fps=args.fps)
        # Only generate side-by-side if both videos succeeded
        if not base_success or not ft_success:
            print(f"   ❌ Skipping side-by-side for {vid} due to failed video creation")
            continue

        # Collect frame ids
        frame_ids = [img_id for img_id in base_video_data.get("images", {}).keys()]
        frame_ids = sorted(frame_ids)

        # Generate side-by-side video
        side_by_side_path = out_dir / f"{vid}_side_by_side.mp4"
        print(f"   Generating side-by-side video...")
        create_side_by_side_video(
            vid,
            base_dir=base_frames_dir,
            ft_dir=ft_frames_dir,
            output_path=str(side_by_side_path),
            frame_ids=frame_ids,
            fps=args.fps
        )

if __name__ == "__main__":
    main()