#!/usr/bin/env python3
import pandas as pd
from collections import Counter
from pathlib import Path
from scripts.utils.grounded_sam_utils import GroundedDinoAnnotations

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ANNOTATIONS_JSON = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations_finetuned.json"  # ← update this if needed
PROMPT           = "individual embryo"
THRESHOLDS       = [
    (0.50, 0.50),  # (confidence, iou)
    (0.45, 0.50),
]
OUTPUT_CSV = "video_mode_changes.csv"
# ────────────────────────────────────────────────────────────────────────────────

def get_hq_detection_count(hq_dict, img_id):
    """Count total HQ detections for a single image_id across all experiments."""
    total = 0
    for exp_block in hq_dict.values():
        filtered = exp_block.get("filtered", {})
        total += len(filtered.get(img_id, []))
    return total

def main():
    # Load annotations
    gdino = GroundedDinoAnnotations(ANNOTATIONS_JSON, verbose=False)

    # Build map: video_id -> sorted list of image_ids
    all_ids = gdino.get_all_image_ids()
    video_map = {}
    for img_id in all_ids:
        parts = img_id.split("_")
        vid = "_".join(parts[:2]) if len(parts) > 1 else img_id
        video_map.setdefault(vid, []).append(img_id)
    for vid in video_map:
        video_map[vid].sort()

    results = []

    # Iterate each video
    for vid, img_ids in sorted(video_map.items()):
        n = len(img_ids)
        if n == 0:
            continue
        sample_n = max(1, int(n * 0.2))
        sample_ids = img_ids[:sample_n]

        # Generate HQ dict for each threshold
        modes = {}
        for conf, iou in THRESHOLDS:
            key = f"{conf:.2f}_{iou:.2f}"
            hq = gdino.get_or_generate_high_quality_annotations(
                sample_ids,
                prompt=PROMPT,
                confidence_threshold=conf,
                iou_threshold=iou,
                save_to_self=False
            )
            counts = [ get_hq_detection_count(hq, img) for img in sample_ids ]
            mode = Counter(counts).most_common(1)[0][0] if counts else None
            modes[key] = mode

        # Compare modes
        keys = list(modes.keys())
        if modes[keys[0]] != modes[keys[1]]:
            results.append({
                "video_id": vid,
                f"mode_{keys[0]}": modes[keys[0]],
                f"mode_{keys[1]}": modes[keys[1]],
            })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {len(results)} differing videos to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
