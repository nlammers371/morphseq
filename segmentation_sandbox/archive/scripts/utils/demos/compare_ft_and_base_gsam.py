from scripts.utils.sam2_utils import GroundedSamAnnotations

import json
from pathlib import Path

def print_gsam_summary(annotation_path, seed_annotations_path, metadata_path, config_path, sam2_config, sam2_checkpoint, verbose=True):
    print(f"\n🔎 Summary for: {annotation_path}")
    grounded_sam = GroundedSamAnnotations(
        filepath=annotation_path,
        seed_annotations_path=seed_annotations_path,
        experiment_metadata_path=metadata_path,
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        device="cuda",
        verbose=verbose
    )
    summary = grounded_sam.get_summary()
    print(f"Videos processed: {summary.get('videos_processed', 0)}")
    print(f"Videos failed: {summary.get('videos_failed', 0)}")
    print(f"Frames processed: {summary.get('total_frames_processed', 0)}")
    print(f"Embryos tracked: {summary.get('total_embryos_tracked', 0)}")

    # Compare embryo count disagreements between base and finetuned annotations
    print("\n🤼 Embryo count disagreements:")
    # Load base and finetuned JSONs
    base_json = json.loads(Path(annotation_path).read_text())
    ft_path = Path(annotation_path).with_name(Path(annotation_path).stem + "_finetuned.json")
    ft_json = json.loads(ft_path.read_text()) if ft_path.exists() else {}
    base_counts = {}
    ft_counts = {}
    # Collect counts from base
    for exp in base_json.get("experiments", {}).values():
        for vid, v in exp.get("videos", {}).items():
            base_counts[vid] = v.get("num_embryos", len(v.get("embryo_ids", [])))
    # Collect counts from finetuned
    for exp in ft_json.get("experiments", {}).values():
        for vid, v in exp.get("videos", {}).items():
            ft_counts[vid] = v.get("num_embryos", len(v.get("embryo_ids", [])))
    # Print mismatches
    for vid in sorted(set(base_counts) & set(ft_counts)):
        b = base_counts[vid]
        f = ft_counts[vid]
        if b != f:
            print(f"{vid}: base={b} vs finetuned={f}")

    print(f"\n📁 Results file: {annotation_path}")
    grounded_sam.print_summary()

if __name__ == "__main__":
    # Use the same paths as in 04_sam2_video_processing.py

    metadata_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json"

    base_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations.json"
    finetuned_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json"
    seed_annotations_path_ft = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations_finetuned.json"
    seed_annotations_path_base = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations.json"

    config_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/configs/pipeline_config.yaml"
    # These are pulled from config in 04_sam2_video_processing.py
    # If you want to load from config.yaml, you can use load_config(config_path) as in 04
    sam2_config = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/models/checkpoints/sam2.1_hiera_large.pt"

    print_gsam_summary(base_path, seed_annotations_path_base, metadata_path, config_path, sam2_config, sam2_checkpoint)
    # print_gsam_summary(finetuned_path, seed_annotations_path_ft, metadata_path, config_path, sam2_config, sam2_checkpoint)