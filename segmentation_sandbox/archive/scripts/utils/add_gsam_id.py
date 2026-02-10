#!/usr/bin/env python3
"""
Dummy script to load GSAM annotations JSON and set the gsam_annotation_id to 9545.
"""
import json
from pathlib import Path

def main():
    # Path to GSAM annotations file
    json_path = Path(
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json"
    )
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return

    # Load existing data
    data = json.loads(json_path.read_text())

    # Set the GSAM annotation ID
    data['gsam_annotation_id'] = 9545

     data['gsam_annotation_id'] = 9545

    # Save changes atomically
    temp_path = json_path.with_suffix('.tmp')
    json.dump(data, temp_path.open('w'), indent=2)
    temp_path.replace(json_path)

    print(f"Updated GSAM annotation ID to 9545 in {json_path}")

if __name__ == "__main__":
    main()