# Pipeline Scripts Structure

## Script Organization

```
segmentation_sandbox/
├── scripts/
│   ├── 01_prepare_videos.py         # Organize raw data
│   ├── 02_image_quality_check.py    # QC on images (optional)
│   ├── 03_gdino_detection.py        # GroundedDINO detection
│   ├── 04_sam2_segmentation.py      # SAM2 video processing
│   ├── 05_gsam_quality_control.py   # Technical QC
│   ├── 06_export_masks.py           # Export mask images
│   └── 07_embryo_metadata.py        # Initialize biological annotations
│
├── run_pipeline.sh                   # Master batch script
└── configs/
    └── pipeline_config.yaml          # Shared configuration
```

## Script 1: `01_prepare_videos.py`

```python
#!/usr/bin/env python
"""
Organize raw stitched images into standard structure and create videos.
"""
import argparse
from pathlib import Path
from data_organization import DataOrganizer

def main():
    parser = argparse.ArgumentParser(description="Organize raw data and create videos")
    parser.add_argument("--directory_with_experiments", required=True, help="Root directory with stitched images")
    parser.add_argument("--output_parent_dir", required=True, help="Output directory")
    parser.add_argument("--experiments_to_process", help="Comma-separated experiment names")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Parse experiment list
    experiment_names = None
    if args.experiments_to_process:
        experiment_names = [e.strip() for e in args.experiments_to_process.split(",")]
    
    # Run organizer
    organizer = DataOrganizer(verbose=args.verbose)
    organizer.process_experiments(
        source_dir=Path(args.directory_with_experiments),
        output_dir=Path(args.output_parent_dir),
        experiment_names=experiment_names,
        n_workers=args.workers
    )
    
    print(f"✅ Data organization complete. Metadata saved to: {args.output_parent_dir}/raw_data_organized/experiment_metadata.json")

if __name__ == "__main__":
    main()
```

## Script 2: `02_image_quality_check.py` (Optional)

```python
#!/usr/bin/env python
"""
Quality check on organized images (blur detection, etc).
"""
import argparse
from pathlib import Path
from metadata import ExperimentMetadata

def main():
    parser = argparse.ArgumentParser(description="Image quality check")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--output", help="QC report output path")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # TODO: Implement image quality checks
    print("⚠️  Image QC not yet implemented")

if __name__ == "__main__":
    main()
```

## Script 3: `03_gdino_detection.py`

```python
#!/usr/bin/env python
"""
Run GroundedDINO detection and high-quality filtering.
"""
import argparse
import yaml
from pathlib import Path
from detection_segmentation import GroundedDinoAnnotations

def load_gdino_model(config_path):
    """Load model from config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Model loading logic
    from detection_segmentation.grounded_dino_utils import load_groundingdino_model
    return load_groundingdino_model(config)

def main():
    parser = argparse.ArgumentParser(description="GroundedDINO detection")
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--output", required=True, help="Output annotations JSON")
    parser.add_argument("--experiments", help="Comma-separated experiment IDs")
    parser.add_argument("--prompt", default="individual embryo", help="Detection prompt")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold for HQ filtering")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for HQ filtering")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GDINO text threshold")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Load model
    print("🔧 Loading GroundedDINO model...")
    model = load_gdino_model(args.config)
    
    # Initialize annotations
    gdino = GroundedDinoAnnotations(args.output, verbose=args.verbose)
    gdino.set_metadata_path(args.metadata)
    
    # Parse experiments
    experiment_ids = None
    if args.experiments:
        experiment_ids = [e.strip() for e in args.experiments.split(",")]
    
    # Run detection
    print(f"🔍 Running detection with prompt: '{args.prompt}'")
    gdino.process_missing_annotations(
        model=model,
        prompts=args.prompt,
        experiment_ids=experiment_ids,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        auto_save_interval=100
    )
    
    # Generate high-quality annotations
    print(f"🎯 Generating high-quality annotations (conf>{args.confidence_threshold}, iou>{args.iou_threshold})")
    all_image_ids = gdino.get_all_image_ids()
    gdino.generate_high_quality_annotations(
        image_ids=all_image_ids,
        prompt=args.prompt,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        save_to_self=True
    )
    
    # Final save
    gdino.save()
    gdino.print_summary()

if __name__ == "__main__":
    main()
```

## Script 4: `04_sam2_segmentation.py`

```python
#!/usr/bin/env python
"""
Run SAM2 video segmentation using GDINO detections.
"""
import argparse
import yaml
from pathlib import Path
from detection_segmentation import GroundedSamAnnotations

def main():
    parser = argparse.ArgumentParser(description="SAM2 video segmentation")
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--annotations", required=True, help="GDINO annotations JSON")
    parser.add_argument("--output", required=True, help="Output SAM2 annotations JSON")
    parser.add_argument("--target-prompt", default="individual embryo", help="Target prompt from GDINO")
    parser.add_argument("--segmentation-format", default="rle", choices=["rle", "polygon"])
    parser.add_argument("--experiments", help="Comma-separated experiment IDs")
    parser.add_argument("--video-ids", help="Comma-separated video IDs")
    parser.add_argument("--max-videos", type=int, help="Maximum videos to process")
    parser.add_argument("--save-interval", type=int, default=5, help="Auto-save interval")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize SAM2
    print("🎬 Initializing SAM2...")
    gsam = GroundedSamAnnotations(
        filepath=args.output,
        seed_annotations_path=args.annotations,
        experiment_metadata_path=args.metadata,
        sam2_config=config["models"]["sam2"]["config"],
        sam2_checkpoint=config["models"]["sam2"]["checkpoint"],
        target_prompt=args.target_prompt,
        segmentation_format=args.segmentation_format,
        verbose=args.verbose
    )
    
    # Parse filters
    experiment_ids = None
    if args.experiments:
        experiment_ids = [e.strip() for e in args.experiments.split(",")]
    
    video_ids = None
    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(",")]
    
    # Process videos
    print(f"🔄 Processing videos...")
    gsam.process_missing_annotations(
        experiment_ids=experiment_ids,
        video_ids=video_ids,
        max_videos=args.max_videos,
        auto_save_interval=args.save_interval
    )
    
    gsam.print_summary()

if __name__ == "__main__":
    main()
```

## Script 5: `05_gsam_quality_control.py`

```python
#!/usr/bin/env python
"""
Run quality control on SAM2 annotations.
"""
import argparse
from pathlib import Path
from detection_segmentation import GSAMQualityControl

def main():
    parser = argparse.ArgumentParser(description="GSAM quality control")
    parser.add_argument("--annotations", required=True, help="SAM2 annotations JSON")
    parser.add_argument("--author", default="auto_qc", help="QC author identifier")
    parser.add_argument("--process-all", action="store_true", help="Reprocess all entities")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Run QC
    print("🔍 Running quality control checks...")
    qc = GSAMQualityControl(args.annotations, verbose=args.verbose)
    qc.run_all_checks(
        author=args.author,
        process_all=args.process_all,
        save_in_place=True
    )
    
    qc.print_summary()

if __name__ == "__main__":
    main()
```

## Script 6: `06_export_masks.py`

```python
#!/usr/bin/env python
"""
Export segmentation masks as image files.
"""
import argparse
from pathlib import Path
from detection_segmentation import SimpleMaskExporter

def main():
    parser = argparse.ArgumentParser(description="Export masks as images")
    parser.add_argument("--annotations", required=True, help="SAM2 annotations JSON")
    parser.add_argument("--output", required=True, help="Output directory for masks")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "tiff"])
    parser.add_argument("--experiments", help="Comma-separated experiment IDs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Parse experiments
    experiment_ids = None
    if args.experiments:
        experiment_ids = [e.strip() for e in args.experiments.split(",")]
    
    # Export masks
    print(f"📦 Exporting masks to {args.output}")
    exporter = SimpleMaskExporter(
        sam2_path=args.annotations,
        output_dir=Path(args.output),
        format=args.format
    )
    
    exported = exporter.process_missing_masks(
        experiment_ids=experiment_ids,
        overwrite=args.overwrite
    )
    
    print(f"✅ Exported {len(exported)} mask images")

if __name__ == "__main__":
    main()
```

## Script 7: `07_embryo_metadata.py`

```python
#!/usr/bin/env python
"""
Initialize embryo metadata for biological annotations.
"""
import argparse
from pathlib import Path
from annotations import EmbryoMetadata

def main():
    parser = argparse.ArgumentParser(description="Initialize embryo metadata")
    parser.add_argument("--annotations", required=True, help="SAM2 annotations JSON")
    parser.add_argument("--output", required=True, help="Output embryo metadata JSON")
    parser.add_argument("--schema", help="Custom schema path")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize embryo metadata
    print("🧬 Initializing embryo metadata...")
    em = EmbryoMetadata(
        sam_annotation_path=args.annotations,
        embryo_metadata_path=args.output,
        gen_if_no_file=True,
        verbose=args.verbose,
        schema_path=args.schema
    )
    
    # Save
    em.save()
    
    # Print summary
    summary = em.get_summary()
    print(f"✅ Created embryo metadata:")
    print(f"   Embryos: {summary['entity_counts']['embryos']}")
    print(f"   Snips: {summary['entity_counts']['snips']}")
    print(f"   Ready for biological annotations at: {args.output}")

if __name__ == "__main__":
    main()
```

## Master Batch Script: `run_pipeline.sh`

```bash
#!/bin/bash
#$ -N morphseq_pipeline
#$ -cwd
#$ -q trapnell-login.q
#$ -l gpgpu=TRUE,cuda=1
#$ -l mfree=30G
#$ -pe serial 1
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err

set -uo pipefail

# Activate environment
source /net/trapnell/vol1/home/mdcolon/software/miniconda3/etc/profile.d/conda.sh
conda activate segmentation_grounded_sam

# Paths
ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox
CONFIG=$ROOT/configs/pipeline_config.yaml
DATA_DIR=$ROOT/data
STITCHED_DIR=/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images

# Output paths
METADATA=$DATA_DIR/raw_data_organized/experiment_metadata.json
GDINO_OUT=$DATA_DIR/detections/gdino_annotations.json
SAM2_OUT=$DATA_DIR/segmentations/grounded_sam_annotations.json
EMBRYO_META=$DATA_DIR/annotations/embryo_metadata.json
MASK_DIR=$DATA_DIR/masks/embryo_masks

mkdir -p logs $DATA_DIR/{detections,segmentations,annotations,masks}

# Optional: specific experiments
# EXPERIMENTS="20231206,20240418,20250612_30hpf_ctrl_atf6"

echo "=== STEP 1: Organize raw data ==="
python $ROOT/scripts/01_prepare_videos.py \
  --directory_with_experiments "$STITCHED_DIR" \
  --output_parent_dir "$DATA_DIR" \
  --workers 8 \
  --verbose

echo "=== STEP 3: GroundedDINO detection ==="
python $ROOT/scripts/03_gdino_detection.py \
  --config "$CONFIG" \
  --metadata "$METADATA" \
  --output "$GDINO_OUT" \
  --confidence-threshold 0.45 \
  --iou-threshold 0.5 \
  --verbose

echo "=== STEP 4: SAM2 segmentation ==="
python $ROOT/scripts/04_sam2_segmentation.py \
  --config "$CONFIG" \
  --metadata "$METADATA" \
  --annotations "$GDINO_OUT" \
  --output "$SAM2_OUT" \
  --save-interval 10 \
  --verbose

echo "=== STEP 5: Quality control ==="
python $ROOT/scripts/05_gsam_quality_control.py \
  --annotations "$SAM2_OUT" \
  --verbose

echo "=== STEP 6: Export masks ==="
python $ROOT/scripts/06_export_masks.py \
  --annotations "$SAM2_OUT" \
  --output "$MASK_DIR" \
  --format png \
  --workers 8 \
  --verbose

echo "=== STEP 7: Initialize embryo metadata ==="
python $ROOT/scripts/07_embryo_metadata.py \
  --annotations "$SAM2_OUT" \
  --output "$EMBRYO_META" \
  --verbose

echo "✅ Pipeline complete!"
```

## Benefits of This Structure

1. **Modular**: Each step can be run independently
2. **Resumable**: Can restart from any step
3. **Debuggable**: Easy to isolate issues
4. **Configurable**: Each script has its own arguments
5. **Batch-friendly**: Works well with SGE/SLURM
6. **Parallel-ready**: Can run different experiments on different nodes