#!/usr/bin/env python3
"""
Live vs Dead GroundingDINO Inference on All Metadata Images
===========================================================

Process ALL images in experiment metadata with both base and finetuned models.
Saves annotations to live_vs_dead.json with auto-save functionality.
"""

import os
import sys
import random
from pathlib import Path
import torch

# Add sandbox root to PYTHONPATH
SANDBOX_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox")
sys.path.append(str(SANDBOX_ROOT))

from scripts.utils.grounded_sam_utils import (
    load_config,
    load_groundingdino_model,
    GroundedDinoAnnotations,
)

from scripts.utils.experiment_metadata_utils import (
    load_experiment_metadata,
)

def main():
    print("🚀 Starting GroundingDINO inference pipeline on ALL metadata images")
    print("=" * 70)
    
    # 1) Load config + metadata
    config_path = SANDBOX_ROOT / "configs" / "pipeline_config.yaml" 
    config = load_config(config_path)
    
    expr_metadata_path = SANDBOX_ROOT / "data" / "raw_data_organized" / "experiment_metadata.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📱 Using device: {device}")
    print(f"📁 Config: {config_path}")
    print(f"📂 Metadata: {expr_metadata_path}")
    
    # Load experiment metadata
    metadata = load_experiment_metadata(expr_metadata_path)
    image_ids = metadata.get("image_ids", [])
    print(f"📊 Total images in metadata: {len(image_ids)}")
    
    # 2) Initialize annotation manager with metadata integration
    annotation_path = SANDBOX_ROOT / "data" / "annotation_and_masks" / "gdino_annotations" / "gdino_annotations.json"
    annotations = GroundedDinoAnnotations(annotation_path, verbose=True)
    annotations.set_metadata_path(expr_metadata_path)  # Set metadata path for annotations manager
    
    print(f"💾 Annotations will be saved to: {annotation_path}")
    
    # Load model
    model = load_groundingdino_model(config, device=device)
    print("✅ Model loaded successfully and fixed memory issue")
    
    # Print initial summary
    annotations.print_processing_summary(["individual embryo", "live", "dead"])
    
    # PHASE 0: Process "individual embryo" prompt first
    print("\n" + "="*60)
    print("🔍 PHASE 0: Processing 'individual embryo' detection FIRST")
    print("="*60)
    
    try:
        # Save any new annotations for "individual embryo"
        annotations.process_missing_annotations(
            model, 
            "individual embryo",
            auto_save_interval=100,
            store_image_source=False,
            show_anno=False,
            overwrite=False
        )
        annotations.save()  # Persist annotations to disk
        
        print("✅ Phase 0 complete: 'individual embryo' processing finished")
        
    except Exception as e:
        print(f"❌ Error in Phase 0 (individual embryo): {e}")
        return
    
    # 3) Process with base (pretrained) model - "embryo" prompt
    # PHASE 1: Process with basse (pretrained) model - "embryo" prompt (if needed)
    print("\n" + "="*50)
    print("🔍 PHASE 1: Base GroundingDINO - 'embryo' detection")
    print("="*50)

    print(f"✅ Phase 1 complete: Base model processing finished! Processed images")
        

    # PHASE 2: Process with finetuned model - "live" and "dead" prompts
    print("\n" + "="*50)
    print("🔍 PHASE 2: Finetuned GroundingDINO - 'live' and 'dead' detection")
    print("="*50)
    # 2) Initialize annotation manager with metadata integration
    annotation_path = SANDBOX_ROOT / "data" / "annotation_and_masks" / "gdino_annotations" / "gdino_annotations_(finetuned)_live_vs_dead.json"
    annotations_ft = GroundedDinoAnnotations(annotation_path, verbose=True)
    annotations_ft.set_metadata_path(expr_metadata_path)  # Set metadata path for annotations manager

    try:
        # Update config to use finetuned weights
        config["models"]["groundingdino"]["weights"] = (
            "/net/trapnell/vol1/home/mdcolon/proj/"
            "image_segmentation/Open-GroundingDino/"
            "finetune_output/finetune_output_run_nick_masks_20250308/"
            "checkpoint_best_regular.pth"
        )
        
        # Load finetuned model
        model_ft = load_groundingdino_model(config, device=device)
        
        # Process both "live" and "dead" prompts
        annotations_ft.process_missing_annotations(
            model=model_ft,
            prompts=["live", "dead"],  # Process both prompts
            box_threshold=0.15,
            text_threshold=0.01,
            show_anno=False,  # Don't display during batch processing
            auto_save_interval=100,  # Auto-save every 25 processed images (more frequent for finetuned)
            overwrite=False,  # Don't overwrite existing annotations
            store_image_source=False,  # Save memory during batch processing
            consider_different_if_different_weights=True,  # Treat different models as separate
        )

        annotations_ft.save()

        print(f"✅ Phase 2 complete: Finetuned model processing finished! Processed images")

    except Exception as e:
        print(f"❌ Error with finetuned model in Phase 2: {e}")
    
    # Final summary and save
    print("\n" + "="*50)
    print("📊 FINAL SUMMARY")
    print("="*50)
    
    # Print comprehensive summary
    annotations_ft.print_processing_summary(["individual embryo", "live", "dead"], consider_different_if_different_weights=True)

    # Ensure final save
    if annotations_ft.has_unsaved_changes:
        print("\n💾 Performing final save...")
        annotations_ft.save()
    
    print(f"\n✅ ALL PROCESSING COMPLETE!")
    print(f"📂 Annotations saved to: {annotation_path}")
    print(f"🎯 You can now analyze the results or run additional inference")
    
    # Optional: Print some example annotations
    print(f"\n📋 Quick sample of annotations:")
    all_image_ids = annotations.get_all_image_ids()
    if len(all_image_ids) > 0:
        sample_id = random.choice(all_image_ids)
        annotation_list = annotations.get_annotations_for_image(sample_id)
        print(f"   📸 Sample image {sample_id}: {len(annotation_list)} annotations")
        for ann in annotation_list[:3]:  # Show first 3
            model_name = Path(ann.get("model_metadata", {}).get("model_weights_path", "unknown")).name
            print(f"      • '{ann['prompt']}': {ann['num_detections']} detections (model: {model_name})")

def main_with_sampling(max_images: int = 1000):
    """
    Alternative version that samples a subset of images for testing.
    
    Args:
        max_images: Maximum number of images to process (for testing)
    """
    print(f"🚀 Starting GroundingDINO inference pipeline on {max_images} sampled images")
    print("=" * 70)
    
    # Load metadata to get image IDs for sampling
    expr_metadata_path = SANDBOX_ROOT / "data" / "raw_data_organized" / "experiment_metadata.json"
    metadata = load_experiment_metadata(expr_metadata_path)
    
    # Sample random image IDs
    all_ids = metadata["image_ids"]
    sampled_ids = random.sample(all_ids, min(max_images, len(all_ids)))
    
    print(f"📊 Sampled {len(sampled_ids)} images from {len(all_ids)} total images")
    
    # Initialize annotation manager
    annotation_path = SANDBOX_ROOT / "data" / "annotation_and_masks" / "gdino_annotations" / "gdino_annotations_test.json"
    annotations = GroundedDinoAnnotations(annotation_path, verbose=True)
    annotations.set_metadata_path(expr_metadata_path)
    
    # Load config
    config = load_config(SANDBOX_ROOT / "configs" / "pipeline_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_groundingdino_model(config, device=device)
    
    # Process "individual embryo" first
    print("\n🔍 Phase 0: Processing 'individual embryo'...")
    annotations.process_missing_annotations(
        model=model,
        prompts="individual embryo",
        image_ids=sampled_ids,  # Only process sampled images
        auto_save_interval=20,
        show_anno=False,
        store_image_source=False,
    )

    # Process with finetuned model
    print("\n🔍 Phase 2: Processing with finetuned model (live/dead)...")
    config["models"]["groundingdino"]["weights"] = (
        "/net/trapnell/vol1/home/mdcolon/proj/"
        "image_segmentation/Open-GroundingDino/"
        "finetune_output/finetune_output_run_nick_masks_20250308/"
        "checkpoint_best_regular.pth"
    )
    model_ft = load_groundingdino_model(config, device=device)
    
    results_ft = annotations.process_missing_annotations(
        model=model_ft,
        prompts=["live", "dead"],
        image_ids=sampled_ids,  # Only process sampled images
        auto_save_interval=10,
        show_anno=False,
        store_image_source=False,
        consider_different_if_different_weights=True,
    )
    
    print(f"\n✅ Sampling complete! Results saved to {annotation_path}")
    annotations.print_summary()

if __name__ == "__main__":
    # Choose which version to run:
    
    # For ALL images (production run):
    main()

    # For testing with sampled images (uncomment to use):
    # main_with_sampling(max_images=100)  # Test with 100 images