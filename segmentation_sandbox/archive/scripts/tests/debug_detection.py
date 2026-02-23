#!/usr/bin/env python3
"""
Debug GroundingDINO Detection Issues
===================================

This script helps debug why GroundingDINO isn't detecting embryos by:
1. Showing empty log status
2. Testing different thresholds
3. Visualizing results on sample images
4. Providing options to clear empty log for reprocessing
"""

import sys
import argparse
from pathlib import Path
import torch

# Add project root to path
SANDBOX_ROOT = Path(__file__).parent
sys.path.append(str(SANDBOX_ROOT))

from scripts.detection_segmentation.grounded_dino_utils import (
    load_config, load_groundingdino_model, GroundedDinoAnnotations, run_inference
)
from scripts.metadata.experiment_metadata import ExperimentMetadata


def main():
    parser = argparse.ArgumentParser(description="Debug GroundingDINO detection issues")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--annotations", required=True, help="Path to annotations JSON")
    
    # Debug options
    parser.add_argument("--clear-empty-log", action="store_true", 
                       help="Clear the empty annotations log")
    parser.add_argument("--test-image", type=str,
                       help="Test detection on a specific image ID")
    parser.add_argument("--experiment-id", type=str, default="20250612_30hpf_ctrl_atf6",
                       help="Experiment ID to work with")
    parser.add_argument("--prompt", default="individual embryo",
                       help="Detection prompt")
    
    # Detection parameters for testing
    parser.add_argument("--box-threshold", type=float, default=0.35,
                       help="Box threshold for detection")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                       help="Text threshold for detection")
    
    args = parser.parse_args()

    print("🔍 DEBUG: GroundingDINO Detection Issues")
    print("=" * 50)

    # Load components
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")

    # Load metadata
    metadata_manager = ExperimentMetadata(args.metadata, verbose=True)
    
    # Initialize annotations manager
    annotations = GroundedDinoAnnotations(args.annotations, verbose=True, metadata_path=args.metadata)
    
    # Show empty log status  
    print("\n📋 EMPTY LOG STATUS")
    print("-" * 30)
    empty_info = annotations.get_empty_log_info()
    print(f"Log file: {empty_info['log_path']}")
    print(f"Exists: {empty_info['exists']}")
    print(f"Empty images recorded: {empty_info['count']}")
    if empty_info['sample_ids']:
        print(f"Sample IDs: {empty_info['sample_ids']}")
    
    # Clear empty log if requested
    if args.clear_empty_log:
        print(f"\n🗑️  CLEARING EMPTY LOG")
        print("-" * 30)
        annotations.clear_empty_log()
        print("✅ Empty log cleared")
        return
    
    # Load model for testing
    try:
        model = load_groundingdino_model(config, device=device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get sample images from the experiment
    exp_images = metadata_manager.list_images(experiment_id=args.experiment_id)
    if not exp_images:
        print(f"❌ No images found for experiment: {args.experiment_id}")
        return
        
    print(f"\n📊 EXPERIMENT INFO")
    print("-" * 30)
    print(f"Experiment: {args.experiment_id}")
    print(f"Total images: {len(exp_images)}")
    
    # Test specific image or use first few
    if args.test_image:
        if args.test_image in exp_images:
            test_images = [args.test_image]
        else:
            print(f"❌ Image {args.test_image} not found in experiment")
            return
    else:
        test_images = exp_images[:3]  # Test first 3 images
    
    print(f"Testing images: {test_images}")
    
    # Test different thresholds
    threshold_combos = [
        (0.35, 0.25),  # Default
        (0.20, 0.20),  # Lower
        (0.10, 0.15),  # Much lower
        (0.05, 0.10),  # Very low
    ]
    
    print(f"\n🔬 TESTING DETECTION WITH DIFFERENT THRESHOLDS")
    print("-" * 50)
    
    for box_thresh, text_thresh in threshold_combos:
        print(f"\n📊 Testing: box_threshold={box_thresh}, text_threshold={text_thresh}")
        print("-" * 40)
        
        total_detections = 0
        for img_id in test_images:
            # Get image path
            image_data_list = metadata_manager.get_images_for_detection(image_ids=[img_id])
            if not image_data_list:
                print(f"   ❌ Could not find path for {img_id}")
                continue
                
            img_path = Path(image_data_list[0]['image_path'])
            
            try:
                # Run inference
                boxes, logits, phrases, image_source = run_inference(
                    model, img_path, args.prompt, 
                    box_threshold=box_thresh, 
                    text_threshold=text_thresh
                )
                
                print(f"   {img_id}: {len(boxes)} detections")
                if len(boxes) > 0:
                    confidences = [f"{logit:.3f}" for logit in logits]
                    print(f"     Confidences: {confidences}")
                    
                total_detections += len(boxes)
                
            except Exception as e:
                print(f"   ❌ Error processing {img_id}: {e}")
        
        print(f"   📈 Total detections: {total_detections}")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)
    if empty_info['count'] > 0:
        print("• Many images were logged as having zero detections")
        print("• Try lower detection thresholds or different prompts")
        print("• Use --clear-empty-log to force reprocessing")
    
    print("• Check if the fine-tuned model is working correctly")
    print("• Consider using different prompts like 'embryo', 'cell', 'organism'")
    print("• Test with much lower thresholds (0.05, 0.10)")


if __name__ == "__main__":
    main()
