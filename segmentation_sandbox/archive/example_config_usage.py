#!/usr/bin/env python
"""
Example usage of the pipeline configuration with morphseq_data_dir base directory.

This script demonstrates how to use the configuration system with the new
base directory feature for easier path management.
"""

from utils.config_utils import load_config

def main():
    """Demonstrate configuration usage."""
    # Load configuration
    config = load_config()
    
    print("=== MorphSeq Pipeline Configuration Example ===\n")
    
    # Basic configuration access
    print("1. Basic parameter access:")
    print(f"   Detection box threshold: {config.get('detection.box_threshold')}")
    print(f"   Processing max frames: {config.get('processing.max_frames_per_video')}")
    print()
    
    # Path management with base directory
    print("2. Path management with morphseq_data_dir:")
    print(f"   Base morphseq directory: {config.get_morphseq_data_path()}")
    print(f"   Stitched images: {config.get_stitched_images_dir()}")
    print()
    
    # Mask directories
    print("3. Mask directory access:")
    for mask_type in ['embryo', 'yolk', 'via']:
        print(f"   {mask_type.capitalize()} masks: {config.get_mask_dir(mask_type)}")
    print()
    
    # Model directories (sandbox-relative)
    print("4. Model directories (sandbox-relative):")
    print(f"   Models base: {config.get_model_dir()}")
    print(f"   GroundingDINO: {config.get('paths.groundingdino_path')}")
    print(f"   SAM2: {config.get('paths.sam2_path')}")
    print()
    
    # Output directories (sandbox-relative)
    print("5. Output directories (sandbox-relative):")
    print(f"   Intermediate: {config.get('paths.intermediate_dir')}")
    print(f"   Final: {config.get('paths.final_dir')}")
    print(f"   Logs: {config.get('paths.logs_dir')}")
    print()
    
    # Convenience methods for file paths
    print("6. Convenience methods for file paths:")
    print(f"   Example intermediate file: {config.get_intermediate_path('detections.json')}")
    print(f"   Example final file: {config.get_final_path('embryo_trajectories.csv')}")
    print(f"   Example log file: {config.get_log_path('pipeline.log')}")
    print()
    
    # Custom subpaths within morphseq data
    print("7. Custom subpaths within morphseq data:")
    print(f"   Custom experiment: {config.get_morphseq_data_path('experiments/test_batch_1')}")
    print(f"   Custom analysis: {config.get_morphseq_data_path('analysis_results/2024')}")
    print()
    
    # Model configurations
    print("8. Model configurations:")
    grounding_config = config.get_model_config('groundingdino')
    sam2_config = config.get_model_config('sam2')
    print(f"   GroundingDINO config file: {grounding_config.get('config', 'Not found')}")
    print(f"   SAM2 checkpoint: {sam2_config.get('checkpoint', 'Not found')}")
    print()
    
    print("=== Configuration loaded successfully! ===")

if __name__ == "__main__":
    main()
