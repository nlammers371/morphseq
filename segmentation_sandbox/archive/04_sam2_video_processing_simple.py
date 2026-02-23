#!/usr/bin/env python3
"""
Simple SAM2 Video Processing Script
==================================

Simplified version that avoids complex import issues.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
import torch

def load_config(config_path):
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="SAM2 video processing for embryo segmentation")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--annotations", required=True, help="Path to gdino_high_quality_annotations.json (seed annotations)")
    parser.add_argument("--output", required=True, help="Path to output grounded_sam_annotations.json")
    
    # SAM2 model configuration
    parser.add_argument("--sam2-config", help="Path to SAM2 config file (overrides config file)")
    parser.add_argument("--sam2-checkpoint", help="Path to SAM2 checkpoint (overrides config file)")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    # Processing configuration
    parser.add_argument("--target-prompt", default="individual embryo", 
                       help="Target prompt for embryo detection (default: 'individual embryo')")
    parser.add_argument("--segmentation-format", default="rle", choices=["rle", "polygon"],
                       help="Format for storing segmentation masks (rle is much more compact)")
    
    # Processing limits (for testing)
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Maximum number of videos to process (for testing)")
    parser.add_argument("--video-ids", nargs="+", default=None,
                       help="Specific video IDs to process")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save-interval", type=int, default=5,
                       help="Save results every N videos (default: 5)")
    
    args = parser.parse_args()
    
    print("🎬 SAM2 Video Processing for Embryo Segmentation")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Seed annotations: {args.annotations}")
    print(f"Output: {args.output}")
    print(f"Target prompt: '{args.target_prompt}'")
    print(f"Device: {args.device}")
    print(f"Segmentation format: {args.segmentation_format}")
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
        
    if not os.path.exists(args.annotations):
        print(f"❌ Annotations file not found: {args.annotations}")
        return 1
    
    # Load configuration
    print("\n📁 Loading configuration...")
    try:
        config = load_config(args.config)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    # Check SAM2 model paths
    sam2_config_path = args.sam2_config or config.get('models', {}).get('sam2', {}).get('config')
    sam2_checkpoint_path = args.sam2_checkpoint or config.get('models', {}).get('sam2', {}).get('checkpoint')
    
    if not sam2_config_path:
        print("❌ SAM2 config path not found in config file or arguments")
        return 1
        
    if not sam2_checkpoint_path:
        print("❌ SAM2 checkpoint path not found in config file or arguments")
        return 1
    
    print(f"🔧 SAM2 Configuration:")
    print(f"   Config: {sam2_config_path}")
    print(f"   Checkpoint: {sam2_checkpoint_path}")
    
    # Now try to import and use GroundedSamAnnotations
    try:
        print("\n🚀 Attempting to import GroundedSamAnnotations...")
        
        # Add sandbox root to path
        sandbox_root = Path(__file__).parent.parent
        sys.path.insert(0, str(sandbox_root))
        
        # Try different import approaches
        try:
            from scripts.utils.sam2_utils import GroundedSamAnnotations
            print("✅ Imported GroundedSamAnnotations using scripts.utils import")
        except ImportError as e1:
            print(f"❌ scripts.utils import failed: {e1}")
            try:
                # Direct file import
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "sam2_utils", 
                    sandbox_root / "scripts/utils/sam2_utils.py"
                )
                sam2_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sam2_utils)
                GroundedSamAnnotations = sam2_utils.GroundedSamAnnotations
                print("✅ Imported GroundedSamAnnotations using direct file import")
            except Exception as e2:
                print(f"❌ Direct import also failed: {e2}")
                print("\n🔧 Troubleshooting information:")
                print(f"   Python path: {sys.path}")
                print(f"   Sam2_utils file exists: {(sandbox_root / 'scripts/utils/sam2_utils.py').exists()}")
                return 1
        
        # Initialize GroundedSamAnnotations
        print(f"\n🎯 Initializing GroundedSamAnnotations...")
        
        grounded_sam = GroundedSamAnnotations(
            filepath=args.output,
            seed_annotations_path=args.annotations,
            sam2_config=sam2_config_path,
            sam2_checkpoint=sam2_checkpoint_path,
            device=args.device,
            target_prompt=args.target_prompt,
            segmentation_format=args.segmentation_format,
            verbose=args.verbose or True
        )
        
        print("✅ GroundedSamAnnotations initialized successfully")
        
        # Continue with processing...
        print("\n📊 Starting processing pipeline...")
        print("(Full processing implementation would continue here)")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n🎉 Script completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
