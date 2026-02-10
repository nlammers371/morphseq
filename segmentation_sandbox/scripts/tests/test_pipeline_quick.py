#!/usr/bin/env python3
"""
Quick Test of Module 2 GroundingDINO Pipeline
==============================================

Test the new Module 2 pipeline script with a small subset of real data.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🧪 Testing Module 2 GroundingDINO Pipeline")
    print("=" * 50)
    
    # Test parameters
    config_path = "configs/pipeline_config.yaml"
    metadata_path = "data/raw_data_organized/experiment_metadata.json"
    annotations_path = "temp/test_pipeline_annotations.json"
    
    # Make sure temp directory exists
    Path("temp").mkdir(exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/pipelines/03_gdino_detection_with_filtering_modern.py",
        "--config", config_path,
        "--metadata", metadata_path,
        "--annotations", annotations_path,
        "--experiment-ids", "20250612_30hpf_ctrl_atf6",  # Focus on one experiment
        "--max-images", "3",  # Just test 3 images
        "--confidence-threshold", "0.4",
        "--iou-threshold", "0.3",
        "--auto-save-interval", "1"
    ]
    
    print("🚀 Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("📤 STDERR:")
            print(result.stderr)
        
        print(f"\n✅ Command completed with exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("🎉 Pipeline test successful!")
            
            # Check if output file was created
            if Path(annotations_path).exists():
                print(f"📄 Output file created: {annotations_path}")
                
                # Quick peek at the file
                with open(annotations_path, 'r') as f:
                    import json
                    data = json.load(f)
                    images = data.get("images", {})
                    hq_annotations = data.get("high_quality_annotations", {})
                    entity_tracker = data.get("entity_tracker", {})
                    
                    print(f"   📊 Images processed: {len(images)}")
                    print(f"   📊 High-quality experiments: {len(hq_annotations)}")
                    print(f"   📊 Entity tracker step: {entity_tracker.get('pipeline_step', 'None')}")
            else:
                print("⚠️  Output file not created")
        else:
            print("❌ Pipeline test failed!")
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running command: {e}")

if __name__ == "__main__":
    main()
