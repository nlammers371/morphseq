#!/usr/bin/env python3
"""
Debug script to test QC on grounded_sam_segmentations.json with comprehensive debug output.
"""

import json
import sys
from pathlib import Path

# Add script paths
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def remove_qc_flags_from_file(gsam_path):
    """Remove existing QC flags from the GSAM file."""
    print(f"DEBUG: Loading file {gsam_path}")
    with open(gsam_path, 'r') as f:
        data = json.load(f)
    
    print(f"DEBUG: Original top-level keys: {list(data.keys())}")
    
    # Remove flags if they exist
    if "flags" in data:
        print("DEBUG: Removing existing 'flags' section")
        del data["flags"]
    else:
        print("DEBUG: No 'flags' section found to remove")
    
    # Create backup
    backup_path = f"{gsam_path}.backup_no_flags"
    print(f"DEBUG: Creating backup at {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Write back the cleaned version
    print(f"DEBUG: Writing cleaned version back to {gsam_path}")
    with open(gsam_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("DEBUG: Flags removed successfully")
    return data

def explore_data_structure(gsam_path, max_samples=2):
    """Explore the GSAM data structure to understand its contents."""
    print(f"\nDEBUG: Exploring data structure of {gsam_path}")
    print("=" * 60)
    
    with open(gsam_path, 'r') as f:
        data = json.load(f)
    
    print(f"Top-level keys: {list(data.keys())}")
    
    experiments = data.get("experiments", {})
    print(f"Total experiments: {len(experiments)}")
    
    if not experiments:
        print("❌ No experiments found!")
        return
    
    # Sample experiments
    sample_count = 0
    for exp_id, exp_data in experiments.items():
        if sample_count >= max_samples:
            print(f"... (showing first {max_samples} experiments)")
            break
            
        print(f"\n📁 Experiment: {exp_id}")
        videos = exp_data.get("videos", {})
        print(f"   Videos: {len(videos)}")
        
        if not videos:
            print("   ❌ No videos found!")
            continue
        
        # Sample videos
        video_sample_count = 0
        for video_id, video_data in videos.items():
            if video_sample_count >= max_samples:
                print(f"   ... (showing first {max_samples} videos)")
                break
                
            print(f"   📹 Video: {video_id}")
            images = video_data.get("images", {})
            print(f"      Images: {len(images)}")
            
            if not images:
                print("      ❌ No images found!")
                continue
            
            # Sample first few images
            image_list = list(images.items())[:3]
            total_embryos = 0
            
            for image_id, image_data in image_list:
                embryos = image_data.get("embryos", {})
                total_embryos += len(embryos)
                print(f"      🖼️ {image_id}: {len(embryos)} embryos")
                
                if embryos:
                    # Show first embryo details
                    first_embryo_id, first_embryo_data = list(embryos.items())[0]
                    snip_id = first_embryo_data.get("snip_id")
                    area = first_embryo_data.get("area")
                    has_segmentation = "segmentation" in first_embryo_data
                    print(f"         Example embryo {first_embryo_id}: snip_id={snip_id}, area={area}, has_seg={has_segmentation}")
            
            print(f"      Total embryos in first 3 images: {total_embryos}")
            video_sample_count += 1
        
        sample_count += 1
    
    print("\n" + "=" * 60)

def main():
    gsam_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/segmentation/grounded_sam_segmentations.json"
    
    print("DEBUG: Starting QC debug test")
    print(f"DEBUG: Target file: {gsam_path}")
    
    # Check if file exists
    if not Path(gsam_path).exists():
        print(f"ERROR: File {gsam_path} does not exist!")
        return
    
    # Explore data structure first
    explore_data_structure(gsam_path)
    
    # Remove existing QC flags
    print("\nDEBUG: Removing existing QC flags...")
    data = remove_qc_flags_from_file(gsam_path)
    
    # Import QC class
    try:
        from detection_segmentation.gsam_quality_control import GSAMQualityControl
        print("DEBUG: Successfully imported GSAMQualityControl")
    except ImportError as e:
        print(f"ERROR: Failed to import GSAMQualityControl: {e}")
        return
    
    # Initialize QC with debug output
    print("\nDEBUG: Initializing GSAMQualityControl...")
    qc = GSAMQualityControl(gsam_path, verbose=True, progress=True)
    
    # Get actual experiment IDs from the file
    actual_experiments = list(data.get("experiments", {}).keys())
    print(f"DEBUG: Actual experiment IDs in file: {actual_experiments}")
    
    # Force processing by providing the correct experiment IDs
    target_entities = {
        "experiment_ids": actual_experiments,
        "video_ids": [],
        "image_ids": [],
        "snip_ids": []
    }
    
    print(f"DEBUG: Using target_entities: {target_entities}")
    
    # Run QC with the correct experiment IDs
    print("\nDEBUG: Running QC checks...")
    try:
        qc.run_all_checks(
            author="debug_test", 
            target_entities=target_entities,
            save_in_place=False  # Don't save yet, just test
        )
        print("DEBUG: QC checks completed successfully!")
        
        # Print summary
        qc.print_summary()
        
    except Exception as e:
        print(f"ERROR during QC processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
