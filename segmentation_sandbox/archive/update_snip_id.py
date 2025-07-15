#!/usr/bin/env python3
"""
Update GroundedSamAnnotations JSON files to add snip_id fields.

This script adds snip_id to existing GroundedSamAnnotations JSON files that were
created before the snip_id feature was implemented.

Usage:
    python update_snip_id.py input_file.json [output_file.json]
    
If output_file is not specified, the input file will be backed up and updated in place.
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

def extract_frame_suffix(image_id: str) -> str:
    """Extract frame suffix from image_id (e.g., '0000' from '20240411_A01_0000')."""
    return image_id.split('_')[-1]

def create_snip_id(embryo_id: str, image_id: str) -> str:
    """Create snip_id by combining embryo_id with frame suffix from image_id."""
    frame_suffix = extract_frame_suffix(image_id)
    return f"{embryo_id}_{frame_suffix}"

def update_grounded_sam_annotations_with_snip_id(data: dict, verbose: bool = True) -> dict:
    """
    Update GroundedSamAnnotations data structure to include snip_id fields.
    
    Args:
        data: The loaded JSON data from GroundedSamAnnotations file
        verbose: Print progress information
        
    Returns:
        Updated data with snip_id fields added
    """
    if verbose:
        print("🔄 Updating GroundedSamAnnotations with snip_id fields...")
    
    total_embryos_updated = 0
    total_videos_processed = 0
    
    # Iterate through experiments
    experiments = data.get("experiments", {})
    for exp_id, exp_data in experiments.items():
        if verbose:
            print(f"   📂 Processing experiment: {exp_id}")
        
        # Iterate through videos in this experiment
        videos = exp_data.get("videos", {})
        for video_id, video_data in videos.items():
            if verbose:
                print(f"      🎬 Processing video: {video_id}")
            
            embryos_in_video = 0
            
            # Iterate through images in this video
            images = video_data.get("images", {})
            for image_id, image_data in images.items():
                
                # Iterate through embryos in this image
                embryos = image_data.get("embryos", {})
                for embryo_id, embryo_data in embryos.items():
                    
                    # Check if snip_id already exists
                    if "snip_id" not in embryo_data:
                        # Create and add snip_id
                        snip_id = create_snip_id(embryo_id, image_id)
                        embryo_data["snip_id"] = snip_id
                        total_embryos_updated += 1
                        embryos_in_video += 1
                        
                        if verbose and embryos_in_video <= 3:  # Show first few examples
                            print(f"         🧬 Added snip_id '{snip_id}' to embryo '{embryo_id}' in image '{image_id}'")
                    
            if verbose and embryos_in_video > 3:
                print(f"         🧬 ... and {embryos_in_video - 3} more embryos in this video")
                
            total_videos_processed += 1
    
    # Update metadata
    data["last_updated"] = datetime.now().isoformat()
    
    # Add snip_ids to global list if not present
    if "snip_ids" not in data:
        data["snip_ids"] = []
    
    # Collect all snip_ids from the data
    all_snip_ids = set()
    for exp_data in experiments.values():
        for video_data in exp_data.get("videos", {}).values():
            for image_data in video_data.get("images", {}).values():
                for embryo_data in image_data.get("embryos", {}).values():
                    if "snip_id" in embryo_data:
                        all_snip_ids.add(embryo_data["snip_id"])
    
    # Update the global snip_ids list
    data["snip_ids"] = sorted(list(all_snip_ids))
    
    if verbose:
        print(f"✅ Update complete!")
        print(f"   Videos processed: {total_videos_processed}")
        print(f"   Embryos updated: {total_embryos_updated}")
        print(f"   Total snip_ids: {len(data['snip_ids'])}")
    
    return data

def main():
    """Main function to handle command line arguments and file processing."""
    if len(sys.argv) < 2:
        print("Usage: python update_snip_id.py input_file.json [output_file.json]")
        print("If output_file is not specified, input file will be backed up and updated in place.")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file
    
    # Validate input file
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    
    if not input_file.suffix.lower() == '.json':
        print(f"❌ Error: Input file must be a JSON file: {input_file}")
        sys.exit(1)
    
    print(f"📁 Loading GroundedSamAnnotations from: {input_file}")
    
    try:
        # Load the JSON data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Validate that this looks like a GroundedSamAnnotations file
        if "experiments" not in data:
            print(f"❌ Error: File doesn't appear to be a GroundedSamAnnotations file (missing 'experiments' key)")
            sys.exit(1)
        
        print(f"✅ Loaded {len(data.get('experiments', {}))} experiments")
        
        # Create backup if updating in place
        if output_file == input_file:
            backup_file = input_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            print(f"💾 Creating backup: {backup_file}")
            shutil.copy2(input_file, backup_file)
        
        # Update the data
        updated_data = update_grounded_sam_annotations_with_snip_id(data, verbose=True)
        
        # Save the updated data
        print(f"💾 Saving updated data to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        print(f"✅ Successfully updated GroundedSamAnnotations file!")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")
        if output_file == input_file:
            print(f"   Backup: {backup_file}")
            
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()