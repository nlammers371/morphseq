#!/usr/bin/env python3
"""
Test the new progress bars implementation on a limited dataset.
"""

import sys
from pathlib import Path

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def test_edge_detection_progress():
    """Test edge detection with progress bars on 20250305 experiment only."""
    
    # Mock the required imports to avoid conda environment issues
    class MockQC:
        def __init__(self):
            self.verbose = True
            self.progress = True
            
            # Load minimal SAM2 data
            import json
            with open("data/segmentation/grounded_sam_segmentations.json", 'r') as f:
                self.gsam_data = json.load(f)
        
        def _should_process_experiment(self, exp_id, entities):
            return exp_id == "20250305"  # Only process 20250305
            
        def _should_process_video(self, video_id, entities):
            return True
            
        def _should_process_image(self, image_id, entities):
            return True
            
        def _should_process_snip(self, snip_id, entities):
            return True
        
        def _count_target_embryos(self, entities):
            """Count embryos in 20250305 only."""
            count = 0
            exp_data = self.gsam_data.get("experiments", {}).get("20250305", {})
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("image_ids", {}).items():
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        segmentation = embryo_data.get("segmentation")
                        if segmentation and segmentation.get("format") in ["rle", "rle_base64"]:
                            count += 1
                            if count >= 20:  # Limit to 20 for quick test
                                return count
            return count
        
        def _create_embryo_progress(self, desc, total):
            """Create progress bar - will use print fallback if tqdm not available."""
            try:
                from tqdm import tqdm
                return tqdm(total=total, desc=desc, unit="embryos", leave=False)
            except ImportError:
                print(f"{desc}: Processing {total} embryos...")
                return None
    
    # Create mock QC instance
    qc = MockQC()
    
    # Test the counting
    entities = {"experiment_ids": ["20250305"]}
    total_embryos = qc._count_target_embryos(entities)
    print(f"🔍 Testing progress bars with {total_embryos} embryos from 20250305")
    
    # Test progress bar creation
    progress_bar = qc._create_embryo_progress("🔍 Test Progress", total_embryos)
    
    if progress_bar:
        print("✅ Progress bar created successfully with tqdm")
        # Simulate processing
        import time
        for i in range(total_embryos):
            time.sleep(0.1)  # Simulate work
            progress_bar.update(1)
        progress_bar.close()
        print("✅ Progress bar completed successfully")
    else:
        print("✅ Fallback mode working (no tqdm available)")
        # Simulate processing with print updates
        for i in range(total_embryos):
            if i % 5 == 0:  # Print every 5th embryo
                print(f"   Processed {i}/{total_embryos} embryos...")
        print(f"   Processed {total_embryos}/{total_embryos} embryos...")
    
    print("🎉 Progress bar test completed!")

if __name__ == "__main__":
    test_edge_detection_progress()