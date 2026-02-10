#!/usr/bin/env python3
"""
Test EmbryoMetadata with real SAM2 segmentation data
"""
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_real_sam2_import():
    """Test importing real SAM2 segmentation data"""
    print("🔍 Testing real SAM2 data import...")
    
    sam2_path = project_root / "data" / "segmentation" / "grounded_sam_segmentations.json"
    
    if not sam2_path.exists():
        print(f"❌ SAM2 file not found: {sam2_path}")
        return False
    
    try:
        from scripts.annotations.embryo_metadata import EmbryoMetadata
        
        # Create temporary metadata file path to avoid overwriting anything
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_metadata_path = Path(temp_dir) / "test_embryo_metadata.json"
            
            print(f"📂 Using SAM2 file: {sam2_path}")
            print(f"📂 Creating metadata at: {temp_metadata_path}")
            
            # Create metadata from SAM2 - this tests the _create_from_sam method
            metadata = EmbryoMetadata(
                sam_annotation_path=sam2_path,
                embryo_metadata_path=temp_metadata_path, 
                gen_if_no_file=True,
                verbose=False
            )
            
            print("✅ EmbryoMetadata created from real SAM2 data")
            print(f"   Embryo count: {metadata.embryo_count}")
            print(f"   Snip count: {metadata.snip_count}")
            
            # Test entity counts
            entity_counts = metadata.get_entity_counts()
            print(f"✅ Entity counts: {entity_counts}")
            
            # Test some basic queries
            available_snips = metadata.get_available_snips()
            print(f"✅ Available snips: {len(available_snips)} (showing first 3: {available_snips[:3]})")
            
            # Test data access helpers with real data
            if available_snips:
                first_snip = available_snips[0]
                embryo_id = metadata.get_embryo_id_from_snip(first_snip)
                
                if embryo_id:
                    print(f"✅ First snip: {first_snip} → embryo: {embryo_id}")
                    
                    # Test helpers
                    embryo_data = metadata._get_embryo_data(embryo_id)
                    snip_data = metadata._get_snip_data(first_snip)
                    
                    print(f"✅ Embryo data keys: {list(embryo_data.keys())}")
                    print(f"✅ Snip data keys: {list(snip_data.keys())}")
                else:
                    print("⚠️ Could not get embryo ID from first snip")
            
            # Test save functionality
            metadata.save(backup=False)
            print("✅ Save functionality working")
            
            # Verify file was created and has content
            if temp_metadata_path.exists():
                size = temp_metadata_path.stat().st_size
                print(f"✅ Metadata file created: {size / 1024:.1f} KB")
            else:
                print("❌ Metadata file was not created")
                return False
            
            print("✅ All real SAM2 import tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run real SAM2 import test"""
    print("=" * 60)
    print("Real SAM2 Import Test")
    print("=" * 60)
    
    success = test_real_sam2_import()
    
    print("=" * 60)
    if success:
        print("✅ EmbryoMetadata works correctly with real SAM2 data!")
        print("✅ Ready to implement DEAD safety workflow")
    else:
        print("❌ Real SAM2 import needs fixes before proceeding")
    print("=" * 60)

if __name__ == "__main__":
    main()