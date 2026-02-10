#!/usr/bin/env python3
"""
Test Module 7 Integration Layer with Real Files

This script tests the SAM annotation integration using the real files
specified by the user.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from embryo_metadata_refactored import EmbryoMetadata
from embryo_metadata_integration import (
    SamAnnotationIntegration, 
    GsamIdManager,
    create_embryo_metadata_from_sam
)

def test_integration_with_real_files():
    """Test integration with the real SAM and metadata files."""
    
    # File paths as specified by user
    sam_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json")
    metadata_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/embryo_metadata/embryo_metadata_finetuned.json")
    
    print("🧪 Testing Module 7 Integration Layer with Real Files")
    print("=" * 60)
    
    # Test 1: Check if files exist
    print(f"📁 SAM file exists: {sam_path.exists()}")
    print(f"📁 Metadata file exists: {metadata_path.exists()}")
    
    if not sam_path.exists():
        print(f"❌ SAM file not found: {sam_path}")
        return False
    
    # Test 2: Load SAM annotation
    try:
        print("\n🔍 Loading SAM annotation...")
        sam_data = SamAnnotationIntegration.load_sam_annotations(sam_path)
        print(f"✅ SAM annotation loaded successfully")
        
        # Basic info
        experiments = sam_data.get("experiments", {})
        print(f"📊 Experiments found: {len(experiments)}")
        
        # Extract structure
        embryo_structure = SamAnnotationIntegration.extract_embryo_structure(sam_data)
        total_snips = sum(len(emb['snips']) for emb in embryo_structure.values())
        print(f"🔬 Embryos found: {len(embryo_structure)}")
        print(f"📷 Snips found: {total_snips}")
        
    except Exception as e:
        print(f"❌ Failed to load SAM annotation: {e}")
        return False
    
    # Test 3: GSAM ID management
    try:
        print("\n🆔 Testing GSAM ID management...")
        existing_id = GsamIdManager.get_gsam_id_from_sam(sam_path)
        
        if existing_id:
            print(f"✅ Existing GSAM ID found: {existing_id}")
        else:
            print("🆕 No existing GSAM ID, will create one")
            
    except Exception as e:
        print(f"❌ Failed GSAM ID check: {e}")
        return False
    
    # Test 4: Test with existing metadata or create new
    try:
        if metadata_path.exists():
            print(f"\n📖 Loading existing metadata from: {metadata_path}")
            em = EmbryoMetadata(sam_path, metadata_path, verbose=True)
            print("✅ Existing metadata loaded")
            
            # Test linking
            gsam_id = em.link_to_sam_annotation(sam_path)
            print(f"🔗 Linked with GSAM ID: {gsam_id}")
            
        else:
            print(f"\n🆕 Creating new metadata at: {metadata_path}")
            
            # Ensure directory exists
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create from SAM
            em = create_embryo_metadata_from_sam(sam_path, metadata_path, verbose=True)
            print("✅ New metadata created and linked")
        
        # Test SAM features
        snip_ids = em.get_snip_ids()
        if snip_ids:
            test_snip = snip_ids[0]
            print(f"\n🔍 Testing SAM features for snip: {test_snip}")
            features = em.get_sam_features_for_snip(test_snip)
            if features:
                print(f"✅ SAM features retrieved: {list(features.keys())}")
            else:
                print("⚠️ No SAM features found (this may be normal)")
        
        # Test configuration inheritance
        print("\n🔧 Testing configuration inheritance...")
        em.inherit_sam_configs()
        
        config = em.data.get("config", {})
        if config:
            print(f"✅ Configuration sections: {list(config.keys())}")
        else:
            print("⚠️ No configuration inherited")
        
        # Save
        em.save()
        print(f"💾 Metadata saved to: {metadata_path}")
        
        # Final verification
        print(f"\n📊 Final statistics:")
        print(f"   Embryos: {len(em.data['embryos'])}")
        total_snips = sum(len(emb['snips']) for emb in em.data['embryos'].values())
        print(f"   Snips: {total_snips}")
        
        gsam_id = em.data.get("file_info", {}).get("gsam_annotation_id")
        if gsam_id:
            print(f"   GSAM ID: {gsam_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed metadata operations: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration_with_real_files()
    
    if success:
        print("\n🎉 Integration test completed successfully!")
        print("✅ Module 7 Integration Layer working correctly")
    else:
        print("\n❌ Integration test failed")
        sys.exit(1)
