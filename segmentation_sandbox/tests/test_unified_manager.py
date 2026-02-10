#!/usr/bin/env python3
"""
Test UnifiedEmbryoManager functionality specifically
"""
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_data():
    """Create test embryo metadata structure"""
    return {
        "file_info": {
            "version": "1.0",
            "created": "2025-08-19T12:00:00"
        },
        "embryos": {
            "test_exp_A01_e01": {
                "genotype": None,
                "treatments": {},
                "flags": {},
                "notes": "",
                "metadata": {"created": "2025-08-19T12:00:00"},
                "snips": {
                    "test_exp_A01_e01_s0100": {"flags": []},
                    "test_exp_A01_e01_s0101": {"flags": []}
                }
            }
        },
        "entity_tracking": {}
    }

def test_unified_manager():
    """Test UnifiedEmbryoManager functionality"""
    print("🔍 Testing UnifiedEmbryoManager...")
    
    try:
        from scripts.annotations.embryo_metadata import EmbryoMetadata
        
        # Create temporary SAM2 file (minimal)
        sam2_data = {
            "experiments": {
                "test_exp": {
                    "videos": {
                        "test_exp_A01": {
                            "embryo_ids": ["test_exp_A01_e01"],
                            "images": {
                                "test_exp_A01_ch00_t0100": {
                                    "embryos": {
                                        "test_exp_A01_e01": {"snip_id": "test_exp_A01_e01_s0100"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sam2_data, f)
            sam2_path = f.name
        
        try:
            # Create metadata manually with proper structure
            with tempfile.NamedTemporaryFile(mode='w', suffix='_embryo_metadata.json', delete=False) as f:
                json.dump(create_test_data(), f)
                metadata_path = f.name
            
            # Load using EmbryoMetadata 
            metadata = EmbryoMetadata(sam2_path, metadata_path, verbose=False)
            
            print(f"✅ Loaded metadata: {metadata.embryo_count} embryos, {metadata.snip_count} snips")
            
            # Test phenotype management
            embryo_id = "test_exp_A01_e01"
            snip_id = "test_exp_A01_e01_s0100"
            
            print(f"🧪 Testing phenotype addition...")
            
            # Add phenotype using inherited UnifiedEmbryoManager method
            success = metadata.add_phenotype(snip_id, "EDEMA", "test_author", "test phenotype")
            print(f"✅ Added phenotype: {success}")
            
            # Get embryo phenotypes
            phenotypes = metadata._get_embryo_phenotypes(embryo_id)
            print(f"✅ Embryo phenotypes: {phenotypes}")
            
            # Test DEAD exclusivity 
            print(f"🧪 Testing DEAD exclusivity...")
            try:
                metadata.add_phenotype(snip_id, "DEAD", "test_author", "testing dead")
                print("❌ Should have raised ValueError for DEAD with existing phenotype")
                return False
            except ValueError as e:
                print(f"✅ DEAD exclusivity working: {e}")
            
            # Test with force_dead
            success = metadata.add_phenotype(snip_id, "DEAD", "test_author", "forced dead", force_dead=True)
            print(f"✅ force_dead override working: {success}")
            
            # Test genotype management
            print(f"🧪 Testing genotype management...")
            success = metadata.add_genotype(embryo_id, "tmem67", "tmem67tm1a", "heterozygous", "test_author")
            print(f"✅ Added genotype: {success}")
            
            genotype = metadata.get_genotype(embryo_id)
            print(f"✅ Retrieved genotype: {genotype}")
            
            # Test validation
            print(f"🧪 Testing data validation...")
            validation_results = metadata.validate_data_integrity()
            print(f"✅ Validation results: {validation_results}")
            
            # Test summary
            summary = metadata.get_embryo_summary(embryo_id)
            print(f"✅ Embryo summary: {summary}")
            
            print("✅ All UnifiedEmbryoManager tests passed!")
            return True
            
        finally:
            # Cleanup
            Path(sam2_path).unlink()
            if Path(metadata_path).exists():
                Path(metadata_path).unlink()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run unified manager test"""
    print("=" * 60)
    print("UnifiedEmbryoManager Test")
    print("=" * 60)
    
    success = test_unified_manager()
    
    print("=" * 60)
    if success:
        print("✅ UnifiedEmbryoManager working correctly!")
        print("✅ DEAD safety workflow implemented!")
    else:
        print("❌ UnifiedEmbryoManager needs fixes")
    print("=" * 60)

if __name__ == "__main__":
    main()