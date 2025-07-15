"""
Debug script for batch processing issues
Quick diagnostic to identify the problem
"""

import sys
sys.path.append('/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/utils/embryo_metada_dev_instruction')

import tempfile
import json
from pathlib import Path

def debug_batch_operations():
    """Debug the batch operations issues."""
    print("🔍 Debugging batch operations...")
    
    # Create test setup
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    
    # Create minimal SAM annotation
    sam_data = {
        "experiments": {
            "20240411": {
                "videos": {
                    "20240411_A01": {
                        "embryo_ids": ["20240411_A01_e01"],
                        "images": {
                            "20240411_A01_0000": {
                                "embryos": {
                                    "20240411_A01_e01": {
                                        "snip_id": "20240411_A01_e01_0000"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "embryo_ids": ["20240411_A01_e01"],
        "snip_ids": ["20240411_A01_e01_0000"]
    }
    
    sam_path = temp_path / "sam_annotations.json"
    with open(sam_path, 'w') as f:
        json.dump(sam_data, f)
    
    try:
        # Import and create metadata
        from embryo_metadata_refactored import EmbryoMetadata
        
        print("📋 Creating EmbryoMetadata instance...")
        em = EmbryoMetadata(
            sam_path,
            temp_path / "embryo_metadata.json",
            gen_if_no_file=True,
            verbose=True
        )
        
        print("✅ EmbryoMetadata created successfully")
        print(f"   Embryos: {len(em.data['embryos'])}")
        print(f"   Snips: {em.snip_ids}")
        
        # Test single phenotype addition first
        print("\n🧪 Testing single phenotype addition...")
        success = em.add_phenotype("20240411_A01_e01_0000", "EDEMA", "test_author")
        print(f"   Single add result: {success}")
        
        # Test single genotype addition
        print("\n🧪 Testing single genotype addition...")
        success = em.add_genotype("20240411_A01_e01", "WT", "test_author")
        print(f"   Single add result: {success}")
        
        # Test batch phenotype (minimal)
        print("\n🧪 Testing batch phenotype assignment...")
        assignments = [{
            "embryo_id": "20240411_A01_e01",
            "phenotype": "CONVERGENCE_EXTENSION", 
            "frames": "all"
        }]
        
        results = em.batch_add_phenotypes(assignments, "test_author")
        print(f"   Batch phenotype results: {results}")
        
        # Test batch genotype (minimal)
        print("\n🧪 Testing batch genotype assignment...")
        assignments = [{
            "embryo_id": "20240411_A01_e01",
            "genotype": "lmx1b"
        }]
        
        results = em.batch_add_genotypes(assignments, "test_author", overwrite=True)
        print(f"   Batch genotype results: {results}")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        temp_dir.cleanup()

if __name__ == "__main__":
    debug_batch_operations()
