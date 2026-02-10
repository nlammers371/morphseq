#!/usr/bin/env python3
"""
Debug script to test batch operations and identify issues
"""

import tempfile
import json
from pathlib import Path
import sys
import traceback

# Add current directory to path
sys.path.insert(0, '.')

def debug_batch_operations():
    """Debug the batch operations to see what's failing."""
    print("🔍 Debugging batch operations...")
    
    try:
        # Create temporary test data
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create mock SAM annotation
        sam_data = {
            "experiments": {
                "20240411": {
                    "videos": {
                        "20240411_A01": {
                            "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
                            "images": {
                                "20240411_A01_0010": {
                                    "embryos": {
                                        "20240411_A01_e01": {"snip_id": "20240411_A01_e01_0010"},
                                        "20240411_A01_e02": {"snip_id": "20240411_A01_e02_0010"}
                                    }
                                },
                                "20240411_A01_0011": {
                                    "embryos": {
                                        "20240411_A01_e01": {"snip_id": "20240411_A01_e01_0011"},
                                        "20240411_A01_e02": {"snip_id": "20240411_A01_e02_0011"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
            "snip_ids": ["20240411_A01_e01_0010", "20240411_A01_e01_0011", 
                        "20240411_A01_e02_0010", "20240411_A01_e02_0011"]
        }
        
        sam_path = temp_path / "sam_annotations.json"
        with open(sam_path, 'w') as f:
            json.dump(sam_data, f, indent=2)
        
        print(f"📁 Created test SAM file: {sam_path}")
        
        # Import and test
        from embryo_metadata_refactored import EmbryoMetadata
        
        print("📦 Creating EmbryoMetadata instance...")
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=True)
        
        print(f"✅ EmbryoMetadata created: {em}")
        print(f"🔍 Embryo IDs: {em.get_embryo_ids()}")
        print(f"🔍 Snip IDs: {em.get_snip_ids()}")
        
        # Test basic phenotype addition
        print("\n🧪 Testing basic phenotype addition...")
        try:
            result = em.add_phenotype("20240411_A01_e01_0010", "EDEMA", "test_user")
            print(f"✅ Basic phenotype addition: {result}")
        except Exception as e:
            print(f"❌ Basic phenotype addition failed: {e}")
            traceback.print_exc()
        
        # Test basic genotype addition
        print("\n🧪 Testing basic genotype addition...")
        try:
            result = em.add_genotype("20240411_A01_e01", "WT", "test_user")
            print(f"✅ Basic genotype addition: {result}")
        except Exception as e:
            print(f"❌ Basic genotype addition failed: {e}")
            traceback.print_exc()
        
        # Test batch phenotype assignment
        print("\n🧪 Testing batch phenotype assignment...")
        try:
            assignments = [
                {
                    "embryo_id": "20240411_A01_e01",
                    "phenotype": "CONVERGENCE_EXTENSION",
                    "frames": "all"
                }
            ]
            
            print(f"📋 Assignment: {assignments}")
            results = em.batch_add_phenotypes(assignments, "test_user")
            print(f"✅ Batch phenotype results: {results}")
            
        except Exception as e:
            print(f"❌ Batch phenotype assignment failed: {e}")
            traceback.print_exc()
        
        # Test batch genotype assignment
        print("\n🧪 Testing batch genotype assignment...")
        try:
            assignments = [
                {"embryo_id": "20240411_A01_e02", "genotype": "mutant"}
            ]
            
            results = em.batch_add_genotypes(assignments, "test_user")
            print(f"✅ Batch genotype results: {results}")
            
        except Exception as e:
            print(f"❌ Batch genotype assignment failed: {e}")
            traceback.print_exc()
        
        print("\n✅ Debug completed!")
        
    except Exception as e:
        print(f"💥 Debug failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_batch_operations()
