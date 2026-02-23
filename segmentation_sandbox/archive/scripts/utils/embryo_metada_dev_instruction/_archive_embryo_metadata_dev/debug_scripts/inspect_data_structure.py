#!/usr/bin/env python3
"""
Quick script to inspect the actual data structure after batch operations
"""

import tempfile
import json
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, '.')

def inspect_data_structure():
    """Inspect the actual data structure after operations."""
    print("🔍 Inspecting data structure after batch operations...")
    
    # Create temporary test data
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create mock SAM annotation
    sam_data = {
        "experiments": {
            "20240411": {
                "videos": {
                    "20240411_A01": {
                        "embryo_ids": ["20240411_A01_e01"],
                        "images": {
                            "20240411_A01_0010": {
                                "embryos": {
                                    "20240411_A01_e01": {"snip_id": "20240411_A01_e01_0010"}
                                }
                            },
                            "20240411_A01_0011": {
                                "embryos": {
                                    "20240411_A01_e01": {"snip_id": "20240411_A01_e01_0011"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "embryo_ids": ["20240411_A01_e01"],
        "snip_ids": ["20240411_A01_e01_0010", "20240411_A01_e01_0011"]
    }
    
    sam_path = temp_path / "sam_annotations.json"
    with open(sam_path, 'w') as f:
        json.dump(sam_data, f, indent=2)
    
    # Import and test
    from embryo_metadata_refactored import EmbryoMetadata
    
    em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
    
    # Add a phenotype
    em.add_phenotype("20240411_A01_e01_0010", "EDEMA", "test_user")
    
    # Add a genotype
    em.add_genotype("20240411_A01_e01", "WT", "mutant")
    
    # Inspect structures
    embryo_data = em.data["embryos"]["20240411_A01_e01"]
    
    print("\n📋 EMBRYO DATA STRUCTURE:")
    print(json.dumps(embryo_data, indent=2))
    
    print("\n📋 GENOTYPES STRUCTURE:")
    print(f"genotypes keys: {list(embryo_data.get('genotypes', {}).keys())}")
    if embryo_data.get('genotypes'):
        for key, value in embryo_data['genotypes'].items():
            print(f"  {key}: {value}")
    
    print("\n📋 SNIPS STRUCTURE:")
    snip_data = embryo_data.get("snips", {})
    for snip_id, snip_info in snip_data.items():
        print(f"Snip {snip_id}:")
        print(f"  phenotypes: {snip_info.get('phenotypes', {})}")
        if snip_info.get('phenotypes'):
            for key, value in snip_info['phenotypes'].items():
                print(f"    {key}: {value}")

if __name__ == "__main__":
    inspect_data_structure()
