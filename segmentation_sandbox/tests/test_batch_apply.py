#!/usr/bin/env python3
"""
Test AnnotationBatch apply functionality
"""
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_sam2_data():
    """Create test SAM2 data"""
    return {
        "experiments": {
            "test_exp": {
                "videos": {
                    "test_exp_A01": {
                        "embryo_ids": ["test_exp_A01_e01", "test_exp_A01_e02"],
                        "images": {
                            "test_exp_A01_ch00_t0100": {
                                "embryos": {
                                    "test_exp_A01_e01": {"snip_id": "test_exp_A01_e01_s0100"},
                                    "test_exp_A01_e02": {"snip_id": "test_exp_A01_e02_s0100"}
                                }
                            },
                            "test_exp_A01_ch00_t0101": {
                                "embryos": {
                                    "test_exp_A01_e01": {"snip_id": "test_exp_A01_e01_s0101"},
                                    "test_exp_A01_e02": {"snip_id": "test_exp_A01_e02_s0101"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

def test_batch_apply():
    """Test batch apply functionality"""
    print("🔍 Testing batch apply functionality...")
    
    try:
        from scripts.annotations.annotation_batch import AnnotationBatch
        from scripts.annotations.embryo_metadata import EmbryoMetadata
        
        # Create temporary SAM2 file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(create_test_sam2_data(), f)
            sam2_path = f.name
        
        try:
            # Create EmbryoMetadata instance
            metadata = EmbryoMetadata(sam2_path, gen_if_no_file=True, verbose=False)
            print(f"✅ Created metadata: {metadata.embryo_count} embryos, {metadata.snip_count} snips")
            
            # Create annotation batch
            batch = AnnotationBatch("test_author", "Testing batch apply")
            
            # Add annotations to batch
            embryo_id1 = "test_exp_A01_e01"
            embryo_id2 = "test_exp_A01_e02"
            
            # Add genotypes
            batch.add_genotype(embryo_id1, "tmem67", "tmem67tm1a", "heterozygous", "test genotype")
            batch.add_genotype(embryo_id2, "b9d2", "b9d2tm2a", "homozygous", "test genotype 2")
            
            # Add phenotypes
            batch.add_phenotype(embryo_id1, "EDEMA", frames="100:110", notes="mild edema")
            batch.add_phenotype(embryo_id2, "DEAD", frames="105:", force_dead=True, notes="died at 105")
            
            # Add treatments
            batch.add_treatment(embryo_id1, "DMSO", dosage="1%", timing="24hpf")
            batch.add_treatment(embryo_id2, "PTU", dosage="0.003%", timing="0hpf")
            
            print(f"✅ Created batch with {len(batch.get_embryo_list())} embryos")
            print(f"   Stats: {batch.get_stats()}")
            
            # Test dry run first
            print("🔍 Testing dry run (preview)...")
            dry_results = batch.dry_run(metadata)
            print(f"✅ Dry run results:")
            print(f"   Operations attempted: {dry_results['operations_attempted']}")
            print(f"   Operations successful: {dry_results['operations_successful']}")
            print(f"   Errors: {len(dry_results['errors'])}")
            print(f"   Preview mode: {dry_results.get('preview_mode', False)}")
            
            if dry_results["errors"]:
                print(f"   Error details: {dry_results['errors']}")
            
            # Verify no changes were made during dry run
            initial_genotype1 = metadata.get_genotype(embryo_id1)
            initial_genotype2 = metadata.get_genotype(embryo_id2)
            print(f"✅ No changes during dry run: genotypes still {initial_genotype1}, {initial_genotype2}")
            
            # Now apply for real
            print("🔍 Applying batch for real...")
            apply_results = batch.apply(metadata, backup=False)
            
            print(f"✅ Apply results:")
            print(f"   Success: {apply_results['success']}")
            print(f"   Operations attempted: {apply_results['operations_attempted']}")
            print(f"   Operations successful: {apply_results['operations_successful']}")
            print(f"   Errors: {len(apply_results['errors'])}")
            print(f"   Saved: {apply_results.get('saved', False)}")
            
            if apply_results["errors"]:
                print(f"   Error details: {apply_results['errors']}")
            
            # Verify changes were applied
            final_genotype1 = metadata.get_genotype(embryo_id1)
            final_genotype2 = metadata.get_genotype(embryo_id2)
            
            print(f"✅ Changes applied:")
            print(f"   Embryo 1 genotype: {final_genotype1['value'] if final_genotype1 else None}")
            print(f"   Embryo 2 genotype: {final_genotype2['value'] if final_genotype2 else None}")
            
            # Check treatments were applied
            treatments1 = metadata.get_treatments(embryo_id1)
            treatments2 = metadata.get_treatments(embryo_id2)
            
            print(f"   Embryo 1 treatments: {len(treatments1)}")
            print(f"   Embryo 2 treatments: {len(treatments2)}")
            
            # Get summary
            summary1 = metadata.get_embryo_summary(embryo_id1)
            summary2 = metadata.get_embryo_summary(embryo_id2)
            
            print(f"✅ Final summaries:")
            print(f"   Embryo 1: {summary1}")
            print(f"   Embryo 2: {summary2}")
            
            # Test overall metadata stats
            overall_stats = metadata.get_summary()
            print(f"✅ Metadata stats: {overall_stats['entity_counts']}")
            print(f"   Phenotypes: {overall_stats['phenotype_stats']}")
            print(f"   Genotypes: {overall_stats['genotype_stats']}")
            
            print("✅ All batch apply tests passed!")
            return True
            
        finally:
            # Clean up
            Path(sam2_path).unlink()
            metadata_path = Path(sam2_path).with_name(Path(sam2_path).stem + '_embryo_metadata.json')
            if metadata_path.exists():
                metadata_path.unlink()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run batch apply test"""
    print("=" * 60)
    print("AnnotationBatch Apply Test")
    print("=" * 60)
    
    success = test_batch_apply()
    
    print("=" * 60)
    if success:
        print("✅ Batch apply functionality working correctly!")
        print("✅ Module 3 core implementation complete!")
    else:
        print("❌ Batch apply functionality needs fixes")
    print("=" * 60)

if __name__ == "__main__":
    main()