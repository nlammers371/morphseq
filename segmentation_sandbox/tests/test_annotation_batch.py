#!/usr/bin/env python3
"""
Test AnnotationBatch functionality
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

def test_annotation_batch():
    """Test AnnotationBatch functionality"""
    print("🔍 Testing AnnotationBatch...")
    
    try:
        from scripts.annotations.annotation_batch import AnnotationBatch, EmbryoQuery
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
            batch = AnnotationBatch("test_author", "Testing batch operations")
            print("✅ Created AnnotationBatch")
            
            # Test adding annotations
            embryo_id = "test_exp_A01_e01"
            
            # Add genotype
            batch.add_genotype(embryo_id, "tmem67", "tmem67tm1a", "heterozygous", "genotype notes")
            print("✅ Added genotype to batch")
            
            # Add phenotype with DEAD safety
            batch.add_phenotype(embryo_id, "EDEMA", frames="100:110", notes="test phenotype")
            print("✅ Added phenotype to batch")
            
            # Add treatment
            batch.add_treatment(embryo_id, "DMSO", dosage="1%", timing="24hpf", notes="vehicle control")
            print("✅ Added treatment to batch")
            
            # Add flag
            batch.add_flag(embryo_id, "QUALITY_CHECK", level="embryo", priority="medium", description="needs review")
            print("✅ Added flag to batch")
            
            # Test mark_dead functionality
            embryo_id2 = "test_exp_A01_e02"
            batch.mark_dead(embryo_id2, start_frame=105)
            print("✅ Marked embryo as dead")
            
            # Test batch preview
            print("🔍 Batch preview:")
            print(batch.preview()[:500] + "..." if len(str(batch)) > 500 else batch.preview())
            
            # Test stats
            stats = batch.get_stats()
            print(f"✅ Batch stats: {stats}")
            
            # Test embryo list
            embryo_list = batch.get_embryo_list()
            print(f"✅ Embryos in batch: {embryo_list}")
            
            # Test persistence
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                batch_path = f.name
            
            batch.save_json(batch_path)
            print("✅ Saved batch to JSON")
            
            # Load batch
            loaded_batch = AnnotationBatch.load_json(batch_path)
            print(f"✅ Loaded batch: {loaded_batch.get_stats()}")
            
            # Test to_dict/from_dict
            batch_dict = batch.to_dict()
            restored_batch = AnnotationBatch.from_dict(batch_dict)
            print(f"✅ Dict serialization: {restored_batch.get_stats()}")
            
            # Test query system
            query = EmbryoQuery(metadata)
            
            # Can't test actual queries since metadata is empty, but test creation
            query_batch = query.to_batch("query_author")
            print(f"✅ Created query batch: {len(query_batch.get_embryo_list())} embryos")
            
            # Clean up
            Path(batch_path).unlink()
            
            print("✅ All AnnotationBatch tests passed!")
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
    """Run annotation batch test"""
    print("=" * 60)
    print("AnnotationBatch Test")
    print("=" * 60)
    
    success = test_annotation_batch()
    
    print("=" * 60)
    if success:
        print("✅ AnnotationBatch system working correctly!")
    else:
        print("❌ AnnotationBatch system needs fixes")
    print("=" * 60)

if __name__ == "__main__":
    main()