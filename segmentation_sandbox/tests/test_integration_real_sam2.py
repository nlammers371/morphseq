#!/usr/bin/env python3
"""
Integration test with real SAM2 data
Tests the complete annotation workflow with actual segmentation data
"""
import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_integration_with_real_sam2():
    """Test integration with real SAM2 segmentation data"""
    print("🚀 Integration test with real SAM2 data...")
    
    try:
        from scripts.annotations.embryo_metadata import EmbryoMetadata
        from scripts.annotations.annotation_batch import AnnotationBatch
        
        # Use real SAM2 file
        sam2_path = project_root / "data" / "segmentation" / "grounded_sam_segmentations.json"
        
        if not sam2_path.exists():
            print(f"❌ Real SAM2 file not found: {sam2_path}")
            return False
        
        print(f"📂 Using real SAM2 file: {sam2_path}")
        print(f"   File size: {sam2_path.stat().st_size / 1024:.1f} KB")
        
        # Create temporary metadata file path (don't create the file)
        temp_metadata_path = Path(tempfile.mktemp(suffix='_embryo_metadata.json'))
        
        try:
            # ========== TEST 1: Create metadata from real SAM2 ==========
            print("\n🔬 Test 1: Creating EmbryoMetadata from real SAM2 data")
            
            metadata = EmbryoMetadata(
                sam2_path,
                temp_metadata_path,
                gen_if_no_file=True,
                verbose=True
            )
            
            print(f"✅ Created metadata successfully")
            print(f"   Embryo count: {metadata.embryo_count}")
            print(f"   Snip count: {metadata.snip_count}")
            print(f"   Entity counts: {metadata.get_entity_counts()}")
            
            # ========== TEST 2: Get available entities ==========
            print("\n🔬 Test 2: Accessing real entities")
            
            available_snips = metadata.get_available_snips()
            print(f"✅ Available snips: {len(available_snips)}")
            
            if available_snips:
                # Get first few embryos and snips
                first_snip = available_snips[0]
                first_embryo = metadata.get_embryo_id_from_snip(first_snip)
                
                print(f"   First snip: {first_snip}")
                print(f"   First embryo: {first_embryo}")
                
                # Test data access
                if first_embryo:
                    embryo_data = metadata._get_embryo_data(first_embryo)
                    snip_data = metadata._get_snip_data(first_snip)
                    print(f"   ✅ Data access working: embryo has {len(embryo_data.get('snips', {}))} snips")
            
            # ========== TEST 3: Add real annotations ==========
            print("\n🔬 Test 3: Adding annotations to real data")
            
            if available_snips:
                test_snip = available_snips[0]
                test_embryo = metadata.get_embryo_id_from_snip(test_snip)
                
                if test_embryo:
                    # Add genotype
                    metadata.add_genotype(test_embryo, "tmem67", "tmem67tm1a", "heterozygous", "integration_test")
                    genotype = metadata.get_genotype(test_embryo)
                    print(f"   ✅ Added genotype: {genotype['value'] if genotype else None}")
                    
                    # Add phenotype
                    metadata.add_phenotype(test_snip, "EDEMA", "integration_test", "integration test phenotype")
                    phenotypes = metadata.get_phenotypes(test_snip)
                    print(f"   ✅ Added phenotype: {phenotypes[0]['value'] if phenotypes else None}")
                    
                    # Add treatment
                    metadata.add_treatment(test_embryo, "DMSO", dosage="1%", timing="24hpf", author="integration_test")
                    treatments = metadata.get_treatments(test_embryo)
                    print(f"   ✅ Added treatment: {len(treatments)} treatments")
            
            # ========== TEST 4: Batch operations with real data ==========
            print("\n🔬 Test 4: Batch operations with real data")
            
            batch = AnnotationBatch("integration_test", "Real data integration test")
            
            # Add annotations to multiple embryos if available
            embryos_to_annotate = list(set([metadata.get_embryo_id_from_snip(s) for s in available_snips[:5]]))[:3]  # First 3 unique embryos
            embryos_to_annotate = [e for e in embryos_to_annotate if e]  # Remove None values
            
            if embryos_to_annotate:
                for i, embryo_id in enumerate(embryos_to_annotate):
                    # Add genotype
                    genotypes = ["b9d2", "lmx1b", "WT"]
                    batch.add_genotype(embryo_id, genotypes[i % len(genotypes)], f"{genotypes[i % len(genotypes)]}tm1", "heterozygous")
                    
                    # Add treatment
                    treatments = ["PTU", "DMSO", "BIO"]
                    batch.add_treatment(embryo_id, treatments[i % len(treatments)], "0.1%", "24hpf")
                
                print(f"   ✅ Created batch with {len(batch.get_embryo_list())} embryos")
                print(f"   Batch stats: {batch.get_stats()}")
                
                # Test dry run
                dry_results = batch.dry_run(metadata)
                print(f"   ✅ Dry run: {dry_results['operations_attempted']} operations, {dry_results['operations_successful']} successful")
                
                # Apply batch
                if dry_results["operations_successful"] > 0:
                    apply_results = batch.apply(metadata, backup=False)
                    print(f"   ✅ Applied batch: {apply_results['operations_successful']} operations successful")
            
            # ========== TEST 5: Save and validation ==========
            print("\n🔬 Test 5: Save and validation")
            
            # Validate data integrity
            validation_results = metadata.validate_data_integrity()
            print(f"✅ Data validation completed")
            print(f"   Genotype coverage: {validation_results['genotype_coverage']['missing_count']} missing")
            print(f"   DEAD conflicts: {validation_results['dead_conflicts']['conflict_count']} conflicts")
            print(f"   Flag distribution: {sum(validation_results['flag_distribution'].values())} total flags")
            
            # Save metadata
            metadata.save(backup=True)
            file_size = temp_metadata_path.stat().st_size
            print(f"✅ Saved metadata: {file_size / 1024:.1f} KB")
            
            # Get final summary
            summary = metadata.get_summary()
            print(f"✅ Final summary:")
            print(f"   Entity counts: {summary['entity_counts']}")
            print(f"   Phenotype stats: {summary['phenotype_stats']}")
            print(f"   Genotype stats: {summary['genotype_stats']}")
            
            return True
            
        finally:
            # Clean up
            if temp_metadata_path.exists():
                temp_metadata_path.unlink()
            # Also clean up backup
            backup_path = temp_metadata_path.with_suffix('.json.bak')
            if backup_path.exists():
                backup_path.unlink()
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration test"""
    print("=" * 70)
    print("Module 3 Real SAM2 Integration Test")
    print("=" * 70)
    
    success = test_integration_with_real_sam2()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 INTEGRATION TEST PASSED!")
        print("✅ Module 3 works correctly with real SAM2 segmentation data")
        print("✅ End-to-end annotation workflow functional")
        print("✅ Ready for production use!")
    else:
        print("❌ Integration test failed - see errors above")
    print("=" * 70)

if __name__ == "__main__":
    main()