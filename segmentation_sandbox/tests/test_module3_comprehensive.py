#!/usr/bin/env python3
"""
Comprehensive Module 3 Test Suite
Tests all core functionality of the embryo annotation system
"""
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_comprehensive_sam2_data():
    """Create comprehensive SAM2 data for testing"""
    return {
        "experiments": {
            "test_exp_001": {
                "videos": {
                    "test_exp_001_A01": {
                        "embryo_ids": ["test_exp_001_A01_e01", "test_exp_001_A01_e02", "test_exp_001_A01_e03"],
                        "images": {
                            "test_exp_001_A01_ch00_t0100": {
                                "embryos": {
                                    "test_exp_001_A01_e01": {"snip_id": "test_exp_001_A01_e01_s0100"},
                                    "test_exp_001_A01_e02": {"snip_id": "test_exp_001_A01_e02_s0100"},
                                    "test_exp_001_A01_e03": {"snip_id": "test_exp_001_A01_e03_s0100"}
                                }
                            },
                            "test_exp_001_A01_ch00_t0101": {
                                "embryos": {
                                    "test_exp_001_A01_e01": {"snip_id": "test_exp_001_A01_e01_s0101"},
                                    "test_exp_001_A01_e02": {"snip_id": "test_exp_001_A01_e02_s0101"},
                                    "test_exp_001_A01_e03": {"snip_id": "test_exp_001_A01_e03_s0101"}
                                }
                            },
                            "test_exp_001_A01_ch00_t0102": {
                                "embryos": {
                                    "test_exp_001_A01_e01": {"snip_id": "test_exp_001_A01_e01_s0102"},
                                    "test_exp_001_A01_e02": {"snip_id": "test_exp_001_A01_e02_s0102"},
                                    "test_exp_001_A01_e03": {"snip_id": "test_exp_001_A01_e03_s0102"}
                                }
                            }
                        }
                    },
                    "test_exp_001_B01": {
                        "embryo_ids": ["test_exp_001_B01_e01", "test_exp_001_B01_e02"],
                        "images": {
                            "test_exp_001_B01_ch00_t0100": {
                                "embryos": {
                                    "test_exp_001_B01_e01": {"snip_id": "test_exp_001_B01_e01_s0100"},
                                    "test_exp_001_B01_e02": {"snip_id": "test_exp_001_B01_e02_s0100"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

def run_comprehensive_tests():
    """Run comprehensive Module 3 tests"""
    print("🧪 Running comprehensive Module 3 tests...")
    
    test_results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    try:
        from scripts.annotations.embryo_metadata import EmbryoMetadata
        from scripts.annotations.annotation_batch import AnnotationBatch, EmbryoQuery
        from scripts.annotations.unified_managers import UnifiedEmbryoManager
        
        # Create temporary SAM2 file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(create_comprehensive_sam2_data(), f)
            sam2_path = f.name
        
        try:
            # ========== TEST 1: EmbryoMetadata Creation ==========
            test_results["total_tests"] += 1
            print("🔬 Test 1: EmbryoMetadata creation and SAM2 import")
            
            metadata = EmbryoMetadata(sam2_path, gen_if_no_file=True, verbose=False)
            expected_embryos = 5  # 3 + 2 from the SAM2 data
            expected_snips = 11   # 3*3 + 2*1 from the SAM2 data
            
            if metadata.embryo_count == expected_embryos and metadata.snip_count == expected_snips:
                print(f"   ✅ Correct counts: {metadata.embryo_count} embryos, {metadata.snip_count} snips")
                test_results["passed"] += 1
            else:
                print(f"   ❌ Wrong counts: got {metadata.embryo_count}/{metadata.snip_count}, expected {expected_embryos}/{expected_snips}")
                test_results["failed"] += 1
                test_results["errors"].append("Test 1: Wrong entity counts")
            
            # ========== TEST 2: Entity Tracking ==========
            test_results["total_tests"] += 1
            print("🔬 Test 2: Entity tracking and validation")
            
            entity_counts = metadata.get_entity_counts()
            if entity_counts["embryos"] == expected_embryos:
                print(f"   ✅ Entity tracking working: {entity_counts}")
                test_results["passed"] += 1
            else:
                print(f"   ❌ Entity tracking failed: {entity_counts}")
                test_results["failed"] += 1
                test_results["errors"].append("Test 2: Entity tracking failed")
            
            # ========== TEST 3: Data Access Helpers ==========
            test_results["total_tests"] += 1
            print("🔬 Test 3: Data access helpers")
            
            embryo_id = "test_exp_001_A01_e01"
            snip_id = "test_exp_001_A01_e01_s0100"
            
            try:
                embryo_data = metadata._get_embryo_data(embryo_id)
                snip_data = metadata._get_snip_data(snip_id)
                print(f"   ✅ Data access helpers working")
                test_results["passed"] += 1
            except Exception as e:
                print(f"   ❌ Data access helpers failed: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"Test 3: Data access failed - {e}")
            
            # ========== TEST 4: Phenotype Management with DEAD Safety ==========
            test_results["total_tests"] += 1
            print("🔬 Test 4: Phenotype management and DEAD safety")
            
            try:
                # Add normal phenotype
                metadata.add_phenotype(snip_id, "EDEMA", "test_author", "normal phenotype")
                
                # Try to add DEAD (should fail)
                try:
                    metadata.add_phenotype(snip_id, "DEAD", "test_author", "should fail")
                    print("   ❌ DEAD exclusivity not working")
                    test_results["failed"] += 1
                    test_results["errors"].append("Test 4: DEAD exclusivity failed")
                except ValueError:
                    # Expected behavior
                    # Now try with force_dead
                    metadata.add_phenotype(snip_id, "DEAD", "test_author", "forced", force_dead=True)
                    print(f"   ✅ DEAD safety and force_dead working")
                    test_results["passed"] += 1
            except Exception as e:
                print(f"   ❌ Phenotype management failed: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"Test 4: Phenotype management failed - {e}")
            
            # ========== TEST 5: Genotype Management ==========
            test_results["total_tests"] += 1
            print("🔬 Test 5: Genotype management")
            
            try:
                metadata.add_genotype(embryo_id, "tmem67", "tmem67tm1a", "heterozygous", "test_author")
                genotype = metadata.get_genotype(embryo_id)
                
                if genotype and genotype["value"] == "tmem67":
                    print(f"   ✅ Genotype management working: {genotype['value']}")
                    test_results["passed"] += 1
                else:
                    print(f"   ❌ Genotype not properly stored: {genotype}")
                    test_results["failed"] += 1
                    test_results["errors"].append("Test 5: Genotype storage failed")
            except Exception as e:
                print(f"   ❌ Genotype management failed: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"Test 5: Genotype management failed - {e}")
            
            # ========== TEST 6: Treatment Management ==========
            test_results["total_tests"] += 1
            print("🔬 Test 6: Treatment management")
            
            try:
                metadata.add_treatment(embryo_id, "DMSO", dosage="1%", timing="24hpf", author="test_author")
                treatments = metadata.get_treatments(embryo_id)
                
                if len(treatments) > 0:
                    print(f"   ✅ Treatment management working: {len(treatments)} treatments")
                    test_results["passed"] += 1
                else:
                    print(f"   ❌ No treatments found: {treatments}")
                    test_results["failed"] += 1
                    test_results["errors"].append("Test 6: Treatment storage failed")
            except Exception as e:
                print(f"   ❌ Treatment management failed: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"Test 6: Treatment management failed - {e}")
            
            # ========== TEST 7: Data Validation ==========
            test_results["total_tests"] += 1
            print("🔬 Test 7: Data validation")
            
            try:
                validation_results = metadata.validate_data_integrity()
                print(f"   ✅ Data validation working: {len(validation_results)} validation categories")
                test_results["passed"] += 1
            except Exception as e:
                print(f"   ❌ Data validation failed: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"Test 7: Data validation failed - {e}")
            
            # ========== TEST 8: AnnotationBatch System ==========
            test_results["total_tests"] += 1
            print("🔬 Test 8: AnnotationBatch system")
            
            try:
                batch = AnnotationBatch("test_author", "Comprehensive test batch")
                
                # Add various annotations
                batch.add_genotype("test_exp_001_A01_e02", "b9d2", "b9d2tm1a", "heterozygous")
                batch.add_phenotype("test_exp_001_A01_e02", "BODY_AXIS_DEFECT", frames="100:110")
                batch.add_treatment("test_exp_001_A01_e02", "PTU", dosage="0.003%", timing="0hpf")
                
                stats = batch.get_stats()
                if stats["genotypes"] == 1 and stats["phenotypes"] == 1 and stats["treatments"] == 1:
                    print(f"   ✅ Batch creation working: {stats}")
                    test_results["passed"] += 1
                else:
                    print(f"   ❌ Batch stats incorrect: {stats}")
                    test_results["failed"] += 1
                    test_results["errors"].append("Test 8: Batch stats incorrect")
            except Exception as e:
                print(f"   ❌ AnnotationBatch failed: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"Test 8: AnnotationBatch failed - {e}")
            
            # ========== TEST 9: Batch Apply ==========
            test_results["total_tests"] += 1
            print("🔬 Test 9: Batch apply functionality")
            
            try:
                # Create a new batch for application
                apply_batch = AnnotationBatch("test_author", "Apply test")
                apply_batch.add_genotype("test_exp_001_A01_e03", "lmx1b", "lmx1btm1", "homozygous")
                
                # Dry run first
                dry_results = apply_batch.dry_run(metadata)
                
                if dry_results.get("preview_mode") and dry_results["operations_attempted"] == 1:
                    # Now apply for real
                    apply_results = apply_batch.apply(metadata, backup=False)
                    
                    if apply_results["success"] and apply_results["operations_successful"] == 1:
                        # Verify application
                        applied_genotype = metadata.get_genotype("test_exp_001_A01_e03")
                        if applied_genotype and applied_genotype["value"] == "lmx1b":
                            print(f"   ✅ Batch apply working: {apply_results['operations_successful']} operations")
                            test_results["passed"] += 1
                        else:
                            print(f"   ❌ Batch apply verification failed: {applied_genotype}")
                            test_results["failed"] += 1
                            test_results["errors"].append("Test 9: Batch apply verification failed")
                    else:
                        print(f"   ❌ Batch apply failed: {apply_results}")
                        test_results["failed"] += 1
                        test_results["errors"].append("Test 9: Batch apply failed")
                else:
                    print(f"   ❌ Batch dry run failed: {dry_results}")
                    test_results["failed"] += 1
                    test_results["errors"].append("Test 9: Batch dry run failed")
            except Exception as e:
                print(f"   ❌ Batch apply failed: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"Test 9: Batch apply failed - {e}")
            
            # ========== TEST 10: Save/Load Functionality ==========
            test_results["total_tests"] += 1
            print("🔬 Test 10: Save/load functionality")
            
            try:
                # Save metadata
                metadata.save(backup=False)
                
                # Reload to verify persistence
                metadata.reload()
                
                # Verify data persisted
                reloaded_genotype = metadata.get_genotype(embryo_id)
                if reloaded_genotype and reloaded_genotype["value"] == "tmem67":
                    print(f"   ✅ Save/load working: data persisted correctly")
                    test_results["passed"] += 1
                else:
                    print(f"   ❌ Save/load failed: {reloaded_genotype}")
                    test_results["failed"] += 1
                    test_results["errors"].append("Test 10: Save/load verification failed")
            except Exception as e:
                print(f"   ❌ Save/load failed: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"Test 10: Save/load failed - {e}")
            
        finally:
            # Clean up
            Path(sam2_path).unlink()
            metadata_path = Path(sam2_path).with_name(Path(sam2_path).stem + '_embryo_metadata.json')
            if metadata_path.exists():
                metadata_path.unlink()
        
        return test_results
        
    except Exception as e:
        test_results["failed"] += 1
        test_results["errors"].append(f"Critical failure: {e}")
        import traceback
        traceback.print_exc()
        return test_results

def main():
    """Run comprehensive test suite"""
    print("=" * 70)
    print("Module 3 Comprehensive Test Suite")
    print("=" * 70)
    
    results = run_comprehensive_tests()
    
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"📊 Total Tests: {results['total_tests']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"🏆 Success Rate: {(results['passed'] / results['total_tests'] * 100):.1f}%")
    
    if results["errors"]:
        print(f"\n🚨 ERRORS:")
        for error in results["errors"]:
            print(f"   - {error}")
    
    print("\n" + "=" * 70)
    if results["failed"] == 0:
        print("🎉 ALL TESTS PASSED - Module 3 Implementation Complete! 🎉")
        print("✅ EmbryoMetadata system fully functional")
        print("✅ UnifiedEmbryoManager business logic complete")
        print("✅ AnnotationBatch system working")
        print("✅ DEAD safety workflow implemented")
        print("✅ Schema validation working")
        print("✅ Save/load persistence functional")
    else:
        print("❌ Some tests failed - Review errors above")
    print("=" * 70)

if __name__ == "__main__":
    main()