#!/usr/bin/env python3
"""
Phase 1 Validation Tests

Simple tests without pytest dependency to validate MVP functionality.
"""

import json
import tempfile
import time
from pathlib import Path
import sys
import traceback

# Add scripts to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from annotations.embryo_metadata import EmbryoMetadata


def create_test_sam2_data():
    """Create test SAM2 data with multiple embryos and frames."""
    return {
        "experiments": {
            "20240418": {
                "videos": {
                    "20240418_A01": {
                        "images": {
                            "20240418_A01_t0100": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e03": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0101": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0150": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e03": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    },
                    "20240418_B01": {
                        "images": {
                            "20240418_B01_t0200": {
                                "embryos": {
                                    "20240418_B01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_B01_e02": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            },
            "20240419": {
                "videos": {
                    "20240419_C01": {
                        "images": {
                            "20240419_C01_t0300": {
                                "embryos": {
                                    "20240419_C01_e01": {"segmentation": {"counts": "test"}},
                                    "20240419_C01_e02": {"segmentation": {"counts": "test"}},
                                    "20240419_C01_e03": {"segmentation": {"counts": "test"}},
                                    "20240419_C01_e04": {"segmentation": {"counts": "test"}},
                                    "20240419_C01_e05": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            }
        }
    }


def test_basic_initialization():
    """Test basic EmbryoMetadata initialization."""
    print("Testing basic initialization...")
    
    # Create temp SAM2 file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        # Initialize metadata
        start_time = time.time()
        metadata = EmbryoMetadata(str(sam2_file))
        init_time = time.time() - start_time
        
        # Check basic structure
        assert "metadata" in metadata.data
        assert "embryos" in metadata.data
        assert metadata.data["metadata"]["version"] == "simplified_v1"
        
        # Check embryos were extracted (should be 10 total)
        embryos = metadata.data["embryos"]
        expected_count = 10  # 3+2+5 from the test data
        assert len(embryos) == expected_count, f"Expected {expected_count} embryos, got {len(embryos)}"
        
        # Check some specific embryos
        assert "20240418_A01_e01" in embryos
        assert "20240418_B01_e01" in embryos
        assert "20240419_C01_e01" in embryos
        
        # Check embryo structure
        e01 = embryos["20240418_A01_e01"]
        assert e01["embryo_id"] == "20240418_A01_e01"
        assert e01["experiment_id"] == "20240418"
        assert e01["video_id"] == "20240418_A01"
        assert e01["genotype"] is None
        assert e01["treatments"] == []
        
        # Check snips were created correctly
        expected_snips = ["20240418_A01_e01_s0100", "20240418_A01_e01_s0101", "20240418_A01_e01_s0150"]
        for snip_id in expected_snips:
            assert snip_id in e01["snips"], f"Missing snip: {snip_id}"
        
        print(f"✅ PASS - Initialized {len(embryos)} embryos in {init_time:.3f}s")
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_add_phenotype_functionality():
    """Test basic phenotype addition."""
    print("Testing add_phenotype functionality...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test adding phenotype
        start_time = time.time()
        result = metadata.add_phenotype("EDEMA", "test_user", "20240418_A01_e01", target="all")
        add_time = time.time() - start_time
        
        # Check result
        assert result["operation"] == "add_phenotype"
        assert result["phenotype"] == "EDEMA"
        assert result["embryo_id"] == "20240418_A01_e01"
        assert result["count"] == 3  # Should be applied to 3 snips
        
        # Check phenotype was actually added
        e01 = metadata.data["embryos"]["20240418_A01_e01"]
        for snip_data in e01["snips"].values():
            phenotypes = snip_data["phenotypes"]
            assert len(phenotypes) == 1
            assert phenotypes[0]["value"] == "EDEMA"
            assert phenotypes[0]["author"] == "test_user"
            assert "timestamp" in phenotypes[0]
        
        print(f"✅ PASS - Added phenotype to {result['count']} snips in {add_time:.3f}s")
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_add_genotype_functionality():
    """Test genotype addition."""
    print("Testing add_genotype functionality...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test adding genotype
        result = metadata.add_genotype("tmem67", "test_user", "20240418_A01_e01", 
                                     allele="sa1423", zygosity="homozygous")
        
        # Check result
        assert result["operation"] == "add_genotype"
        assert result["gene"] == "tmem67"
        assert result["embryo_id"] == "20240418_A01_e01"
        
        # Check genotype was actually added
        genotype = metadata.data["embryos"]["20240418_A01_e01"]["genotype"]
        assert genotype["gene"] == "tmem67"
        assert genotype["allele"] == "sa1423"
        assert genotype["zygosity"] == "homozygous"
        assert genotype["author"] == "test_user"
        
        print("✅ PASS - Added genotype successfully")
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_validation_errors():
    """Test validation error handling."""
    print("Testing validation errors...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test invalid phenotype
        try:
            metadata.add_phenotype("INVALID_PHENOTYPE", "test_user", "20240418_A01_e01")
            assert False, "Should have raised ValueError for invalid phenotype"
        except ValueError as e:
            assert "Invalid phenotype" in str(e)
        
        # Test invalid embryo ID
        try:
            metadata.add_phenotype("EDEMA", "test_user", "invalid_embryo_id")
            assert False, "Should have raised ValueError for invalid embryo"
        except ValueError as e:
            assert "not found" in str(e)
        
        # Test invalid target (frame range with no matching snips)
        try:
            metadata.add_phenotype("EDEMA", "test_user", "20240418_A01_e01", target="30:50")
            assert False, "Should have raised ValueError for invalid target"
        except ValueError as e:
            assert "No snips found in range" in str(e)
        
        # Test invalid gene
        try:
            metadata.add_genotype("invalid_gene", "test_user", "20240418_A01_e01")
            assert False, "Should have raised ValueError for invalid gene"
        except ValueError as e:
            assert "Invalid gene" in str(e)
        
        print("✅ PASS - All validation errors handled correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_save_and_reload():
    """Test save/reload cycle."""
    print("Testing save and reload...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        # Create and populate metadata
        metadata1 = EmbryoMetadata(str(sam2_file))
        metadata1.add_phenotype("EDEMA", "test_user", "20240418_A01_e01")
        metadata1.add_genotype("tmem67", "test_user", "20240418_A01_e01", zygosity="homozygous")
        
        # Save
        start_time = time.time()
        metadata1.save()
        save_time = time.time() - start_time
        
        # Reload
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        start_time = time.time()
        metadata2 = EmbryoMetadata(str(sam2_file), str(biology_path))
        reload_time = time.time() - start_time
        
        # Check data integrity (ignoring timestamp differences)
        # Compare embryo count and structure
        assert len(metadata1.data["embryos"]) == len(metadata2.data["embryos"])
        assert set(metadata1.data["embryos"].keys()) == set(metadata2.data["embryos"].keys())
        
        # Check specific annotations survived
        e01 = metadata2.data["embryos"]["20240418_A01_e01"]
        assert e01["genotype"]["gene"] == "tmem67"
        
        # Check phenotypes
        for snip_data in e01["snips"].values():
            assert len(snip_data["phenotypes"]) == 1
            assert snip_data["phenotypes"][0]["value"] == "EDEMA"
        
        print(f"✅ PASS - Save in {save_time:.3f}s, reload in {reload_time:.3f}s")
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_performance_target():
    """Test that we meet the performance target: process 10 embryos in <2 seconds."""
    print("Testing performance target (10 embryos in <2 seconds)...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        # Time the full workflow
        start_time = time.time()
        
        # Initialize
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Add annotations to all embryos
        embryo_ids = list(metadata.data["embryos"].keys())
        for i, embryo_id in enumerate(embryo_ids):
            metadata.add_phenotype("NORMAL", f"user{i}", embryo_id)
            if i % 2 == 0:  # Add genotype to half
                metadata.add_genotype("tmem67", f"user{i}", embryo_id, zygosity="homozygous")
        
        # Save
        metadata.save()
        
        total_time = time.time() - start_time
        
        # Check we processed the expected number of embryos
        assert len(embryo_ids) == 10, f"Expected 10 embryos, got {len(embryo_ids)}"
        
        # Check performance target
        target_time = 2.0
        assert total_time < target_time, f"Processing took {total_time:.3f}s, target was <{target_time}s"
        
        print(f"✅ PASS - Processed {len(embryo_ids)} embryos in {total_time:.3f}s (target: <{target_time}s)")
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_auto_update():
    """Test auto-update functionality."""
    print("Testing auto-update from SAM2...")
    
    # Create initial SAM2 data
    initial_data = {
        "experiments": {
            "20240418": {
                "videos": {
                    "20240418_A01": {
                        "images": {
                            "20240418_A01_t0100": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(initial_data, f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        # Create initial metadata
        metadata1 = EmbryoMetadata(str(sam2_file))
        metadata1.add_phenotype("NORMAL", "user1", "20240418_A01_e01")
        metadata1.save()
        
        # Update SAM2 with new embryos/frames
        updated_data = {
            "experiments": {
                "20240418": {
                    "videos": {
                        "20240418_A01": {
                            "images": {
                                "20240418_A01_t0100": {
                                    "embryos": {
                                        "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                        "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                    }
                                },
                                "20240418_A01_t0150": {
                                    "embryos": {
                                        "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                        "20240418_A01_e03": {"segmentation": {"counts": "test"}}  # New embryo
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # Update SAM2 file
        with open(sam2_file, 'w') as f:
            json.dump(updated_data, f)
        
        # Load metadata again - should auto-detect
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        metadata2 = EmbryoMetadata(str(sam2_file), str(biology_path))
        
        # Check new embryo detected
        assert "20240418_A01_e03" in metadata2.data["embryos"]
        
        # Check new snip added to existing embryo
        e01_snips = metadata2.data["embryos"]["20240418_A01_e01"]["snips"]
        assert "20240418_A01_e01_s0150" in e01_snips
        
        # Check existing annotations preserved
        assert len(e01_snips["20240418_A01_e01_s0100"]["phenotypes"]) == 1
        assert e01_snips["20240418_A01_e01_s0100"]["phenotypes"][0]["value"] == "NORMAL"
        
        print("✅ PASS - Auto-update detected new embryos and frames")
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def main():
    """Run all Phase 1 validation tests."""
    print("=" * 60)
    print("PHASE 1 MVP VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_basic_initialization,
        test_add_phenotype_functionality,
        test_add_genotype_functionality,
        test_validation_errors,
        test_save_and_reload,
        test_performance_target,
        test_auto_update
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAIL - {test_func.__name__}: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL PHASE 1 TESTS PASSED!")
        print("✅ MVP implementation is working correctly")
        print("✅ Performance targets met")
        print("✅ Ready to proceed to Phase 2")
    else:
        print("❌ Some tests failed - need to fix issues before Phase 2")
    
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)