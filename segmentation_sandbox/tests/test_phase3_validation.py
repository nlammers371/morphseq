#!/usr/bin/env python3
"""
Phase 3 Validation Tests

Tests business rules implementation:
- DEAD phenotype exclusivity
- DEAD permanence 
- DEAD safety with overwrite_dead parameter
- Controlled vocabulary validation
- Enhanced error messages
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
    """Create test SAM2 data with sequential frames."""
    return {
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
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0200": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0250": {
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


def test_dead_exclusivity():
    """Test DEAD exclusivity validation."""
    print("Testing DEAD exclusivity validation...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Add normal phenotype first
        metadata.add_phenotype("EDEMA", "user1", embryo_id="20240418_A01_e01", target="150")
        
        # Try to add DEAD to same snip - should fail
        try:
            metadata.add_phenotype("DEAD", "user1", 
                                 snip_ids=["20240418_A01_e01_s0150"])
            assert False, "Should have failed DEAD exclusivity"
        except ValueError as e:
            assert "DEAD exclusivity violation" in str(e)
            print("✓ DEAD exclusivity prevents adding DEAD to snip with other phenotypes")
        
        # Add DEAD to clean snip
        result = metadata.add_phenotype("DEAD", "user1", embryo_id="20240418_A01_e01", target="200")
        assert result["count"] == 1
        print("✓ DEAD can be added to clean snip")
        
        # Try to add other phenotype to DEAD snip - should be silently skipped (range approach)
        result = metadata.add_phenotype("NORMAL", "user1", 
                                       embryo_id="20240418_A01_e01", target="200")
        assert result["count"] == 0
        assert "skipped_dead_frames" in result
        assert len(result["skipped_dead_frames"]) == 1
        print("✓ Non-DEAD phenotypes silently skip DEAD frames (range approach)")
        
        # Override DEAD frame with overwrite_dead=True
        result = metadata.add_phenotype("NORMAL", "user1", 
                                       snip_ids=["20240418_A01_e01_s0200"],
                                       overwrite_dead=True)
        assert result["count"] == 1
        print("✓ overwrite_dead=True allows phenotypes on DEAD frames")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_dead_permanence():
    """Test DEAD permanence validation."""
    print("Testing DEAD permanence validation...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Mark embryo as DEAD at frame 200
        metadata.add_phenotype("DEAD", "user1", embryo_id="20240418_A01_e01", target="200")
        
        # Try to add normal phenotype after death using snip approach - should fail
        try:
            metadata.add_phenotype("NORMAL", "user1", snip_ids=["20240418_A01_e01_s0250"])
            assert False, "Should have failed DEAD permanence"
        except ValueError as e:
            assert "DEAD permanence violation" in str(e)
            print("✓ DEAD permanence prevents phenotypes after death (snip approach)")
        
        # Try to add normal phenotype at same frame as death using snip approach - should fail  
        try:
            metadata.add_phenotype("EDEMA", "user1", snip_ids=["20240418_A01_e01_s0200"])
            assert False, "Should have failed DEAD exclusivity"
        except ValueError as e:
            assert "DEAD exclusivity violation" in str(e)
            print("✓ DEAD exclusivity prevents phenotypes at DEAD frame (snip approach)")
        
        # Range operations should silently skip
        result = metadata.add_phenotype("NORMAL", "user1", embryo_id="20240418_A01_e01", target="250")
        assert result["count"] == 0  # Should skip the DEAD frame
        assert "skipped_dead_frames" in result
        print("✓ Range operations silently skip frames after death")
        
        # Add phenotype before death - should work
        result = metadata.add_phenotype("NORMAL", "user1", embryo_id="20240418_A01_e01", target="150")
        assert result["count"] == 1
        print("✓ Phenotypes before death are allowed")
        
        # Override with overwrite_dead=True
        result = metadata.add_phenotype("EDEMA", "user1", embryo_id="20240418_A01_e01", 
                                       target="250", overwrite_dead=True)
        assert result["count"] == 1
        print("✓ overwrite_dead=True allows phenotypes after death")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_dead_safety_ranges():
    """Test DEAD safety behavior with ranges."""
    print("Testing DEAD safety with ranges...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Mark embryo as DEAD at frame 200
        metadata.add_phenotype("DEAD", "user1", embryo_id="20240418_A01_e01", target="200")
        
        # Apply phenotype to range that includes DEAD frame
        result = metadata.add_phenotype("EDEMA", "user1", embryo_id="20240418_A01_e01", target="all")
        
        # Should apply to frames before death, skip DEAD frame
        assert result["count"] == 2  # frames 100, 150
        assert "skipped_dead_frames" in result
        assert "20240418_A01_e01_s0200" in result["skipped_dead_frames"]
        print("✓ Range operations safely skip DEAD frames")
        
        # Range that spans death with overwrite
        result = metadata.add_phenotype("BLUR", "user1", embryo_id="20240418_A01_e01", 
                                       target="150:250", overwrite_dead=True)
        assert result["count"] == 2  # frames 150, 250 (200 has DEAD which gets overwritten)
        assert "skipped_dead_frames" not in result
        print("✓ Range operations with overwrite_dead=True work")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_genotype_validation():
    """Test genotype validation and overwrite protection."""
    print("Testing genotype validation...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test valid genotype
        result = metadata.add_genotype("tmem67", "user1", "20240418_A01_e01", 
                                     allele="sa1423", zygosity="homozygous")
        assert result["operation"] == "add_genotype"
        assert result["previous_genotype"] is None
        print("✓ Valid genotype added successfully")
        
        # Test duplicate genotype without overwrite - should fail
        try:
            metadata.add_genotype("lmx1b", "user2", "20240418_A01_e01", zygosity="heterozygous")
            assert False, "Should have failed genotype overwrite protection"
        except ValueError as e:
            assert "already has genotype" in str(e)
            print("✓ Genotype overwrite protection works")
        
        # Test duplicate genotype with overwrite - should work
        result = metadata.add_genotype("lmx1b", "user2", "20240418_A01_e01", 
                                     zygosity="heterozygous", overwrite=True)
        assert result["gene"] == "lmx1b"
        assert result["previous_genotype"]["gene"] == "tmem67"
        print("✓ Genotype overwrite with overwrite=True works")
        
        # Test invalid gene
        try:
            metadata.add_genotype("invalid_gene", "user1", "20240418_A01_e02")
            assert False, "Should have failed invalid gene validation"
        except ValueError as e:
            assert "Invalid gene" in str(e)
            print("✓ Invalid gene validation works")
        
        # Test invalid zygosity
        try:
            metadata.add_genotype("tmem67", "user1", "20240418_A01_e02", zygosity="invalid")
            assert False, "Should have failed invalid zygosity validation"
        except ValueError as e:
            assert "Invalid zygosity" in str(e)
            print("✓ Invalid zygosity validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_treatment_validation():
    """Test treatment validation."""
    print("Testing treatment validation...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test valid treatment
        result = metadata.add_treatment("PTU", "user1", "20240418_A01_e01",
                                      temperature_celsius=28.5, 
                                      concentration="200μM",
                                      notes="24-48hpf pigment inhibition")
        assert result["operation"] == "add_treatment"
        assert result["treatment"] == "PTU"
        print("✓ Valid treatment added successfully")
        
        # Test multiple treatments on same embryo
        result = metadata.add_treatment("heat_shock", "user1", "20240418_A01_e01",
                                      temperature_celsius=37.0,
                                      notes="1hr at 30hpf")
        assert result["treatment"] == "heat_shock"
        
        # Check both treatments exist
        embryo_data = metadata.data["embryos"]["20240418_A01_e01"]
        assert len(embryo_data["treatments"]) == 2
        treatment_values = [t["value"] for t in embryo_data["treatments"]]
        assert "PTU" in treatment_values
        assert "heat_shock" in treatment_values
        print("✓ Multiple treatments per embryo work")
        
        # Test invalid treatment
        try:
            metadata.add_treatment("invalid_treatment", "user1", "20240418_A01_e02")
            assert False, "Should have failed invalid treatment validation"
        except ValueError as e:
            assert "Invalid treatment" in str(e)
            print("✓ Invalid treatment validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_error_message_quality():
    """Test that error messages are informative and actionable."""
    print("Testing error message quality...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Set up DEAD frame
        metadata.add_phenotype("DEAD", "user1", embryo_id="20240418_A01_e01", target="200")
        
        # Test DEAD permanence error message using snip approach (strict mode)
        try:
            metadata.add_phenotype("NORMAL", "user1", snip_ids=["20240418_A01_e01_s0250"])
            assert False, "Should have failed"
        except ValueError as e:
            error_msg = str(e)
            assert "DEAD permanence violation" in error_msg
            assert "already dead at frame" in error_msg
            assert "overwrite_dead=True" in error_msg
            print("✓ DEAD permanence error message is informative")
        
        # Test genotype conflict error message
        metadata.add_genotype("tmem67", "user1", "20240418_A01_e01", zygosity="homozygous")
        try:
            metadata.add_genotype("lmx1b", "user2", "20240418_A01_e01", zygosity="heterozygous")
            assert False, "Should have failed"
        except ValueError as e:
            error_msg = str(e)
            assert "already has genotype" in error_msg
            assert "overwrite=True" in error_msg
            print("✓ Genotype conflict error message is informative")
        
        # Test vocabulary error messages contain valid options
        try:
            metadata.add_phenotype("INVALID_PHENOTYPE", "user1", embryo_id="20240418_A01_e02")
            assert False, "Should have failed"
        except ValueError as e:
            error_msg = str(e)
            assert "Valid options:" in error_msg
            assert "NORMAL" in error_msg  # Should show valid phenotypes
            print("✓ Vocabulary error messages show valid options")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_complex_dead_scenarios():
    """Test complex DEAD scenarios."""
    print("Testing complex DEAD scenarios...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Scenario: Mark DEAD, then try to backfill earlier phenotypes
        metadata.add_phenotype("DEAD", "user1", embryo_id="20240418_A01_e01", target="200")
        
        # Should be able to add phenotypes before death
        result = metadata.add_phenotype("NORMAL", "user1", embryo_id="20240418_A01_e01", target="100:200")
        assert result["count"] == 2  # frames 100, 150 (200 skipped because DEAD)
        if "skipped_dead_frames" in result:
            assert "20240418_A01_e01_s0200" in result["skipped_dead_frames"]
        print("✓ Can backfill phenotypes before death, DEAD frame skipped")
        
        # Scenario: Change death frame (move earlier)
        result = metadata.add_phenotype("DEAD", "user1", embryo_id="20240418_A01_e01", 
                                       target="150", overwrite_dead=True)
        assert result["count"] == 1
        print("✓ Can change death frame with overwrite_dead=True")
        
        # Now frame 200 should be silently skipped (after death at frame 150)
        result = metadata.add_phenotype("EDEMA", "user1", embryo_id="20240418_A01_e01", target="200")
        assert result["count"] == 0  # Should be skipped
        assert "skipped_dead_frames" in result
        print("✓ DEAD permanence updated correctly after death frame change")
        
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
    """Run all Phase 3 validation tests."""
    print("=" * 60)
    print("PHASE 3 BUSINESS RULES VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_dead_exclusivity,
        test_dead_permanence,
        test_dead_safety_ranges,
        test_genotype_validation,
        test_treatment_validation,
        test_error_message_quality,
        test_complex_dead_scenarios
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
        print("🎉 ALL PHASE 3 TESTS PASSED!")
        print("✅ DEAD exclusivity validation working")
        print("✅ DEAD permanence validation working")
        print("✅ DEAD safety (silent skipping) working")
        print("✅ Controlled vocabulary validation working")
        print("✅ Enhanced error messages are clear and actionable")
        print("✅ Complex DEAD scenarios handled correctly")
        print("✅ Ready to proceed to Phase 4")
    else:
        print("❌ Some tests failed - need to fix issues before Phase 4")
    
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)