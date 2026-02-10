#!/usr/bin/env python3
"""
Test MVP Basic Functionality

Tests core EmbryoMetadata functionality:
- Loading SAM2 data
- Basic phenotype addition
- Save/reload cycle
- Data integrity
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add scripts to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from annotations.embryo_metadata import EmbryoMetadata


def create_test_sam2_data():
    """Create minimal SAM2 data for testing."""
    return {
        "experiments": {
            "20240418": {
                "videos": {
                    "20240418_A01": {
                        "images": {
                            "20240418_A01_t0100": {
                                "embryos": {
                                    "20240418_A01_e01": {
                                        "segmentation": {"counts": "test", "size": [100, 100]},
                                        "mask_confidence": 0.85
                                    },
                                    "20240418_A01_e02": {
                                        "segmentation": {"counts": "test", "size": [100, 100]},
                                        "mask_confidence": 0.92
                                    }
                                }
                            },
                            "20240418_A01_t0101": {
                                "embryos": {
                                    "20240418_A01_e01": {
                                        "segmentation": {"counts": "test", "size": [100, 100]},
                                        "mask_confidence": 0.88
                                    }
                                }
                            },
                            "20240418_A01_t0150": {
                                "embryos": {
                                    "20240418_A01_e01": {
                                        "segmentation": {"counts": "test", "size": [100, 100]},
                                        "mask_confidence": 0.90
                                    },
                                    "20240418_A01_e02": {
                                        "segmentation": {"counts": "test", "size": [100, 100]},
                                        "mask_confidence": 0.95
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def temp_sam2_file():
    """Create temporary SAM2 file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)
    # Also cleanup potential biology file
    biology_path = Path(f.name).parent / f"{Path(f.name).stem}_biology.json"
    biology_path.unlink(missing_ok=True)


def test_embryo_metadata_initialization(temp_sam2_file):
    """Test basic initialization from SAM2 data."""
    metadata = EmbryoMetadata(str(temp_sam2_file))
    
    # Check basic structure
    assert "metadata" in metadata.data
    assert "embryos" in metadata.data
    assert metadata.data["metadata"]["version"] == "simplified_v1"
    
    # Check embryos were extracted
    embryos = metadata.data["embryos"]
    assert "20240418_A01_e01" in embryos
    assert "20240418_A01_e02" in embryos
    
    # Check embryo structure
    e01 = embryos["20240418_A01_e01"]
    assert e01["embryo_id"] == "20240418_A01_e01"
    assert e01["experiment_id"] == "20240418"
    assert e01["video_id"] == "20240418_A01"
    assert e01["genotype"] is None
    assert e01["treatments"] == []
    assert isinstance(e01["snips"], dict)
    
    # Check snips were created
    snips = e01["snips"]
    assert "20240418_A01_e01_s0100" in snips
    assert "20240418_A01_e01_s0101" in snips
    assert "20240418_A01_e01_s0150" in snips
    
    # Check snip structure
    snip = snips["20240418_A01_e01_s0100"]
    assert snip["snip_id"] == "20240418_A01_e01_s0100"
    assert snip["frame_number"] == 100
    assert snip["phenotypes"] == []
    assert snip["flags"] == []


def test_add_phenotype_basic(temp_sam2_file):
    """Test basic phenotype addition."""
    metadata = EmbryoMetadata(str(temp_sam2_file))
    
    # Add phenotype to embryo
    result = metadata.add_phenotype("EDEMA", "test_user", "20240418_A01_e01", target="all")
    
    # Check result
    assert result["operation"] == "add_phenotype"
    assert result["phenotype"] == "EDEMA"
    assert result["embryo_id"] == "20240418_A01_e01"
    assert result["target"] == "all"
    assert result["count"] == 3  # Should be applied to 3 snips
    assert len(result["applied_to"]) == 3
    
    # Check that phenotype was actually added
    e01 = metadata.data["embryos"]["20240418_A01_e01"]
    for snip_data in e01["snips"].values():
        phenotypes = snip_data["phenotypes"]
        assert len(phenotypes) == 1
        assert phenotypes[0]["value"] == "EDEMA"
        assert phenotypes[0]["author"] == "test_user"
        assert "timestamp" in phenotypes[0]


def test_add_genotype(temp_sam2_file):
    """Test genotype addition."""
    metadata = EmbryoMetadata(str(temp_sam2_file))
    
    # Add genotype
    result = metadata.add_genotype("tmem67", "test_user", "20240418_A01_e01", 
                                 allele="sa1423", zygosity="homozygous")
    
    # Check result
    assert result["operation"] == "add_genotype"
    assert result["gene"] == "tmem67"
    assert result["embryo_id"] == "20240418_A01_e01"
    assert result["zygosity"] == "homozygous"
    
    # Check that genotype was actually added
    genotype = metadata.data["embryos"]["20240418_A01_e01"]["genotype"]
    assert genotype["gene"] == "tmem67"
    assert genotype["allele"] == "sa1423"
    assert genotype["zygosity"] == "homozygous"
    assert genotype["author"] == "test_user"
    assert "timestamp" in genotype


def test_validation_errors(temp_sam2_file):
    """Test validation error handling."""
    metadata = EmbryoMetadata(str(temp_sam2_file))
    
    # Test invalid phenotype
    with pytest.raises(ValueError, match="Invalid phenotype"):
        metadata.add_phenotype("INVALID_PHENOTYPE", "test_user", "20240418_A01_e01")
    
    # Test invalid embryo ID
    with pytest.raises(ValueError, match="Embryo .* not found"):
        metadata.add_phenotype("EDEMA", "test_user", "invalid_embryo_id")
    
    # Test invalid target (MVP only supports 'all')
    with pytest.raises(ValueError, match="MVP only supports target='all'"):
        metadata.add_phenotype("EDEMA", "test_user", "20240418_A01_e01", target="30:50")
    
    # Test invalid gene
    with pytest.raises(ValueError, match="Invalid gene"):
        metadata.add_genotype("invalid_gene", "test_user", "20240418_A01_e01")
    
    # Test invalid zygosity
    with pytest.raises(ValueError, match="Invalid zygosity"):
        metadata.add_genotype("tmem67", "test_user", "20240418_A01_e01", zygosity="invalid")


def test_save_and_reload(temp_sam2_file):
    """Test save/reload cycle maintains data integrity."""
    # Create and populate metadata
    metadata1 = EmbryoMetadata(str(temp_sam2_file))
    metadata1.add_phenotype("EDEMA", "test_user", "20240418_A01_e01")
    metadata1.add_genotype("tmem67", "test_user", "20240418_A01_e01", zygosity="homozygous")
    
    # Save
    metadata1.save()
    
    # Reload from saved file
    biology_path = Path(str(temp_sam2_file)).parent / f"{Path(str(temp_sam2_file)).stem}_biology.json"
    metadata2 = EmbryoMetadata(str(temp_sam2_file), str(biology_path))
    
    # Check data integrity
    assert metadata1.data == metadata2.data
    
    # Check specific annotations survived
    e01 = metadata2.data["embryos"]["20240418_A01_e01"]
    assert e01["genotype"]["gene"] == "tmem67"
    
    # Check phenotypes
    for snip_data in e01["snips"].values():
        assert len(snip_data["phenotypes"]) == 1
        assert snip_data["phenotypes"][0]["value"] == "EDEMA"


def test_embryo_summary(temp_sam2_file):
    """Test embryo summary functionality."""
    metadata = EmbryoMetadata(str(temp_sam2_file))
    
    # Add some annotations
    metadata.add_phenotype("EDEMA", "test_user", "20240418_A01_e01")
    metadata.add_phenotype("NORMAL", "test_user", "20240418_A01_e02")
    metadata.add_genotype("tmem67", "test_user", "20240418_A01_e01", zygosity="homozygous")
    
    # Get summary
    summary = metadata.get_embryo_summary("20240418_A01_e01")
    
    assert summary["embryo_id"] == "20240418_A01_e01"
    assert summary["experiment_id"] == "20240418"
    assert summary["video_id"] == "20240418_A01"
    assert summary["total_snips"] == 3
    assert summary["phenotype_counts"]["EDEMA"] == 3
    assert summary["genotype"]["gene"] == "tmem67"


def test_auto_update_from_sam2(temp_sam2_file):
    """Test auto-detection of new embryos/snips from SAM2."""
    # Create initial metadata
    metadata1 = EmbryoMetadata(str(temp_sam2_file))
    metadata1.add_phenotype("NORMAL", "test_user", "20240418_A01_e01")
    metadata1.save()
    
    # Modify SAM2 data to add new frames
    sam2_data = create_test_sam2_data()
    sam2_data["experiments"]["20240418"]["videos"]["20240418_A01"]["images"]["20240418_A01_t0200"] = {
        "embryos": {
            "20240418_A01_e01": {
                "segmentation": {"counts": "test", "size": [100, 100]},
                "mask_confidence": 0.87
            },
            "20240418_A01_e03": {  # New embryo
                "segmentation": {"counts": "test", "size": [100, 100]},
                "mask_confidence": 0.91
            }
        }
    }
    
    # Save updated SAM2 data
    with open(temp_sam2_file, 'w') as f:
        json.dump(sam2_data, f)
    
    # Load metadata again - should auto-detect changes
    biology_path = Path(str(temp_sam2_file)).parent / f"{Path(str(temp_sam2_file)).stem}_biology.json"
    metadata2 = EmbryoMetadata(str(temp_sam2_file), str(biology_path))
    
    # Check new embryo was detected
    assert "20240418_A01_e03" in metadata2.data["embryos"]
    
    # Check new snip was added to existing embryo
    e01_snips = metadata2.data["embryos"]["20240418_A01_e01"]["snips"]
    assert "20240418_A01_e01_s0200" in e01_snips
    
    # Check existing annotations were preserved
    original_snips = ["20240418_A01_e01_s0100", "20240418_A01_e01_s0101", "20240418_A01_e01_s0150"]
    for snip_id in original_snips:
        assert len(e01_snips[snip_id]["phenotypes"]) == 1
        assert e01_snips[snip_id]["phenotypes"][0]["value"] == "NORMAL"
    
    # Check new snip has no annotations yet
    assert len(e01_snips["20240418_A01_e01_s0200"]["phenotypes"]) == 0


def test_get_stats(temp_sam2_file):
    """Test overall statistics functionality."""
    metadata = EmbryoMetadata(str(temp_sam2_file))
    
    # Add some annotations
    metadata.add_phenotype("EDEMA", "test_user", "20240418_A01_e01")
    metadata.add_genotype("tmem67", "test_user", "20240418_A01_e01")
    
    stats = metadata.get_stats()
    
    assert stats["embryo_count"] == 2
    assert stats["total_snips"] == 5  # e01: 3 snips, e02: 2 snips
    assert stats["total_phenotypes"] == 3  # EDEMA applied to 3 snips
    assert stats["genotyped_embryos"] == 1
    assert "source_sam2" in stats
    assert "created" in stats


if __name__ == "__main__":
    # Run tests manually for development
    import tempfile
    import traceback
    
    print("Running MVP basic functionality tests...")
    
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
            json.dump(create_test_sam2_data(), f)
            temp_file = Path(f.name)
        
        # Run individual tests
        tests = [
            test_embryo_metadata_initialization,
            test_add_phenotype_basic,
            test_add_genotype,
            test_validation_errors,
            test_save_and_reload,
            test_embryo_summary,
            test_auto_update_from_sam2,
            test_get_stats
        ]
        
        passed = 0
        failed = 0
        
        for test_func in tests:
            try:
                print(f"Running {test_func.__name__}...", end=" ")
                test_func(temp_file)
                print("PASS")
                passed += 1
            except Exception as e:
                print(f"FAIL: {e}")
                traceback.print_exc()
                failed += 1
            
            # Cleanup biology file between tests
            biology_path = temp_file.parent / f"{temp_file.stem}_biology.json"
            biology_path.unlink(missing_ok=True)
        
        print(f"\nTest Results: {passed} passed, {failed} failed")
        
        # Cleanup
        temp_file.unlink(missing_ok=True)
        biology_path.unlink(missing_ok=True)
        
    except Exception as e:
        print(f"Test setup failed: {e}")
        traceback.print_exc()