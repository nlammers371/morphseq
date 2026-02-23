#!/usr/bin/env python3
"""
Test Embryo Detection and Auto-Update Functionality

Tests the auto-detection and merging of new embryos from SAM2:
- New embryo detection
- New snip detection for existing embryos
- Preservation of existing annotations
- Update metadata tracking
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


def create_initial_sam2_data():
    """Create initial SAM2 data."""
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
                            "20240418_A01_t0101": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            }
        }
    }


def create_updated_sam2_data():
    """Create updated SAM2 data with new embryos and frames."""
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
                            "20240418_A01_t0101": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0150": {  # New frame
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e03": {"segmentation": {"counts": "test"}}  # New embryo
                                }
                            },
                            "20240418_A01_t0200": {  # Another new frame
                                "embryos": {
                                    "20240418_A01_e03": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    },
                    "20240418_B01": {  # New video
                        "images": {
                            "20240418_B01_t0050": {
                                "embryos": {
                                    "20240418_B01_e01": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            },
            "20240419": {  # New experiment
                "videos": {
                    "20240419_A01": {
                        "images": {
                            "20240419_A01_t0075": {
                                "embryos": {
                                    "20240419_A01_e01": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def initial_sam2_file():
    """Create temporary initial SAM2 file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_initial_sam2_data(), f)
        f.flush()
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)
    biology_path = Path(f.name).parent / f"{Path(f.name).stem}_biology.json"
    biology_path.unlink(missing_ok=True)


def test_new_embryo_detection(initial_sam2_file):
    """Test detection of completely new embryos."""
    # Create initial metadata with annotations
    metadata1 = EmbryoMetadata(str(initial_sam2_file))
    metadata1.add_phenotype("NORMAL", "user1", "20240418_A01_e01")
    metadata1.add_genotype("tmem67", "user1", "20240418_A01_e01", zygosity="homozygous")
    metadata1.save()
    
    # Update SAM2 file with new data
    with open(initial_sam2_file, 'w') as f:
        json.dump(create_updated_sam2_data(), f)
    
    # Load metadata again - should auto-detect new embryos
    biology_path = initial_sam2_file.parent / f"{initial_sam2_file.stem}_biology.json"
    metadata2 = EmbryoMetadata(str(initial_sam2_file), str(biology_path))
    
    # Check new embryos were detected
    new_embryos = ["20240418_A01_e03", "20240418_B01_e01", "20240419_A01_e01"]
    for embryo_id in new_embryos:
        assert embryo_id in metadata2.data["embryos"], f"New embryo not detected: {embryo_id}"
        
        # Check embryo structure
        embryo = metadata2.data["embryos"][embryo_id]
        assert embryo["embryo_id"] == embryo_id
        assert embryo["genotype"] is None  # New embryos start with no annotations
        assert embryo["treatments"] == []
    
    # Check new snips for new embryos
    e03 = metadata2.data["embryos"]["20240418_A01_e03"]
    assert "20240418_A01_e03_s0150" in e03["snips"]
    assert "20240418_A01_e03_s0200" in e03["snips"]
    assert len(e03["snips"]["20240418_A01_e03_s0150"]["phenotypes"]) == 0
    
    # Check existing annotations were preserved
    e01 = metadata2.data["embryos"]["20240418_A01_e01"]
    assert e01["genotype"]["gene"] == "tmem67"
    
    # Check existing phenotypes preserved
    for snip_id in ["20240418_A01_e01_s0100", "20240418_A01_e01_s0101"]:
        phenotypes = e01["snips"][snip_id]["phenotypes"]
        assert len(phenotypes) == 1
        assert phenotypes[0]["value"] == "NORMAL"


def test_new_snip_detection_existing_embryo(initial_sam2_file):
    """Test detection of new snips for existing embryos."""
    # Create initial metadata with annotations
    metadata1 = EmbryoMetadata(str(initial_sam2_file))
    metadata1.add_phenotype("EDEMA", "user1", "20240418_A01_e01")
    metadata1.add_phenotype("NORMAL", "user1", "20240418_A01_e02")
    metadata1.save()
    
    # Update SAM2 file
    with open(initial_sam2_file, 'w') as f:
        json.dump(create_updated_sam2_data(), f)
    
    # Load metadata again
    biology_path = initial_sam2_file.parent / f"{initial_sam2_file.stem}_biology.json"
    metadata2 = EmbryoMetadata(str(initial_sam2_file), str(biology_path))
    
    # Check new snips were added to existing embryos
    e01 = metadata2.data["embryos"]["20240418_A01_e01"]
    assert "20240418_A01_e01_s0150" in e01["snips"]  # New snip
    
    e02 = metadata2.data["embryos"]["20240418_A01_e02"]
    assert "20240418_A01_e02_s0150" in e02["snips"]  # New snip
    
    # Check existing snips still have annotations
    assert len(e01["snips"]["20240418_A01_e01_s0100"]["phenotypes"]) == 1
    assert e01["snips"]["20240418_A01_e01_s0100"]["phenotypes"][0]["value"] == "EDEMA"
    
    assert len(e02["snips"]["20240418_A01_e02_s0100"]["phenotypes"]) == 1
    assert e02["snips"]["20240418_A01_e02_s0100"]["phenotypes"][0]["value"] == "NORMAL"
    
    # Check new snips have no annotations
    assert len(e01["snips"]["20240418_A01_e01_s0150"]["phenotypes"]) == 0
    assert len(e02["snips"]["20240418_A01_e02_s0150"]["phenotypes"]) == 0


def test_metadata_tracking_updates(initial_sam2_file):
    """Test that metadata tracks update information."""
    # Create initial metadata
    metadata1 = EmbryoMetadata(str(initial_sam2_file))
    metadata1.save()
    
    # Check initial metadata
    initial_created = metadata1.data["metadata"]["created"]
    assert "updated" not in metadata1.data["metadata"]
    
    # Update SAM2 file
    with open(initial_sam2_file, 'w') as f:
        json.dump(create_updated_sam2_data(), f)
    
    # Load metadata again
    biology_path = initial_sam2_file.parent / f"{initial_sam2_file.stem}_biology.json"
    metadata2 = EmbryoMetadata(str(initial_sam2_file), str(biology_path))
    
    # Check metadata was updated
    assert metadata2.data["metadata"]["created"] == initial_created  # Preserved
    assert "updated" in metadata2.data["metadata"]  # Added update timestamp
    assert metadata2.data["metadata"]["updated"] != initial_created  # Different timestamp


def test_no_changes_scenario(initial_sam2_file):
    """Test behavior when SAM2 data hasn't changed."""
    # Create initial metadata with annotations
    metadata1 = EmbryoMetadata(str(initial_sam2_file))
    metadata1.add_phenotype("NORMAL", "user1", "20240418_A01_e01")
    metadata1.save()
    
    original_data = metadata1.data.copy()
    
    # Load metadata again with same SAM2 data
    biology_path = initial_sam2_file.parent / f"{initial_sam2_file.stem}_biology.json"
    metadata2 = EmbryoMetadata(str(initial_sam2_file), str(biology_path))
    
    # Check that data structure is essentially the same
    # (only timestamp may differ)
    assert len(metadata2.data["embryos"]) == len(original_data["embryos"])
    assert set(metadata2.data["embryos"].keys()) == set(original_data["embryos"].keys())
    
    # Check annotations preserved
    e01 = metadata2.data["embryos"]["20240418_A01_e01"]
    for snip_data in e01["snips"].values():
        if snip_data["phenotypes"]:  # If has phenotypes
            assert snip_data["phenotypes"][0]["value"] == "NORMAL"


def test_partial_overlap_scenarios(initial_sam2_file):
    """Test scenarios where only some embryos/frames change."""
    # Create initial metadata
    metadata1 = EmbryoMetadata(str(initial_sam2_file))
    metadata1.add_phenotype("EDEMA", "user1", "20240418_A01_e01")
    metadata1.add_phenotype("NORMAL", "user1", "20240418_A01_e02")
    metadata1.save()
    
    # Create partially updated SAM2 data (add frames but not new embryos)
    partial_update_data = create_initial_sam2_data()
    partial_update_data["experiments"]["20240418"]["videos"]["20240418_A01"]["images"]["20240418_A01_t0150"] = {
        "embryos": {
            "20240418_A01_e01": {"segmentation": {"counts": "test"}},
            "20240418_A01_e02": {"segmentation": {"counts": "test"}}
        }
    }
    
    # Update SAM2 file
    with open(initial_sam2_file, 'w') as f:
        json.dump(partial_update_data, f)
    
    # Load metadata again
    biology_path = initial_sam2_file.parent / f"{initial_sam2_file.stem}_biology.json"
    metadata2 = EmbryoMetadata(str(initial_sam2_file), str(biology_path))
    
    # Check only expected changes occurred
    assert len(metadata2.data["embryos"]) == 2  # No new embryos
    
    # Check new snips added
    e01 = metadata2.data["embryos"]["20240418_A01_e01"]
    e02 = metadata2.data["embryos"]["20240418_A01_e02"]
    assert "20240418_A01_e01_s0150" in e01["snips"]
    assert "20240418_A01_e02_s0150" in e02["snips"]
    
    # Check existing annotations preserved
    assert len(e01["snips"]["20240418_A01_e01_s0100"]["phenotypes"]) == 1
    assert e01["snips"]["20240418_A01_e01_s0100"]["phenotypes"][0]["value"] == "EDEMA"
    
    # Check new snips have no annotations
    assert len(e01["snips"]["20240418_A01_e01_s0150"]["phenotypes"]) == 0


def test_complex_annotation_preservation():
    """Test preservation of complex annotations during updates."""
    # Create SAM2 file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_initial_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        # Create metadata with complex annotations
        metadata1 = EmbryoMetadata(str(sam2_file))
        
        # Add multiple phenotypes to same embryo
        metadata1.add_phenotype("NORMAL", "user1", "20240418_A01_e01")
        metadata1.add_phenotype("EDEMA", "user2", "20240418_A01_e01")
        
        # Add genotype
        metadata1.add_genotype("tmem67", "user1", "20240418_A01_e01", 
                              allele="sa1423", zygosity="homozygous")
        
        # Add different annotations to other embryo
        metadata1.add_phenotype("DEAD", "user3", "20240418_A01_e02")
        metadata1.add_genotype("WT", "user3", "20240418_A01_e02", zygosity="homozygous")
        
        metadata1.save()
        
        # Update SAM2 with new data
        with open(sam2_file, 'w') as f:
            json.dump(create_updated_sam2_data(), f)
        
        # Load metadata again
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        metadata2 = EmbryoMetadata(str(sam2_file), str(biology_path))
        
        # Verify all complex annotations preserved
        e01 = metadata2.data["embryos"]["20240418_A01_e01"]
        e02 = metadata2.data["embryos"]["20240418_A01_e02"]
        
        # Check genotypes preserved
        assert e01["genotype"]["gene"] == "tmem67"
        assert e01["genotype"]["allele"] == "sa1423"
        assert e01["genotype"]["zygosity"] == "homozygous"
        assert e01["genotype"]["author"] == "user1"
        
        assert e02["genotype"]["gene"] == "WT"
        assert e02["genotype"]["zygosity"] == "homozygous"
        assert e02["genotype"]["author"] == "user3"
        
        # Check multiple phenotypes preserved
        for snip_data in e01["snips"].values():
            if snip_data["phenotypes"]:
                phenotype_values = [p["value"] for p in snip_data["phenotypes"]]
                assert "NORMAL" in phenotype_values
                assert "EDEMA" in phenotype_values
        
        for snip_data in e02["snips"].values():
            if snip_data["phenotypes"]:
                phenotype_values = [p["value"] for p in snip_data["phenotypes"]]
                assert "DEAD" in phenotype_values
        
        # Check new embryos have no annotations
        e03 = metadata2.data["embryos"]["20240418_A01_e03"]
        assert e03["genotype"] is None
        for snip_data in e03["snips"].values():
            assert len(snip_data["phenotypes"]) == 0
    
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests manually for development
    import tempfile
    import traceback
    
    print("Running embryo detection and auto-update tests...")
    
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
            json.dump(create_initial_sam2_data(), f)
            temp_file = Path(f.name)
        
        # Run individual tests
        tests = [
            (test_new_embryo_detection, temp_file),
            (test_new_snip_detection_existing_embryo, temp_file),
            (test_metadata_tracking_updates, temp_file),
            (test_no_changes_scenario, temp_file),
            (test_partial_overlap_scenarios, temp_file),
            test_complex_annotation_preservation
        ]
        
        passed = 0
        failed = 0
        
        for test_item in tests:
            if isinstance(test_item, tuple):
                test_func, *args = test_item
            else:
                test_func = test_item
                args = []
            
            try:
                print(f"Running {test_func.__name__}...", end=" ")
                test_func(*args)
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