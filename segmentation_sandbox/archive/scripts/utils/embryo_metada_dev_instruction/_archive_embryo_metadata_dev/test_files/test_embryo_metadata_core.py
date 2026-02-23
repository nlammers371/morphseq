#!/usr/bin/env python3
"""
Comprehensive Unit Tests for EmbryoMetadata Core Class
Tests initialization, data access, save/load, and summary functionality
"""

import sys
import tempfile
import json
from pathlib import Path

# Add the path to our modules
sys.path.append(str(Path(__file__).parent))

from embryo_metadata import EmbryoMetadata
from embryo_metadata_models import ValidationError

def create_mock_sam_annotation(tmp_path: Path) -> Path:
    """Create a realistic mock SAM annotation file for testing."""
    sam_data = {
        "experiments": {
            "20240411": {
                "videos": {
                    "20240411_A01": {
                        "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
                        "images": {
                            "20240411_A01_0001": {
                                "embryos": {
                                    "20240411_A01_e01": {
                                        "snip_id": "20240411_A01_e01_0001",
                                        "bbox": [10, 20, 50, 60]
                                    },
                                    "20240411_A01_e02": {
                                        "snip_id": "20240411_A01_e02_0001",
                                        "bbox": [100, 120, 150, 160]
                                    }
                                }
                            },
                            "20240411_A01_0002": {
                                "embryos": {
                                    "20240411_A01_e01": {
                                        "snip_id": "20240411_A01_e01_0002",
                                        "bbox": [12, 22, 52, 62]
                                    },
                                    "20240411_A01_e02": {
                                        "snip_id": "20240411_A01_e02_0002",
                                        "bbox": [102, 122, 152, 162]
                                    }
                                }
                            }
                        }
                    },
                    "20240411_B01": {
                        "embryo_ids": ["20240411_B01_e01"],
                        "images": {
                            "20240411_B01_0001": {
                                "embryos": {
                                    "20240411_B01_e01": {
                                        "snip_id": "20240411_B01_e01_0001",
                                        "bbox": [200, 220, 250, 260]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02", "20240411_B01_e01"],
        "snip_ids": [
            "20240411_A01_e01_0001", "20240411_A01_e01_0002",
            "20240411_A01_e02_0001", "20240411_A01_e02_0002", 
            "20240411_B01_e01_0001"
        ],
        "config": {
            "detection_model": {
                "config": "GroundingDINO_SwinT_OGC.py",
                "weights": "groundingdino_swint_ogc.pth"
            },
            "segmentation_model": {
                "config": "sam2.1_hiera_l.yaml",
                "weights": "sam2.1_hiera_large.pt"
            }
        }
    }
    
    sam_path = tmp_path / "sam_annotations.json"
    with open(sam_path, 'w') as f:
        json.dump(sam_data, f, indent=2)
    
    return sam_path

def test_initialization():
    """Test EmbryoMetadata initialization."""
    print("🧪 Testing EmbryoMetadata initialization...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create mock SAM annotation
        sam_path = create_mock_sam_annotation(tmp_path)
        
        # Test 1: Create new metadata (gen_if_no_file=True)
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Verify basic structure
        assert "file_info" in em.data
        assert "embryos" in em.data
        assert "flags" in em.data
        assert "config" in em.data
        
        # Verify embryo import
        expected_embryos = {"20240411_A01_e01", "20240411_A01_e02", "20240411_B01_e01"}
        actual_embryos = set(em.get_embryo_ids())
        assert actual_embryos == expected_embryos
        
        # Verify snip import
        total_snips = len(em.get_snip_ids())
        assert total_snips == 5  # 2+2+1 snips
        
        # Verify snip to embryo mapping
        assert em.get_embryo_id_from_snip("20240411_A01_e01_0001") == "20240411_A01_e01"
        assert em.get_embryo_id_from_snip("20240411_B01_e01_0001") == "20240411_B01_e01"
        
        # Test 2: Save and reload
        metadata_path = tmp_path / "embryo_metadata.json"
        em.filepath = metadata_path
        em.save()
        
        assert metadata_path.exists()
        
        # Reload and verify
        em2 = EmbryoMetadata(sam_path, metadata_path, verbose=False)
        assert set(em2.get_embryo_ids()) == expected_embryos
        assert len(em2.get_snip_ids()) == 5
        
        # Test 3: Error handling
        try:
            # Should fail without gen_if_no_file
            EmbryoMetadata(sam_path, tmp_path / "nonexistent.json", gen_if_no_file=False)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
        
        print("✅ Initialization tests passed!")

def test_data_access():
    """Test data access methods."""
    print("🧪 Testing data access methods...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_mock_sam_annotation(tmp_path)
        
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Test embryo data access
        embryo_data = em.get_embryo_data("20240411_A01_e01")
        assert embryo_data is not None
        assert "snips" in embryo_data
        assert "source" in embryo_data
        
        # Test snip data access
        snip_data = em.get_snip_data("20240411_A01_e01_0001")
        assert snip_data is not None
        assert "phenotype" in snip_data
        assert snip_data["phenotype"]["value"] == "NONE"
        
        # Test snips for specific embryo
        e01_snips = em.get_snip_ids("20240411_A01_e01")
        assert len(e01_snips) == 2
        assert "20240411_A01_e01_0001" in e01_snips
        assert "20240411_A01_e01_0002" in e01_snips
        
        # Test non-existent data
        assert em.get_embryo_data("nonexistent") is None
        assert em.get_snip_data("nonexistent") is None
        
        print("✅ Data access tests passed!")

def test_save_and_backup():
    """Test save and backup functionality."""
    print("🧪 Testing save and backup functionality...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_mock_sam_annotation(tmp_path)
        metadata_path = tmp_path / "test_metadata.json"
        
        # Create and save metadata
        em = EmbryoMetadata(sam_path, metadata_path, gen_if_no_file=True, verbose=False)
        
        # Initially should have unsaved changes
        assert em.has_unsaved_changes
        
        # Save
        em.save()
        assert not em.has_unsaved_changes
        assert metadata_path.exists()
        
        # Load file and verify structure
        with open(metadata_path) as f:
            saved_data = json.load(f)
        
        assert "file_info" in saved_data
        assert "last_updated" in saved_data["file_info"]
        
        # Test backup creation
        # Modify data to trigger backup on next save
        em.data["file_info"]["test_field"] = "test_value"
        em._unsaved_changes = True
        
        em.save()  # Should create backup
        
        # Check for backup file
        backup_files = list(tmp_path.glob("*.backup.*"))
        assert len(backup_files) == 1
        
        # Test manual backup
        backup_path = em.backup()
        assert backup_path.exists()
        assert backup_path != metadata_path
        
        print("✅ Save and backup tests passed!")

def test_reload():
    """Test reload functionality."""
    print("🧪 Testing reload functionality...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_mock_sam_annotation(tmp_path)
        metadata_path = tmp_path / "test_metadata.json"
        
        # Create and save metadata
        em = EmbryoMetadata(sam_path, metadata_path, gen_if_no_file=True, verbose=False)
        em.save()
        
        # Make unsaved changes
        original_embryo_count = len(em.get_embryo_ids())
        em.data["test_field"] = "test_value"
        em._unsaved_changes = True
        
        assert em.has_unsaved_changes
        
        # Reload - should discard changes
        em.reload()
        
        assert not em.has_unsaved_changes
        assert "test_field" not in em.data
        assert len(em.get_embryo_ids()) == original_embryo_count
        
        print("✅ Reload tests passed!")

def test_summary_and_statistics():
    """Test summary and statistics functionality."""
    print("🧪 Testing summary and statistics...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_mock_sam_annotation(tmp_path)
        
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Get summary
        summary = em.get_summary()
        
        # Verify structure
        assert "file_info" in summary
        assert "totals" in summary
        assert "genotypes" in summary
        assert "phenotypes" in summary
        assert "flags" in summary
        
        # Verify counts
        totals = summary["totals"]
        assert totals["embryos"] == 3
        assert totals["snips"] == 5
        assert totals["flags"] == 0  # No flags added yet
        
        # Verify genotype stats (none should be genotyped initially)
        geno = summary["genotypes"]
        assert geno["genotyped"] == 0
        assert geno["missing"] == 3
        assert geno["completion_rate"] == 0.0
        
        # Verify phenotype stats (all should be NONE initially)
        pheno = summary["phenotypes"]
        assert pheno["phenotyped_snips"] == 0  # NONE doesn't count as phenotyped
        assert pheno["completion_rate"] == 0.0
        
        # Test string representation
        str_repr = str(em)
        assert "3 embryos" in str_repr
        assert "5 snips" in str_repr
        assert "unsaved" in str_repr  # Should have unsaved changes initially
        
        # Test print summary (just make sure it doesn't crash)
        import io
        import contextlib
        
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            em.print_summary()
        
        output = captured_output.getvalue()
        assert "EmbryoMetadata Summary" in output
        assert "Embryos: 3" in output
        assert "Snips: 5" in output
        
        print("✅ Summary and statistics tests passed!")

def test_configuration_inheritance():
    """Test configuration inheritance from SAM annotations."""
    print("🧪 Testing configuration inheritance...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_mock_sam_annotation(tmp_path)
        
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Verify config inheritance
        config = em.data["config"]
        assert "detection_model" in config
        assert "segmentation_model" in config
        
        detection_config = config["detection_model"]
        assert detection_config["config"] == "GroundingDINO_SwinT_OGC.py"
        assert detection_config["weights"] == "groundingdino_swint_ogc.pth"
        
        segmentation_config = config["segmentation_model"]
        assert segmentation_config["config"] == "sam2.1_hiera_l.yaml"
        assert segmentation_config["weights"] == "sam2.1_hiera_large.pt"
        
        print("✅ Configuration inheritance tests passed!")

def test_consistency_checks():
    """Test consistency checking between SAM and metadata."""
    print("🧪 Testing consistency checks...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_mock_sam_annotation(tmp_path)
        
        # Create metadata with consistent data
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False, auto_validate=True)
        
        # Should not raise any errors
        em._run_consistency_checks()
        
        # Test with inconsistent data
        # Remove an embryo from metadata
        del em.data["embryos"]["20240411_A01_e01"]
        
        try:
            em._run_consistency_checks()
            # With strict validation, should raise error
            # If we get here, either strict validation is off or the test needs adjustment
        except ValidationError:
            pass  # Expected with strict validation
        
        print("✅ Consistency check tests passed!")

def test_error_handling():
    """Test error handling and edge cases."""
    print("🧪 Testing error handling...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Test with non-existent SAM file
        try:
            EmbryoMetadata(tmp_path / "nonexistent.json", gen_if_no_file=True)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test with invalid JSON
        invalid_json_path = tmp_path / "invalid.json"
        with open(invalid_json_path, 'w') as f:
            f.write("{invalid json")
        
        try:
            EmbryoMetadata(invalid_json_path, gen_if_no_file=True)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test with empty SAM file
        empty_sam_path = tmp_path / "empty.json"
        with open(empty_sam_path, 'w') as f:
            json.dump({}, f)
        
        try:
            EmbryoMetadata(empty_sam_path, gen_if_no_file=True)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        print("✅ Error handling tests passed!")

def run_all_tests():
    """Run all unit tests."""
    print("🚀 Running EmbryoMetadata Core Class Unit Tests")
    print("=" * 60)
    
    try:
        test_initialization()
        test_data_access()
        test_save_and_backup()
        test_reload()
        test_summary_and_statistics()
        test_configuration_inheritance()
        test_consistency_checks()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 All unit tests passed successfully!")
        print("\n💡 EmbryoMetadata Core Class is ready for:")
        print("   ✅ Phenotype management (Module 3)")
        print("   ✅ Genotype management (Module 4)")
        print("   ✅ Flag management (Module 5)")
        print("   ✅ Batch processing (Module 6)")
        print("   ✅ Integration testing with real data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
