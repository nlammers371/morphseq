"""
Unit tests for critical validation logic in EmbryoMetadata and AnnotationBatch.

Tests focus on:
- DEAD frame validation and skipping logic
- Configuration loading and fallback behavior  
- Input validation for files and JSON structure
- Batch operations and conflict resolution
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from .embryo_metadata import EmbryoMetadata
from .annotation_batch import AnnotationBatch


class TestEmbryoMetadata:
    """Test critical validation logic in EmbryoMetadata."""
    
    def test_dead_frame_skipping_logic(self):
        """Test that DEAD frames are properly skipped without exceptions."""
        # Create minimal SAM2 data structure
        sam2_data = {
            "experiments": {
                "exp1": {
                    "videos": {
                        "vid1": {
                            "images": {
                                "test_e01_t0100": {"embryos": {"test_e01": {}}},
                                "test_e01_t0200": {"embryos": {"test_e01": {}}}
                            }
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sam2_data, f)
            sam2_path = f.name
        
        try:
            # Initialize metadata
            metadata = EmbryoMetadata(sam2_path)
            
            # Mark first frame as DEAD
            metadata.add_phenotype("DEAD", "test_author", embryo_id="test_e01", target="100")
            
            # Try to add NORMAL to same frame range - should skip dead frame
            result = metadata.add_phenotype("NORMAL", "test_author", embryo_id="test_e01", target="100:200")
            
            # Should have skipped the dead frame (100) but applied to frame 200  
            assert result["skipped_count"] == 1
            assert result["count"] == 1  # Only applied to frame 200
            assert any("100" in snip_id for snip_id in result["skipped_dead_frames"])
            
        finally:
            Path(sam2_path).unlink()
    
    def test_should_skip_dead_frame_method(self):
        """Test the _should_skip_dead_frame boolean method works correctly."""
        sam2_data = {
            "experiments": {
                "exp1": {
                    "videos": {
                        "vid1": {
                            "images": {
                                "test_e01_t0100": {"embryos": {"test_e01": {}}}
                            }
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sam2_data, f)
            sam2_path = f.name
        
        try:
            metadata = EmbryoMetadata(sam2_path)
            
            # Add DEAD phenotype first
            metadata.add_phenotype("DEAD", "test_author", embryo_id="test_e01", target="100")
            
            # Test the skip method directly
            should_skip = metadata._should_skip_dead_frame("test_e01_s0100", "NORMAL")
            assert should_skip == True
            
            # Test that DEAD can be added to DEAD frame (no skip)
            should_skip_dead = metadata._should_skip_dead_frame("test_e01_s0100", "DEAD")
            assert should_skip_dead == False
            
        finally:
            Path(sam2_path).unlink()
    
    def test_config_loading_with_fallback(self):
        """Test configuration loading with fallback to defaults."""
        # Test with valid config
        test_config = {
            "phenotypes": ["TEST_PHENOTYPE", "ANOTHER_PHENOTYPE"],
            "genes": ["test_gene"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = Path(f.name)
        
        try:
            # Load config
            EmbryoMetadata._load_config(config_path)
            
            # Check that config was loaded
            assert "TEST_PHENOTYPE" in EmbryoMetadata.VALID_PHENOTYPES
            assert "test_gene" in EmbryoMetadata.VALID_GENES
            
        finally:
            config_path.unlink()
            # Reset to defaults
            EmbryoMetadata._load_config()
    
    def test_config_fallback_on_invalid_json(self):
        """Test fallback to defaults when config JSON is invalid."""
        # Store original values
        original_phenotypes = EmbryoMetadata.VALID_PHENOTYPES.copy()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            config_path = Path(f.name)
        
        try:
            # Should not crash, should fall back to defaults
            EmbryoMetadata._load_config(config_path)
            
            # Should still have default phenotypes
            assert "NORMAL" in EmbryoMetadata.VALID_PHENOTYPES
            
        finally:
            config_path.unlink()
    
    def test_input_validation_missing_sam2_file(self):
        """Test that missing SAM2 file raises proper error."""
        with pytest.raises(FileNotFoundError, match="SAM2 file not found"):
            EmbryoMetadata("/nonexistent/path.json")
    
    def test_input_validation_invalid_sam2_json(self):
        """Test that invalid SAM2 JSON raises proper error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            sam2_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid JSON in SAM2 file"):
                EmbryoMetadata(sam2_path)
        finally:
            Path(sam2_path).unlink()
    
    def test_input_validation_missing_experiments_key(self):
        """Test that SAM2 file without experiments key raises proper error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"wrong_key": "value"}, f)
            sam2_path = f.name
        
        try:
            with pytest.raises(ValueError, match="missing 'experiments' key"):
                EmbryoMetadata(sam2_path)
        finally:
            Path(sam2_path).unlink()


class TestAnnotationBatch:
    """Test critical validation logic in AnnotationBatch."""
    
    def test_batch_config_loading(self):
        """Test that AnnotationBatch loads same config as EmbryoMetadata."""
        # Create test config
        test_config = {
            "phenotypes": ["BATCH_TEST_PHENOTYPE"],
            "genes": ["batch_gene"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = Path(f.name)
        
        # Create batch data structure
        batch_data = {
            "metadata": {"version": "test"},
            "embryos": {
                "test_e01": {
                    "embryo_id": "test_e01",
                    "snips": {
                        "test_e01_s0100": {
                            "snip_id": "test_e01_s0100",
                            "frame_number": 100,
                            "phenotypes": []
                        }
                    }
                }
            }
        }
        
        try:
            # Temporarily replace config location
            with patch('builtins.open', mock_open(read_data=json.dumps(test_config))):
                with patch.object(Path, 'exists', return_value=True):
                    batch = AnnotationBatch(batch_data, "test_author")
                    
                    # Should have loaded test config
                    assert "BATCH_TEST_PHENOTYPE" in batch.VALID_PHENOTYPES
                    assert "batch_gene" in batch.VALID_GENES
        
        finally:
            if config_path.exists():
                config_path.unlink()
    
    def test_batch_validation_uses_loaded_config(self):
        """Test that batch validation uses loaded configuration."""
        batch_data = {
            "metadata": {"version": "test"},
            "embryos": {
                "test_e01": {
                    "embryo_id": "test_e01",
                    "snips": {
                        "test_e01_s0100": {
                            "snip_id": "test_e01_s0100",
                            "frame_number": 100,
                            "phenotypes": []
                        }
                    }
                }
            }
        }
        
        batch = AnnotationBatch(batch_data, "test_author")
        
        # Override config with test data
        batch.VALID_PHENOTYPES = ["ONLY_THIS_PHENOTYPE"]
        
        # Should accept valid phenotype
        result = batch.add_phenotype("ONLY_THIS_PHENOTYPE", embryo_id="test_e01", target="100")
        assert result["count"] == 1
        
        # Should reject invalid phenotype
        with pytest.raises(ValueError, match="Invalid phenotype"):
            batch.add_phenotype("INVALID_PHENOTYPE", embryo_id="test_e01", target="100")
    
    def test_batch_dead_frame_skipping(self):
        """Test that batch operations properly skip DEAD frames."""
        batch_data = {
            "metadata": {"version": "test"},
            "embryos": {
                "test_e01": {
                    "embryo_id": "test_e01",
                    "snips": {
                        "test_e01_s0100": {
                            "snip_id": "test_e01_s0100",
                            "frame_number": 100,
                            "phenotypes": [{"value": "DEAD", "author": "test", "timestamp": "2024-01-01"}]
                        },
                        "test_e01_s0200": {
                            "snip_id": "test_e01_s0200", 
                            "frame_number": 200,
                            "phenotypes": []
                        }
                    }
                }
            }
        }
        
        batch = AnnotationBatch(batch_data, "test_author")
        
        # Try to add NORMAL to range including dead frame
        result = batch.add_phenotype("NORMAL", embryo_id="test_e01", target="100:300")
        
        # Should skip frame 100 (dead) but apply to frame 200
        assert result["skipped_count"] == 1
        assert result["count"] == 1
        assert "test_e01_s0100" in result["skipped_dead_frames"]


def test_integration_metadata_and_batch():
    """Test integration between EmbryoMetadata and AnnotationBatch."""
    # Create minimal SAM2 data
    sam2_data = {
        "experiments": {
            "exp1": {
                "videos": {
                    "vid1": {
                        "images": {
                            "test_e01_t0100": {"embryos": {"test_e01": {}}},
                            "test_e01_t0200": {"embryos": {"test_e01": {}}}
                        }
                    }
                }
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sam2_data, f)
        sam2_path = f.name
    
    try:
        # Create metadata
        metadata = EmbryoMetadata(sam2_path)
        
        # Initialize batch
        batch = metadata.initialize_batch(mode="skeleton", author="test_author")
        
        # Add some annotations to batch
        batch.add_phenotype("NORMAL", embryo_id="test_e01", target="all")
        batch.add_genotype("WT", embryo_id="test_e01")
        
        # Apply batch back to metadata
        result = metadata.apply_batch(batch)
        
        # Should have successfully applied annotations
        assert result["applied_count"] > 0
        assert result["errors"] == []
        
        # Verify annotations were applied
        summary = metadata.get_embryo_summary("test_e01")
        assert summary["phenotype_counts"]["NORMAL"] >= 1
        assert summary["genotype"]["gene"] == "WT"
        
    finally:
        Path(sam2_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])