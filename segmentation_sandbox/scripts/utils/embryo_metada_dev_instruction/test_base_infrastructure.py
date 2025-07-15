#!/usr/bin/env python3
"""
Test the BaseAnnotationParser and embryo_metadata_utils
"""

import sys
import tempfile
import json
from pathlib import Path

# Add the path to our modules
sys.path.append(str(Path(__file__).parent))

from base_annotation_parser import BaseAnnotationParser
from embryo_metadata_utils import (
    validate_path, load_json, save_json, IdParser, 
    PerformanceUtils, validate_embryo_metadata_structure
)

def test_base_annotation_parser():
    """Test the BaseAnnotationParser functionality."""
    print("🧪 Testing BaseAnnotationParser...")
    
    # Create temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        test_data = {"test": "data", "timestamp": "2024-01-01T00:00:00"}
        json.dump(test_data, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        # Test initialization
        parser = BaseAnnotationParser(tmp_path, verbose=False)
        assert parser.filepath == tmp_path
        assert not parser.has_unsaved_changes
        
        # Test JSON loading
        data = parser.load_json()
        assert data["test"] == "data"
        
        # Test ID parsing
        test_ids = [
            ("20240411", "experiment"),
            ("20240411_A01", "video"),
            ("20240411_A01_0001", "image"),
            ("20240411_A01_e01", "embryo"),
            ("20240411_A01_e01_0001", "snip"),
            ("invalid_id", "unknown")
        ]
        
        for test_id, expected_type in test_ids:
            parsed = parser.parse_id(test_id)
            assert parsed["type"] == expected_type, f"ID {test_id} should be {expected_type}, got {parsed['type']}"
        
        # Test embryo ID extraction from snip
        embryo_id = parser.get_embryo_id_from_snip("20240411_A01_e01_0001")
        assert embryo_id == "20240411_A01_e01"
        
        # Test frame extraction
        frame = parser.extract_frame_number("20240411_A01_e01_0001")
        assert frame == 1
        
        # Test change tracking
        parser._add_change_log("test_operation", {"detail": "test"})
        assert parser.has_unsaved_changes
        assert len(parser.get_recent_changes()) == 1
        
        # Test marking as saved
        parser.mark_saved()
        assert not parser.has_unsaved_changes
        assert len(parser._change_log) == 0
        
        print("✅ BaseAnnotationParser tests passed!")
        
    finally:
        # Cleanup
        tmp_path.unlink()

def test_utilities():
    """Test the utility functions."""
    print("🧪 Testing utility functions...")
    
    # Test path validation
    valid_path = validate_path("/tmp")
    assert isinstance(valid_path, Path)
    
    # Test ID parsing utilities
    parser_result = IdParser.parse_id("20240411_A01_e01_0001")
    assert parser_result["type"] == "snip"
    
    # Test sequential ID generation
    sequential_ids = IdParser.generate_sequential_ids("20240411_A01_e01_0001", 3)
    expected = ["20240411_A01_e01_0001", "20240411_A01_e01_0002", "20240411_A01_e01_0003"]
    assert sequential_ids == expected
    
    # Test ID consistency validation
    consistent_ids = ["20240411_A01_e01_0001", "20240411_A01_e01_0002"]
    result = IdParser.validate_id_consistency(consistent_ids)
    assert result["valid"] == True
    
    # Test embryo metadata structure validation
    valid_structure = {
        "file_info": {},
        "embryos": {
            "20240411_A01_e01": {"snips": {}}
        },
        "flags": {}
    }
    issues = validate_embryo_metadata_structure(valid_structure)
    assert len(issues) == 0
    
    invalid_structure = {"missing_key": "value"}
    issues = validate_embryo_metadata_structure(invalid_structure)
    assert len(issues) > 0
    
    print("✅ Utility function tests passed!")

def test_performance_utils():
    """Test performance utilities."""
    print("🧪 Testing performance utilities...")
    
    # Test operation profiling
    def dummy_operation(x):
        return x * 2
    
    result, exec_time = PerformanceUtils.profile_operation(dummy_operation, 5)
    assert result == 10
    assert exec_time >= 0
    
    # Test timing decorator
    @PerformanceUtils.time_it(iterations=2)
    def timed_function():
        return "test"
    
    result = timed_function()
    assert result == "test"
    
    print("✅ Performance utility tests passed!")

def test_integration():
    """Test integration between BaseAnnotationParser and utilities."""
    print("🧪 Testing integration...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_integration.json"
        
        # Create a simple annotation parser class
        class TestParser(BaseAnnotationParser):
            def __init__(self, filepath):
                super().__init__(filepath, verbose=False)
                self.data = self._load_or_create()
            
            def _load_or_create(self):
                if self.filepath.exists():
                    return self.load_json()
                else:
                    return {"entities": {}, "metadata": {}}
            
            def add_entity(self, entity_id: str, data: dict):
                # Validate ID format
                parsed = self.parse_id(entity_id)
                if parsed["type"] == "unknown":
                    raise ValueError(f"Invalid ID format: {entity_id}")
                
                self.data["entities"][entity_id] = data
                self.log_operation("add_entity", entity_id, data_keys=list(data.keys()))
            
            def save(self):
                self.data["metadata"]["last_updated"] = self.get_timestamp()
                self.save_json(self.data)
                self.mark_saved()
        
        # Test the integration
        parser = TestParser(tmp_path)
        
        # Add some test entities
        parser.add_entity("20240411_A01_e01", {"phenotype": "normal"})
        parser.add_entity("20240411_A01_e01_0001", {"annotation": "test"})
        
        assert len(parser.data["entities"]) == 2
        assert parser.has_unsaved_changes
        
        # Save and reload
        parser.save()
        assert not parser.has_unsaved_changes
        
        # Create new parser and verify data persisted
        parser2 = TestParser(tmp_path)
        assert len(parser2.data["entities"]) == 2
        assert "20240411_A01_e01" in parser2.data["entities"]
        
        print("✅ Integration tests passed!")

if __name__ == "__main__":
    print("🚀 Running tests for BaseAnnotationParser and utilities...")
    
    try:
        test_base_annotation_parser()
        test_utilities()
        test_performance_utils()
        test_integration()
        
        print("\n🎉 All tests passed! The base infrastructure is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
