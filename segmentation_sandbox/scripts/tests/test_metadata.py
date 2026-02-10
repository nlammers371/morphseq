"""
Test Mo# Add the scripts directory to the path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts')

# Import modules with absolute paths
from utils.parsing_utils import parse_entity_id, build_image_id, extract_experiment_id
from utils.entity_id_tracker import EntityIDTracker 
from utils.base_file_handler import BaseFileHandler
from metadata.schema_manager import SchemaManager
from metadata.experiment_metadata import ExperimentMetadataxperiment Metadata Management

Tests for ExperimentMetadata and SchemaManager classes.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts')

# Import modules with absolute paths
from utils.parsing_utils import parse_entity_id
from utils.entity_id_tracker import EntityIDTracker 
from utils.base_file_handler import BaseFileHandler
from metadata.schema_manager import SchemaManager
from metadata.experiment_metadata import ExperimentMetadata


def test_schema_manager():
    """Test SchemaManager functionality."""
    print("🧪 Testing SchemaManager...")
    
    # Create temporary schema file that doesn't exist yet
    with tempfile.TemporaryDirectory() as temp_dir:
        schema_path = Path(temp_dir) / "test_schema.json"
    
        try:
            # Test schema creation
            schema = SchemaManager(str(schema_path))
            print(f"   ✅ Created schema: {schema}")
            
            # Test adding phenotypes
            schema.add_phenotype("TEST_PHENOTYPE", "Test phenotype for validation")
            assert schema.validate_phenotype("TEST_PHENOTYPE"), "Phenotype validation failed"
            print("   ✅ Phenotype management works")
            
            # Test adding genotypes  
            schema.add_genotype("test_gene", "Test gene", gene="test")
            assert schema.validate_genotype("test_gene"), "Genotype validation failed"
            print("   ✅ Genotype management works")
            
            # Test saving and loading
            schema.save_schema()
            schema2 = SchemaManager(str(schema_path))
            assert schema2.validate_phenotype("TEST_PHENOTYPE"), "Schema persistence failed"
            print("   ✅ Schema persistence works")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Schema test failed: {e}")
            return False


def test_experiment_metadata():
    """Test ExperimentMetadata functionality."""
    print("🧪 Testing ExperimentMetadata...")
    
    # Create temporary metadata file 
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata_path = Path(temp_dir) / "test_metadata.json"
    
        try:
            # Test metadata creation
            meta = ExperimentMetadata(str(metadata_path))
            print(f"   ✅ Created metadata: {meta}")
            
            # Test adding experiment
            meta.add_experiment("20250731_test_exp", temperature=28, medium="E2")
            experiments = meta.list_experiments()
            assert "20250731_test_exp" in experiments, "Experiment not added"
            print("   ✅ Experiment addition works")
            
            # Test adding video
            meta.add_video_to_experiment("20250731_test_exp", "20250731_test_exp_A01", well="A01")
            videos = meta.list_videos("20250731_test_exp")
            assert "20250731_test_exp_A01" in videos, "Video not added"
            print("   ✅ Video addition works")
            
            # Test adding images
            image_ids = ["20250731_test_exp_A01_t0000", "20250731_test_exp_A01_t0001", "20250731_test_exp_A01_t0002"]
            meta.add_images_to_video("20250731_test_exp", "20250731_test_exp_A01", image_ids)
            images = meta.list_images("20250731_test_exp", "20250731_test_exp_A01")
            assert len(images) == 3, f"Expected 3 images, got {len(images)}"
            print("   ✅ Image addition works")
            
            # Test entity tracking
            summary = meta.get_entity_summary()
            counts = summary["counts"]
            assert counts["experiments"] == 1, f"Expected 1 experiment, got {counts['experiments']}"
            assert counts["videos"] == 1, f"Expected 1 video, got {counts['videos']}"
            assert counts["images"] == 3, f"Expected 3 images, got {counts['images']}"
            print("   ✅ Entity tracking works")
            
            # Test saving and loading
            meta.save()
            meta2 = ExperimentMetadata(str(metadata_path))
            experiments2 = meta2.list_experiments()
            assert "20250731_test_exp" in experiments2, "Metadata persistence failed"
            print("   ✅ Metadata persistence works")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Metadata test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_metadata_driven_paths():
    """Test that path resolution uses stored metadata efficiently."""
    print("🧪 Testing metadata-driven path resolution...")
    
    # Create temporary metadata file
    with tempfile.TemporaryDirectory() as temp_dir:
        metadata_path = Path(temp_dir) / "test_metadata.json"
        
        try:
            meta = ExperimentMetadata(str(metadata_path))
            
            # Add experiment with proper metadata structure (like module_0_2 produces)
            experiment_id = "20250731_test"
            video_id = "20250731_test_A01"
            images_dir = Path(temp_dir) / "processed_images" / video_id
            images_dir.mkdir(parents=True)
            
            # Create actual image files
            image_files = []
            for i in range(3):
                image_file = images_dir / f"{i:04d}.jpg"
                image_file.touch()
                image_files.append(f"{video_id}_t{i:04d}")
            
            # Add experiment with metadata structure that includes paths
            meta.metadata["experiments"][experiment_id] = {
                "experiment_id": experiment_id,
                "videos": {
                    video_id: {
                        "video_id": video_id,
                        "well_id": "A01",
                        "mp4_path": str(Path(temp_dir) / "videos" / f"{video_id}.mp4"),
                        "processed_jpg_images_dir": str(images_dir),
                        "image_ids": image_files,
                        "total_frames": len(image_files)
                    }
                }
            }
            
            # Test that get_image_path uses stored metadata
            image_id = "20250731_test_A01_t0000"
            actual_path = meta.get_image_path(image_id)
            expected_path = images_dir / "0000.jpg"
            assert actual_path == expected_path, f"Path mismatch: {actual_path} != {expected_path}"
            print("   ✅ get_image_path uses stored metadata paths")
            
            # Test that verify_image_exists works with metadata
            exists = meta.verify_image_exists(image_id)
            assert exists, "verify_image_exists should return True for existing tracked file"
            print("   ✅ verify_image_exists uses metadata efficiently")
            
            # Test list_existing_images_in_video uses metadata
            existing_images = meta.list_existing_images_in_video(video_id)
            assert len(existing_images) == 3, f"Expected 3 images, got {len(existing_images)}"
            assert existing_images == image_files, "Should return stored image_ids from metadata"
            print("   ✅ list_existing_images_in_video uses stored metadata")
            
            # Test get_images_for_detection
            detection_images = meta.get_images_for_detection([experiment_id])
            assert len(detection_images) == 3, f"Expected 3 detection images, got {len(detection_images)}"
            
            first_image = detection_images[0]
            assert 'image_id' in first_image, "Detection image should have image_id"
            assert 'image_path' in first_image, "Detection image should have image_path"
            assert 'video_id' in first_image, "Detection image should have video_id"
            assert first_image['experiment_id'] == experiment_id, "Detection image should have correct experiment_id"
            print("   ✅ get_images_for_detection provides complete metadata")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Path test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all tests."""
    print("🚀 Starting Module 1 Tests...")
    print()
    
    try:
        # Test individual components
        success1 = test_schema_manager()
        print()
        
        success2 = test_experiment_metadata()
        print()
        
        success3 = test_metadata_driven_paths()
        print()
        
        if success1 and success2 and success3:
            print("✅ All Module 1 tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
