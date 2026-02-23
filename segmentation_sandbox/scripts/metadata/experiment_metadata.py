"""
Experiment Metadata File Structure:

{
    "file_info": {
        "creation_time": <ISO timestamp>,
        "version": <str>,
        "created_by": <str>,
        "last_updated": <ISO timestamp>
    },
    "experiments": {
        <experiment_id>: {
            "experiment_id": <str>,
            "videos": {
                <video_id>: {
                    "video_id": <str>,
                    "images": {
                        <image_id>: {
                            "image_id": <str>,
                            "metadata": { ... },
                            "created_time": <ISO timestamp>
                        },
                        ...
                    },
                    "metadata": { ... },
                    "created_time": <ISO timestamp>
                },
                ...
            },
            "metadata": { ... },
            "created_time": <ISO timestamp>
        },
        ...
    },
    "entity_tracking": {
        "experiments": [<experiment_id>, ...],
        "videos": [<video_id>, ...],
        "images": [<image_id>, ...],
        "embryos": [ ... ], #empty
        "snips": [ ... ] #empty
    }
}
"""
"""
Experiment Metadata Management - Module 1

Manages experiment metadata with entity tracking, validation, and directory scanning.
Uses Module 0 utilities for consistent ID handling and validation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import json

# Import Module 0 utilities
from scripts.utils.parsing_utils import (
    parse_entity_id, 
    extract_frame_number, 
    get_entity_type,
    extract_experiment_id,
    build_image_id,
    normalize_frame_number
)
from scripts.utils.entity_id_tracker import EntityIDTracker
from scripts.utils.base_file_handler import BaseFileHandler
from scripts.metadata.schema_manager import SchemaManager


class ExperimentMetadata(BaseFileHandler):
    """Manage experiment metadata with entity tracking and validation."""
    
    def __init__(self, metadata_path: Union[str, Path], verbose: bool = True, 
                 auto_save_interval: Optional[int] = None):
        super().__init__(metadata_path, verbose)
        self.metadata = self._load_or_initialize()
        self._validate_and_update_tracking()
        
        # Autosave configuration
        self.auto_save_interval = auto_save_interval
        self._operations_since_save = 0
        
        # Initialize schema manager
        schema_path = self.filepath.parent / "morphseq_schema.json"
        self.schema_manager = SchemaManager(schema_path)
    
    def _load_or_initialize(self) -> Dict:
        """Load existing metadata or create new structure."""
        if self.filepath.exists():
            return self.load_json()
        else:
            return self._create_empty_metadata()
    
    def _create_empty_metadata(self) -> Dict:
        """Create empty metadata structure."""
        return {
            "file_info": {
                "creation_time": datetime.now().isoformat(),
                "version": "1.0",
                "created_by": "ExperimentMetadata"
            },
            "experiments": {},
            "entity_tracking": {
                "experiments": [], 
                "videos": [], 
                "images": [], 
                "embryos": [], 
                "snips": []
            }
        }
    
    def _validate_and_update_tracking(self):
        """Validate hierarchy and update tracking on load."""
        entities = EntityIDTracker.extract_entities(self.metadata)
        validation_result = EntityIDTracker.validate_hierarchy(entities)
        if not validation_result["valid"] and self.verbose:
            print(f"⚠️  Entity validation warnings: {len(validation_result['violations'])} issues")
            for violation in validation_result['violations'][:3]:  # Show first 3
                print(f"   - {violation}")
        self.metadata["entity_tracking"] = {k: list(v) for k, v in entities.items()}
    
    def save(self, force: bool = False):
        """Save with validation and autosave tracking."""
        # Update tracking before save
        entities = EntityIDTracker.extract_entities(self.metadata)
        validation_result = EntityIDTracker.validate_hierarchy(entities)
        if not validation_result["valid"] and self.verbose:
            print(f"⚠️  Entity validation warnings: {len(validation_result['violations'])} issues")
            for violation in validation_result['violations'][:3]:  # Show first 3
                print(f"   - {violation}")
        self.metadata["entity_tracking"] = {k: list(v) for k, v in entities.items()}
        
        # Update timestamp
        self.metadata["file_info"]["last_updated"] = datetime.now().isoformat()
        
        # Save using parent method
        self.save_json(self.metadata)
        
        # Reset autosave counter
        self._operations_since_save = 0
        
        if self.verbose:
            print(f"💾 Saved experiment metadata: {len(self.metadata['experiments'])} experiments")
    
    def _check_autosave(self):
        """Check if autosave should be triggered."""
        if self.auto_save_interval and self._operations_since_save >= self.auto_save_interval:
            if self.verbose:
                print(f"💾 Auto-saving after {self._operations_since_save} operations...")
            self.save()
    
    def _increment_operations(self):
        """Increment operations counter for autosave."""
        self._operations_since_save += 1
        self._check_autosave()
    
    # ========== EXPERIMENT MANAGEMENT ==========
    
    def add_experiment(self, experiment_id: str, **metadata_fields) -> None:
        """Add new experiment with metadata."""
        if experiment_id in self.metadata["experiments"]:
            if self.verbose:
                print(f"Experiment {experiment_id} already exists, updating metadata")
        
        # Initialize experiment structure
        if experiment_id not in self.metadata["experiments"]:
            self.metadata["experiments"][experiment_id] = {
                "experiment_id": experiment_id,
                "videos": {},
                "metadata": {},
                "created_time": datetime.now().isoformat()
            }
        
        # Update metadata fields
        self.metadata["experiments"][experiment_id]["metadata"].update(metadata_fields)
        
        # Update entity tracking
        self._update_entity_tracking(experiment_id, None, [])
        
        # Trigger autosave check
        self._increment_operations()
    
    def add_video_to_experiment(self, experiment_id: str, video_id: str, **metadata_fields) -> None:
        """Add video to experiment."""
        # Ensure experiment exists
        if experiment_id not in self.metadata["experiments"]:
            self.add_experiment(experiment_id)
        
        # Add video
        if video_id not in self.metadata["experiments"][experiment_id]["videos"]:
            self.metadata["experiments"][experiment_id]["videos"][video_id] = {
                "video_id": video_id,
                "image_ids": [],
                "metadata": {},
                "created_time": datetime.now().isoformat()
            }
        
        # Update metadata
        self.metadata["experiments"][experiment_id]["videos"][video_id]["metadata"].update(metadata_fields)
        
        # Update entity tracking
        self._update_entity_tracking(experiment_id, video_id, [])
        
        # Trigger autosave check
        self._increment_operations()
    
    def add_images_to_video(self, experiment_id: str, video_id: str, image_ids: List[str], **metadata_fields) -> None:
        """Add images to video - handles both list and dictionary image_ids formats."""
        # Ensure video exists
        if experiment_id not in self.metadata["experiments"]:
            self.add_experiment(experiment_id)
        if video_id not in self.metadata["experiments"][experiment_id]["videos"]:
            self.add_video_to_experiment(experiment_id, video_id)
        
        # Get current image_ids (handle both list and dictionary formats)
        video_data = self.metadata["experiments"][experiment_id]["videos"][video_id]
        current_image_ids = video_data["image_ids"]
        
        if isinstance(current_image_ids, dict):
            # Dictionary format: add new images as dictionary entries
            for image_id in image_ids:
                if image_id not in current_image_ids:
                    current_image_ids[image_id] = {
                        "frame_index": len(current_image_ids),
                        "raw_image_data_info": {}  # Will be populated by data_organizer
                    }
        else:
            # List format: maintain backwards compatibility
            for image_id in image_ids:
                if image_id not in current_image_ids:
                    current_image_ids.append(image_id)
            current_image_ids.sort()
        
        # Update entity tracking
        self._update_entity_tracking(experiment_id, video_id, image_ids)
        
        # Trigger autosave check
        self._increment_operations()
    
    def _update_entity_tracking(self, experiment_id: str, video_id: Optional[str], image_ids: List[str]):
        """Update tracking lists with new entities."""
        tracking = self.metadata["entity_tracking"]
        
        # Add experiment if new
        if experiment_id not in tracking["experiments"]:
            tracking["experiments"].append(experiment_id)
            tracking["experiments"].sort()
        
        # Add video if new
        if video_id and video_id not in tracking["videos"]:
            tracking["videos"].append(video_id)
            tracking["videos"].sort()
        
        # Add images if new
        for image_id in image_ids:
            if image_id not in tracking["images"]:
                tracking["images"].append(image_id)
        
        # Sort images
        tracking["images"].sort()
    
    # ========== DIRECTORY SCANNING ==========
    
    def scan_organized_experiments(self, raw_data_dir: Path) -> Dict:
        """
        Scan organized directory structure and compare with tracking.
        
        Returns:
            Dict with found experiments, videos, and images compared to tracking
        """
        raw_data_path = Path(raw_data_dir)
        organized_path = raw_data_path / "raw_data_organized"
        
        if not organized_path.exists():
            raise FileNotFoundError(f"Organized data directory not found: {organized_path}")
        
        found_experiments = []
        found_videos = []
        found_images = []
        
        # Scan experiment directories
        for exp_dir in organized_path.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                experiment_id = exp_dir.name
                found_experiments.append(experiment_id)
                
                # Scan for images directory
                images_dir = exp_dir / "images"
                if images_dir.exists():
                    # Scan video directories within images
                    for video_dir in images_dir.iterdir():
                        if video_dir.is_dir() and not video_dir.name.startswith('.'):
                            video_id = video_dir.name
                            found_videos.append(video_id)
                            
                            # Scan for image files
                            for img_file in video_dir.glob("*.jpg"):
                                # Extract frame number and build image_id with 't' prefix
                                frame_str = img_file.stem  # e.g., "0042"
                                frame_num = int(frame_str)
                                image_id = build_image_id(video_id, frame_num)
                                found_images.append(image_id)
        
        # Compare with current tracking
        current_tracking = self.metadata["entity_tracking"]
        
        return {
            "found_experiments": sorted(found_experiments),
            "found_videos": sorted(found_videos),
            "found_images": sorted(found_images),
            "new_experiments": sorted(set(found_experiments) - set(current_tracking["experiments"])),
            "new_videos": sorted(set(found_videos) - set(current_tracking["videos"])),
            "new_images": sorted(set(found_images) - set(current_tracking["images"])),
            "tracked_experiments": len(current_tracking["experiments"]),
            "tracked_videos": len(current_tracking["videos"]),
            "tracked_images": len(current_tracking["images"])
        }
    
    def discover_new_content(self, raw_data_dir: Path) -> Dict:
        """Find new experiments/videos/images since last scan."""
        scan_results = self.scan_organized_experiments(raw_data_dir)
        
        return {
            "new_experiments": scan_results["new_experiments"],
            "new_videos": scan_results["new_videos"], 
            "new_images": scan_results["new_images"],
            "counts": {
                "new_experiments": len(scan_results["new_experiments"]),
                "new_videos": len(scan_results["new_videos"]),
                "new_images": len(scan_results["new_images"])
            }
        }
    
    def update_from_scan(self, raw_data_dir: Path, add_missing: bool = True) -> Dict:
        """Update metadata from directory scan."""
        scan_results = self.scan_organized_experiments(raw_data_dir)
        
        if add_missing:
            # Add new experiments
            for exp_id in scan_results["new_experiments"]:
                self.add_experiment(exp_id)
            
            # Add new videos (group by experiment)
            video_to_exp = {}
            for video_id in scan_results["new_videos"]:
                exp_id = extract_experiment_id(video_id)
                if exp_id not in video_to_exp:
                    video_to_exp[exp_id] = []
                video_to_exp[exp_id].append(video_id)
            
            for exp_id, videos in video_to_exp.items():
                for video_id in videos:
                    self.add_video_to_experiment(exp_id, video_id)
            
            # Add new images (group by video and experiment)
            image_to_video = {}
            for image_id in scan_results["new_images"]:
                parsed = parse_entity_id(image_id)
                video_id = parsed["video_id"]
                if video_id not in image_to_video:
                    image_to_video[video_id] = []
                image_to_video[video_id].append(image_id)
            
            for video_id, images in image_to_video.items():
                exp_id = extract_experiment_id(video_id)
                self.add_images_to_video(exp_id, video_id, images)
        
        return scan_results
    
    # ========== QUERY METHODS ==========
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment metadata."""
        return self.metadata["experiments"].get(experiment_id)
    
    def get_video(self, experiment_id: str, video_id: str) -> Optional[Dict]:
        """Get video metadata."""
        exp = self.get_experiment(experiment_id)
        if exp:
            return exp["videos"].get(video_id)
        return None
    
    def get_image(self, experiment_id: str, video_id: str, image_id: str) -> Optional[Dict]:
        """Get image metadata - handles both list and dictionary image_ids formats."""
        video = self.get_video(experiment_id, video_id)
        if not video:
            return None
            
        image_ids = video.get("image_ids", [])
        
        if isinstance(image_ids, dict):
            # Dictionary format: return enhanced metadata
            if image_id in image_ids:
                image_info = image_ids[image_id]
                return {
                    "image_id": image_id,
                    "video_id": video_id,
                    "experiment_id": experiment_id,
                    "exists": True,
                    "frame_index": image_info.get("frame_index", 0),
                    "raw_image_data_info": image_info.get("raw_image_data_info", {})
                }
        else:
            # List format: return basic info
            if image_id in image_ids:
                return {
                    "image_id": image_id,
                    "video_id": video_id,
                    "experiment_id": experiment_id,
                    "exists": True
                }
        return None
    
    def list_experiments(self) -> List[str]:
        """Get list of all experiment IDs."""
        return list(self.metadata["experiments"].keys())
    
    def list_videos(self, experiment_id: str = None) -> List[str]:
        """Get list of video IDs (optionally filtered by experiment)."""
        if experiment_id:
            exp = self.get_experiment(experiment_id)
            return list(exp["videos"].keys()) if exp else []
        else:
            return self.metadata["entity_tracking"]["videos"].copy()
    
    def list_images(self, experiment_id: str = None, video_id: str = None) -> List[str]:
        """Get list of image IDs (optionally filtered by experiment/video) - handles both formats."""
        if experiment_id and video_id:
            video = self.get_video(experiment_id, video_id)
            if not video:
                return []
            image_ids = video.get("image_ids", [])
            return list(image_ids.keys()) if isinstance(image_ids, dict) else image_ids
        elif experiment_id:
            exp = self.get_experiment(experiment_id)
            if exp:
                all_images = []
                for video in exp["videos"].values():
                    image_ids = video.get("image_ids", [])
                    if isinstance(image_ids, dict):
                        all_images.extend(image_ids.keys())
                    else:
                        all_images.extend(image_ids)
                return all_images
            return []
        else:
            return self.metadata["entity_tracking"]["images"].copy()

    def get_all_image_ids(self) -> List[str]:
        """Get all image IDs - alias for list_images() for backward compatibility."""
        return self.list_images()
    
    def _image_exists_in_video_data(self, video_data: Dict, image_id: str) -> bool:
        """Helper function to check if image exists in video_data (handles both formats)."""
        image_ids = video_data.get("image_ids", [])
        if isinstance(image_ids, dict):
            return image_id in image_ids
        else:
            return image_id in image_ids
    
    # ========== ENTITY SUMMARY ==========
    
    def get_entity_summary(self) -> Dict:
        """Get entity counts and validation summary."""
        entities = EntityIDTracker.extract_entities(self.metadata)
        counts = EntityIDTracker.get_counts(entities)
        
        return {
            "counts": counts,
            "validation": {
                "hierarchy_valid": True,  # If we get here, it's valid
                "last_validated": datetime.now().isoformat()
            },
            "schema_summary": self.schema_manager.get_schema_summary()
        }
    
    def get_experiment_summary(self, experiment_id: str) -> Dict:
        """Get summary for specific experiment."""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return {"error": f"Experiment {experiment_id} not found"}
        
        videos = exp["videos"]
        total_images = sum(len(video["images"]) for video in videos.values())
        
        return {
            "experiment_id": experiment_id,
            "videos": len(videos),
            "total_images": total_images,
            "created_time": exp.get("created_time"),
            "metadata_fields": list(exp.get("metadata", {}).keys())
        }
    
    # ========== PATH RESOLUTION ==========
    
    def get_base_data_path(self) -> Optional[Path]:
        """Get the base data path if configured."""
        return getattr(self, '_base_data_path', None)
    
    def set_base_data_path(self, base_path: Union[str, Path]) -> None:
        """Set the base data directory path."""
        self._base_data_path = Path(base_path)
    
    def get_image_path(self, image_id: str, extension: str = "jpg", 
                      base_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Get full path to image file using metadata or provided base_path.
        
        Args:
            image_id: Image ID with 't' prefix (e.g., "20240411_A01_t0000")
            extension: File extension (default: jpg)
            base_path: Override base path, otherwise uses stored metadata paths
            
        Returns:
            Full path to image file
        """
        # First try to use stored metadata paths
        parsed = parse_entity_id(image_id)
        experiment_id = parsed["experiment_id"]
        video_id = parsed["video_id"]
        frame_number = parsed["frame_number"]
        
        # Check if we have this image in our metadata
        exp_data = self.metadata.get("experiments", {}).get(experiment_id)
        if exp_data:
            video_data = exp_data.get("videos", {}).get(video_id)
            if video_data:
                # Check if this image_id is tracked
                if self._image_exists_in_video_data(video_data, image_id):
                    # Use the stored processed_jpg_images_dir with parsing utils
                    from scripts.utils.parsing_utils import get_image_id_path
                    images_dir = video_data["processed_jpg_images_dir"]
                    return get_image_id_path(images_dir, image_id, extension)
        
        # Fallback to base path construction if not in metadata
        if base_path:
            use_base = Path(base_path)
        elif hasattr(self, '_base_data_path') and self._base_data_path:
            use_base = self._base_data_path
        else:
            raise ValueError(
                f"Image {image_id} not found in metadata and no base path configured. "
                f"Use set_base_data_path() or provide base_path parameter."
            )
        
        # Use parsing utils for path components (fallback)
        from scripts.utils.parsing_utils import get_relative_image_path
        relative_path = get_relative_image_path(image_id, extension)
        
        return use_base / experiment_id / relative_path

    
    def get_video_directory_path(self, video_id: str, 
                               base_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Get full path to video directory.
        
        Args:
            video_id: Video ID 
            base_path: Override base path, otherwise uses configured base
            
        Returns:
            Full path to video directory
        """
        if base_path:
            use_base = Path(base_path)
        elif hasattr(self, '_base_data_path') and self._base_data_path:
            use_base = self._base_data_path
        else:
            raise ValueError("No base path configured. Use set_base_data_path() or provide base_path parameter.")
        
        # Use parsing utils for path components
        from scripts.utils.parsing_utils import get_relative_video_path
        relative_path = get_relative_video_path(video_id)
        experiment_id = extract_experiment_id(video_id)
        
        return use_base / experiment_id / relative_path
    
    def get_experiment_directory_path(self, experiment_id: str,
                                    base_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Get full path to experiment directory.
        
        Args:
            experiment_id: Experiment ID
            base_path: Override base path, otherwise uses configured base
            
        Returns:
            Full path to experiment directory
        """
        if base_path:
            use_base = Path(base_path)
        elif hasattr(self, '_base_data_path') and self._base_data_path:
            use_base = self._base_data_path
        else:
            raise ValueError("No base path configured. Use set_base_data_path() or provide base_path parameter.")
        
        return use_base / experiment_id
    
    def verify_image_exists(self, image_id: str, extension: str = "jpg",
                          base_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Check if image file exists on disk using metadata or base path.
        
        Leverages stored "processed_jpg_images_dir" and "image_ids" from metadata
        for faster lookups when available.
        """
        # First check if image is tracked in metadata
        parsed = parse_entity_id(image_id)
        experiment_id = parsed["experiment_id"]
        video_id = parsed["video_id"]
        
        exp_data = self.metadata.get("experiments", {}).get(experiment_id)
        if exp_data:
            video_data = exp_data.get("videos", {}).get(video_id)
            if video_data:
                # Quick check: if it's in image_ids, assume it exists
                # (since metadata should reflect actual files)
                if self._image_exists_in_video_data(video_data, image_id):
                    # Optional: verify file actually exists
                    try:
                        image_path = self.get_image_path(image_id, extension, base_path)
                        return image_path.exists()
                    except (ValueError, Exception):
                        return False
        
        # Fallback to path construction and file check
        try:
            image_path = self.get_image_path(image_id, extension, base_path)
            return image_path.exists()
        except (ValueError, Exception):
            return False
    
    def list_existing_images_in_video(self, video_id: str, extension: str = "jpg",
                                    base_path: Optional[Union[str, Path]] = None) -> List[str]:
        """
        Get list of image_ids that exist for a video.
        
        Uses stored "image_ids" from metadata when available for fast lookup.
        
        Returns:
            List of image_ids (with 't' prefix) that have corresponding files
        """
        experiment_id = extract_experiment_id(video_id)
        
        # Try to use stored metadata first (much faster)
        exp_data = self.metadata.get("experiments", {}).get(experiment_id)
        if exp_data:
            video_data = exp_data.get("videos", {}).get(video_id)
            if video_data:
                stored_image_ids = video_data.get("image_ids", [])
                if stored_image_ids:
                    # Optional: verify a few exist to ensure metadata is accurate
                    if len(stored_image_ids) > 0:
                        # Sample check on first image - handle both list and dict formats
                        if isinstance(stored_image_ids, dict):
                            first_image = sorted(stored_image_ids.keys())[0]
                            if self.verify_image_exists(first_image, extension, base_path):
                                return sorted(stored_image_ids.keys())
                        else:
                            first_image = stored_image_ids[0]
                            if self.verify_image_exists(first_image, extension, base_path):
                                return stored_image_ids
                    
        # Fallback to directory scanning
        try:
            video_dir = self.get_video_directory_path(video_id, base_path)
            if not video_dir.exists():
                return []
            
            existing_images = []
            for img_file in video_dir.glob(f"*.{extension}"):
                # Convert filename back to image_id
                # New format: filename is the image_id itself (e.g., "20250612_30hpf_ctrl_atf6_F11_ch00_t0042.jpg")
                potential_image_id = img_file.stem
                
                # Validate that this looks like an image_id and matches the video_id
                if potential_image_id.startswith(video_id):
                    existing_images.append(potential_image_id)
                else:
                    # Fallback for old numeric format (e.g., "0042.jpg") 
                    try:
                        frame_num = int(img_file.stem)
                        image_id = build_image_id(video_id, frame_num)
                        existing_images.append(image_id)
                    except ValueError:
                        # Skip files that don't match expected patterns
                        continue
            
            return sorted(existing_images)
            
        except (ValueError, Exception):
            return []
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """Get stored metadata for a video, including paths and image_ids."""
        experiment_id = extract_experiment_id(video_id)
        exp_data = self.metadata.get("experiments", {}).get(experiment_id)
        if exp_data:
            return exp_data.get("videos", {}).get(video_id)
        return None
    
    def get_images_for_detection(self, experiment_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Get image information optimized for detection pipelines.
        
        Uses stored metadata for fast path resolution.
        
        Returns:
            List of dicts with image_id, image_path, video_id, etc.
        """
        images = []
        
        target_experiments = experiment_ids or list(self.metadata.get("experiments", {}).keys())
        
        for exp_id in target_experiments:
            exp_data = self.metadata.get("experiments", {}).get(exp_id)
            if not exp_data:
                continue
                
            for video_id, video_data in exp_data.get("videos", {}).items():
                images_dir = Path(video_data.get("processed_jpg_images_dir", ""))
                
                # Handle both list and dictionary formats for image_ids
                image_ids = video_data.get("image_ids", [])
                image_id_list = list(image_ids.keys()) if isinstance(image_ids, dict) else image_ids
                
                for image_id in image_id_list:
                    # Parse image_id for metadata and use parsing utils for path construction
                    parsed = parse_entity_id(image_id)
                    frame_number = parsed["frame_number"]
                    from scripts.utils.parsing_utils import get_image_id_path
                    image_path = get_image_id_path(images_dir, image_id)
                    
                    if image_path.exists():
                        images.append({
                            'image_id': image_id,
                            'image_path': str(image_path),
                            'video_id': video_id,
                            'well_id': video_data.get('well_id', parsed.get('well_id', '')),
                            'experiment_id': exp_id,
                            'frame_number': int(frame_number)
                        })
        
        return images

    # ========== VALIDATION WITH SCHEMA ==========
    
    def validate_metadata_field(self, field_name: str, value: str, level: str = None) -> bool:
        """Validate metadata field against schema."""
        if field_name == "phenotype":
            return self.schema_manager.validate_phenotype(value)
        elif field_name == "genotype":
            return self.schema_manager.validate_genotype(value)
        elif field_name == "treatment":
            return self.schema_manager.validate_treatment(value)
        elif field_name == "zygosity":
            return self.schema_manager.validate_zygosity(value)
        elif field_name.endswith("_flag") and level:
            return self.schema_manager.validate_flag(value, level)
        else:
            # Unknown field, allow anything
            return True
    
    def get_processed_images_dir_for_video(self, video_id: str) -> str:
        """Get the processed images directory path for a specific video."""
        
        video_info = self.get_video_metadata(video_id)
        
        if video_info and "processed_jpg_images_dir" in video_info:
            path = video_info["processed_jpg_images_dir"]
            return path
        else:
            # Fallback: construct from metadata file location
            # Remove the experiment metadata filename and go to images/video_id
            metadata_dir = self.filepath.parent
            fallback_path = str(metadata_dir / "images" / video_id)
            return fallback_path
    
    def __str__(self) -> str:
        summary = self.get_entity_summary()
        counts = summary["counts"]
        return f"ExperimentMetadata({counts['experiments']} experiments, {counts['videos']} videos, {counts['images']} images)"
