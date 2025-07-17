#!/usr/bin/env python3
"""
SAM2 Video Processing Utilities for Embryo Segmentation
=======================================================

This module provides SAM2 integration with GroundedDINO annotations for embryo tracking:
- GroundedSamAnnotations class for managing SAM2 video segmentation
- High-quality annotation processing for seed frame selection
- Video-based embryo tracking and mask propagation
- Structured output format matching experiment metadata hierarchy
- Pipeline management class for orchestrating the entire workflow

Output Structure:
================

GroundedSam2Annotations.json format:
{
  "script_version": "sam2_utils.py",
  "creation_time": "YYYY-MM-DDThh:mm:ss",
  "last_updated": "YYYY-MM-DDThh:mm:ss",
  # ... other fields ...
  "snip_ids": ["20240411_A01_e01_0000", "20240411_A01_e01_0001", ...],
  "experiments": {
    "20240411": {
      # ... experiment structure ...
      "images": {
        "20240411_A01_0000": {
          "image_id": "20240411_A01_0000",
          "frame_index": 0,
          "is_seed_frame": true,
          "embryos": {
            "20240411_A01_e01": {
              "embryo_id": "20240411_A01_e1",
              "snip_id": "20240411_A01_e1_0000",  # ADDED: Unique snippet identifier
              "segmentation": {...},
              "segmentation_format": "rle",
              "bbox": [x, y, x, y],
              "area": 1234.5,
              "mask_confidence": 0.85
            }
          }
        }
      }
    }
  }
}
"""

import os
import sys
import json
import yaml
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
import numpy as np
from collections import Counter, defaultdict
import cv2
import tempfile
import shutil

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure the project root is in the path
SANDBOX_ROOT = Path(__file__).parent.parent.parent
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.append(str(SANDBOX_ROOT))

# Add SAM2 to path - using your working path structure
SAM2_MODELS_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/models/sam2")
SAM2_ROOT = SAM2_MODELS_ROOT / "sam2"  # The actual sam2 directory

# Add the models directory to path (matches your working approach)
if str(SAM2_MODELS_ROOT) not in sys.path:
    sys.path.append(str(SAM2_MODELS_ROOT))

# Import from other utils
from scripts.utils.experiment_metadata_utils import load_experiment_metadata, get_image_id_paths


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sam2_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load SAM2 model using exact working approach - simple and clean."""
    print(f"🔧 Loading SAM2 model...")
    
    # Store current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to SAM2 directory (your working approach)
        os.chdir(SAM2_ROOT)
        
        # Use the relative paths exactly like your working code
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        print(f"   Config: {model_cfg}")
        print(f"   Checkpoint: {sam2_checkpoint}")
        
        # Import and build predictor (exactly like your working approach)
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
        print(f"✅ SAM2 model loaded successfully")
        return predictor
        
    finally:
        # Always restore original working directory
        os.chdir(original_cwd)


class GroundedSamAnnotations:
    """
    SAM2 video processing manager that integrates with GroundedDINO annotations.
    
    Handles:
    - Loading high-quality GroundedDINO annotations
    - Video grouping and seed frame selection
    - SAM2 video segmentation and tracking
    - Structured output generation matching experiment metadata format
    - Autosave functionality and progress tracking
    """


    def __init__(self, 
                filepath: Union[str, Path],
                seed_annotations_path: Union[str, Path],          # Required
                experiment_metadata_path: Union[str, Path],       # Required (remove Optional)
                sam2_config: Optional[str] = None,                # Actually optional
                sam2_checkpoint: Optional[str] = None,            # Actually optional
                device: str = "cuda",
                target_prompt: str = "individual embryo",
                segmentation_format: str = "rle",
                verbose: bool = True):
        """
        Initialize GroundedSamAnnotations.
        
        Args:
            filepath: Path where SAM2 results will be saved
            seed_annotations_path: Path to GroundedDINO annotations JSON with high_quality_annotations
            experiment_metadata_path: Path to experiment_metadata.json file
            sam2_config: SAM2 model config path (optional, can be set later)
            sam2_checkpoint: SAM2 model checkpoint path (optional, can be set later)
            device: Device for SAM2 model ('cuda' or 'cpu')
            target_prompt: Prompt to use from annotations (default: 'individual embryo')
            segmentation_format: Output format ('rle' or 'polygon')
            verbose: Enable verbose output
        """
        self.filepath = Path(filepath)
        self.seed_annotations_path = Path(seed_annotations_path) if seed_annotations_path else None
        self.experiment_metadata_path = Path(experiment_metadata_path) if experiment_metadata_path else None
        self.target_prompt = target_prompt
        self.segmentation_format = segmentation_format
        self.verbose = verbose
        self.device = device
        self._unsaved_changes = False
        # Explicit validation for seed annotations path
        if self.seed_annotations_path is None:
            raise ValueError(
                "Missing required argument: seed_annotations_path. "
                "Please provide the path to your GroundedDINO annotations JSON as seed_annotations_path."
            )
        if not self.seed_annotations_path.exists():
            raise FileNotFoundError(
                f"GroundedDINO annotations JSON not found at: {self.seed_annotations_path}. "
                "Please ensure the file exists and the path is correct."
            )
        
        if self.verbose:
            print(f"🎬 Initializing GroundedSamAnnotations...")
            print(f"   Target prompt: '{self.target_prompt}'")
            print(f"   Segmentation format: {self.segmentation_format}")
            print(f"   Output file: {self.filepath}")
        
        # EARLY VALIDATION - Check paths before loading anything
        validation_errors = []
        
        # Check seed annotations path
        if not self.seed_annotations_path:
            validation_errors.append("No seed annotations path provided")
        elif not self.seed_annotations_path.exists():
            validation_errors.append(f"Seed annotations file not found: {self.seed_annotations_path}")
        
        # Check experiment metadata path
        if not self.experiment_metadata_path:
            validation_errors.append(
                "Missing experiment metadata path. "
                "Please provide the path to your experiment_metadata.json as experiment_metadata_path."
            )
        elif not self.experiment_metadata_path.exists():
            validation_errors.append(f"Experiment metadata file not found: {self.experiment_metadata_path}")
        
        # If we have validation errors, stop immediately
        if validation_errors:
            if self.verbose:
                print("\n❌ Cannot initialize GroundedSamAnnotations:")
                for i, error in enumerate(validation_errors, 1):
                    print(f"   {i}. {error}")
                print("\nRequired files:")
                print("  • seed_annotations_path: GroundedDINO annotations JSON with 'high_quality_annotations'")
                print("  • experiment_metadata_path: experiment_metadata.json file")
            
            raise ValueError(f"Missing required files: {', '.join(validation_errors)}")
        
        # Now load the files (we know they exist)
        if self.verbose:
            print("✅ All required paths provided and files exist")
        
        # Load experiment metadata 
        self.experiment_metadata = self._load_experiment_metadata()
        if not self.experiment_metadata:
            raise ValueError("Failed to load experiment metadata")
        # Validate experiment metadata structure
        if not isinstance(self.experiment_metadata, dict) or 'experiments' not in self.experiment_metadata:
            raise ValueError(f"Invalid experiment metadata format: missing 'experiments' in {self.experiment_metadata_path}")
        
        # Load seed annotations with validation
        self.seed_annotations = self._load_seed_annotations()
        if not self.seed_annotations:
            raise ValueError("Failed to load seed annotations")
        # Validate seed annotations structure
        if not isinstance(self.seed_annotations, dict) or 'high_quality_annotations' not in self.seed_annotations:
            raise ValueError(f"Invalid seed annotations format: missing 'high_quality_annotations' in {self.seed_annotations_path}")
        
        # Group video annotations only if we have valid seed annotations
        self.video_annotations = self._group_video_annotations()
        if not self.video_annotations:
            raise ValueError(
                f"No video annotations found for prompt '{self.target_prompt}'. "
                f"Please verify that {self.seed_annotations_path} contains valid annotations "
                "and that the prompt matches your data."
            )
        
        # Initialize SAM2 model (lazy loading)
        self.sam2_predictor = None
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        
        # Initialize or load results structure
        self.results = self._load_or_initialize_results()
        
        if self.verbose:
            seed_info = extract_seed_annotations_info(self.seed_annotations, self.target_prompt)
            print(f"🔧 Seed model: {seed_info['model_architecture']} ({Path(seed_info['model_weights']).name})")
            print(f"📋 Ready to process {len(self.video_annotations)} videos")
            print("✅ GroundedSamAnnotations initialized successfully")

    def _load_seed_annotations(self) -> Optional[Dict]:
        """Load GroundedDINO seed annotations."""
        if not self.seed_annotations_path or not self.seed_annotations_path.exists():
            if self.verbose:
                print("⚠️  No seed annotations provided")
            return None
        
        if self.verbose:
            print(f"📁 Loading seed annotations from: {self.seed_annotations_path}")
        
        with open(self.seed_annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Validate that high-quality annotations exist for target prompt
        hq_annotations = annotations.get("high_quality_annotations", {})
        valid_experiments = 0
        for exp_id, exp_data in hq_annotations.items():
            if exp_data.get("prompt") == self.target_prompt:
                valid_experiments += 1
        
        if valid_experiments == 0 and hq_annotations:
            if self.verbose:
                print(f"⚠️  No high-quality annotations found for prompt '{self.target_prompt}'")
        
        return annotations
            
    def _load_experiment_metadata(self) -> Optional[Dict]:
        """Load experiment metadata with clear requirements."""
        if not self.experiment_metadata_path:
            if self.verbose:
                print("❌ No experiment metadata path provided")
                print("   Please provide path to experiment_metadata.json file")
            return None
        
        # Check if file exists
        if not self.experiment_metadata_path.exists():
            if self.verbose:
                print(f"❌ Experiment metadata file not found: {self.experiment_metadata_path}")
                print("   Please provide a valid path to experiment_metadata.json")
            return None
        
        try:
            if self.verbose:
                print(f"📁 Loading experiment metadata from: {self.experiment_metadata_path}")
            return load_experiment_metadata(self.experiment_metadata_path)
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading experiment metadata: {e}")
                print("   Please check that the file is a valid experiment metadata JSON")
            return None

    def _load_or_initialize_results(self) -> Dict:
        """Load existing results file or initialize a new one."""
        
        #default behavior if no annotation is give is to create one to start
        if not self.filepath.exists():
            if self.verbose:
                print(f"🆕 Initializing new SAM2 results file at: {self.filepath}")
            
            # Extract seed annotations info
            seed_info = {}
            if self.seed_annotations:
                seed_annotations_info = extract_seed_annotations_info(self.seed_annotations, self.target_prompt)
                seed_info = {
                    "source_file": str(self.seed_annotations_path),
                    "model_architecture": seed_annotations_info["model_architecture"],
                    "model_weights": seed_annotations_info["model_weights"],
                    "model_config": seed_annotations_info["model_config"],
                    "target_prompt": self.target_prompt,
                    "has_high_quality_annotations": seed_annotations_info["has_high_quality_annotations"],
                    "experiments_with_hq": seed_annotations_info["experiments_with_hq"]
                }
            
            return {
                "script_version": "sam2_utils.py",
                "creation_time": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "seed_annotations_info": seed_info,
                "sam2_model_info": {
                    "config_path": str(self.sam2_config) if self.sam2_config else None,
                    "checkpoint_path": str(self.sam2_checkpoint) if self.sam2_checkpoint else None,
                    "model_architecture": "SAM2"
                },
                "target_prompt": self.target_prompt,
                "segmentation_format": self.segmentation_format,
                "experiment_ids": [],
                "video_ids": [],
                "embryo_ids": [],
                "snip_ids": [],
                "experiments": {}
            }
        
        try:
            if self.verbose:
                print(f"📁 Loading existing SAM2 results from: {self.filepath}")
            
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            if self.verbose:
                total_videos = len(data.get('video_ids', []))
                print(f"✅ Loaded {total_videos} videos successfully")
            return data
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ JSON corruption detected: {e}")
            
            backup_path = self.filepath.with_suffix('.json.backup')
            shutil.move(self.filepath, backup_path)
            
            if self.verbose:
                print(f"📋 Moved corrupted file to backup: {backup_path.name}")
                print(f"🆕 Starting with fresh SAM2 results")
            
            return self._initialize_results()
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Unexpected error: {e}")
            return self._initialize_results()

    def save(self):
        """Save results to file with atomic write."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.results["last_updated"] = datetime.now().isoformat()
        
        temp_path = self.filepath.with_suffix('.json.tmp')
        backup_path = self.filepath.with_suffix('.json.backup')
        
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            shutil.move(temp_path, self.filepath)
            
            if backup_path.exists():
                backup_path.unlink()
                if self.verbose:
                    print(f"🗑️  Removed corrupted backup (save successful)")
            
            self._unsaved_changes = False
            if self.verbose:
                print(f"💾 Saved SAM2 results to: {self.filepath}")
                
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            if self.verbose:
                print(f"❌ Failed to save SAM2 results: {e}")
            raise

    def set_seed_annotations_path(self, seed_annotations_path: Union[str, Path]):
        """Set or update the seed annotations path."""
        self.seed_annotations_path = Path(seed_annotations_path)
        if self.seed_annotations_path.exists():
            self.seed_annotations = self._load_seed_annotations()
            
            # Update cached video annotations
            self.video_annotations = self._group_video_annotations()
            
            # Update results with new seed info
            if self.seed_annotations:
                seed_annotations_info = extract_seed_annotations_info(self.seed_annotations, self.target_prompt)
                self.results["seed_annotations_info"] = {
                    "source_file": str(self.seed_annotations_path),
                    "model_architecture": seed_annotations_info["model_architecture"],
                    "model_weights": seed_annotations_info["model_weights"],
                    "model_config": seed_annotations_info["model_config"],
                    "target_prompt": self.target_prompt,
                    "has_high_quality_annotations": seed_annotations_info["has_high_quality_annotations"],
                    "experiments_with_hq": seed_annotations_info["experiments_with_hq"]
                }
                self._unsaved_changes = True
                
                if self.verbose:
                    if seed_annotations_info["has_high_quality_annotations"]:
                        print(f"📂 Updated seed annotations: {seed_annotations_info['experiments_with_hq']} experiments with high-quality data")
                    else:
                        print(f"📂 Updated seed annotations: using regular annotations")
        else:
            if self.verbose:
                print(f"⚠️  Seed annotations file not found: {seed_annotations_path}")
            self.seed_annotations = None
            self.video_annotations = {}

    def _load_sam2_model(self):
        """Lazy load SAM2 model."""
        if self.sam2_predictor is None:
            if not self.sam2_config or not self.sam2_checkpoint:
                raise ValueError("SAM2 config and checkpoint paths required")
            
            self.sam2_predictor = load_sam2_model(
                self.sam2_config, 
                self.sam2_checkpoint, 
                self.device
            )

    def set_sam2_model_paths(self, config_path: str, checkpoint_path: str):
        """Set SAM2 model paths."""
        self.sam2_config = config_path
        self.sam2_checkpoint = checkpoint_path
        self.sam2_predictor = None  # Reset to force reload
        
        # Update model info in results
        if hasattr(self, 'results') and self.results:
            self.results["sam2_model_info"] = {
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
                "model_architecture": "SAM2"
            }
            self._unsaved_changes = True

    def group_annotations_by_video(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Get cached video annotations (grouped during initialization)."""
        return self.video_annotations

    def get_processed_video_ids(self) -> List[str]:
        """Get video_ids that have been successfully processed."""
        processed_videos = []
        for exp_data in self.results.get("experiments", {}).values():
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_data.get("sam2_success", False):
                    processed_videos.append(video_id)
        return processed_videos

    def get_missing_videos(self, video_ids: Optional[List[str]] = None,
                        experiment_ids: Optional[List[str]] = None) -> List[str]:
        """
        Get video_ids that haven't been processed yet.
        
        Args:
            video_ids: Specific video IDs to check (optional)
            experiment_ids: Specific experiment IDs to check (optional)
            
        Returns:
            List of unprocessed video_ids
        """
        if not self.video_annotations:
            if self.verbose:
                print("❌ No video annotations available")
                print("   Please ensure valid seed annotations are loaded with high-quality data")
            return []
        
        # Get all available videos from cached annotations
        available_videos = set(self.video_annotations.keys())
        
        # Filter by experiment_ids if specified
        if experiment_ids:
            filtered_videos = set()
            for video_id in available_videos:
                exp_id = video_id.split('_')[0]
                if exp_id in experiment_ids:
                    filtered_videos.add(video_id)
            available_videos = filtered_videos
        
        # Filter by video_ids if specified
        if video_ids:
            available_videos = available_videos.intersection(set(video_ids))
        
        # Get processed videos
        processed_videos = set(self.get_processed_video_ids())
        
        # Find missing videos
        missing_videos = list(available_videos - processed_videos)
        
        if self.verbose:
            print(f"📊 Video processing status:")
            print(f"   Available videos: {len(available_videos)}")
            print(f"   Processed videos: {len(processed_videos & available_videos)}")
            print(f"   Missing videos: {len(missing_videos)}")
        
        return missing_videos
    def process_missing_annotations(self, 
                                  video_ids: Optional[List[str]] = None,
                                  experiment_ids: Optional[List[str]] = None,
                                  max_videos: Optional[int] = None,
                                  auto_save_interval: Optional[int] = 5,
                                  overwrite: bool = False) -> Dict:
        """
        Process missing SAM2 annotations by running video segmentation on unprocessed videos.
        
        Args:
            video_ids: Specific video IDs to process (optional)
            experiment_ids: Specific experiment IDs to process (optional)
            max_videos: Maximum number of videos to process
            auto_save_interval: How often to auto-save during processing
            overwrite: Whether to overwrite existing results
            
        Returns:
            Dict of processing results
        """
        if not self.seed_annotations:
            if self.verbose:
                print("❌ No seed annotations loaded")
            return {}
        
        # Load SAM2 model if needed
        self._load_sam2_model()
        
        # Get missing videos
        missing_videos = self.get_missing_videos(video_ids, experiment_ids)
        
        if not missing_videos:
            if self.verbose:
                print("✅ No missing videos found!")
            return {}
        
        # Apply max_videos limit
        if max_videos:
            missing_videos = missing_videos[:max_videos]
        
        if self.verbose:
            print(f"🔄 Processing {len(missing_videos)} missing videos...")
        
        # Initialize processing statistics
        processing_stats = {
            "videos_processed": 0,
            "videos_failed": 0,
            "total_frames_processed": 0,
            "total_embryos_tracked": 0,
            "videos_with_non_first_seed": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Get video annotations (use cached)
        if not self.video_annotations:
            if self.verbose:
                print("❌ No video annotations available")
            return {}
        
        all_results = {}
        processed_count = 0
        
        for video_idx, video_id in enumerate(missing_videos, 1):
            if self.verbose:
                print(f"\n{'='*15} Video {video_idx}/{len(missing_videos)} {'='*15}")
            
            try:
                # Check if video should be overwritten
                if not overwrite and video_id in self.get_processed_video_ids():
                    if self.verbose:
                        print(f"⏭️  Skipping {video_id} (already processed, use overwrite=True)")
                    continue
                
                # Get video annotations (from cached data)
                if video_id not in self.video_annotations:
                    if self.verbose:
                        print(f"⚠️  No annotations found for {video_id}")
                    continue
                
                video_ann = self.video_annotations[video_id]
                
                # Process video using utility function
                sam2_results, video_metadata = process_single_video_from_annotations(
                    video_id, video_ann, self, self.sam2_predictor,  # PASS SELF instead of self.seed_annotations
                    processing_stats, self.segmentation_format, self.verbose
                )
                
                # Store results if successful
                if sam2_results and video_metadata.get("sam2_success"):
                    # Extract experiment_id from video_id
                    experiment_id = video_id.split('_')[0]
                    
                    # Initialize experiment structure if needed
                    if experiment_id not in self.results["experiments"]:
                        self.results["experiments"][experiment_id] = {
                            "experiment_id": experiment_id,
                            "first_processed_time": datetime.now().isoformat(),
                            "last_processed_time": datetime.now().isoformat(),
                            "videos": {}
                        }
                    
                    # Create video result structure
                    video_result = {
                        "video_id": video_id,
                        "well_id": video_id.split('_')[-1],
                        "seed_frame_info": video_metadata.get("seed_info", {}),
                        "embryo_ids": video_metadata.get("embryo_ids", []),
                        "num_embryos": video_metadata.get("num_embryos", 0),
                        "frames_processed": video_metadata.get("frames_processed", 0),
                        "sam2_success": video_metadata.get("sam2_success", False),
                        "processing_timestamp": video_metadata.get("processing_timestamp"),
                        "requires_bidirectional_propagation": video_metadata.get("requires_bidirectional_propagation", False),
                        "images": {}
                    }
                    
                    # Add image results with proper structure and corrected frame indexing
                    seed_frame_id = video_metadata.get("seed_info", {}).get("seed_frame")
                    all_image_ids = video_metadata.get("seed_info", {}).get("all_frames", [])
                    
                    # Create a robust frame index mapping that handles bidirectional results
                    # First, create the original mapping from all_image_ids
                    original_image_id_to_frame_idx = {image_id: idx for idx, image_id in enumerate(all_image_ids)}
                    
                    # Then, ensure all sam2_results have valid frame indices
                    # For any missing image_ids, assign sequential indices based on sorted order
                    sam2_image_ids = list(sam2_results.keys())
                    
                    # Sort sam2_image_ids to maintain temporal order (assuming image_id format includes temporal info)
                    sam2_image_ids.sort()
                    
                    # Create final frame index mapping
                    final_image_id_to_frame_idx = {}
                    
                    for image_id in sam2_image_ids:
                        if image_id in original_image_id_to_frame_idx:
                            # Use original frame index if available
                            final_image_id_to_frame_idx[image_id] = original_image_id_to_frame_idx[image_id]
                        else:
                            # For missing image_ids, find their correct position in the sequence
                            # This can happen when bidirectional propagation creates additional results
                            if all_image_ids:
                                # Find where this image_id would fit in the sorted sequence
                                insertion_point = 0
                                for i, orig_image_id in enumerate(all_image_ids):
                                    if image_id < orig_image_id:
                                        insertion_point = i
                                        break
                                    elif image_id > orig_image_id:
                                        insertion_point = i + 1
                                
                                final_image_id_to_frame_idx[image_id] = insertion_point
                            else:
                                # Fallback: use position in sorted sam2_image_ids
                                final_image_id_to_frame_idx[image_id] = sam2_image_ids.index(image_id)
                    
                    # Verify no negative frame indices
                    if any(idx < 0 for idx in final_image_id_to_frame_idx.values()):
                        if self.verbose:
                            print(f"   ⚠️  Warning: Found negative frame indices, using sorted order as fallback")
                        # Fallback: assign sequential indices based on sorted order
                        final_image_id_to_frame_idx = {image_id: idx for idx, image_id in enumerate(sam2_image_ids)}
                    
                    if self.verbose:
                        frame_indices = list(final_image_id_to_frame_idx.values())
                        print(f"   📋 Frame indices: {min(frame_indices)} to {max(frame_indices)} ({len(frame_indices)} frames)")
                    
                    for image_id, embryo_data in sam2_results.items():
                        frame_idx = final_image_id_to_frame_idx.get(image_id, sam2_image_ids.index(image_id))
                        
                        video_result["images"][image_id] = {
                            "image_id": image_id,
                            "frame_index": frame_idx,
                            "is_seed_frame": image_id == seed_frame_id,
                            "embryos": embryo_data
                        }
                    
                    # Store video result
                    self.results["experiments"][experiment_id]["videos"][video_id] = video_result
                    self.results["experiments"][experiment_id]["last_processed_time"] = datetime.now().isoformat()
                    
                    # Update global lists
                    if experiment_id not in self.results["experiment_ids"]:
                        self.results["experiment_ids"].append(experiment_id)

                    if video_id not in self.results["video_ids"]:
                        self.results["video_ids"].append(video_id)

                    # Add embryo_ids and snip_ids to global lists
                    embryo_ids = video_metadata.get("embryo_ids", [])
                    for embryo_id in embryo_ids:
                        if embryo_id not in self.results["embryo_ids"]:
                            self.results["embryo_ids"].append(embryo_id)

                    # Extract snip_ids from sam2_results
                    for image_id, embryo_data in sam2_results.items():
                        for embryo_id, embryo_info in embryo_data.items():
                            snip_id = embryo_info.get("snip_id")
                            if snip_id and snip_id not in self.results["snip_ids"]:
                                self.results["snip_ids"].append(snip_id)
                    
                    self.results["last_updated"] = datetime.now().isoformat()
                    self._unsaved_changes = True
                    
                    all_results[video_id] = video_result
                    processed_count += 1
                    
                    if self.verbose:
                        frames = video_metadata.get("frames_processed", 0)
                        embryos = video_metadata.get("num_embryos", 0)
                        print(f"   ✅ Success: {frames} frames, {embryos} embryos")
                
                else:
                    # Store failed result
                    experiment_id = video_id.split('_')[0]
                    if experiment_id not in self.results["experiments"]:
                        self.results["experiments"][experiment_id] = {
                            "experiment_id": experiment_id,
                            "first_processed_time": datetime.now().isoformat(),
                            "last_processed_time": datetime.now().isoformat(),
                            "videos": {}
                        }
                    
                    self.results["experiments"][experiment_id]["videos"][video_id] = video_metadata
                    self._unsaved_changes = True
                    
                    if self.verbose:
                        error_msg = video_metadata.get("error_message", "Unknown error")
                        print(f"   ❌ Failed: {error_msg}")
                
                # Auto-save periodically
                if auto_save_interval and processed_count % auto_save_interval == 0:
                    if self.verbose:
                        print(f"💾 Auto-saving after {processed_count} processed videos...")
                    self.save()
                
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Unexpected error processing {video_id}: {e}")
                processing_stats["videos_failed"] += 1
                continue
        
        # Final save
        if self._unsaved_changes:
            if self.verbose:
                print(f"💾 Final save...")
            self.save()
        
        # Update final statistics
        processing_stats["end_time"] = datetime.now().isoformat()
        self.results["processing_stats"] = processing_stats
        
        if self.verbose:
            print(f"\n🎯 Processing Complete!")
            print(f"Videos processed: {processing_stats['videos_processed']}")
            print(f"Videos failed: {processing_stats['videos_failed']}")
            print(f"Success rate: {processing_stats['videos_processed']/len(missing_videos)*100:.1f}%" if missing_videos else "0%")
            print(f"Frames processed: {processing_stats['total_frames_processed']}")
            print(f"Embryos tracked: {processing_stats['total_embryos_tracked']}")
        
        return all_results

    def process_video(self, video_id: str) -> Dict:
        """Process a single video with SAM2 segmentation."""
        if self.verbose:
            print(f"\n🎬 Processing video: {video_id}")
        
        if not self.video_annotations:
            raise ValueError("No video annotations available - ensure seed annotations are loaded")
        
        # Load SAM2 model if needed
        self._load_sam2_model()
        
        # Get video annotations (from cached data)
        if video_id not in self.video_annotations:
            raise ValueError(f"No annotations found for video {video_id}")
        
        video_ann = self.video_annotations[video_id]
        
        # Initialize processing stats
        processing_stats = {
            "videos_processed": 0,
            "videos_failed": 0,
            "videos_with_non_first_seed": 0,
            "total_frames_processed": 0,
            "total_embryos_tracked": 0
        }
        
        # Process video using utility function
        sam2_results, video_metadata = process_single_video_from_annotations(
            video_id, video_ann, self, self.sam2_predictor,
            processing_stats, self.segmentation_format, self.verbose
        )
        
        # Structure results for this video
        if sam2_results:
            # Extract experiment_id from video_id
            experiment_id = video_id.split('_')[0]
            
            # Initialize experiment structure if needed
            if experiment_id not in self.results["experiments"]:
                self.results["experiments"][experiment_id] = {
                    "experiment_id": experiment_id,
                    "first_processed_time": datetime.now().isoformat(),
                    "last_processed_time": datetime.now().isoformat(),
                    "videos": {}
                }
            
            # Create video result structure
            video_result = {
                "video_id": video_id,
                "well_id": video_metadata.get("well_id", video_id.split('_')[-1]),
                "seed_frame_info": video_metadata.get("seed_info", {}),
                "embryo_ids": video_metadata.get("embryo_ids", []),
                "num_embryos": video_metadata.get("num_embryos", 0),
                "frames_processed": video_metadata.get("frames_processed", 0),
                "sam2_success": video_metadata.get("sam2_success", False),
                "processing_timestamp": video_metadata.get("processing_timestamp"),
                "requires_bidirectional_propagation": video_metadata.get("requires_bidirectional_propagation", False),
                "images": {}
            }
            
            # Add image results with proper structure and corrected frame indexing
            seed_frame_id = video_metadata.get("seed_info", {}).get("seed_frame")
            all_image_ids = video_metadata.get("seed_info", {}).get("all_frames", [])
            
            # Create a robust frame index mapping that handles bidirectional results
            original_image_id_to_frame_idx = {image_id: idx for idx, image_id in enumerate(all_image_ids)}
            
            # Ensure all sam2_results have valid frame indices
            sam2_image_ids = list(sam2_results.keys())
            sam2_image_ids.sort()  # Sort to maintain temporal order
            
            # Create final frame index mapping
            final_image_id_to_frame_idx = {}
            
            for image_id in sam2_image_ids:
                if image_id in original_image_id_to_frame_idx:
                    # Use original frame index if available
                    final_image_id_to_frame_idx[image_id] = original_image_id_to_frame_idx[image_id]
                else:
                    # For missing image_ids, find their correct position in the sequence
                    if all_image_ids:
                        insertion_point = 0
                        for i, orig_image_id in enumerate(all_image_ids):
                            if image_id < orig_image_id:
                                insertion_point = i
                                break
                            elif image_id > orig_image_id:
                                insertion_point = i + 1
                        final_image_id_to_frame_idx[image_id] = insertion_point
                    else:
                        # Fallback: use position in sorted sam2_image_ids
                        final_image_id_to_frame_idx[image_id] = sam2_image_ids.index(image_id)
            
            # Verify no negative frame indices
            if any(idx < 0 for idx in final_image_id_to_frame_idx.values()):
                if self.verbose:
                    print(f"   ⚠️  Warning: Found negative frame indices, using sorted order as fallback")
                # Fallback: assign sequential indices based on sorted order
                final_image_id_to_frame_idx = {image_id: idx for idx, image_id in enumerate(sam2_image_ids)}
            
            if self.verbose:
                frame_indices = list(final_image_id_to_frame_idx.values())
                print(f"   📋 Frame indices: {min(frame_indices)} to {max(frame_indices)} ({len(frame_indices)} frames)")
            
            for image_id, embryo_data in sam2_results.items():
                frame_idx = final_image_id_to_frame_idx.get(image_id, sam2_image_ids.index(image_id))
                
                video_result["images"][image_id] = {
                    "image_id": image_id,
                    "frame_index": frame_idx,
                    "is_seed_frame": image_id == seed_frame_id,
                    "embryos": embryo_data
                }
            
            # Store video result
            self.results["experiments"][experiment_id]["videos"][video_id] = video_result
            self.results["experiments"][experiment_id]["last_processed_time"] = datetime.now().isoformat()
            
            # Update global lists
            if experiment_id not in self.results["experiment_ids"]:
                self.results["experiment_ids"].append(experiment_id)

            if video_id not in self.results["video_ids"]:
                self.results["video_ids"].append(video_id)

            # Add embryo_ids and snip_ids to global lists
            embryo_ids = video_metadata.get("embryo_ids", [])
            for embryo_id in embryo_ids:
                if embryo_id not in self.results["embryo_ids"]:
                    self.results["embryo_ids"].append(embryo_id)

            # Extract snip_ids from sam2_results
            for image_id, embryo_data in sam2_results.items():
                for embryo_id, embryo_info in embryo_data.items():
                    snip_id = embryo_info.get("snip_id")
                    if snip_id and snip_id not in self.results["snip_ids"]:
                        self.results["snip_ids"].append(snip_id)
            
            self.results["last_updated"] = datetime.now().isoformat()
            self._unsaved_changes = True
            
            return video_result
        else:
            return video_metadata

    def process_videos(self, video_ids: Optional[List[str]] = None, max_videos: Optional[int] = None,
                      auto_save_interval: Optional[int] = 5) -> Dict:
        """Process multiple videos with autosave."""
        return self.process_missing_annotations(
            video_ids=video_ids,
            max_videos=max_videos,
            auto_save_interval=auto_save_interval,
            overwrite=False
        )

    def get_summary(self) -> Dict:
        """Get processing summary."""
        summary = {
            "total_experiments": len(self.results.get("experiment_ids", [])),
            "total_videos": len(self.results.get("video_ids", [])),
            "total_embryos": len(self.results.get("embryo_ids", [])),
            "target_prompt": self.results.get("target_prompt"),
            "segmentation_format": self.results.get("segmentation_format"),
            "last_updated": self.results.get("last_updated")
        }
        
        stats = self.results.get("processing_stats", {})
        if stats:
            summary.update({
                "videos_processed": stats.get("videos_processed", 0),
                "videos_failed": stats.get("videos_failed", 0),
                "total_frames_processed": stats.get("total_frames_processed", 0),
                "total_embryos_tracked": stats.get("total_embryos_tracked", 0)
            })
        
        # Add seed annotations info
        seed_info = self.results.get("seed_annotations_info", {})
        if seed_info:
            summary.update({
                "seed_model": seed_info.get("model_architecture", "unknown"),
                "seed_weights": Path(seed_info.get("model_weights", "unknown")).name,
                "has_high_quality_seed": seed_info.get("has_high_quality_annotations", False)
            })
        
        return summary

    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        print(f"\n📊 GROUNDED SAM2 SUMMARY")
        print(f"=" * 35)
        print(f"🧪 Experiments: {summary.get('total_experiments', 0)}")
        print(f"🎬 Videos: {summary.get('total_videos', 0)}")
        print(f"🧬 Embryos: {summary.get('total_embryos', 0)}")
        print(f"🎯 Seed prompt: '{summary.get('target_prompt', '')}'")
        print(f"📦 Format: {summary.get('segmentation_format', '')}")
        
        # Seed annotations info
        if summary.get('seed_model'):
            hq_status = "✅ High-quality" if summary.get('has_high_quality_seed') else "⚠️  Regular"
            print(f"🌱 Seed model: {summary.get('seed_model')} ({summary.get('seed_weights')})")
            print(f"🌱 Seed quality: {hq_status}")
        
        if 'videos_processed' in summary:
            print(f"✅ Videos processed: {summary['videos_processed']}")
            print(f"❌ Videos failed: {summary['videos_failed']}")
            print(f"🖼️  Frames processed: {summary['total_frames_processed']}")
            print(f"🔬 Embryos tracked: {summary['total_embryos_tracked']}")
        
        print(f"🕒 Last updated: {summary.get('last_updated', '')}")

    @property
    def has_unsaved_changes(self) -> bool:
        """Check for unsaved changes."""
        return self._unsaved_changes

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        status = "✅ saved" if not self._unsaved_changes else "⚠️ unsaved"
        seed_status = f", 🌱 {summary.get('seed_model', 'no-seed')}"
        return (f"GroundedSamAnnotations(experiments={summary.get('total_experiments', 0)}, "
                f"videos={summary.get('total_videos', 0)}, embryos={summary.get('total_embryos', 0)}, "
                f"prompt='{summary.get('target_prompt', '')}', {status}{seed_status})")
    
    def _group_video_annotations(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Group video annotations once during initialization."""
        if not self.seed_annotations:
            if self.verbose:
                print("⚠️  No seed annotations loaded - video grouping skipped")
            return {}
        
        video_annotations = group_annotations_by_video(self.seed_annotations, self.target_prompt)
        
        if self.verbose:
            total_videos = len(video_annotations)
            total_images = sum(len(video_data) for video_data in video_annotations.values())
            print(f"📋 Grouped annotations: {total_videos} videos, {total_images} images")
        
        return video_annotations

def run_sam2_propagation(predictor, video_dir: Path, seed_frame_idx: int, 
                        seed_detections: List[Dict], embryo_ids: List[str],
                        image_ids: List[str], segmentation_format: str = 'rle',
                        verbose: bool = True) -> Dict:
    """
    Run SAM2 propagation from seed frame using the actual processed images directory.
    FIXED: Updated to use corrected bbox format.
    """
    if verbose:
        print(f"🔄 Running SAM2 propagation from frame {seed_frame_idx}...")
        print(f"   Video directory: {video_dir}")
        print(f"   Seed frame image_id: {image_ids[seed_frame_idx]}")
    
    # Create temporary directory with properly named symlinks for SAM2
    with tempfile.TemporaryDirectory() as temp_dir_str:
        sam2_video_dir = Path(temp_dir_str) / "sam2_frames"
        sam2_video_dir.mkdir(parents=True)
        
        if verbose:
            print(f"   📁 Creating SAM2-compatible frame directory with {len(image_ids)} frames")
        
        # Create sequentially named symlinks (000000.jpg, 000001.jpg, ...)
        for idx, image_id in enumerate(image_ids):
            src_frame = video_dir / f"{image_id}.jpg"
            dst_frame = sam2_video_dir / f"{idx:06d}.jpg"
            
            if src_frame.exists():
                try:
                    dst_frame.symlink_to(src_frame.absolute())
                except OSError:
                    # Fallback to copying if symlink fails
                    shutil.copy2(src_frame, dst_frame)
            else:
                raise FileNotFoundError(f"Source image not found: {src_frame}")
        
        # Initialize SAM2 inference state with properly named directory
        inference_state = predictor.init_state(video_path=str(sam2_video_dir))
        predictor.reset_state(inference_state)
        
        # Add bounding boxes from seed frame detections
        for embryo_idx, (detection, embryo_id) in enumerate(zip(seed_detections, embryo_ids)):
            # Detections are now in xyxy format (normalized coordinates)
            x1, y1, x2, y2 = detection["box_xyxy"]
            bbox_xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            
            # Add box to SAM2
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=seed_frame_idx,
                obj_id=embryo_idx + 1,  # SAM2 object IDs start from 1
                box=bbox_xyxy
            )
            
            if verbose:
                print(f"   Added embryo {embryo_id} (SAM2 obj_id: {embryo_idx + 1})")
        
        # Propagate through video
        video_segments = {}
        if verbose:
            print(f"   Propagating through {len(image_ids)} video frames...")
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # Map frame index back to image_id
            if out_frame_idx < len(image_ids):
                image_id = image_ids[out_frame_idx]
                frame_results = {}
                
                for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                    if obj_id <= len(embryo_ids):  # Valid embryo ID
                        embryo_id = embryo_ids[obj_id - 1]  # Convert back to 0-based indexing
                        
                        # Convert mask logits to binary mask
                        binary_mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                        
                        # Extract segmentation in specified format
                        if segmentation_format == 'rle':
                            segmentation = convert_sam2_mask_to_rle(binary_mask)
                        elif segmentation_format == 'polygon':
                            segmentation = convert_sam2_mask_to_polygon(binary_mask)
                        else:
                            raise ValueError(f"Unknown segmentation_format: {segmentation_format}")
                        
                        bbox = extract_bbox_from_mask(binary_mask)  # Now returns xyxy format
                        area = float(np.sum(binary_mask))
                        
                        # Calculate mask confidence (mean of positive logits)
                        positive_logits = mask_logits[0][mask_logits[0] > 0]
                        mask_confidence = float(torch.mean(positive_logits)) if len(positive_logits) > 0 else 0.0
                        
                        # ADDED: Create snip_id using embryo_id and frame suffix from image_id
                        snip_id = create_snip_id(embryo_id, image_id)

                        frame_results[embryo_id] = {
                            "embryo_id": embryo_id,
                            "snip_id": snip_id,  # ADDED: Unique identifier for this embryo snippet
                            "segmentation": segmentation,
                            "segmentation_format": segmentation_format,
                            "bbox": bbox,  # Now in xyxy format
                            "area": area,
                            "mask_confidence": mask_confidence
                        }
                                        
                video_segments[image_id] = frame_results
    
    if verbose:
        print(f"✅ SAM2 propagation complete for {len(video_segments)} frames")
    
    return video_segments

def run_bidirectional_propagation(predictor, video_dir: Path, seed_frame_idx: int,
                                 seed_detections: List[Dict], embryo_ids: List[str],
                                 image_ids: List[str], segmentation_format: str = 'rle',
                                 verbose: bool = True) -> Dict:
    """
    Run bidirectional SAM2 propagation when seed frame is not the first frame.
    FIXED: Properly maintains frame ordering in final results.
    
    Args:
        predictor: SAM2 video predictor
        video_dir: Directory containing processed video frames (from metadata)
        seed_frame_idx: Index of seed frame
        seed_detections: Detections from seed frame
        embryo_ids: Assigned embryo IDs
        image_ids: Ordered list of image IDs corresponding to frame indices
        segmentation_format: 'rle' (recommended) or 'polygon' for segmentation storage
        verbose: Enable verbose output
        
    Returns:
        Combined segmentation results from both directions with proper frame ordering
    """
    if verbose:
        print(f"🔄 Running bidirectional SAM2 propagation...")
        print(f"   Seed frame: {seed_frame_idx} ({image_ids[seed_frame_idx]})")
        print(f"   Total frames: {len(image_ids)}")
    
    # Forward propagation (seed to end) - use original video directory
    if verbose:
        print("   🔜 Forward propagation (seed → end)")
    forward_results = run_sam2_propagation(predictor, video_dir, seed_frame_idx, 
                                          seed_detections, embryo_ids, image_ids, 
                                          segmentation_format, verbose=verbose)
    
    # Backward propagation (seed to beginning) - only if there are frames before seed
    backward_results = {}
    if seed_frame_idx > 0:
        if verbose:
            print("   🔙 Backward propagation (seed → beginning)")
        
        # Create temporary directory with properly ordered frames
        with tempfile.TemporaryDirectory() as temp_dir_str:
            backward_video_dir = Path(temp_dir_str) / "backward_frames"
            backward_video_dir.mkdir(parents=True)
            
            # Create frames to reverse: from seed_frame_idx down to 0
            frames_to_reverse = list(range(seed_frame_idx + 1))  # [0, 1, 2, ..., seed_frame_idx]
            frames_to_reverse.reverse()  # [seed_frame_idx, seed_frame_idx-1, ..., 1, 0]
            
            if verbose:
                print(f"   📁 Reordering {len(frames_to_reverse)} frames for backward propagation")
            
            # Create sequentially named symlinks so SAM2 processes them in correct order
            backward_image_ids = []
            for new_idx, original_idx in enumerate(frames_to_reverse):
                original_image_id = image_ids[original_idx]
                backward_image_ids.append(original_image_id)
                
                # Source: original image file
                src_frame = video_dir / f"{original_image_id}.jpg"
                
                # Destination: sequential numbering (000000.jpg, 000001.jpg, ...)
                # This ensures SAM2 processes them in the order we want
                dst_frame = backward_video_dir / f"{new_idx:06d}.jpg"
                
                if src_frame.exists() and not dst_frame.exists():
                    try:
                        dst_frame.symlink_to(src_frame.absolute())
                    except OSError:
                        # Fallback to copying if symlink fails
                        shutil.copy2(src_frame, dst_frame)
            
            if verbose:
                print(f"   🎯 Seed frame ({image_ids[seed_frame_idx]}) is now at index 0")
            
            # Run backward propagation with properly ordered frames
            
            # Initialize SAM2 for backward directory
            inference_state = predictor.init_state(video_path=str(backward_video_dir))
            predictor.reset_state(inference_state)
            
            # Add bounding boxes from seed frame (now at index 0 in backward directory)
            if verbose:
                print(f"   🎯 Adding {len(seed_detections)} embryo detections at frame 0 (seed)")
            for embryo_idx, (detection, embryo_id) in enumerate(zip(seed_detections, embryo_ids)):
                # Detections are now in xyxy format (normalized coordinates)
                x1, y1, x2, y2 = detection["box_xyxy"]
                bbox_xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,  # Seed frame is now at index 0
                    obj_id=embryo_idx + 1,
                    box=bbox_xyxy
                )
                if verbose:
                    print(f"      Added {embryo_id} (SAM2 obj_id: {embryo_idx + 1})")
            
            # Propagate through backward video
            if verbose:
                print(f"   🔄 Propagating through {len(backward_image_ids)} frames in backward order")
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                # Map SAM2 frame index back to original image_id
                if out_frame_idx < len(backward_image_ids):
                    original_image_id = backward_image_ids[out_frame_idx]
                    frame_results = {}
                    
                    for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                        if obj_id <= len(embryo_ids):
                            embryo_id = embryo_ids[obj_id - 1]
                            
                            # Convert mask and extract features
                            binary_mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                            
                            if segmentation_format == 'rle':
                                segmentation = convert_sam2_mask_to_rle(binary_mask)
                            else:
                                segmentation = convert_sam2_mask_to_polygon(binary_mask)
                            
                            bbox = extract_bbox_from_mask(binary_mask)  # Now returns xyxy format
                            area = float(np.sum(binary_mask))
                            
                            positive_logits = mask_logits[0][mask_logits[0] > 0]
                            mask_confidence = float(torch.mean(positive_logits)) if len(positive_logits) > 0 else 0.0
                            
                            frame_results[embryo_id] = {
                                "embryo_id": embryo_id,
                                "segmentation": segmentation,
                                "segmentation_format": segmentation_format,
                                "bbox": bbox,  # Now in xyxy format
                                "area": area,
                                "mask_confidence": mask_confidence
                            }
                    
                    backward_results[original_image_id] = frame_results
    
    # FIXED: Properly combine results maintaining original frame order
    if verbose:
        print("   🧵 Stitching bidirectional results with proper frame ordering...")
    
    # Create combined results in original frame order using OrderedDict to maintain sequence
    from collections import OrderedDict
    combined_results = OrderedDict()
    seed_image_id = image_ids[seed_frame_idx]
    
    # Process all frames in strict original temporal order
    for frame_idx, image_id in enumerate(image_ids):
        if frame_idx < seed_frame_idx:
            # Frames before seed: use backward results (excluding seed frame to avoid duplication)
            if image_id in backward_results and image_id != seed_image_id:
                combined_results[image_id] = backward_results[image_id]
        else:
            # Frames from seed onwards: use forward results (including seed frame)
            if image_id in forward_results:
                combined_results[image_id] = forward_results[image_id]
    
    # Convert back to regular dict but maintain order
    combined_results = dict(combined_results)
    
    if verbose:
        frames_from_backward = sum(1 for frame_idx, image_id in enumerate(image_ids) 
                                 if frame_idx < seed_frame_idx and image_id in combined_results)
        frames_from_forward = sum(1 for frame_idx, image_id in enumerate(image_ids) 
                                if frame_idx >= seed_frame_idx and image_id in combined_results)
        
        print(f"   📊 Combined results: {len(combined_results)} total frames")
        print(f"      • From backward: {frames_from_backward} frames (before seed)")
        print(f"      • From forward: {frames_from_forward} frames (seed onwards)")
        
        # Verify frame ordering is maintained by checking that keys are in temporal order
        result_image_ids = list(combined_results.keys())
        result_frame_indices = [image_ids.index(img_id) for img_id in result_image_ids]
        if result_frame_indices == sorted(result_frame_indices):
            print(f"   ✅ Frame ordering verified: properly sequential")
            print(f"   📋 Frame sequence: {result_frame_indices[:5]}{'...' if len(result_frame_indices) > 5 else ''}")
        else:
            print(f"   ⚠️  Frame ordering issue detected!")
            print(f"   📋 Expected order: {sorted(result_frame_indices)[:5]}...")
            print(f"   📋 Actual order: {result_frame_indices[:5]}...")
            
            # Fix ordering by recreating the dict in correct order
            print(f"   🔧 Fixing frame ordering...")
            ordered_results = OrderedDict()
            for frame_idx in sorted(result_frame_indices):
                image_id = image_ids[frame_idx]
                if image_id in combined_results:
                    ordered_results[image_id] = combined_results[image_id]
            combined_results = dict(ordered_results)
            print(f"   ✅ Frame ordering fixed")
    
    return combined_results

def process_single_video_from_annotations(video_id: str, video_annotations: Dict, grounded_sam_instance,
                                         predictor, processing_stats: Dict, segmentation_format: str = 'rle',
                                         verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Process a single video with SAM2 segmentation using class experiment metadata.
    
    Args:
        video_id: Video identifier
        video_annotations: Annotations for this video (image_id -> detections)
        grounded_sam_instance: GroundedSamAnnotations instance (contains experiment metadata)
        predictor: SAM2 video predictor
        processing_stats: Dictionary to update with processing statistics
        segmentation_format: 'rle' (recommended) or 'polygon' for segmentation storage
        verbose: Enable verbose output
        
    Returns:
        Tuple of (sam2_results, video_metadata)
    """
    if verbose:
        print(f"\n🎬 Processing video: {video_id}")
    
    try:
        # Get video info from class experiment metadata
        video_info = None
        if grounded_sam_instance.experiment_metadata:
            for exp_data in grounded_sam_instance.experiment_metadata.get("experiments", {}).values():
                for vid_id, vid_data in exp_data.get("videos", {}).items():
                    if vid_id == video_id:
                        video_info = vid_data
                        break
                if video_info:
                    break
        
        if not video_info:
            raise ValueError(f"Video {video_id} not found in experiment metadata")
        
        # Get video directory and image info from metadata
        video_dir = Path(video_info["processed_jpg_images_dir"])
        if not video_dir.exists():
            raise FileNotFoundError(f"Processed images directory not found: {video_dir}")
        
        image_ids = sorted(video_info.get("image_ids", []))
        if not image_ids:
            raise ValueError(f"No images found for video_id: {video_id}")
        
        image_id_to_frame_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
        
        # Find seed frame using video annotations
        seed_frame_id, seed_info = find_seed_frame_from_video_annotations(video_annotations, video_id)
        
        # Get seed frame index and detections
        seed_frame_idx = image_id_to_frame_idx[seed_frame_id]
        seed_detections = video_annotations[seed_frame_id]
        
        # Assign embryo IDs
        num_embryos = len(seed_detections)
        embryo_ids = assign_embryo_ids(video_id, num_embryos)
        
        if verbose:
            print(f"   📍 Seed frame: {seed_frame_id} (index {seed_frame_idx})")
            print(f"   🧬 Embryos: {num_embryos} ({', '.join(embryo_ids)})")
            print(f"   📁 Using processed images: {video_dir}")
        
        if seed_frame_idx != 0 and verbose:
            print(f"   ⚠️  Seed frame is not first frame - will use bidirectional propagation")
        
        # Verify that the seed frame image exists
        seed_image_path = video_dir / f"{seed_frame_id}.jpg"
        if not seed_image_path.exists():
            raise FileNotFoundError(f"Seed frame image not found: {seed_image_path}")
        
        # Run SAM2 propagation
        if seed_frame_idx == 0:
            # Simple forward propagation (seed frame is first frame)
            sam2_results = run_sam2_propagation(
                predictor, video_dir, seed_frame_idx, seed_detections, embryo_ids, 
                image_ids, segmentation_format, verbose=verbose
            )
        else:
            # Bidirectional propagation (seed frame is not first frame)
            sam2_results = run_bidirectional_propagation(
                predictor, video_dir, seed_frame_idx, seed_detections, embryo_ids, 
                image_ids, segmentation_format, verbose=verbose
            )
        
        # Update processing statistics
        processing_stats["videos_processed"] += 1
        processing_stats["total_frames_processed"] += len(sam2_results)
        processing_stats["total_embryos_tracked"] += num_embryos
        
        if seed_frame_idx != 0:
            processing_stats["videos_with_non_first_seed"] += 1
        
        # Create video metadata
        video_metadata = {
            "video_id": video_id,
            "seed_info": seed_info,
            "embryo_ids": embryo_ids,
            "num_embryos": num_embryos,
            "frames_processed": len(sam2_results),
            "processed_jpg_images_dir": str(video_dir),
            "requires_bidirectional_propagation": seed_frame_idx != 0,
            "processing_timestamp": datetime.now().isoformat(),
            "sam2_success": True
        }
        
        if verbose:
            print(f"   ✅ Processed {len(sam2_results)} frames with {num_embryos} embryos")
        
        return sam2_results, video_metadata
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Error processing video {video_id}: {e}")
        processing_stats["videos_failed"] += 1
        
        # Return empty results with error info
        error_metadata = {
            "video_id": video_id,
            "sam2_success": False,
            "error_message": str(e),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return {}, error_metadata

def extract_frame_suffix(image_id: str) -> str:
    """Extract frame suffix from image_id (e.g., '0000' from '20240411_A01_0000')."""
    return image_id.split('_')[-1]

def create_snip_id(embryo_id: str, image_id: str) -> str:
    """Create snip_id by combining embryo_id with frame suffix from image_id."""
    frame_suffix = extract_frame_suffix(image_id)
    return f"{embryo_id}_{frame_suffix}"
    
def convert_sam2_mask_to_rle(binary_mask: np.ndarray) -> Dict:
    """Convert SAM2 binary mask to RLE format for compact storage."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        print("Warning: pycocotools not available, using simple mask storage")
        return {
            'format': 'simple_mask',
            'size': binary_mask.shape,
            'data': binary_mask.flatten().tolist()
        }
    
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    binary_mask_fortran = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(binary_mask_fortran)
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle

def convert_sam2_mask_to_polygon(binary_mask: np.ndarray) -> List[List[float]]:
    """Convert SAM2 binary mask to polygon format."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.flatten().astype(float).tolist()
            polygons.append(polygon)
    
    return polygons

def extract_bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    """Extract bounding box from binary mask in normalized xyxy format."""
    y_indices, x_indices = np.where(binary_mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    h, w = binary_mask.shape
    bbox_xyxy = [
        x_min / w,  # x1 (normalized)
        y_min / h,  # y1 (normalized)
        x_max / w,  # x2 (normalized)
        y_max / h   # y2 (normalized)
    ]
    
    return bbox_xyxy

def extract_seed_annotations_info(annotations: Dict, target_prompt: str = "individual embryo") -> Dict:
    """Extract information about the seed annotations (model, prompt, etc.)."""
    seed_info = {
        "target_prompt": target_prompt,
        "model_architecture": "unknown",
        "model_weights": "unknown",
        "model_config": "unknown",
        "has_high_quality_annotations": False,
        "experiments_with_hq": 0,
        "total_images_with_detections": 0
    }
    
    # Check for high-quality annotations first
    hq_annotations = annotations.get("high_quality_annotations", {})
    if hq_annotations:
        seed_info["has_high_quality_annotations"] = True
        seed_info["experiments_with_hq"] = len(hq_annotations)
        
        # Get model info from any experiment with the target prompt
        for exp_id, exp_data in hq_annotations.items():
            if exp_data.get("prompt") == target_prompt:
                seed_info["total_images_with_detections"] += len(exp_data.get("filtered", {}))
                break
    
    # Get model metadata from any annotation in the regular annotations
    for image_id, image_data in annotations.get("images", {}).items():
        for annotation in image_data.get("annotations", []):
            if annotation.get("prompt") == target_prompt:
                model_meta = annotation.get("model_metadata", {})
                seed_info["model_architecture"] = model_meta.get("model_architecture", "unknown")
                seed_info["model_weights"] = model_meta.get("model_weights_path", "unknown")
                seed_info["model_config"] = model_meta.get("model_config_path", "unknown")
                break
        if seed_info["model_architecture"] != "unknown":
            break
    
    return seed_info

def group_annotations_by_video(annotations: Dict, target_prompt: str = "individual embryo") -> Dict[str, Dict]:
    """
    Group high-quality annotations by video_id.
    
    Args:
        annotations: Annotations dictionary (with high_quality_annotations section)
        target_prompt: Target prompt to filter for
        
    Returns:
        Dictionary mapping video_id to image_id -> detections
    """
    print("🔗 Grouping high-quality annotations by video...")
    
    video_annotations = defaultdict(dict)
    
    # Check for high-quality annotations first
    hq_annotations = annotations.get("high_quality_annotations", {})
    if hq_annotations:
        print(f"   Using high-quality annotations for prompt: '{target_prompt}'")
        
        for exp_id, exp_data in hq_annotations.items():
            if exp_data.get("prompt") == target_prompt:
                filtered_data = exp_data.get("filtered", {})
                
                for image_id, detections in filtered_data.items():
                    # Extract video_id from image_id (format: experiment_well_frame)
                    parts = image_id.split('_')
                    if len(parts) >= 3:
                        video_id = '_'.join(parts[:2])  # experiment_well
                        
                        if detections:  # Only include if there are detections
                            video_annotations[video_id][image_id] = detections
    else:
        # Fallback to regular annotations
        print(f"   No high-quality annotations found, using regular annotations for prompt: '{target_prompt}'")
        
        for image_id, image_data in annotations.get("images", {}).items():
            # Extract video_id from image_id
            parts = image_id.split('_')
            if len(parts) >= 3:
                video_id = '_'.join(parts[:2])
                
                # Extract detections for target prompt
                detections = []
                for annotation in image_data.get('annotations', []):
                    if annotation.get('prompt') == target_prompt:
                        detections.extend(annotation.get('detections', []))
                
                if detections:
                    video_annotations[video_id][image_id] = detections
    
    print(f"📊 Found annotations for {len(video_annotations)} videos")
    for video_id, image_annotations in video_annotations.items():
        total_detections = sum(len(dets) for dets in image_annotations.values())
        print(f"  {video_id}: {len(image_annotations)} images, {total_detections} detections")
    
    return dict(video_annotations)

def get_video_metadata_from_annotations(video_id: str, annotations: Dict) -> Optional[Dict]:
    """
    Extract video metadata from experiment metadata stored in annotations.
    
    Args:
        video_id: Video identifier
        annotations: Annotations dictionary that should contain experiment metadata
        
    Returns:
        Video metadata if found, None otherwise
    """
    # Look for experiment metadata in the annotations file
    experiment_metadata = annotations.get("experiment_metadata")
    if not experiment_metadata:
        print(f"   ⚠️  No experiment_metadata found in annotations file")
        return None
    
    # Find video metadata
    for exp_data in experiment_metadata.get("experiments", {}).values():
        for vid_id, vid_data in exp_data.get("videos", {}).items():
            if vid_id == video_id:
                return vid_data
    
    # Debug: Show what videos are available in metadata
    available_videos = []
    for exp_data in experiment_metadata.get("experiments", {}).values():
        available_videos.extend(exp_data.get("videos", {}).keys())
    
    print(f"   ⚠️  Video {video_id} not found in experiment metadata")
    print(f"   📋 Available videos in metadata: {len(available_videos)} total")
    if available_videos:
        # Show some examples
        exp_id = video_id.split('_')[0]
        same_exp_videos = [v for v in available_videos if v.startswith(exp_id)]
        if same_exp_videos:
            print(f"   📋 Same experiment ({exp_id}) videos available: {same_exp_videos[:5]}...")
        else:
            print(f"   📋 No videos found for experiment {exp_id}")
            # Show available experiments
            available_exps = set(v.split('_')[0] for v in available_videos)
            print(f"   📋 Available experiments: {sorted(list(available_exps))[:10]}...")
    
    return None

def prepare_video_frames_from_annotations(video_id: str, annotations: Dict) -> Tuple[Path, List[str], Dict[str, int], Dict]:
    """
    Prepare video information for SAM2 processing using metadata from annotations file.
    
    Args:
        video_id: Video identifier
        annotations: Annotations dictionary containing experiment metadata
        
    Returns:
        Tuple of (video_directory, image_ids, image_id_to_frame_index_mapping, video_metadata)
    """
    video_info = get_video_metadata_from_annotations(video_id, annotations)
    
    if not video_info:
        raise ValueError(f"Video {video_id} not found in annotations metadata")
    
    # Get the processed images directory from metadata
    processed_jpg_images_dir = Path(video_info["processed_jpg_images_dir"])
    if not processed_jpg_images_dir.exists():
        raise FileNotFoundError(f"Processed images directory not found: {processed_jpg_images_dir}")
    
    # Get all image IDs for this video in correct order
    image_ids = video_info.get("image_ids", [])
    if not image_ids:
        raise ValueError(f"No images found for video_id: {video_id}")
    
    # Verify that the images exist in the processed directory
    missing_images = []
    for image_id in image_ids:
        image_path = processed_jpg_images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            missing_images.append(image_id)
    
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images in {processed_jpg_images_dir}: {missing_images[:5]}...")
    
    # Create mapping from image_id to frame index (SAM2 uses indices)
    image_id_to_frame_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
    
    return processed_jpg_images_dir, image_ids, image_id_to_frame_idx, video_info

def prepare_video_frames_from_image_paths(video_id: str, video_annotations: Dict[str, List[Dict]]) -> Tuple[Path, List[str], Dict[str, int]]:
    """
    Prepare video information for SAM2 processing using experiment metadata to find image paths.
    This function uses the existing experiment_metadata_utils to find the correct image directory.
    
    Args:
        video_id: Video identifier
        video_annotations: Annotations for this video (image_id -> detections)
        
    Returns:
        Tuple of (video_directory, image_ids, image_id_to_frame_index_mapping)
    """
    if not video_annotations:
        raise ValueError(f"No annotations found for video {video_id}")
    
    # Get all image IDs for this video
    image_ids = list(video_annotations.keys())
    if not image_ids:
        raise ValueError(f"No images found for video_id: {video_id}")
    
    # Sort image IDs to ensure correct temporal order
    # Assuming format: experiment_well_frameXXXX
    image_ids.sort()
    
    # Use experiment_metadata_utils to find the video info and get the processed_jpg_images_dir
    try:
        # Try to load experiment metadata from standard location
        sandbox_root = Path(__file__).parent.parent.parent
        metadata_path = sandbox_root / "data" / "raw_data_organized" / "experiment_metadata.json"
        
        if not metadata_path.exists():
            # Try alternative location
            metadata_path = sandbox_root / "data" / "experiment_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError("Could not find experiment_metadata.json")
        
        # Use the existing function to get video info
        video_info = get_video_info(video_id, metadata_path)
        
        if video_info and "processed_jpg_images_dir" in video_info:
            video_dir = Path(video_info["processed_jpg_images_dir"])
            
            # Verify that the directory exists
            if not video_dir.exists():
                raise FileNotFoundError(f"Processed images directory does not exist: {video_dir}")
            
            # Get image_ids from metadata if available, otherwise use what we have
            metadata_image_ids = video_info.get("image_ids", [])
            if metadata_image_ids:
                # Use metadata image_ids, sorted
                image_ids = sorted(metadata_image_ids)
            
            print(f"   📁 Found video directory from metadata: {video_dir}")
            
        else:
            raise ValueError(f"Video {video_id} not found in experiment metadata")
            
    except (FileNotFoundError, ValueError) as e:
        raise FileNotFoundError(f"Could not find video directory for {video_id} using experiment metadata: {e}")
    
    # Verify that all images exist using the exact approach from experiment_metadata_utils
    missing_images = []
    for image_id in image_ids:
        image_path = video_dir / f"{image_id}.jpg"
        if not image_path.exists():
            missing_images.append(image_id)
    
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images in {video_dir}: {missing_images[:5]}...")
    
    # Create mapping from image_id to frame index (SAM2 uses indices)
    image_id_to_frame_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
    
    print(f"   📄 Found {len(image_ids)} images")
    
    return video_dir, image_ids, image_id_to_frame_idx


def find_seed_frame_from_video_annotations(video_annotations: Dict[str, List[Dict]], video_id: str) -> Tuple[str, Dict]:
    """
    Find the optimal seed frame for a video using just the video annotations (no metadata required).
    
    Args:
        video_annotations: Annotations for this video (image_id -> detections)
        video_id: Video identifier
        
    Returns:
        Tuple of (seed_frame_id, seed_info)
    """
    if not video_annotations:
        raise ValueError(f"No annotations found for video {video_id}")
    
    # Get all image IDs and sort them
    all_image_ids = list(video_annotations.keys())
    all_image_ids.sort()
    
    if not all_image_ids:
        raise ValueError(f"No image_ids found for video {video_id}")
    
    # Consider first 20% of frames
    first_20_percent = max(1, len(all_image_ids) // 5)
    early_frames = all_image_ids[:first_20_percent]
    
    # Count detections in early frames
    detection_counts = []
    frame_detection_info = {}
    
    for image_id in early_frames:
        detections = video_annotations.get(image_id, [])
        count = len(detections)
        
        detection_counts.append(count)
        frame_detection_info[image_id] = {
            'count': count,
            'detections': detections
        }
    
    if not detection_counts or max(detection_counts) == 0:
        raise ValueError(f"No detections found in early frames for {video_id}")
    
    # Find mode of detection counts
    count_freq = Counter(detection_counts)
    mode_count = count_freq.most_common(1)[0][0]
    
    # Find earliest frame with mode count
    seed_frame = None
    for image_id in early_frames:
        if frame_detection_info[image_id]['count'] == mode_count and mode_count > 0:
            seed_frame = image_id
            break
    
    if not seed_frame:
        raise ValueError(f"No suitable seed frame found for {video_id}")
    
    seed_info = {
        'video_id': video_id,
        'seed_frame': seed_frame,
        'num_embryos': mode_count,
        'detections': frame_detection_info[seed_frame]['detections'],
        'is_first_frame': seed_frame == all_image_ids[0],
        'all_frames': all_image_ids,
        'seed_frame_index': all_image_ids.index(seed_frame)
    }
    
    return seed_frame, seed_info

def assign_embryo_ids(video_id: str, num_embryos: int) -> List[str]:
    """
    Generate unique embryo IDs for a video.
    
    Args:
        video_id: Video identifier
        num_embryos: Number of embryos in the video
        
    Returns:
        List of embryo IDs
    """
    embryo_ids = []
    for i in range(num_embryos):
        # Always use 2-digit formatting for consistency: e01, e02, ..., e10, e11, ...
        embryo_id = f"{video_id}_e{i+1:02d}"
        embryo_ids.append(embryo_id)
    return embryo_ids

