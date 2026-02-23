#!/usr/bin/env python3
"""
GroundingDINO Detection Utilities for Module 2
==============================================

This module provides GroundingDINO detection with integration to the modular pipeline:
- Uses shared parsing utilities and entity tracking from Module 0/1
- Integrates with ExperimentMetadata for efficient image discovery
- Maintains backward compatibility with existing annotation format
- Uses BaseFileHandler for atomic saves and entity validation

Key Integration Points:
- parsing_utils: For consistent ID parsing and validation
- EntityIDTracker: For hierarchy validation on save
- ExperimentMetadata: For image path resolution and filtering
- BaseFileHandler: For atomic JSON operations

ANNOTATION FILE FORMAT
======================
{
  "file_info": {
    "creation_time": "2025-08-04T12:00:00.000000",
    "last_updated": "2025-08-04T12:30:00.000000"
  },
  "images": {
    "20250612_30hpf_ctrl_atf6_A01_t0000": {
      "annotations": [
        {
          "annotation_id": "ann_20250804120000123456",
          "prompt": "individual embryo",
          "model_metadata": {
            "model_config_path": "GroundingDINO_SwinT_OGC.py",
            "model_weights_path": "groundingdino_swint_ogc.pth",
            "loading_timestamp": "2025-08-04T12:00:00.000000",
            "model_architecture": "GroundedDINO"
          },
          "inference_params": {
            "box_threshold": 0.35,
            "text_threshold": 0.25
          },
          "timestamp": "2025-08-04T12:00:00.000000",
          "num_detections": 2,
          "detections": [
            {
              "box_xyxy": [0.3, 0.1, 0.7, 0.5],
              "confidence": 0.85,
              "phrase": "individual embryo"
            }
          ]
        }
      ]
    }
  },
  "high_quality_annotations": {
    "20250612_30hpf_ctrl_atf6": {
      "prompt": "individual embryo",
      "confidence_threshold": 0.5,
      "iou_threshold": 0.5,
      "timestamp": "2025-08-04T12:00:00.000000",
      "filtered": {
        "20250612_30hpf_ctrl_atf6_A01_t0000": [
          {
            "box_xyxy": [0.3, 0.1, 0.7, 0.5],
            "confidence": 0.85,
            "phrase": "individual embryo"
          }
        ]
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
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure the project root is in the path
SANDBOX_ROOT = Path(__file__).parent.parent.parent
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.append(str(SANDBOX_ROOT))

# Import from our modular utilities
from scripts.utils.parsing_utils import parse_entity_id, validate_id_format, get_entity_type, extract_experiment_id
from scripts.utils.entity_id_tracker import EntityIDTracker
from scripts.utils.base_file_handler import BaseFileHandler
from scripts.metadata.experiment_metadata import ExperimentMetadata


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load pipeline configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_groundingdino_model(config: Dict, device: str = "cuda") -> "torch.nn.Module":
    """Load GroundingDINO model using paths from the pipeline configuration."""
    groundingdino_path = SANDBOX_ROOT / "models" / "GroundingDINO"
    if str(groundingdino_path) not in sys.path:
        sys.path.insert(0, str(groundingdino_path))
    
    try:
        from groundingdino.util.inference import load_model
        import torch
        import argparse
        
        model_config_path = SANDBOX_ROOT / config["models"]["groundingdino"]["config"]
        
        # Handle weights path - if absolute, use as-is; if relative, make relative to SANDBOX_ROOT
        weights_path = config["models"]["groundingdino"]["weights"]
        if Path(weights_path).is_absolute():
            model_weights_path = Path(weights_path)
        else:
            model_weights_path = SANDBOX_ROOT / weights_path
        
        # Load model with standard GroundingDINO interface
        model = load_model(str(model_config_path), str(model_weights_path), device=device)
        print(f"GroundedDINO model loaded successfully on {device}.")
        
        # Store model metadata for annotations
        model._annotation_metadata = {
            "model_config_path": str(model_config_path),
            "model_weights_path": str(model_weights_path), 
            "loading_timestamp": datetime.now().isoformat(),
            "model_architecture": "GroundedDINO"
        }
        
        return model
    except ImportError as e:
        raise ImportError(f"Failed to import GroundingDINO. Make sure it's installed: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load GroundingDINO model: {e}")


def get_model_metadata(model) -> Dict:
    """Extract model metadata for annotation storage."""
    if hasattr(model, '_annotation_metadata'):
        base_metadata = model._annotation_metadata.copy()
    else:
        base_metadata = {
            "model_config_path": "unknown",
            "model_weights_path": "unknown",
            "loading_timestamp": datetime.now().isoformat(),
            "model_architecture": "GroundedDINO"
        }
    
    return {
        "model_config_path": Path(base_metadata.get("model_config_path", "unknown")).name,
        "model_weights_path": Path(base_metadata.get("model_weights_path", "unknown")).name,
        "loading_timestamp": base_metadata.get("loading_timestamp", datetime.now().isoformat()),
        "model_architecture": base_metadata.get("model_architecture", "GroundedDINO")
    }


def calculate_detection_iou(box1_xyxy: List[float], box2_xyxy: List[float]) -> float:
    """Calculate IoU between two bounding boxes in xyxy format (normalized coordinates)."""
    
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# ========== UTILITY FUNCTIONS ==========

def calculate_detection_iou(box1_xyxy: List[float], box2_xyxy: List[float]) -> float:
    """Calculate IoU between two bounding boxes in xyxy format (normalized coordinates)."""
    
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


class GroundedDinoAnnotations(BaseFileHandler):
    """
    GroundingDINO annotation manager with experiment metadata integration.
    
    Provides annotation storage, retrieval, batch processing, and high-quality filtering.
    Integrates with Module 0/1 utilities for consistent entity tracking and validation.
    """

    def __init__(self, filepath: Union[str, Path], verbose: bool = True, 
                 metadata_path: Optional[Union[str, Path]] = None):
        """Initialize the annotation manager."""
        # Initialize BaseFileHandler
        super().__init__(filepath, verbose=verbose)
        
        # Load or initialize the annotation data
        self.annotations = self._load_or_initialize()
        
        # Set metadata path and load if available
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self._metadata_manager = None
        if self.metadata_path and self.metadata_path.exists():
            self._metadata_manager = ExperimentMetadata(self.metadata_path)
            if self.verbose:
                all_images = self._metadata_manager.list_images()
                print(f"📂 Loaded experiment metadata: {len(all_images)} total images")

    def _load_or_initialize(self) -> Dict:
        """Load existing annotations or create new structure."""
        if self.filepath.exists():
            return self.load_json()
        else:
            return self._get_initial_data()

    def _get_initial_data(self) -> Dict:
        """Initialize the data structure for new files."""
        return {
            "file_info": {
                "creation_time": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            },
            "images": {}
        }

    def set_metadata_path(self, metadata_path: Union[str, Path]):
        """Set or update the experiment metadata path."""
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.exists():
            self._metadata_manager = ExperimentMetadata(self.metadata_path)
            if self.verbose:
                all_images = self._metadata_manager.list_images()
                print(f"📂 Updated metadata: {len(all_images)} total images")
        else:
            if self.verbose:
                print(f"⚠️  Metadata file not found: {metadata_path}")
            self._metadata_manager = None

    def get_all_metadata_image_ids(self) -> List[str]:
        """Get all image IDs from experiment metadata."""
        if not self._metadata_manager:
            return []
        return self._metadata_manager.list_images()

    def get_annotated_image_ids(self, prompt: Optional[str] = None, 
                              model_metadata: Optional[Dict] = None,
                              consider_different_if_different_weights: bool = False) -> List[str]:
        """Get image IDs that already have annotations."""
        annotated_ids = []
        for image_id, image_data in self.annotations.get("images", {}).items():
            annotations = image_data.get("annotations", [])
            if prompt:
                for ann in annotations:
                    if ann.get("prompt") == prompt:
                        # If considering different weights, check model metadata
                        if consider_different_if_different_weights and model_metadata:
                            ann_model_metadata = ann.get("model_metadata", {})
                            current_weights = Path(model_metadata.get("model_weights_path", "")).name
                            ann_weights = Path(ann_model_metadata.get("model_weights_path", "")).name
                            
                            # Only consider annotated if weights match
                            if current_weights == ann_weights:
                                annotated_ids.append(image_id)
                                break
                        else:
                            # Standard case - any annotation with this prompt counts
                            annotated_ids.append(image_id)
                            break
            else:
                if len(annotations) > 0:
                    annotated_ids.append(image_id)
        return annotated_ids

    def get_missing_annotations(self, prompts: List[str], 
                              experiment_ids: Optional[List[str]] = None,
                              video_ids: Optional[List[str]] = None,
                              image_ids: Optional[List[str]] = None,
                              model_metadata: Optional[Dict] = None,
                              consider_different_if_different_weights: bool = False) -> Dict[str, List[str]]:
        """Find images that are missing annotations for given prompts."""
        if not self._metadata_manager:
            if self.verbose:
                print("❌ No metadata loaded. Cannot find missing annotations.")
            return {prompt: [] for prompt in prompts}

        target_image_ids = self._get_filtered_image_ids(experiment_ids, video_ids, image_ids)
        
        missing_by_prompt = {}
        for prompt in prompts:
            annotated_for_prompt = set(self.get_annotated_image_ids(
                prompt, 
                model_metadata=model_metadata,
                consider_different_if_different_weights=consider_different_if_different_weights
            ))
            missing_for_prompt = [img_id for img_id in target_image_ids 
                                if img_id not in annotated_for_prompt]
            missing_by_prompt[prompt] = missing_for_prompt
            
            if self.verbose:
                weights_info = ""
                if consider_different_if_different_weights and model_metadata:
                    weights_name = Path(model_metadata.get("model_weights_path", "unknown")).name
                    weights_info = f" (for {weights_name})"
                print(f"📊 Prompt '{prompt}'{weights_info}: {len(missing_for_prompt)} missing, {len(annotated_for_prompt)} annotated")
        
        return missing_by_prompt

    def _get_filtered_image_ids(self, experiment_ids: Optional[List[str]] = None,
                               video_ids: Optional[List[str]] = None, 
                               image_ids: Optional[List[str]] = None) -> List[str]:
        """Get filtered list of image IDs based on criteria."""
        if not self._metadata_manager:
            return []
        
        if image_ids:
            all_metadata_ids = set(self.get_all_metadata_image_ids())
            return [img_id for img_id in image_ids if img_id in all_metadata_ids]
        
        if video_ids:
            # Get images for specific videos
            filtered_images = []
            for video_id in video_ids:
                # Find experiment for this video and get images
                for exp_id in self._metadata_manager.list_experiments():
                    if video_id in self._metadata_manager.list_videos(exp_id):
                        filtered_images.extend(self._metadata_manager.list_images(exp_id, video_id))
                        break
            return filtered_images
        
        if experiment_ids:
            # Get images for specific experiments
            filtered_images = []
            for exp_id in experiment_ids:
                filtered_images.extend(self._metadata_manager.list_images(exp_id))
            return filtered_images
        
        return self.get_all_metadata_image_ids()

    def get_images_for_detection(self, experiment_ids: Optional[List[str]] = None,
                               video_ids: Optional[List[str]] = None,
                               image_ids: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
        """
        Get (image_id, image_path) tuples for detection processing.
        
        Uses ExperimentMetadata for efficient path resolution.
        
        Returns:
            List of (image_id, image_path) tuples ready for detection
        """
        if not self._metadata_manager:
            if self.verbose:
                print("❌ No metadata loaded. Cannot get images for detection.")
            return []
        
        # Get image data using experiment filtering  
        if experiment_ids:
            image_data_list = self._metadata_manager.get_images_for_detection(experiment_ids=experiment_ids)
        else:
            # Get all images if no specific experiments requested
            image_data_list = self._metadata_manager.get_images_for_detection()
        
        # Convert to (image_id, path) tuples and filter if needed
        results = []
        for img_data in image_data_list:
            img_id = img_data['image_id']
            img_path = Path(img_data['image_path'])
            
            # Filter by video_ids if specified
            if video_ids and img_data.get('video_id') not in video_ids:
                continue
                
            # Filter by image_ids if specified
            if image_ids and img_id not in image_ids:
                continue
                
            results.append((img_id, img_path))
        
        return results

    def process_missing_annotations(
        self, 
        model, 
        prompts: Union[str, List[str]], 
        # Parameters for get_missing_annotations
        experiment_ids: Optional[List[str]] = None,
        video_ids: Optional[List[str]] = None,
        image_ids: Optional[List[str]] = None,
        # Local parameters (not passed to inference)
        consider_different_if_different_weights: bool = False,
        # Parameters for gdino_inference_with_visualization
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        show_anno: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
        text_size: float = 1.0,
        auto_save_interval: Optional[int] = None,
        overwrite: bool = False,
        store_image_source: bool = True
    ):
        """
        Process missing annotations by running inference on unprocessed images.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if not self._metadata_manager:
            if self.verbose:
                print("❌ No metadata loaded.")
            return {}

        # Get model metadata for weight comparison
        model_metadata = get_model_metadata(model)
        
        if self.verbose and consider_different_if_different_weights:
            weights_name = Path(model_metadata.get("model_weights_path", "unknown")).name
            print(f"🔧 Model-specific mode: Only considering annotations from {weights_name}")

        # Get missing annotations with filtering parameters
        missing_by_prompt = self.get_missing_annotations(
            prompts, 
            experiment_ids=experiment_ids,
            video_ids=video_ids, 
            image_ids=image_ids,
            model_metadata=model_metadata,
            consider_different_if_different_weights=consider_different_if_different_weights
        )
        total_missing = sum(len(img_ids) for img_ids in missing_by_prompt.values())
        
        if total_missing == 0:
            if self.verbose:
                print("✅ No missing annotations found!")
            return {}
        
        if self.verbose:
            print(f"📊 Found {total_missing} missing annotations to process")
        
        all_results = {}
        for prompt in prompts:
            missing_image_ids = missing_by_prompt[prompt]
            if len(missing_image_ids) == 0:
                continue
            
            if self.verbose:
                print(f"\n🔄 Processing {len(missing_image_ids)} images for prompt '{prompt}'...")
            
            try:
                # Get image paths and IDs using metadata manager
                image_path_tuples = self.get_images_for_detection(image_ids=missing_image_ids)
                
                if self.verbose:
                    print(f"📁 Found paths for {len(image_path_tuples)} images")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error getting image paths: {e}")
                continue
            
            # Create inference params dict for annotations
            inference_params = {
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
            }
            
            results = gdino_inference_with_visualization(
                model=model,
                images=image_path_tuples,  # Pass tuples instead of just paths
                prompts=prompt, 
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                show_anno=show_anno,
                save_dir=save_dir,
                text_size=text_size,
                verbose=self.verbose,
                annotations_manager=self,
                auto_save_interval=auto_save_interval,
                inference_params=inference_params,
                overwrite=overwrite,
                store_image_source=store_image_source
            )
            
            for image_name, image_results in results.items():
                if image_name not in all_results:
                    all_results[image_name] = {}
                all_results[image_name].update(image_results)

        # Add empty annotations for images with zero detections
        for prompt in prompts:
            for img_id in missing_by_prompt[prompt]:
                if img_id not in all_results:
                    # Add empty annotation to main data structure for entity tracking
                    self.add_annotation(
                        image_id=img_id,
                        prompt=prompt,
                        model=model,
                        boxes=np.array([]).reshape(0, 4),  # Empty boxes array
                        logits=np.array([]),  # Empty logits array
                        phrases=[],  # Empty phrases list
                        inference_params=inference_params or {},
                        overwrite=overwrite
                    )

        return all_results

    def _get_image_to_experiment_map(self) -> Dict[str, str]:
        """Create mapping from image_id to experiment_id using metadata or parsing utilities."""
        # If we have metadata manager, use it for efficiency
        if self._metadata_manager:
            image_to_exp = {}
            for image_id in self._metadata_manager.get_all_image_ids():
                try:
                    parsed = parse_entity_id(image_id)
                    experiment_id = parsed['experiment_id']
                    image_to_exp[image_id] = experiment_id
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Could not parse experiment_id from {image_id}: {e}")
                    continue
            return image_to_exp
        
        # Fallback: extract experiment IDs directly from annotation data
        image_to_exp = {}
        for image_id in self.annotations.get("images", {}):
            try:
                experiment_id = extract_experiment_id(image_id)
                image_to_exp[image_id] = experiment_id
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not extract experiment_id from {image_id}: {e}")
                continue
        
        return image_to_exp

    def calculate_detection_iou(self, box1_xyxy: List[float], box2_xyxy: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        return calculate_detection_iou(box1_xyxy, box2_xyxy)

    def add_annotation(self, image_id: str, prompt: str, model, 
                      boxes: np.ndarray, logits: np.ndarray, phrases: List[str],
                      inference_params: Optional[Dict] = None, image_source: Optional[np.ndarray] = None,
                      overwrite: bool = False):
        """Add a single annotation."""
        # Validate image_id using parsing utilities
        try:
            entity_type = get_entity_type(image_id)
            if entity_type != "image":
                if self.verbose:
                    print(f"⚠️  Expected image ID but got {entity_type}: {image_id}")
            # Also try to parse it to catch format issues
            parse_entity_id(image_id)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  ID validation warning for {image_id}: {e}")
            # Continue processing but log the warning
        
        model_metadata = get_model_metadata(model)
        
        # Ensure numpy arrays
        if hasattr(boxes, 'cpu'):
            boxes = boxes.cpu().numpy()
        elif not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
            
        if hasattr(logits, 'cpu'):
            logits = logits.cpu().numpy()
        elif not isinstance(logits, np.ndarray):
            logits = np.array(logits)
        
        # Initialize image entry if needed
        if image_id not in self.annotations["images"]:
            self.annotations["images"][image_id] = {"annotations": []}

        # Check for existing annotations with same prompt
        existing_annotations = self.annotations["images"][image_id]["annotations"]
        updated_annotations = []
        annotation_replaced = False
        
        for ann in existing_annotations:
            if ann.get("prompt") == prompt:
                if overwrite:
                    annotation_replaced = True
                    if self.verbose:
                        print(f"🔄 Replacing existing annotation for {image_id}, prompt '{prompt}'")
                    continue  # Skip this annotation (effectively replacing it)
                else:
                    updated_annotations.append(ann)  # Keep existing
            else:
                updated_annotations.append(ann)  # Keep different prompts

        # Create new annotation
        new_annotation = {
            "annotation_id": f"ann_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "prompt": prompt,
            "model_metadata": model_metadata,
            "inference_params": inference_params or {},
            "timestamp": datetime.now().isoformat(),
            "num_detections": int(len(boxes)),
            "detections": [
                {
                    "box_xyxy": box.tolist(),
                    "confidence": float(logit),
                    "phrase": str(phrase)
                } for box, logit, phrase in zip(boxes, logits, phrases)
            ]
        }
        
        updated_annotations.append(new_annotation)
        self.annotations["images"][image_id]["annotations"] = updated_annotations
        self._mark_unsaved()
    
        if self.verbose and not annotation_replaced:
            print(f"➕ Added annotation for {image_id}: {len(boxes)} detections (prompt: '{prompt}')")

    def _mark_unsaved(self):
        """Mark that there are unsaved changes."""
        self._unsaved_changes = True

    @property
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return getattr(self, '_unsaved_changes', False)

    def save(self):
        """Save annotations with entity validation and entity tracking."""
        # Update timestamp
        self.annotations["file_info"]["last_updated"] = datetime.now().isoformat()
        
        # Extract and validate entities before saving
        try:
            entities = EntityIDTracker.extract_entities(self.annotations)
            validation_result = EntityIDTracker.validate_hierarchy(entities)
            if validation_result["valid"]:
                if self.verbose:
                    print(f"✅ Entity validation passed: {len(entities.get('images', []))} images")
            else:
                if self.verbose:
                    print(f"⚠️  Entity validation warnings: {len(validation_result['violations'])} issues")
                    for violation in validation_result['violations'][:3]:  # Show first 3
                        print(f"   - {violation}")
        except Exception as e:
            print(f"⚠️  Entity validation error: {e}")
            # Continue with save but log the error
        
        # Add/update entity tracker in the data
        self.annotations = EntityIDTracker.update_entity_tracker(
            self.annotations, 
            pipeline_step="module_2_detection"
        )
        
        # Save using BaseFileHandler methods
        self.save_json(self.annotations)
        self._unsaved_changes = False

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total_images = len(self.annotations["images"])
        total_annotations = sum(len(img_data["annotations"]) for img_data in self.annotations["images"].values())
        total_detections = sum(
            sum(ann["num_detections"] for ann in img_data["annotations"])
            for img_data in self.annotations["images"].values()
        )
        
        summary = {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "total_detections": total_detections,
            "avg_annotations_per_image": total_annotations / max(total_images, 1),
            "avg_detections_per_annotation": total_detections / max(total_annotations, 1),
            "metadata_loaded": self._metadata_manager is not None
        }
        
        if self._metadata_manager:
            total_metadata_images = len(self.get_all_metadata_image_ids())
            summary["metadata_images"] = total_metadata_images
            summary["annotation_coverage"] = total_images / max(total_metadata_images, 1)
        
        # Add high-quality annotation stats
        hq_annotations = self.annotations.get("high_quality_annotations", {})
        if hq_annotations:
            summary["high_quality_experiments"] = len(hq_annotations)
            summary["high_quality_images"] = sum(
                len(exp_data.get("filtered", {})) for exp_data in hq_annotations.values()
            )
        
        return summary

    def print_summary(self):
        """Print a formatted summary."""
        summary = self.get_summary()
        print(f"\n📊 ANNOTATION SUMMARY")
        print(f"=" * 30)
        print(f"🖼️  Annotated images: {summary['total_images']}")
        print(f"📝 Total annotations: {summary['total_annotations']}")
        print(f"📍 Total detections: {summary['total_detections']}")
        print(f"📊 Avg annotations/image: {summary['avg_annotations_per_image']:.1f}")
        print(f"📊 Avg detections/annotation: {summary['avg_detections_per_annotation']:.1f}")
        
        if summary['metadata_loaded']:
            print(f"📂 Metadata images: {summary.get('metadata_images', 0)}")
            print(f"📈 Coverage: {summary.get('annotation_coverage', 0):.1%}")
        else:
            print("📂 No metadata loaded")
        
        if "high_quality_experiments" in summary:
            print(f"🎯 High-quality: {summary['high_quality_experiments']} experiments, {summary['high_quality_images']} images")

    def generate_high_quality_annotations(self,
                                        image_ids: List[str],
                                        prompt: str = "individual embryo",
                                        confidence_threshold: float = 0.5,
                                        iou_threshold: float = 0.5,
                                        overwrite: bool = False,
                                        save_to_self: bool = True) -> Dict:
        """Generate high-quality annotations by filtering existing annotations."""
        if self.verbose:
            print(f"🎯 Generating high-quality annotations for {len(image_ids)} images")
            print(f"   Prompt: '{prompt}', Confidence: {confidence_threshold}, IoU: {iou_threshold}")
        
        image_to_exp = self._get_image_to_experiment_map()
        
        # Step 1: Collect raw detections
        raw_detections = []
        # Handle both list and dictionary formats for image_ids
        image_ids_list = sorted(image_ids.keys()) if isinstance(image_ids, dict) else image_ids
        for image_id in image_ids_list:
            if image_id in self.annotations.get("images", {}):
                annotations = self.annotations["images"][image_id].get("annotations", [])
                for annotation in annotations:
                    if annotation.get("prompt") == prompt:
                        for detection in annotation.get("detections", []):
                            raw_detections.append({
                                'image_id': image_id,
                                'experiment_id': image_to_exp.get(image_id),
                                'box_xyxy': detection['box_xyxy'],
                                'confidence': detection['confidence'],
                                'phrase': detection['phrase']
                            })
        
        if self.verbose:
            print(f"   Found {len(raw_detections)} raw detections")
            
            # Report confidence statistics before filtering
            if raw_detections:
                confidences = [det['confidence'] for det in raw_detections]
                import numpy as np
                mean_conf = np.mean(confidences)
                median_conf = np.median(confidences)
                
                print("   📈 Confidence Statistics (before filtering):")
                print(f"      Total detections: {len(confidences)}")
                print(f"      Mean: {mean_conf:.3f}")
                print(f"      Median: {median_conf:.3f}")
                print(f"      Min: {np.min(confidences):.3f}")
                print(f"      Max: {np.max(confidences):.3f}")
                print(f"      Q90: {np.percentile(confidences, 90):.3f}")

        # Step 2: Filter by confidence
        high_conf_detections = [
            det for det in raw_detections 
            if det['confidence'] >= confidence_threshold
        ]
        
        removed_by_confidence = len(raw_detections) - len(high_conf_detections)
        if self.verbose:
            print(f"   After confidence filter: {len(high_conf_detections)} ({removed_by_confidence} removed)")
        
        # Step 3: Filter by IoU (Non-Maximum Suppression within each image)
        detections_by_image = defaultdict(list)
        for det in high_conf_detections:
            detections_by_image[det['image_id']].append(det)
        
        filtered_detections = []
        total_iou_removed = 0
        
        for image_id, detections in detections_by_image.items():
            if len(detections) <= 1:
                filtered_detections.extend(detections)
                continue
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            keep_detections = []
            for det in detections:
                # Check IoU against all already kept detections
                should_keep = True
                for kept_det in keep_detections:
                    iou = calculate_detection_iou(det['box_xyxy'], kept_det['box_xyxy'])
                    if iou >= iou_threshold:
                        should_keep = False
                        total_iou_removed += 1
                        break
                
                if should_keep:
                    keep_detections.append(det)
            
            filtered_detections.extend(keep_detections)
        
        if self.verbose:
            print(f"   After IoU filtering: {len(filtered_detections)} ({total_iou_removed} removed)")
        
        # Step 4: Group by experiment
        results_by_experiment = defaultdict(lambda: {
            "prompt": prompt,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "timestamp": datetime.now().isoformat(),
            "filtered": {}
        })
        
        for det in filtered_detections:
            exp_id = det['experiment_id']
            if exp_id:  # Only include if we can determine experiment
                image_id = det['image_id']
                if image_id not in results_by_experiment[exp_id]["filtered"]:
                    results_by_experiment[exp_id]["filtered"][image_id] = []
                
                # Add only the detection data (not metadata)
                results_by_experiment[exp_id]["filtered"][image_id].append({
                    "box_xyxy": det['box_xyxy'],
                    "confidence": det['confidence'],
                    "phrase": det['phrase']
                })
        
        results = dict(results_by_experiment)
        
        # Step 5: Save to self if requested
        if save_to_self:
            if "high_quality_annotations" not in self.annotations:
                self.annotations["high_quality_annotations"] = {}
            
            for exp_id, exp_data in results.items():
                if overwrite or exp_id not in self.annotations["high_quality_annotations"]:
                    self.annotations["high_quality_annotations"][exp_id] = exp_data
                    self._mark_unsaved()
        
        # Step 6: Return results with statistics
        total_images = sum(len(exp_data["filtered"]) for exp_data in results.values())
        total_final_detections = sum(
            sum(len(dets) for dets in exp_data["filtered"].values()) 
            for exp_data in results.values()
        )
        
        summary = {
            "filtered": results,
            "statistics": {
                "original_detections": len(raw_detections),
                "confidence_removed": removed_by_confidence,
                "iou_removed": total_iou_removed,
                "final_detections": total_final_detections,
                "final_images": total_images,
                "retention_rate": total_final_detections / len(raw_detections) if len(raw_detections) > 0 else 0,
                "experiments_processed": len(results)
            }
        }
        
        if self.verbose:
            stats = summary["statistics"]
            print(f"   📊 Final Statistics:")
            print(f"      Original: {stats['original_detections']} detections")
            print(f"      Final: {stats['final_detections']} detections ({stats['retention_rate']:.1%} retained)")
            print(f"      Images: {stats['final_images']}")
            print(f"      Experiments: {stats['experiments_processed']}")
        
        return summary

    def get_or_generate_high_quality_annotations(self,
                                               image_ids: List[str],
                                               prompt: str = "individual embryo",
                                               confidence_threshold: float = 0.5,
                                               iou_threshold: float = 0.5,
                                               save_to_self: bool = False) -> Dict:
        """Get high-quality annotations, generating them if they don't exist."""
        result = {}
        existing_hq = self.annotations.get("high_quality_annotations", {})
        
        # Check existing high-quality annotations
        for exp_id, content in existing_hq.items():
            if (content.get("prompt") == prompt and 
                content.get("confidence_threshold") == confidence_threshold and
                content.get("iou_threshold") == iou_threshold):
                # Add existing high-quality annotations for these parameters
                for img_id, detections in content.get("filtered", {}).items():
                    if img_id in image_ids:
                        result[img_id] = detections
        
        # Find missing image_ids
        missing_image_ids = [img_id for img_id in image_ids if img_id not in result]
        
        if missing_image_ids:
            if self.verbose:
                print(f"🔄 Generating high-quality annotations for {len(missing_image_ids)} missing images")
            
            # Generate for missing images
            generated = self.generate_high_quality_annotations(
                missing_image_ids, prompt, confidence_threshold, iou_threshold, 
                save_to_self=save_to_self
            )
            
            # Add generated results
            for exp_data in generated["filtered"].values():
                for img_id, detections in exp_data["filtered"].items():
                    result[img_id] = detections
        
        return result

    def generate_missing_high_quality_annotations(self,
                                                prompt: str = "individual embryo",
                                                confidence_threshold: float = 0.5,
                                                iou_threshold: float = 0.5) -> None:
        """Generate high-quality annotations for all unprocessed images."""
        if self.verbose:
            print(f"🔍 Checking for missing high-quality annotations...")
        
        # Get all processed image IDs
        processed_ids = set()
        for content in self.annotations.get("high_quality_annotations", {}).values():
            if (content.get("prompt") == prompt and 
                content.get("confidence_threshold") == confidence_threshold and
                content.get("iou_threshold") == iou_threshold):
                processed_ids.update(content.get("filtered", {}).keys())
        
        # Get all available image IDs
        all_image_ids = list(self.annotations.get("images", {}).keys())
        unprocessed = [img_id for img_id in all_image_ids if img_id not in processed_ids]
        
        if self.verbose:
            print(f"   Found {len(unprocessed)} unprocessed images out of {len(all_image_ids)} total")
        
        if unprocessed:
            self.generate_high_quality_annotations(
                unprocessed, prompt, confidence_threshold, iou_threshold, save_to_self=True
            )
        else:
            if self.verbose:
                print("   All images already have high-quality annotations")

    def export_high_quality_annotations(self, export_path: Union[str, Path]) -> None:
        """Export high-quality annotations to a JSON file."""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        hq_data = self.annotations.get("high_quality_annotations", {})
        
        with open(export_path, 'w') as f:
            json.dump(hq_data, f, indent=2)
        
        if self.verbose:
            total_experiments = len(hq_data)
            total_images = sum(len(exp_data.get("filtered", {})) for exp_data in hq_data.values())
            print(f"📤 Exported high-quality annotations: {total_experiments} experiments, {total_images} images → {export_path}")

    def import_high_quality_annotations(self, import_path: Union[str, Path], overwrite: bool = False) -> None:
        """Import high-quality annotations from a JSON file."""
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
        
        with open(import_path, 'r') as f:
            imported = json.load(f)
        
        if "high_quality_annotations" not in self.annotations:
            self.annotations["high_quality_annotations"] = {}
        
        imported_count = 0
        skipped_count = 0
        
        for exp_id, content in imported.items():
            if exp_id not in self.annotations["high_quality_annotations"] or overwrite:
                self.annotations["high_quality_annotations"][exp_id] = content
                imported_count += 1
                self._mark_unsaved()
            else:
                skipped_count += 1
        
        if imported_count > 0:
            if self.verbose:
                print(f"📥 Imported {imported_count} high-quality annotation sets, skipped {skipped_count}")
        
        if self.verbose:
            print(f"📊 Total high-quality experiments: {len(self.annotations['high_quality_annotations'])}")

    def has_high_quality(self, image_id: str, prompt: str = "individual embryo") -> bool:
        """Check if an image has high-quality annotations for a given prompt."""
        for exp_data in self.annotations.get("high_quality_annotations", {}).values():
            if exp_data.get("prompt") == prompt:
                if image_id in exp_data.get("filtered", {}):
                    return True
        return False


def run_inference(model, image_path: Union[str, Path], text_prompt: str, 
                  box_threshold: float = 0.35, text_threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Run GroundingDINO inference and return results with boxes in xyxy format."""
    from groundingdino.util.inference import load_image, predict
    
    image_source, image_tensor = load_image(str(image_path))
    
    with torch.no_grad():
        boxes_tensor, logits_tensor, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
    
    if isinstance(boxes_tensor, torch.Tensor):
        boxes_xywh = boxes_tensor.cpu().numpy()
    else:
        boxes_xywh = np.array(boxes_tensor)
    
    if isinstance(logits_tensor, torch.Tensor):
        logits = logits_tensor.cpu().numpy()
    else:
        logits = np.array(logits_tensor)
    
    # Convert boxes from xywh (center format) to xyxy (corner format)
    boxes_xyxy = []
    for box in boxes_xywh:
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy.append([x1, y1, x2, y2])
    
    boxes = np.array(boxes_xyxy) if boxes_xyxy else np.array([]).reshape(0, 4)
        
    return boxes, logits, phrases, image_source


def visualize_detections(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, 
                         phrases: List[str], title: str = "Detections", save_path: Optional[str] = None,
                         text_size: float = 1.0, show_anno: bool = True, verbose: bool = True):
    """Visualize detections on an image."""
    annotated_frame = image_source.copy()
    h, w, _ = annotated_frame.shape
    
    colors = [
        (0, 100, 200), (0, 80, 160), (20, 120, 20), (150, 50, 0),
        (120, 20, 120), (0, 140, 140), (100, 0, 100), (80, 80, 0)
    ]
    
    font_scale = 2.0 * text_size
    thickness = max(1, int(10 * text_size))
    
    for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = box
        
        # Convert normalized coordinates to pixel coordinates
        x1_px, y1_px = int(x1 * w), int(y1 * h)
        x2_px, y2_px = int(x2 * w), int(y2 * h)
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1_px, y1_px), (x2_px, y2_px), color, thickness)
        
        # Add text
        label = f"{phrase}: {logit:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated_frame, (x1_px, y1_px - label_h - 10), (x1_px + label_w, y1_px), color, -1)
        cv2.putText(annotated_frame, label, (x1_px, y1_px - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        if verbose:
            print(f"💾 Saved visualization to: {save_path}")
    
    if show_anno:
        plt.show()
    else:
        plt.close()


def gdino_inference_with_visualization(
    model, 
    images: Union[str, Path, List[Union[str, Path]]], 
    prompts: Union[str, List[str]],
    box_threshold: float = 0.35, 
    text_threshold: float = 0.25,
    show_anno: bool = True, 
    save_dir: Optional[Union[str, Path]] = None,
    text_size: float = 1.0,
    verbose: bool = True,
    annotations_manager: Optional['GroundedDinoAnnotations'] = None,
    auto_save_interval: Optional[int] = None,
    inference_params: Optional[Dict] = None,
    overwrite: bool = False,
    store_image_source: bool = True
) -> Dict[str, Dict[str, Tuple]]:
    """Unified function for GroundingDINO inference with visualization support."""
    # Normalize inputs - handle both paths and (image_id, path) tuples
    if isinstance(images, (str, Path)):
        image_list = [(Path(images).stem, Path(images))]  # Use filename as ID for single image
    elif isinstance(images, list):
        if len(images) > 0 and isinstance(images[0], tuple):
            # Already (image_id, path) tuples
            image_list = [(img_id, Path(img_path)) for img_id, img_path in images]
        else:
            # List of paths - use filename as ID
            image_list = [(Path(img).stem, Path(img)) for img in images]
    else:
        raise ValueError("Images must be a path, list of paths, or list of (image_id, path) tuples")
    
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = list(prompts)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    total_ops = len(image_list) * len(prompt_list)
    current_op = 0
    processed_images_count = 0
    auto_save_enabled = annotations_manager is not None and auto_save_interval is not None
    
    if verbose:
        print(f"🚀 Starting inference on {len(image_list)} images with {len(prompt_list)} prompts ({total_ops} total operations)")
    
    for img_idx, (image_id, image_path) in enumerate(image_list):
        if not image_path.exists():
            if verbose:
                print(f"⚠️  Image not found: {image_path}")
            continue
        
        # Use the image_id directly from the tuple
        image_results = {}
        
        for prompt in prompt_list:
            current_op += 1
            if verbose:
                print(f"🔄 [{current_op}/{total_ops}] Processing {image_path.name} [image_id: {image_id}] with prompt '{prompt}'...")
            
            try:
                boxes, logits, phrases, image_source = run_inference(
                    model, image_path, prompt, box_threshold, text_threshold
                )
                
                if verbose:
                    print(f"   Found {len(boxes)} detections")
                
                # Store results
                image_results[prompt] = (boxes, logits, phrases, image_source if store_image_source else None)
                
                # Add to annotations manager if provided
                if annotations_manager:
                    annotations_manager.add_annotation(
                        image_id=image_id,
                        prompt=prompt,
                        model=model,
                        boxes=boxes,
                        logits=logits,
                        phrases=phrases,
                        inference_params=inference_params,
                        image_source=image_source if store_image_source else None,
                        overwrite=overwrite
                    )
                
                # Visualization
                if show_anno or save_dir:
                    title = f"{image_path.name} - {prompt} ({len(boxes)} detections)"
                    save_path = None
                    if save_dir:
                        save_path = save_dir / f"{image_path.stem}_{prompt.replace(' ', '_')}.png"
                    
                    visualize_detections(
                        image_source, boxes, logits, phrases,
                        title=title, save_path=save_path, text_size=text_size,
                        show_anno=show_anno, verbose=verbose
                    )
                
            except Exception as e:
                if verbose:
                    print(f"❌ Error processing image_id '{image_id}' ({image_path.name}) with prompt '{prompt}': {e}")
                continue
        
        if image_results:
            results[image_id] = image_results
            processed_images_count += 1
            
            # Auto-save check
            if auto_save_enabled and processed_images_count % auto_save_interval == 0:
                if verbose:
                    print(f"💾 Auto-saving after {processed_images_count} images...")
                annotations_manager.save()
    
    # Final auto-save if enabled and there are unsaved changes
    if auto_save_enabled and annotations_manager.has_unsaved_changes:
        if verbose:
            print(f"💾 Final save after processing {processed_images_count} images...")
        annotations_manager.save()
    
    if verbose:
        print(f"✅ Inference complete: processed {processed_images_count}/{len(image_list)} images")
    
    return results
