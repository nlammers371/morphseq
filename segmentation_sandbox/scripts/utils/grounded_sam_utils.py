#!/usr/bin/env python3
"""
Enhanced GroundedDINO Utilities: Inference + Annotation Management
=================================================================

This module provides a complete pipeline for GroundedDINO workflows with:
- Standalone and batch inference with visualization
- Annotation management with model metadata tracking
- High-quality annotation filtering with confidence and IoU thresholds
- Experiment metadata integration for batch processing

ANNOTATION FILE FORMAT
======================

The annotations JSON file has this structure:

{
  "file_info": {
    "creation_time": "2024-12-15T14:30:22.123456",
    "last_updated": "2024-12-15T15:45:30.789012"
  },
  "images": {
    "image_001": {
      "annotations": [
        {
          "annotation_id": "ann_20241215143022123456",
          "prompt": "individual embryo",
          "model_metadata": {
            "model_config_path": "GroundingDINO_SwinT_OGC.py",
            "model_weights_path": "groundingdino_swint_ogc.pth",
            "loading_timestamp": "2024-12-15T14:30:22.123456",
            "model_architecture": "GroundedDINO"
          },
          "inference_params": {
            "box_threshold": 0.35,
            "text_threshold": 0.25
          },
          "timestamp": "2024-12-15T14:30:22.789012",
          "num_detections": 2,
          "detections": [
            {
              "box_xywh": [0.5, 0.3, 0.2, 0.4],  // [x_center, y_center, width, height] normalized
              "confidence": 0.85,
              "phrase": "individual embryo"
            },
            {
              "box_xywh": [0.7, 0.6, 0.15, 0.25],
              "confidence": 0.72,
              "phrase": "individual embryo"
            }
          ]
        }
      ]
    }
  },
  "high_quality_annotations": {
    "experiment_20231206": {
      "prompt": "individual embryo",
      "confidence_threshold": 0.5,
      "iou_threshold": 0.5,
      "timestamp": "2024-12-15T14:30:22.123456",
      "filtered": {
        "image_001": [
          {
            "box_xywh": [0.5, 0.3, 0.2, 0.4],
            "confidence": 0.85,
            "phrase": "individual embryo"
          }
        ],
        "image_002": [
          {
            "box_xywh": [0.3, 0.4, 0.18, 0.22],
            "confidence": 0.78,
            "phrase": "individual embryo"
          }
        ]
      }
    }
  }
}

KEY DIFFERENCES:
- Regular annotations: Organized by image_id → annotations[] 
- High-quality annotations: Organized by experiment_id → filtered{image_id → detections[]}
- High-quality annotations store only the detection objects (no annotation metadata)
- High-quality annotations include filtering parameters for reproducibility
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
import cv2
import matplotlib.pyplot as plt
import shutil
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure the project root is in the path
SANDBOX_ROOT = Path(__file__).parent.parent.parent
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.append(str(SANDBOX_ROOT))

# Import from other utils
from scripts.utils.experiment_metadata_utils import get_image_id_paths, load_experiment_metadata

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
        
        model_config_path = SANDBOX_ROOT / config["models"]["groundingdino"]["config"]
        model_weights_path = SANDBOX_ROOT / config["models"]["groundingdino"]["weights"]
        
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

def calculate_detection_iou(box1_xywh: List[float], box2_xywh: List[float]) -> float:
    """Calculate IoU between two bounding boxes in xywh format (normalized coordinates)."""
    def xywh_to_xyxy(box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    box1_xyxy = xywh_to_xyxy(box1_xywh)
    box2_xyxy = xywh_to_xyxy(box2_xywh)
    
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


class GroundedDinoAnnotations:
    """
    GroundingDINO annotation manager with experiment metadata integration.
    
    Provides annotation storage, retrieval, batch processing, and high-quality filtering.
    """

    def __init__(self, filepath: Union[str, Path], verbose: bool = True, 
                 metadata_path: Optional[Union[str, Path]] = None):
        """Initialize the annotation manager."""
        self.filepath = Path(filepath)
        self.verbose = verbose
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.annotations = self._load_or_initialize()
        self._unsaved_changes = False
        
        # Load metadata if provided
        self._metadata = None
        if self.metadata_path and self.metadata_path.exists():
            self._metadata = load_experiment_metadata(self.metadata_path)
            if self.verbose and self._metadata:
                print(f"📂 Loaded experiment metadata: {len(self._metadata.get('image_ids', []))} total images")

    def _load_or_initialize(self) -> Dict:
        """Load existing annotations file or initialize a new one."""
        if not self.filepath.exists():
            if self.verbose:
                print(f"🆕 Initializing new annotations file at: {self.filepath}")
            return {
                "file_info": {
                    "creation_time": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                },
                "images": {}
            }
        
        try:
            if self.verbose:
                print(f"📁 Loading existing annotations from: {self.filepath}")
            
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            if self.verbose:
                print(f"✅ Loaded {len(data.get('images', {}))} images successfully")
            return data
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ JSON corruption detected: {e}")
            
            backup_path = self.filepath.with_suffix('.json.backup')
            shutil.move(self.filepath, backup_path)
            
            if self.verbose:
                print(f"📋 Moved corrupted file to backup: {backup_path.name}")
                print(f"🆕 Starting with fresh annotations")
            
            return {
                "file_info": {
                    "creation_time": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "recovery_note": f"Recovered from corruption on {datetime.now().isoformat()}"
                },
                "images": {}
            }
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Unexpected error: {e}")
            return {
                "file_info": {
                    "creation_time": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                },
                "images": {}
            }

    def save(self):
        """Save annotations to file with atomic write."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.annotations["file_info"]["last_updated"] = datetime.now().isoformat()
        
        temp_path = self.filepath.with_suffix('.json.tmp')
        backup_path = self.filepath.with_suffix('.json.backup')
        
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            
            shutil.move(temp_path, self.filepath)
            
            if backup_path.exists():
                backup_path.unlink()
                if self.verbose:
                    print(f"🗑️  Removed corrupted backup (save successful)")
            
            self._unsaved_changes = False
            if self.verbose:
                print(f"💾 Saved annotations to: {self.filepath}")
                
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            if self.verbose:
                print(f"❌ Failed to save annotations: {e}")
            raise

    def set_metadata_path(self, metadata_path: Union[str, Path]):
        """Set or update the experiment metadata path."""
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.exists():
            self._metadata = load_experiment_metadata(self.metadata_path)
            if self.verbose:
                print(f"📂 Updated metadata: {len(self._metadata.get('image_ids', []))} total images")
        else:
            if self.verbose:
                print(f"⚠️  Metadata file not found: {metadata_path}")
            self._metadata = None

    def get_all_metadata_image_ids(self) -> List[str]:
        """Get all image IDs from experiment metadata."""
        if not self._metadata:
            return []
        return self._metadata.get("image_ids", [])

    def get_annotated_image_ids(self, prompt: Optional[str] = None) -> List[str]:
        """Get image IDs that already have annotations."""
        annotated_ids = []
        for image_id, image_data in self.annotations.get("images", {}).items():
            annotations = image_data.get("annotations", [])
            if prompt:
                for ann in annotations:
                    if ann.get("prompt") == prompt:
                        annotated_ids.append(image_id)
                        break
            else:
                if len(annotations) > 0:
                    annotated_ids.append(image_id)
        return annotated_ids

    def get_missing_annotations(self, prompts: List[str], 
                              experiment_ids: Optional[List[str]] = None,
                              video_ids: Optional[List[str]] = None,
                              image_ids: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Find images that are missing annotations for given prompts."""
        if not self._metadata:
            if self.verbose:
                print("❌ No metadata loaded. Cannot find missing annotations.")
            return {prompt: [] for prompt in prompts}

        target_image_ids = self._get_filtered_image_ids(experiment_ids, video_ids, image_ids)
        
        missing_by_prompt = {}
        for prompt in prompts:
            annotated_for_prompt = set(self.get_annotated_image_ids(prompt))
            missing_for_prompt = [img_id for img_id in target_image_ids 
                                if img_id not in annotated_for_prompt]
            missing_by_prompt[prompt] = missing_for_prompt
            
            if self.verbose:
                print(f"📊 Prompt '{prompt}': {len(missing_for_prompt)} missing, {len(annotated_for_prompt)} annotated")
        
        return missing_by_prompt

    def _get_filtered_image_ids(self, experiment_ids: Optional[List[str]] = None,
                               video_ids: Optional[List[str]] = None, 
                               image_ids: Optional[List[str]] = None) -> List[str]:
        """Get filtered list of image IDs based on criteria."""
        if not self._metadata:
            return []
        
        if image_ids:
            all_metadata_ids = set(self.get_all_metadata_image_ids())
            return [img_id for img_id in image_ids if img_id in all_metadata_ids]
        
        if video_ids:
            target_image_ids = []
            for video_id in video_ids:
                parts = video_id.split('_')
                if len(parts) >= 2:
                    experiment_id = parts[0]
                    exp_data = self._metadata.get("experiments", {}).get(experiment_id, {})
                    video_data = exp_data.get("videos", {}).get(video_id, {})
                    target_image_ids.extend(video_data.get("image_ids", []))
            return target_image_ids
        
        if experiment_ids:
            target_image_ids = []
            for exp_id in experiment_ids:
                exp_data = self._metadata.get("experiments", {}).get(exp_id, {})
                for video_data in exp_data.get("videos", {}).values():
                    target_image_ids.extend(video_data.get("image_ids", []))
            return target_image_ids
        
        return self.get_all_metadata_image_ids()

    def process_missing_annotations(self, model, prompts: Union[str, List[str]], **kwargs):
        """Process missing annotations by running inference on unprocessed images."""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if not self._metadata:
            if self.verbose:
                print("❌ No metadata loaded.")
            return {}
        
        missing_by_prompt = self.get_missing_annotations(prompts, **kwargs)
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
                image_paths = get_image_id_paths(missing_image_ids, self._metadata)
                if self.verbose:
                    print(f"📁 Found paths for {len(image_paths)} images")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error getting image paths: {e}")
                continue
            
            results = gdino_inference_with_visualization(
                model, image_paths, prompt, verbose=self.verbose,
                annotations_manager=self, **kwargs
            )
            
            for image_name, image_results in results.items():
                if image_name not in all_results:
                    all_results[image_name] = {}
                all_results[image_name].update(image_results)
        
        return all_results

    def _get_image_to_experiment_map(self) -> Dict[str, str]:
        """Create mapping from image_id to experiment_id using metadata."""
        if not self._metadata:
            return {}
        
        image_to_exp = {}
        for exp_id, exp_data in self._metadata.get("experiments", {}).items():
            for video_data in exp_data.get("videos", {}).values():
                for image_id in video_data.get("image_ids", []):
                    image_to_exp[image_id] = exp_id
        
        return image_to_exp

    def calculate_detection_iou(self, box1_xywh: List[float], box2_xywh: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        return calculate_detection_iou(box1_xywh, box2_xywh)

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
        for image_id in image_ids:
            if image_id in self.annotations.get("images", {}):
                annotations = self.annotations["images"][image_id].get("annotations", [])
                for annotation in annotations:
                    if annotation.get("prompt") == prompt:
                        for detection in annotation.get("detections", []):
                            raw_detections.append({
                                'image_id': image_id,
                                'detection': detection,
                                'annotation_id': annotation.get('annotation_id'),
                                'confidence': detection.get('confidence', 0),
                                'experiment_id': image_to_exp.get(image_id, 'unknown')
                            })
        
        if self.verbose:
            print(f"   Found {len(raw_detections)} raw detections")
            
            # Report confidence statistics before filtering
            if raw_detections:
                confidences = [det['confidence'] for det in raw_detections]
                mean_conf = np.mean(confidences)
                median_conf = np.median(confidences)
                
                print("   📈 Confidence Statistics (before filtering):")
                print(f"      Total detections: {len(confidences)}")
                print(f"      Mean: {mean_conf:.3f}")
                print(f"      Median: {median_conf:.3f}")
                print(f"      Min: {np.min(confidences):.3f}")
                print(f"      Max: {np.max(confidences):.3f}")
                print(f"      Q90: {np.percentile(confidences, 90):.3f}")
                print(f"      Q95: {np.percentile(confidences, 95):.3f}")
        
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
            
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            keep_detections = []
            for det in detections:
                should_keep = True
                for kept_det in keep_detections:
                    iou = self.calculate_detection_iou(
                        det['detection']['box_xywh'],
                        kept_det['detection']['box_xywh']
                    )
                    if iou > iou_threshold:
                        should_keep = False
                        total_iou_removed += 1
                        break
                
                if should_keep:
                    keep_detections.append(det)
            
            filtered_detections.extend(keep_detections)
        
        if self.verbose:
            print(f"   After IoU filter: {len(filtered_detections)} ({total_iou_removed} removed)")
        
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
            image_id = det['image_id']
            
            if image_id not in results_by_experiment[exp_id]["filtered"]:
                results_by_experiment[exp_id]["filtered"][image_id] = []
            
            results_by_experiment[exp_id]["filtered"][image_id].append(det['detection'])
        
        results = dict(results_by_experiment)
        
        # Step 5: Save to self if requested
        if save_to_self:
            if "high_quality_annotations" not in self.annotations:
                self.annotations["high_quality_annotations"] = {}
            
            for exp_id, exp_data in results.items():
                if not overwrite and exp_id in self.annotations["high_quality_annotations"]:
                    if self.verbose:
                        print(f"   Skipping experiment {exp_id} (already exists, use overwrite=True)")
                    continue
                
                self.annotations["high_quality_annotations"][exp_id] = exp_data
                if self.verbose:
                    total_images = len(exp_data["filtered"])
                    total_dets = sum(len(dets) for dets in exp_data["filtered"].values())
                    print(f"   Saved experiment {exp_id}: {total_images} images, {total_dets} detections")
            
            self._unsaved_changes = True
        
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
            print(f"   📊 Final summary:")
            print(f"      Original: {stats['original_detections']} detections")
            print(f"      Final: {stats['final_detections']} detections")
            print(f"      Retention: {stats['retention_rate']:.1%}")
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
                
                filtered = {
                    img_id: ann for img_id, ann in content.get("filtered", {}).items() 
                    if img_id in image_ids
                }
                result.update(filtered)
        
        # Find missing image_ids
        missing_image_ids = [img_id for img_id in image_ids if img_id not in result]
        
        if missing_image_ids:
            if self.verbose:
                print(f"🔄 Generating high-quality annotations for {len(missing_image_ids)} missing images")
            
            gen_result = self.generate_high_quality_annotations(
                image_ids=missing_image_ids,
                prompt=prompt,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                overwrite=False,
                save_to_self=save_to_self
            )
            
            # Extract the filtered annotations
            for exp_data in gen_result.get("filtered", {}).values():
                for img_id, detections in exp_data.get("filtered", {}).items():
                    result[img_id] = detections
        
        return result

    def generate_missing_high_quality_annotations(self,
                                                prompt: str = "individual embryo",
                                                confidence_threshold: float = 0.5,
                                                iou_threshold: float = 0.5) -> None:
        """Generate high-quality annotations for all unprocessed images."""
        if self.verbose:
            print(f"🔄 Generating missing high-quality annotations")
            print(f"   Prompt: '{prompt}', Thresholds: confidence={confidence_threshold}, iou={iou_threshold}")
        
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
                image_ids=unprocessed,
                prompt=prompt,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                overwrite=True,
                save_to_self=True
            )
            self._unsaved_changes = True
        else:
            if self.verbose:
                print("   ✅ All images already have high-quality annotations")

    def export_high_quality_annotations(self, export_path: Union[str, Path]) -> None:
        """Export high-quality annotations to a JSON file."""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        hq_data = self.annotations.get("high_quality_annotations", {})
        
        with open(export_path, 'w') as f:
            json.dump(hq_data, f, indent=2)
        
        if self.verbose:
            experiments = len(hq_data)
            total_images = sum(len(exp.get("filtered", {})) for exp in hq_data.values())
            print(f"✅ Exported high-quality annotations to: {export_path}")
            print(f"   Experiments: {experiments}, Images: {total_images}")

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
            if not overwrite and exp_id in self.annotations["high_quality_annotations"]:
                if self.verbose:
                    print(f"🔁 Skipped experiment {exp_id} — already exists")
                skipped_count += 1
                continue
            
            self.annotations["high_quality_annotations"][exp_id] = content
            if self.verbose:
                images = len(content.get("filtered", {}))
                print(f"✅ Imported experiment {exp_id}: {images} images")
            imported_count += 1
        
        if imported_count > 0:
            self._unsaved_changes = True
        
        if self.verbose:
            print(f"📊 Import summary: {imported_count} imported, {skipped_count} skipped")

    def has_high_quality(self, image_id: str, prompt: str = "individual embryo") -> bool:
        """Check if an image has high-quality annotations for a given prompt."""
        for exp_data in self.annotations.get("high_quality_annotations", {}).values():
            if exp_data.get("prompt") == prompt and image_id in exp_data.get("filtered", {}):
                return True
        return False

    def add_annotation(self, image_id: str, prompt: str, model, 
                    boxes: np.ndarray, logits: np.ndarray, phrases: List[str],
                    inference_params: Optional[Dict] = None, image_source: Optional[np.ndarray] = None,
                    overwrite: bool = False):
        """Add a single annotation."""
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
                        print(f"🔄 Overwriting existing annotation for '{image_id}' + '{prompt}'")
                    continue
                else:
                    if self.verbose:
                        print(f"⚠️  Skipping duplicate annotation for '{image_id}' + '{prompt}'")
                    return
            else:
                updated_annotations.append(ann)

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
                    "box_xywh": box.tolist(),
                    "confidence": float(logit),
                    "phrase": str(phrase)
                } for box, logit, phrase in zip(boxes, logits, phrases)
            ]
        }
        
        updated_annotations.append(new_annotation)
        self.annotations["images"][image_id]["annotations"] = updated_annotations
        self._unsaved_changes = True
    
        if self.verbose and not annotation_replaced:
            print(f"✅ Added annotation for '{image_id}' + '{prompt}' ({len(boxes)} detections)")

    def add_from_inference_results(self, results: Dict[str, Dict[str, Tuple]], model, 
                                   inference_params: Optional[Dict] = None, overwrite: bool = False):
        """Add annotations from inference results."""
        total_added = 0
        
        for image_name, image_results in results.items():
            for prompt, (boxes, logits, phrases, image_source) in image_results.items():
                if len(boxes) > 0:
                    self.add_annotation(
                        image_name, prompt, model, boxes, logits, phrases, 
                        inference_params, image_source, overwrite=overwrite
                    )
                    total_added += 1
        
        if self.verbose:
            action = "Added/updated" if overwrite else "Added"
            print(f"📊 {action} {total_added} annotations from inference results")

    def get_annotations_for_image(self, image_id: str) -> List[Dict]:
        """Get all annotations for an image."""
        image_data = self.annotations["images"].get(image_id, {})
        return image_data.get("annotations", [])

    def get_all_image_ids(self) -> List[str]:
        """Get all image IDs that have annotations."""
        return list(self.annotations["images"].keys())

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
            "metadata_loaded": self._metadata is not None
        }
        
        if self._metadata:
            metadata_images = len(self.get_all_metadata_image_ids())
            summary.update({
                "metadata_total_images": metadata_images,
                "annotated_vs_metadata": f"{total_images}/{metadata_images}"
            })
        
        # Add high-quality annotation stats
        hq_annotations = self.annotations.get("high_quality_annotations", {})
        if hq_annotations:
            hq_experiments = len(hq_annotations)
            hq_images = sum(len(exp.get("filtered", {})) for exp in hq_annotations.values())
            summary.update({
                "high_quality_experiments": hq_experiments,
                "high_quality_images": hq_images
            })
        
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
            print(f"📂 Metadata integration: ✅ ({summary['annotated_vs_metadata']} coverage)")
        else:
            print(f"📂 Metadata integration: ❌")
        
        if "high_quality_experiments" in summary:
            print(f"⭐ High-quality annotations: {summary['high_quality_experiments']} experiments, {summary['high_quality_images']} images")

    @property
    def has_unsaved_changes(self) -> bool:
        """Check for unsaved changes."""
        return self._unsaved_changes

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        status = "✅ saved" if not self._unsaved_changes else "⚠️ unsaved"
        metadata_status = "📂 metadata" if summary['metadata_loaded'] else "📂 no-metadata"
        hq_status = f", ⭐ {summary.get('high_quality_experiments', 0)} hq-exp" if "high_quality_experiments" in summary else ""
        return f"GroundedDinoAnnotations(images={summary['total_images']}, annotations={summary['total_annotations']}, {status}, {metadata_status}{hq_status})"


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
        if isinstance(box, torch.Tensor):
            box_norm = box.clone()
        else:
            box_norm = torch.tensor(box)
            
        cx, cy, bw, bh = box_norm
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        color = colors[i % len(colors)]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        confidence = float(logit) if isinstance(logit, torch.Tensor) else logit
        label = f"{phrase}: {confidence:.2f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(annotated_frame, 
                      (x1, y1 - text_height - 10), 
                      (x1 + text_width, y1), 
                      color, -1)
        
        cv2.putText(annotated_frame, label, 
                   (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        if verbose:
            print(f"Visualization saved to: {save_path}")
    
    if show_anno:
        plt.show()
    else:
        plt.close()

def run_inference(model, image_path: Union[str, Path], text_prompt: str, 
                  box_threshold: float = 0.35, text_threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Run GroundingDINO inference and return results."""
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
        boxes = boxes_tensor.cpu().numpy()
    else:
        boxes = np.array(boxes_tensor)
    
    if isinstance(logits_tensor, torch.Tensor):
        logits = logits_tensor.cpu().numpy()
    else:
        logits = np.array(logits_tensor)
        
    return boxes, logits, phrases, image_source

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
    # Normalize inputs
    if isinstance(images, (str, Path)):
        image_list = [Path(images)]
    else:
        image_list = [Path(img) for img in images]
    
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = prompts
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    total_ops = len(image_list) * len(prompt_list)
    current_op = 0
    processed_images_count = 0
    auto_save_enabled = annotations_manager is not None and auto_save_interval is not None
    
    if verbose:
        print(f"🔍 Starting inference for {len(image_list)} image(s) × {len(prompt_list)} prompt(s) = {total_ops} operations")
        if auto_save_enabled:
            print(f"💾 Auto-save enabled: saving annotations every {auto_save_interval} processed images")
    
    for img_idx, image_path in enumerate(image_list):
        image_name = image_path.stem
        results[image_name] = {}
        image_has_new_annotations = False
        
        if verbose:
            print(f"\n📸 Processing image [{img_idx+1}/{len(image_list)}]: {image_name}")
        
        for prompt in prompt_list:
            current_op += 1
            if verbose:
                print(f"   🔍 [{current_op}/{total_ops}] Running prompt: '{prompt}'")
            
            try:
                boxes, logits, phrases, image_source = run_inference(
                    model, image_path, prompt, box_threshold, text_threshold
                )
                
                if store_image_source:
                    results[image_name][prompt] = (boxes, logits, phrases, image_source)
                else:
                    results[image_name][prompt] = (boxes, logits, phrases, None)

                if annotations_manager is not None:
                    if len(boxes) > 0:
                        annotations_manager.add_annotation(
                            image_name, prompt, model, boxes, logits, phrases,
                            inference_params or {"box_threshold": box_threshold, "text_threshold": text_threshold},
                            image_source, overwrite=overwrite
                        )
                        image_has_new_annotations = True
                
                if verbose:
                    print(f"      📍 Found {len(boxes)} detections")
                    if len(phrases) > 0:
                        max_conf = float(max(logits)) if len(logits) > 0 else 0.0
                        print(f"      🏆 Max confidence: {max_conf:.3f}")
                
                if show_anno or save_dir:
                    title = f"Image: {image_name} | Prompt: '{prompt}'\n{len(boxes)} detections found"
                    
                    save_path = None
                    if save_dir:
                        safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
                        save_filename = f"{image_name}_{safe_prompt}.jpg"
                        save_path = save_dir / save_filename
                    
                    visualize_detections(
                        image_source, boxes, logits, phrases,
                        title=title,
                        save_path=str(save_path) if save_path else None,
                        text_size=text_size,
                        show_anno=show_anno,
                        verbose=verbose
                    )
                        
            except Exception as e:
                if verbose:
                    print(f"      ❌ Error processing '{prompt}' on {image_name}: {e}")
                results[image_name][prompt] = (
                    np.array([]), np.array([]), [], 
                    np.zeros((100, 100, 3), dtype=np.uint8)
                )
                continue
        
        if image_has_new_annotations:
            processed_images_count += 1
            
            if auto_save_enabled and processed_images_count >= auto_save_interval:
                if verbose:
                    print(f"      💾 Auto-saving annotations...")
                annotations_manager.save()
                processed_images_count = 0
    
    if auto_save_enabled and annotations_manager.has_unsaved_changes:
        if verbose:
            print(f"💾 Final auto-save...")
        annotations_manager.save()
    
    if verbose:
        total_detections = sum(
            sum(len(results[img][prompt][0]) for prompt in results[img])
            for img in results
        )
        print(f"\n✅ Inference complete! {total_detections} total detections")
        
        if save_dir:
            saved_files = len([f for f in save_dir.glob("*.jpg")])
            print(f"💾 Saved {saved_files} annotated images to: {save_dir}")
    
    return results