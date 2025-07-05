#!/usr/bin/env python3
"""
Enhanced GroundedDINO Utilities: Inference + Annotation Management
=================================================================

🎯 DUAL CAPABILITIES
====================
This module provides a complete pipeline for GroundedDINO workflows with:

**🔍 INFERENCE CAPABILITIES:**
- Standalone inference with immediate visualization
- Batch inference across multiple prompts/images  
- Exploration workflows with save/display options
- Smart confidence-based result filtering

**📝 ANNOTATION MANAGEMENT:**
- Individual and batch annotation operations
- Flexible batch building (accumulate vs. direct application)
- Memory-efficient processing with deferred saves
- Integration with inference results


🔧 WHAT'S STORED WITH EACH ANNOTATION
=====================================

Each annotation now includes complete model information for reproducibility:

{
  "annotation_id": "ann_20241215143022123456",
  "prompt": "person car",
  "model_metadata": {
    "model_weights_path": "/path/to/groundingdino_swint_ogc.pth",
    "model_config_path": "/path/to/GroundingDINO_SwinT_OGC.py", 
    "device": "cuda",
    "total_parameters": 463688768,
    "model_size_mb": 1770.1,
    "loading_timestamp": "2024-12-15T14:30:22.123456",
    "model_architecture": "GroundedDINO",
    "config_content": { ... full pipeline config ... }
  },
  "inference_params": {
    "box_threshold": 0.35,
    "text_threshold": 0.25
  },
  "timestamp": "2024-12-15T14:30:22.789012",
  "num_detections": 3,
  "detections": [...]
}

💡 KEY BENEFITS
===============
✅ Full reproducibility - exact model weights and config saved
✅ No more guessing which model version was used
✅ Separate storage of model info vs. inference parameters
✅ You can apply your own filters later using the raw data
✅ Complete audit trail for scientific reproducibility


**Inference Functions:**
- inference_with_visualization(): Single prompt with display/save options
- batch_inference_with_visualization(): Multiple prompts with display/save
- run_inference(): Core inference without visualization
- explore_and_annotate_workflow(): Complete exploration → annotation pipeline

**Annotation Management:**
- GroundedDinoAnnotations: Main annotation manager class
- gen_annotation_batch(): Most flexible batch building (accumulate vs. direct)
- batch_inference_and_annotate(): High-level batch annotation
- add_annotation_batch(): Apply accumulated batches

💡 RECOMMENDED WORKFLOW
=======================
1. **Explore**: Use gdino_inference_with_visualization() to see what gets detected
2. **Review**: Check confidence scores and detection quality  
3. **Annotate**: Use gen_annotation_batch() to selectively add good results
4. **Save**: Call annotations.save() to persist your curated annotations
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
        
        # Store model metadata for annotations - simplified for JSON serialization
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
    """
    Extract model metadata for annotation storage (simplified to avoid JSON serialization issues).
    
    Returns:
        Dictionary containing only model config and weights filenames
    """
    if hasattr(model, '_annotation_metadata'):
        base_metadata = model._annotation_metadata.copy()
    else:
        # Fallback if metadata not available
        base_metadata = {
            "model_config_path": "unknown",
            "model_weights_path": "unknown",
            "loading_timestamp": datetime.now().isoformat(),
            "model_architecture": "GroundedDINO"
        }
    
    # Return only the filenames, not full paths, and ensure JSON serializable
    return {
        "model_config_path": Path(base_metadata.get("model_config_path", "unknown")).name,
        "model_weights_path": Path(base_metadata.get("model_weights_path", "unknown")).name,
        "loading_timestamp": base_metadata.get("loading_timestamp", datetime.now().isoformat()),
        "model_architecture": base_metadata.get("model_architecture", "GroundedDINO")
    }


class GroundedDinoAnnotations:
    """
    Enhanced GroundingDINO annotation manager with experiment metadata integration.
    
    This class provides efficient annotation management with:
    - Simple annotation storage and retrieval
    - Direct visualization from stored annotations (no re-inference needed)
    - Integration with gdino_inference_with_visualization results
    - Experiment metadata integration for batch processing
    - Smart detection of missing annotations
    - Selective processing by experiment, video, or image
    
    Key Features:
    - Store annotations with full model metadata
    - Visualize annotations directly from stored data
    - Batch import from inference results
    - Controlled verbosity for clean workflows
    - Automatic discovery of unprocessed images from metadata
    - Selective processing with experiment/video/image filters
    
    Basic Usage:
        # Store annotations from inference results
        annotations = GroundedDinoAnnotations("annotations.json")
        results = gdino_inference_with_visualization(model, images, prompts)
        annotations.add_from_inference_results(results, model)
        
        # Visualize stored annotations (no re-inference needed!)
        annotations.visualize_annotation("image_001", "person")
        annotations.save()
    
    Metadata Integration:
        # Process all unprocessed images from experiment metadata
        annotations = GroundedDinoAnnotations("annotations.json", metadata_path="experiment_metadata.json")
        annotations.process_missing_annotations(model, ["person", "vehicle"])
        
        # Process specific experiments only
        annotations.process_missing_annotations(model, ["person"], experiment_ids=["20231206", "20231207"])
        
        # Process specific videos only  
        annotations.process_missing_annotations(model, ["person"], video_ids=["20231206_A01", "20231206_A02"])
    """
    # -------------------------------------------------------------------------
    # Initialization and Save (with temporary backup)
    # -------------------------------------------------------------------------

    def __init__(self, filepath: Union[str, Path], verbose: bool = True, 
                 metadata_path: Optional[Union[str, Path]] = None):
        """
        Initialize the annotation manager.
        
        Args:
            filepath: Path to annotations JSON file
            verbose: Whether to print status messages
            metadata_path: Optional path to experiment_metadata.json for integration
        """
        self.filepath = Path(filepath)
        self.verbose = verbose
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.annotations = self._load_or_initialize()
        self._unsaved_changes = False
        
        # Load metadata if provided
        self._metadata = None
        if self.metadata_path and self.metadata_path.exists():
            self._metadata = load_experiment_metadata_safe(self.metadata_path)
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
        
        # Try to load the file
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
            
            # Move corrupted file to backup and start fresh
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

    # Replace your existing save method with this:
    def save(self):
        """Save annotations to file with atomic write and cleanup backup."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.annotations["file_info"]["last_updated"] = datetime.now().isoformat()
        
        # Atomic write: write to temp file first, then rename
        temp_path = self.filepath.with_suffix('.json.tmp')
        backup_path = self.filepath.with_suffix('.json.backup')
        
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            
            # Atomic rename
            shutil.move(temp_path, self.filepath)
            
            # Save successful - remove the corrupted backup since we have a good file now
            if backup_path.exists():
                backup_path.unlink()
                if self.verbose:
                    print(f"🗑️  Removed corrupted backup (save successful)")
            
            self._unsaved_changes = False
            if self.verbose:
                print(f"💾 Saved annotations to: {self.filepath}")
                
        except Exception as e:
            # Clean up temp file if write failed
            if temp_path.exists():
                temp_path.unlink()
            if self.verbose:
                print(f"❌ Failed to save annotations: {e}")
            raise

    # -------------------------------------------------------------------------
    # Experiment Metadata Integration
    # -------------------------------------------------------------------------

    def set_metadata_path(self, metadata_path: Union[str, Path]):
        """Set or update the experiment metadata path."""
        self.metadata_path = Path(metadata_path)
        if self.metadata_path.exists():
            self._metadata = load_experiment_metadata_safe(self.metadata_path)
            if self.verbose:
                print(f"📂 Updated metadata: {len(self._metadata.get('image_ids', []))} total images")
        else:
            if self.verbose:
                print(f"⚠️  Metadata file not found: {metadata_path}")
            self._metadata = None

    def get_all_metadata_image_ids(self) -> List[str]:
        """Get all image IDs from experiment metadata."""
        if not self._metadata:
            if self.verbose:
                print("❌ No metadata loaded. Use set_metadata_path() first.")
            return []
        return self._metadata.get("image_ids", [])

    def get_annotated_image_ids(self, prompt: Optional[str] = None, 
                              model_metadata: Optional[Dict] = None,
                              consider_different_if_different_weights: bool = False) -> List[str]:
        """
        Get image IDs that already have annotations.
        
        Args:
            prompt: Optional specific prompt to check for
            model_metadata: Optional model metadata to match against
            consider_different_if_different_weights: If True, only count as annotated if same model weights
            
        Returns:
            List of image IDs that have annotations (for the specific prompt if provided)
        """
        annotated_ids = []
        for image_id, image_data in self.annotations.get("images", {}).items():
            annotations = image_data.get("annotations", [])
            if prompt:
                # Check for specific prompt
                for ann in annotations:
                    if ann.get("prompt") == prompt:
                        if consider_different_if_different_weights and model_metadata:
                            # Only consider annotated if model weights match
                            if self._models_match(ann.get("model_metadata", {}), model_metadata):
                                annotated_ids.append(image_id)
                                break
                        else:
                            # Standard behavior - any annotation with this prompt counts
                            annotated_ids.append(image_id)
                            break
            else:
                # Check for any annotations
                if len(annotations) > 0:
                    annotated_ids.append(image_id)
        return annotated_ids

    def get_missing_annotations(self, prompts: List[str], 
                              experiment_ids: Optional[List[str]] = None,
                              video_ids: Optional[List[str]] = None,
                              image_ids: Optional[List[str]] = None,
                              model_metadata: Optional[Dict] = None,
                              consider_different_if_different_weights: bool = False) -> Dict[str, List[str]]:
        """
        Find images that are missing annotations for given prompts.
        
        Args:
            prompts: List of prompts to check for
            experiment_ids: Optional filter for specific experiments
            video_ids: Optional filter for specific videos
            image_ids: Optional filter for specific images
            model_metadata: Optional model metadata to check against
            consider_different_if_different_weights: If True, consider different model weights as missing
            
        Returns:
            Dictionary mapping prompt to list of image_ids that need processing
        """
        if not self._metadata:
            if self.verbose:
                print("❌ No metadata loaded. Cannot find missing annotations.")
            return {prompt: [] for prompt in prompts}

        # Get filtered image IDs based on criteria
        target_image_ids = self._get_filtered_image_ids(experiment_ids, video_ids, image_ids)
        
        missing_by_prompt = {}
        for prompt in prompts:
            annotated_for_prompt = set(self.get_annotated_image_ids(
                prompt, model_metadata, consider_different_if_different_weights
            ))
            missing_for_prompt = [img_id for img_id in target_image_ids 
                                if img_id not in annotated_for_prompt]
            missing_by_prompt[prompt] = missing_for_prompt
            
            if self.verbose:
                model_note = " (same model)" if consider_different_if_different_weights else ""
                print(f"📊 Prompt '{prompt}'{model_note}: {len(missing_for_prompt)} missing, {len(annotated_for_prompt)} annotated")
        
        return missing_by_prompt

    def _get_filtered_image_ids(self, experiment_ids: Optional[List[str]] = None,
                               video_ids: Optional[List[str]] = None, 
                               image_ids: Optional[List[str]] = None) -> List[str]:
        """Get filtered list of image IDs based on criteria."""
        if not self._metadata:
            return []
        
        # If specific image_ids provided, just filter those that exist in metadata
        if image_ids:
            all_metadata_ids = set(self.get_all_metadata_image_ids())
            return [img_id for img_id in image_ids if img_id in all_metadata_ids]
        
        # If video_ids provided, get images from those videos
        if video_ids:
            target_image_ids = []
            for video_id in video_ids:
                video_images = get_video_image_ids_safe(video_id, self._metadata)
                target_image_ids.extend(video_images)
            return target_image_ids
        
        # If experiment_ids provided, get images from those experiments
        if experiment_ids:
            target_image_ids = []
            for exp_id in experiment_ids:
                exp_images = get_experiment_image_ids_safe(exp_id, self._metadata)
                target_image_ids.extend(exp_images)
            return target_image_ids
        
        # No filters - return all images from metadata
        return self.get_all_metadata_image_ids()
    def process_missing_annotations(self, model, prompts: Union[str, List[str]],
                                experiment_ids: Optional[List[str]] = None,
                                video_ids: Optional[List[str]] = None,
                                image_ids: Optional[List[str]] = None,
                                save_dir: Optional[Union[str, Path]] = None,
                                box_threshold: float = 0.35, text_threshold: float = 0.25,
                                show_anno: bool = False, text_size: float = 1.0,
                                max_images_per_prompt: Optional[int] = None,
                                consider_different_if_different_weights: bool = False,
                                overwrite: bool = False,
                                store_image_source: bool = False,
                                auto_save_interval: Optional[int] = None) -> Dict[str, Dict[str, Tuple]]:
                                
        """
        Process missing annotations by running inference on unprocessed images.
        
        Args:
            model: GroundingDINO model
            prompts: Prompt(s) to process
            experiment_ids: Optional filter for specific experiments
            video_ids: Optional filter for specific videos  
            image_ids: Optional filter for specific images
            save_dir: Optional directory to save visualizations
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold
            show_anno: Whether to display annotations during inference
            text_size: Text size for visualizations
            max_images_per_prompt: Optional limit on images processed per prompt
            consider_different_if_different_weights: If True, treat different model weights as different annotations
            overwrite: If True, overwrite existing annotations with same prompt/image/model
            auto_save_interval: If provided, automatically save annotations every N processed images
            
        Returns:
            Results from gdino_inference_with_visualization
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if not self._metadata:
            if self.verbose:
                print("❌ No metadata loaded. Use set_m s stadata_path() first.")
            return {}
        
        # Get model metadata for comparison
        model_metadata = get_model_metadata(model)
        
        if self.verbose:
            print(f"🔍 Finding missing annotations for {len(prompts)} prompt(s)...")
            if consider_different_if_different_weights:
                weights_name = Path(model_metadata.get("model_weights_path", "unknown")).name
                print(f"🔧 Model-specific mode: Only considering annotations from {weights_name}")
        
        # Find missing annotations
        missing_by_prompt = self.get_missing_annotations(
            prompts, experiment_ids, video_ids, image_ids, 
            model_metadata, consider_different_if_different_weights
        )
        
        # Count total missing
        total_missing = sum(len(img_ids) for img_ids in missing_by_prompt.values())
        if total_missing == 0:
            if self.verbose:
                print("✅ No missing annotations found!")
            return {}
        
        if self.verbose:
            print(f"📊 Found {total_missing} missing annotations to process")
        
        # Process each prompt
        all_results = {}
        for prompt in prompts:
            missing_image_ids = missing_by_prompt[prompt]
            
            if len(missing_image_ids) == 0:
                if self.verbose:
                    print(f"✅ No missing annotations for prompt '{prompt}'")
                continue
            
            # Apply limit if specified
            if max_images_per_prompt and len(missing_image_ids) > max_images_per_prompt:
                if self.verbose:
                    print(f"⚠️  Limiting '{prompt}' to {max_images_per_prompt} images (found {len(missing_image_ids)})")
                missing_image_ids = missing_image_ids[:max_images_per_prompt]
            
            if self.verbose:
                print(f"\n🔄 Processing {len(missing_image_ids)} images for prompt '{prompt}'...")
            
            # Get image paths
            try:
                image_paths = get_image_paths_safe(missing_image_ids, self._metadata)
                if self.verbose:
                    print(f"📁 Found paths for {len(image_paths)} images")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error getting image paths: {e}")
                continue
            
            # Prepare inference parameters
            inference_params = {"box_threshold": box_threshold, "text_threshold": text_threshold}
            
            # Run inference with auto-save capability
            results = gdino_inference_with_visualization(
                model, image_paths, prompt,
                box_threshold=box_threshold, text_threshold=text_threshold,
                show_anno=show_anno, save_dir=save_dir, text_size=text_size,
                verbose=self.verbose,
                annotations_manager=self,  # Pass self as the annotation manager
                auto_save_interval=auto_save_interval,  # Pass through auto-save setting
                inference_params=inference_params,
                overwrite=overwrite,
                store_image_source=store_image_source,
            )
            
            # Merge results
            for image_name, image_results in results.items():
                if image_name not in all_results:
                    all_results[image_name] = {}
                all_results[image_name].update(image_results)
        
        if self.verbose:
            total_processed = sum(len(img_results) for img_results in all_results.values())
            print(f"\n✅ Processed {total_processed} image-prompt combinations")
            if auto_save_interval is None:
                print("💾 Remember to call .save() to persist annotations!")
            else:
                print("💾 Annotations auto-saved during processing!")
        
        return all_results
    def sync_with_metadata(self, model, prompts: Union[str, List[str]],
                          save_dir: Optional[Union[str, Path]] = None,
                          consider_different_if_different_weights: bool = False,
                          overwrite: bool = False, **kwargs) -> Dict[str, Dict[str, Tuple]]:
        """
        Sync annotations with experiment metadata (alias for process_missing_annotations).
        
        This method ensures all images in the metadata have annotations for the given prompts.
        
        Args:
            model: GroundingDINO model
            prompts: Prompt(s) to process
            save_dir: Optional directory to save visualizations
            consider_different_if_different_weights: If True, treat different model weights as different annotations
            overwrite: If True, overwrite existing annotations with same prompt/image/model
            **kwargs: Additional arguments passed to process_missing_annotations
        
        Example:
            # Ensure all images have person and vehicle annotations
            annotations.sync_with_metadata(model, ["person", "vehicle"])
            
            # Sync with model-specific consideration
            annotations.sync_with_metadata(
                model, ["person"], consider_different_if_different_weights=True
            )
        """
        return self.process_missing_annotations(
            model, prompts, save_dir=save_dir,
            consider_different_if_different_weights=consider_different_if_different_weights,
            overwrite=overwrite, **kwargs
        )

    def get_processing_summary(self, prompts: List[str], 
                             consider_different_if_different_weights: bool = False) -> Dict:
        """
        Get a summary of annotation coverage for given prompts.
        
        Args:
            prompts: List of prompts to analyze
            consider_different_if_different_weights: If True, provide model-specific breakdown
        
        Returns:
            Dictionary with coverage statistics
        """
        if not self._metadata:
            return {"error": "No metadata loaded"}
        
        total_images = len(self.get_all_metadata_image_ids())
        summary = {
            "total_images_in_metadata": total_images,
            "prompts": {}
        }
        
        for prompt in prompts:
            if consider_different_if_different_weights:
                # Get model-specific breakdown
                models_used = self.list_models_for_prompt(prompt)
                model_breakdown = {}
                
                for model_metadata in models_used:
                    annotated = len(self.get_annotated_image_ids(prompt, model_metadata, True))
                    weights_name = Path(model_metadata.get("model_weights_path", "unknown")).name
                    model_breakdown[weights_name] = {
                        "annotated": annotated,
                        "model_metadata": model_metadata
                    }
                
                # Total annotated (any model)
                total_annotated = len(self.get_annotated_image_ids(prompt))
                missing = total_images - total_annotated
                coverage = (total_annotated / total_images * 100) if total_images > 0 else 0
                
                summary["prompts"][prompt] = {
                    "annotated": total_annotated,
                    "missing": missing,
                    "coverage_percent": coverage,
                    "models_used": len(models_used),
                    "model_breakdown": model_breakdown
                }
            else:
                # Standard summary
                annotated = len(self.get_annotated_image_ids(prompt))
                missing = total_images - annotated
                coverage = (annotated / total_images * 100) if total_images > 0 else 0
                
                summary["prompts"][prompt] = {
                    "annotated": annotated,
                    "missing": missing,
                    "coverage_percent": coverage
                }
        
        return summary

    def print_processing_summary(self, prompts: List[str], 
                               consider_different_if_different_weights: bool = False):
        """
        Print a formatted summary of annotation coverage.
        
        Args:
            prompts: List of prompts to analyze
            consider_different_if_different_weights: If True, show model-specific breakdown
        """
        summary = self.get_processing_summary(prompts, consider_different_if_different_weights)
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"\n📊 ANNOTATION COVERAGE SUMMARY")
        print(f"=" * 40)
        print(f"📸 Total images in metadata: {summary['total_images_in_metadata']}")
        
        for prompt, stats in summary["prompts"].items():
            print(f"\n🏷️  Prompt: '{prompt}'")
            print(f"   ✅ Annotated: {stats['annotated']}")
            print(f"   ❌ Missing: {stats['missing']}")
            print(f"   📊 Coverage: {stats['coverage_percent']:.1f}%")
            
            if consider_different_if_different_weights and "model_breakdown" in stats:
                print(f"   🔧 Models used: {stats['models_used']}")
                for model_name, model_stats in stats["model_breakdown"].items():
                    print(f"      • {model_name}: {model_stats['annotated']} images")

    # -------------------------------------------------------------------------
    # Core Annotation Management (unchanged from previous version)
    # -------------------------------------------------------------------------


    def add_annotation(self, image_id: str, prompt: str, model, 
                    boxes: np.ndarray, logits: np.ndarray, phrases: List[str],
                    inference_params: Optional[Dict] = None, image_source: Optional[np.ndarray] = None,
                    overwrite: bool = False):
        """
        Add a single annotation with enhanced tensor safety.
        
        Args:
            image_id: Identifier for the image
            prompt: Text prompt used for inference
            model: The GroundingDINO model (to capture metadata)
            boxes: Detection boxes (should be numpy array, not tensors)
            logits: Confidence scores (should be numpy array, not tensors)
            phrases: Detected phrases
            inference_params: Optional inference parameters
            image_source: Optional original image for visualization
            overwrite: If True, overwrite existing annotation with same prompt/image/model
        """
        # Capture model metadata
        model_metadata = get_model_metadata(model)
        
        # Ensure boxes and logits are numpy arrays and on CPU
        if hasattr(boxes, 'cpu'):  # It's a tensor
            boxes = boxes.cpu().numpy()
        elif not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
            
        if hasattr(logits, 'cpu'):  # It's a tensor  
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
                # Found existing annotation with same prompt
                if overwrite and self._models_match(ann.get("model_metadata", {}), model_metadata):
                    # Same prompt + same model + overwrite=True -> replace this annotation
                    annotation_replaced = True
                    if self.verbose:
                        weights_name = Path(model_metadata.get("model_weights_path", "unknown")).name
                        print(f"🔄 Overwriting existing annotation for '{image_id}' + '{prompt}' (model: {weights_name})")
                    continue  # Skip adding the old annotation
                elif not self._models_match(ann.get("model_metadata", {}), model_metadata):
                    # Same prompt but different model -> keep both
                    updated_annotations.append(ann)
                else:
                    # Same prompt + same model + overwrite=False -> keep existing, skip new
                    if self.verbose:
                        weights_name = Path(model_metadata.get("model_weights_path", "unknown")).name
                        print(f"⚠️  Skipping duplicate annotation for '{image_id}' + '{prompt}' (model: {weights_name}). Use overwrite=True to replace.")
                    return  # Don't add the new annotation
            else:
                # Different prompt -> keep this annotation
                updated_annotations.append(ann)

        # Create new annotation with guaranteed JSON-serializable data
        new_annotation = {
            "annotation_id": f"ann_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "prompt": prompt,
            "model_metadata": model_metadata,
            "inference_params": inference_params or {},
            "timestamp": datetime.now().isoformat(),
            "num_detections": int(len(boxes)),  # Ensure int
            "detections": [
                {
                    "box_xywh": box.tolist(),  # numpy -> list
                    "confidence": float(logit),  # numpy scalar -> float
                    "phrase": str(phrase)  # Ensure string
                } for box, logit, phrase in zip(boxes, logits, phrases)
            ]
        }
        
        # Add new annotation
        updated_annotations.append(new_annotation)
        self.annotations["images"][image_id]["annotations"] = updated_annotations
        self._unsaved_changes = True
    
        if self.verbose and not annotation_replaced:
            weights_path = model_metadata.get("model_weights_path", "unknown")
            print(f"✅ Added annotation for '{image_id}' + '{prompt}' ({len(boxes)} detections)")
            print(f"   📂 Model: {Path(weights_path).name}")


    def add_from_inference_results(self, results: Dict[str, Dict[str, Tuple]], model, 
                                   inference_params: Optional[Dict] = None, overwrite: bool = False):
        """
        Add annotations from gdino_inference_with_visualization results.
        
        Args:
            results: Output from gdino_inference_with_visualization
            model: The GroundingDINO model used for inference
            inference_params: Optional inference parameters used
            overwrite: If True, overwrite existing annotations with same prompt/image/model
        """
        total_added = 0
        
        for image_name, image_results in results.items():
            for prompt, (boxes, logits, phrases, image_source) in image_results.items():
                if len(boxes) > 0:  # Only add if detections found
                    self.add_annotation(
                        image_name, prompt, model, boxes, logits, phrases, 
                        inference_params, image_source, overwrite=overwrite
                    )
                    total_added += 1
        
        if self.verbose:
            action = "Added/updated" if overwrite else "Added"
            print(f"📊 {action} {total_added} annotations from inference results")

    def inference_and_annotate(self, model, images: Union[str, Path, List[Union[str, Path]]], 
                              prompts: Union[str, List[str]], save_dir: Optional[Union[str, Path]] = None,
                              box_threshold: float = 0.35, text_threshold: float = 0.25,
                              show_anno: bool = False, text_size: float = 1.0) -> Dict[str, Dict[str, Tuple]]:
        """Run inference and automatically store annotations (one-step workflow)."""
        if self.verbose:
            print(f"🔄 Running inference and storing annotations...")
        
        # Run inference with controlled verbosity
        results = gdino_inference_with_visualization(
            model, images, prompts, 
            box_threshold=box_threshold, text_threshold=text_threshold,
            show_anno=show_anno, save_dir=save_dir, text_size=text_size,
            verbose=self.verbose
        )
        
        # Store annotations
        inference_params = {"box_threshold": box_threshold, "text_threshold": text_threshold}
        self.add_from_inference_results(results, model, inference_params)
        
        return results

    # -------------------------------------------------------------------------
    # Visualization from Stored Annotations (unchanged from previous version)
    # -------------------------------------------------------------------------

    def visualize_annotation(self, image_id: str, prompt: str, image_path: Optional[Union[str, Path]] = None,
                           text_size: float = 1.0, show_anno: bool = True, save_path: Optional[str] = None):
        """Visualize a stored annotation without re-running inference."""
        # Get stored annotation
        annotation = self._get_annotation(image_id, prompt)
        if annotation is None:
            if self.verbose:
                print(f"❌ No annotation found for '{image_id}' + '{prompt}'")
            return
        
        # Load image (try to auto-find if not provided)
        if image_path is None:
            if self._metadata:
                try:
                    image_path = get_image_paths_safe([image_id], self._metadata)[0]
                    if self.verbose:
                        print(f"📁 Auto-found image path: {image_path}")
                except:
                    if self.verbose:
                        print(f"❌ Could not auto-find image path for '{image_id}'")
                    return
            else:
                if self.verbose:
                    print(f"❌ Image path required for visualization (no metadata loaded)")
                return
        
        try:
            # Load image using GroundingDINO's load_image function
            from groundingdino.util.inference import load_image
            image_source, _ = load_image(str(image_path))
        except Exception as e:
            if self.verbose:
                print(f"❌ Error loading image {image_path}: {e}")
            return
        
        # Extract detection data from stored annotation
        detections = annotation["detections"]
        if len(detections) == 0:
            if self.verbose:
                print(f"📭 No detections in stored annotation for '{image_id}' + '{prompt}'")
            return
        
        # Convert back to numpy arrays
        boxes = np.array([det["box_xywh"] for det in detections])
        logits = np.array([det["confidence"] for det in detections])
        phrases = [det["phrase"] for det in detections]
        
        # Convert to torch tensors (for compatibility with visualize_detections)
        boxes_tensor = torch.tensor(boxes)
        logits_tensor = torch.tensor(logits)
        
        # Create title
        title = f"Stored Annotation: {image_id} | '{prompt}'\n{len(boxes)} detections (from {annotation['timestamp'][:10]})"
        
        # Visualize using the existing function
        visualize_detections(
            image_source, boxes_tensor, logits_tensor, phrases,
            title=title, save_path=save_path, text_size=text_size, 
            show_anno=show_anno, verbose=self.verbose
        )

    def visualize_all_annotations(self, image_id: str, image_path: Optional[Union[str, Path]] = None,
                                text_size: float = 1.0, show_anno: bool = True, save_dir: Optional[Union[str, Path]] = None):
        """Visualize all stored annotations for an image."""
        image_annotations = self.get_annotations_for_image(image_id)
        
        if len(image_annotations) == 0:
            if self.verbose:
                print(f"📭 No annotations found for image '{image_id}'")
            return
        
        # Auto-find image path if not provided
        if image_path is None and self._metadata:
            try:
                image_path = get_image_paths_safe([image_id], self._metadata)[0]
                if self.verbose:
                    print(f"📁 Auto-found image path: {image_path}")
            except:
                if self.verbose:
                    print(f"❌ Could not auto-find image path for '{image_id}'")
                return
        
        if self.verbose:
            print(f"🎨 Visualizing {len(image_annotations)} annotations for '{image_id}'")
        
        for annotation in image_annotations:
            prompt = annotation["prompt"]
            
            save_path = None
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                safe_prompt = prompt.replace(" ", "_").replace("/", "_")
                save_path = save_dir / f"{image_id}_{safe_prompt}_stored.jpg"
            
            self.visualize_annotation(
                image_id, prompt, image_path, text_size, show_anno, str(save_path) if save_path else None
            )

    # -------------------------------------------------------------------------
    # Data Management (mostly unchanged)
    # -------------------------------------------------------------------------

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
        
        # Add metadata integration info
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
            print(f"📂 Metadata integration: ❌ (use set_metadata_path() to enable)")


    def _get_annotation(self, image_id: str, prompt: str) -> Optional[Dict]:
        """Helper to get a specific annotation."""
        annotations = self.get_annotations_for_image(image_id)
        for ann in annotations:
            if ann.get("prompt") == prompt:
                return ann
        return None

    def _models_match(self, metadata1: Dict, metadata2: Dict) -> bool:
        """
        Check if two model metadata represent the same model.
        
        Compares model weights path and config path to determine if annotations
        are from the same model checkpoint.
        
        Args:
            metadata1: First model metadata dictionary
            metadata2: Second model metadata dictionary
            
        Returns:
            True if models match, False otherwise
        """
        # Compare model weights path (primary identifier)
        weights1 = metadata1.get("model_weights_path", "")
        weights2 = metadata2.get("model_weights_path", "")
        
        # Compare config path (secondary identifier)
        config1 = metadata1.get("model_config_path", "")
        config2 = metadata2.get("model_config_path", "")
        
        # Models match if both weights and config paths are the same
        weights_match = weights1 == weights2 and weights1 != ""
        config_match = config1 == config2 and config1 != ""
        
        return weights_match and config_match

    def get_annotations_by_model(self, image_id: str, prompt: str) -> List[Dict]:
        """
        Get all annotations for a specific image and prompt, grouped by model.
        
        Args:
            image_id: Image identifier
            prompt: Prompt to filter by
            
        Returns:
            List of annotations with the specified prompt for the image
            
        Example:
            # Get all 'person' annotations for an image (may include multiple models)
            person_annotations = annotations.get_annotations_by_model("img_001", "person")
            for ann in person_annotations:
                model_name = Path(ann["model_metadata"]["model_weights_path"]).name
                print(f"Model: {model_name}, Detections: {ann['num_detections']}")
        """
        image_annotations = self.get_annotations_for_image(image_id)
        return [ann for ann in image_annotations if ann.get("prompt") == prompt]

    def list_models_for_prompt(self, prompt: str) -> List[Dict]:
        """
        List all different models that have been used for a specific prompt.
        
        Args:
            prompt: Prompt to analyze
            
        Returns:
            List of unique model metadata dictionaries used for this prompt
        """
        unique_models = []
        seen_model_signatures = set()
        
        for image_id, image_data in self.annotations.get("images", {}).items():
            for ann in image_data.get("annotations", []):
                if ann.get("prompt") == prompt:
                    model_metadata = ann.get("model_metadata", {})
                    
                    # Create signature for uniqueness check
                    weights_path = model_metadata.get("model_weights_path", "")
                    config_path = model_metadata.get("model_config_path", "")
                    signature = (weights_path, config_path)
                    
                    if signature not in seen_model_signatures and weights_path:
                        seen_model_signatures.add(signature)
                        unique_models.append(model_metadata)
        
        return unique_models

    @property
    def has_unsaved_changes(self) -> bool:
        """Check for unsaved changes."""
        return self._unsaved_changes

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        status = "✅ saved" if not self._unsaved_changes else "⚠️ unsaved"
        metadata_status = "📂 metadata" if summary['metadata_loaded'] else "📂 no-metadata"
        return f"GroundedDinoAnnotations(images={summary['total_images']}, annotations={summary['total_annotations']}, {status}, {metadata_status})"


# -------------------------------------------------------------------------
# Utility Functions for Experiment Metadata Integration
# -------------------------------------------------------------------------

def load_experiment_metadata_safe(metadata_path: Union[str, Path]) -> Optional[Dict]:
    """
    Safely load experiment metadata with error handling.
    
    Args:
        metadata_path: Path to experiment_metadata.json
        
    Returns:
        Metadata dictionary or None if loading fails
    """
    try:
        # Import function from experiment metadata utils
        from scripts.utils.experiment_metadata_utils import load_experiment_metadata
        return load_experiment_metadata(metadata_path)
    except Exception as e:
        print(f"⚠️  Could not load experiment metadata from {metadata_path}: {e}")
        return None

def get_image_paths_safe(image_ids: List[str], metadata: Dict) -> List[Path]:
    """
    Safely get image paths with error handling.
    
    Args:
        image_ids: List of image IDs
        metadata: Loaded metadata dictionary
        
    Returns:
        List of image paths
    """
    try:
        # Import function from experiment metadata utils
        from scripts.utils.experiment_metadata_utils import get_image_id_paths
        return get_image_id_paths(image_ids, metadata)
    except Exception as e:
        print(f"⚠️  Error getting image paths: {e}")
        raise

def get_experiment_image_ids_safe(experiment_id: str, metadata: Dict) -> List[str]:
    """
    Get all image IDs for a specific experiment.
    
    Args:
        experiment_id: Experiment identifier (e.g., "20231206")
        metadata: Loaded metadata dictionary
        
    Returns:
        List of image IDs in the experiment
    """
    try:
        experiment_data = metadata.get("experiments", {}).get(experiment_id, {})
        image_ids = []
        
        for video_id, video_data in experiment_data.get("videos", {}).items():
            image_ids.extend(video_data.get("image_ids", []))
        
        return image_ids
    except Exception as e:
        print(f"⚠️  Error getting experiment images: {e}")
        return []

def get_video_image_ids_safe(video_id: str, metadata: Dict) -> List[str]:
    """
    Get all image IDs for a specific video.
    
    Args:
        video_id: Video identifier (e.g., "20231206_A01")
        metadata: Loaded metadata dictionary
        
    Returns:
        List of image IDs in the video
    """
    try:
        # Parse video_id to get experiment_id
        parts = video_id.split('_')
        if len(parts) < 2:
            print(f"⚠️  Invalid video_id format: {video_id}")
            return []
        
        experiment_id = parts[0]
        experiment_data = metadata.get("experiments", {}).get(experiment_id, {})
        video_data = experiment_data.get("videos", {}).get(video_id, {})
        
        return video_data.get("image_ids", [])
    except Exception as e:
        print(f"⚠️  Error getting video images: {e}")
        return []

def validate_entity_ids(entity_ids: List[str], entity_type: str, metadata: Dict) -> Tuple[List[str], List[str]]:
    """
    Validate that entity IDs exist in metadata.
    
    Args:
        entity_ids: List of entity IDs to validate
        entity_type: Type of entity ("experiment", "video", "image")
        metadata: Loaded metadata dictionary
        
    Returns:
        Tuple of (valid_ids, invalid_ids)
    """
    if entity_type == "experiment":
        valid_set = set(metadata.get("experiments", {}).keys())
    elif entity_type == "video":
        valid_set = set(metadata.get("video_ids", []))
    elif entity_type == "image":
        valid_set = set(metadata.get("image_ids", []))
    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")
    
    valid_ids = [eid for eid in entity_ids if eid in valid_set]
    invalid_ids = [eid for eid in entity_ids if eid not in valid_set]
    
    return valid_ids, invalid_ids




def visualize_detections(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, 
                         phrases: List[str], title: str = "Detections", save_path: Optional[str] = None,
                         text_size: float = 1.0, show_anno: bool = True, verbose: bool = True):
    """Visualize detections on an image with improved annotation display."""
    annotated_frame = image_source.copy()
    h, w, _ = annotated_frame.shape
    
    # Define colors for different detections
    colors = [
        (0, 100, 200),    # Dark Red
        (0, 80, 160),     # Dark Orange  
        (20, 120, 20),    # Dark Green
        (150, 50, 0),     # Dark Blue
        (120, 20, 120),   # Dark Purple
        (0, 140, 140),    # Dark Yellow/Brown
        (100, 0, 100),    # Dark Magenta
        (80, 80, 0),      # Dark Cyan
        (60, 20, 140),    # Dark Red-Purple
        (0, 60, 100),     # Dark Orange-Red
        (40, 100, 0),     # Dark Blue-Green
        (100, 60, 20),    # Dark Blue-Purple
    ]
    
    # Calculate font parameters based on text_size
    base_font_scale = 2.0
    base_thickness = 10
    
    font_scale = base_font_scale * text_size
    thickness = max(1, int(base_thickness * text_size))  # Ensure thickness is at least 1
    
    for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        # Convert normalized coordinates to pixel coordinates
        if isinstance(box, torch.Tensor):
            box_norm = box.clone()
        else:
            box_norm = torch.tensor(box)
            
        # Convert from center format to corner format
        cx, cy, bw, bh = box_norm
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label with confidence
        confidence = float(logit) if isinstance(logit, torch.Tensor) else logit
        label = f"{phrase}: {confidence:.2f}"
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(annotated_frame, 
                      (x1, y1 - text_height - 10), 
                      (x1 + text_width, y1), 
                      color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, 
                   (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        if verbose:
            print(f"Visualization saved to: {save_path}")
    
    # Only show if requested
    if show_anno:
        plt.show()
    else:
        plt.close()  # Close the figure to prevent display

def run_inference(model, image_path: Union[str, Path], text_prompt: str, 
                  box_threshold: float = 0.35, text_threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Run GroundingDINO inference and return results and the source image.
    
    Returns:
        Tuple of (boxes, logits, phrases, image_source) where:
        - boxes: numpy array of bounding boxes (converted from CUDA tensors)
        - logits: numpy array of confidence scores (converted from CUDA tensors) 
        - phrases: list of detected phrases
        - image_source: source image as numpy array
    """
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
    
    # Convert CUDA tensors to CPU numpy arrays to ensure JSON serializability
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
    """
    Unified function for GroundingDINO inference with visualization support and optional auto-save.
    
    This single function handles all combinations with optional annotation management:
    - Single image + single prompt
    - Single image + multiple prompts  
    - Multiple images + single prompt
    - Multiple images + multiple prompts
    
    Args:
        model: Loaded GroundingDINO model
        images: Single image path or list of image paths
        prompts: Single prompt or list of prompts
        box_threshold: Box confidence threshold (default: 0.35)
        text_threshold: Text confidence threshold (default: 0.25)
        show_anno: Whether to display annotated images (default: True)
        save_dir: Optional directory to save annotated images
        text_size: Text size multiplier for annotations (default: 1.0)
        verbose: Whether to print progress and detection info (default: True)
        annotations_manager: Optional GroundedDinoAnnotations instance for auto-saving
        auto_save_interval: If provided with annotations_manager, save every N processed images
        inference_params: Optional inference parameters for annotations
        overwrite: If True, overwrite existing annotations with same prompt/image/model
    
    Returns:
        Nested dictionary: {image_name: {prompt: (boxes, logits, phrases, image_source)}}
        
    Examples:
        # Standard usage (no auto-save)
        results = gdino_inference_with_visualization(
            model, ["img1.jpg", "img2.jpg"], ["person", "car"]
        )
        
        # With auto-save every 25 images
        annotations = GroundedDinoAnnotations("annotations.json")
        results = gdino_inference_with_visualization(
            model, image_list, ["person", "vehicle"],
            annotations_manager=annotations,
            auto_save_interval=25,
            inference_params={"box_threshold": 0.35, "text_threshold": 0.25}
        )
        
        # Auto-save with overwrite for existing annotations
        results = gdino_inference_with_visualization(
            model, image_paths, prompts,
            annotations_manager=annotations,
            auto_save_interval=10,
            overwrite=True
        )
    """
    # Normalize inputs to lists
    if isinstance(images, (str, Path)):
        image_list = [Path(images)]
    else:
        image_list = [Path(img) for img in images]
    
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = prompts
    
    # Prepare save directory if needed
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results structure
    results = {}
    
    # Calculate total operations for progress tracking
    total_ops = len(image_list) * len(prompt_list)
    current_op = 0
    
    # Auto-save tracking
    processed_images_count = 0
    auto_save_enabled = annotations_manager is not None and auto_save_interval is not None
    
    if verbose:
        print(f"🔍 Starting inference for {len(image_list)} image(s) × {len(prompt_list)} prompt(s) = {total_ops} operations")
        if text_size != 1.0:
            print(f"📝 Using text size multiplier: {text_size}x")
        if auto_save_enabled:
            print(f"💾 Auto-save enabled: saving annotations every {auto_save_interval} processed images")
    
    # Process each image
    for img_idx, image_path in enumerate(image_list):
        image_name = image_path.stem
        results[image_name] = {}
        image_has_new_annotations = False
        
        if verbose:
            print(f"\n📸 Processing image [{img_idx+1}/{len(image_list)}]: {image_name}")
        
        # Process each prompt for this image
        for prompt in prompt_list:
            current_op += 1
            if verbose:
                print(f"   🔍 [{current_op}/{total_ops}] Running prompt: '{prompt}'")
            
            try:
                # Run inference
                boxes, logits, phrases, image_source = run_inference(
                    model, image_path, prompt, box_threshold, text_threshold
                )
                
                # Store results
                if store_image_source:
                    results[image_name][prompt] = (boxes, logits, phrases, image_source)
                else:
                    results[image_name][prompt] = (boxes, logits, phrases, None)  # Save memory

                # Add to annotations if manager provided
                if annotations_manager is not None:
                    if len(boxes) > 0:  # Only add if detections found
                        annotations_manager.add_annotation(
                            image_name, prompt, model, boxes, logits, phrases,
                            inference_params or {"box_threshold": box_threshold, "text_threshold": text_threshold},
                            image_source, overwrite=overwrite
                        )
                        image_has_new_annotations = True
                
                # Print detection summary
                if verbose:
                    print(f"      📍 Found {len(boxes)} detections")
                    if len(phrases) > 0:
                        max_conf = float(max(logits)) if len(logits) > 0 else 0.0
                        print(f"      🏆 Max confidence: {max_conf:.3f}")
                        print(f"      🏷️  Detected: {', '.join(phrases[:3])}" + 
                              ("..." if len(phrases) > 3 else ""))
                
                # Handle visualization and saving
                if show_anno or save_dir:
                    title = f"Image: {image_name} | Prompt: '{prompt}'\n{len(boxes)} detections found"
                    
                    save_path = None
                    if save_dir:
                        # Create filename: {image_name}_{prompt}.jpg
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
                # Store empty results for failed operations
                results[image_name][prompt] = (
                    np.array([]), np.array([]), [], 
                    np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder image
                )
                continue
        
        # Update processed images counter and auto-save if needed
        if image_has_new_annotations:
            processed_images_count += 1
            
            if auto_save_enabled and processed_images_count >= auto_save_interval:
                if verbose:
                    print(f"      💾 Auto-saving annotations (processed {processed_images_count} images with new annotations)...")
                annotations_manager.save()
                processed_images_count = 0  # Reset counter after saving
    
    # Final save if there are remaining unsaved changes
    if auto_save_enabled and annotations_manager.has_unsaved_changes:
        if verbose:
            print(f"💾 Final auto-save (processed {processed_images_count} additional images with annotations)...")
        annotations_manager.save()
    
    # Print final summary (if verbose)
    if verbose:
        print(f"\n✅ Inference complete!")
        print(f"📊 Summary by image:")
    
    # Calculate summary (always needed for potential return info)
    total_detections = 0
    for image_name, image_results in results.items():
        image_detections = 0
        successful_prompts = 0
        
        for prompt, (boxes, logits, phrases, img) in image_results.items():
            detection_count = len(boxes)
            image_detections += detection_count
            if detection_count > 0:
                successful_prompts += 1
        
        total_detections += image_detections
        
        # Print individual image summary if verbose
        if verbose:
            print(f"   📸 {image_name}: {image_detections} total detections ({successful_prompts}/{len(prompt_list)} prompts found objects)")
    
    # Print final totals if verbose
    if verbose:
        print(f"🎯 Grand total: {total_detections} detections across all images and prompts")
        
        if save_dir:
            saved_files = len([f for f in save_dir.glob("*.jpg")])
            print(f"💾 Saved {saved_files} annotated images to: {save_dir}")
        
        if auto_save_enabled:
            print(f"💾 Annotations auto-saved during processing!")
    
    return results
def compare_annotations(image_source: np.ndarray, annotations_list: List[Dict], 
                       image_id: str, save_path: Optional[str] = None):
    """Compare multiple annotation sets for the same image side by side."""
    num_annotations = len(annotations_list)
    if num_annotations == 0:
        print("No annotations to compare.")
        return
    
    fig, axes = plt.subplots(1, num_annotations, figsize=(6*num_annotations, 6))
    if num_annotations == 1:
        axes = [axes]
    
    for i, annotation in enumerate(annotations_list):
        # Extract detection data
        dets = annotation['detections']
        if len(dets) > 0:
            boxes = torch.tensor([d['box_xywh'] for d in dets])
            logits = torch.tensor([d['confidence'] for d in dets])
            phrases = [d['phrase'] for d in dets]
            
            # Create annotated image
            annotated_frame = image_source.copy()
            h, w, _ = annotated_frame.shape
            
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for j, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                cx, cy, bw, bh = box
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                color = colors[j % len(colors)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{phrase}: {logit:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            annotated_frame = image_source.copy()
        
        # Display
        axes[i].imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Prompt: '{annotation['prompt']}'\n{annotation['num_detections']} detections")
        axes[i].axis('off')
    
    plt.suptitle(f"Annotation Comparison - {image_id}", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Comparison saved to: {save_path}")
    
    plt.show()

def inference_and_visualize(model, image_path: Union[str, Path], text_prompt: str, 
                            box_threshold: float = 0.35, text_threshold: float = 0.25):
    """A wrapper to run inference and visualize the results immediately."""
    boxes, logits, phrases, image_source = run_inference(
        model, image_path, text_prompt, box_threshold, text_threshold
    )
    
    print(f"Found {len(boxes)} detections for prompt: '{text_prompt}'")
    
    visualize_detections(
        image_source, boxes, logits, phrases,
        title=f"Detections for: {Path(image_path).name}\nPrompt: '{text_prompt}'"
    )
    return boxes, logits, phrases
