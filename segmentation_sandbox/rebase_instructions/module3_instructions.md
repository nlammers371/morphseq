# Module 3: Annotation Pipeline

## Overview
Create the unified annotation system with GDINO detection and SAM2 segmentation. Implement the all-in-one GroundedSam class that handles the complete pipeline from detection to tracking. Rename and refactor existing utilities to follow the new architecture.

## Dependencies
- Module 1: Core Foundation (completed)
- Module 2: Metadata System (completed)
- Existing code to refactor: grounded_sam_utils.py → gdino_utils.py
- External: GroundingDINO, SAM2 models

## Files to Create/Modify

```
utils/
└── annotation/
    ├── __init__.py
    ├── detection/
    │   ├── __init__.py
    │   ├── gdino_annotations.py     # Renamed from grounded_sam_utils.py
    │   └── gdino_inference.py       # Split out inference logic
    ├── grounded_sam/
    │   ├── __init__.py
    │   ├── gsam_pipeline.py         # All-in-one GroundedSam class
    │   └── gsam_utils.py            # Helper functions
    └── segmentation/
        ├── __init__.py
        ├── sam2_annotations.py       # SAM2 annotation management
        ├── sam2_predictor.py         # SAM2 video prediction
        └── mask_utils.py             # Mask format conversions
```

## Implementation Steps

### Step 1: Create `utils/annotation/detection/gdino_annotations.py`

```python
"""
GDINO annotation management (renamed from grounded_sam_utils.py).
Handles detection storage, high-quality filtering, and batch operations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from datetime import datetime
from collections import defaultdict

from ...core import (
    BaseAnnotationParser, parse_entity_id, get_parent_ids,
    get_timestamp, validate_path, get_simplified_filename
)

class GdinoAnnotations(BaseAnnotationParser):
    """
    Manages GroundingDINO detection annotations.
    
    Features:
    - Model metadata tracking
    - High-quality annotation filtering
    - Batch inference support
    - Integration with experiment metadata
    """
    
    def __init__(self, filepath: Union[str, Path], 
                 metadata_path: Optional[Union[str, Path]] = None,
                 verbose: bool = True):
        """Initialize GDINO annotation manager."""
        self.metadata_path = validate_path(metadata_path) if metadata_path else None
        self._metadata = None
        
        # Load experiment metadata if provided
        if self.metadata_path and self.metadata_path.exists():
            from ...metadata.experiment import ExperimentMetadata
            self._exp_metadata = ExperimentMetadata(self.metadata_path, verbose=False)
            self._metadata = self._exp_metadata.data
        
        super().__init__(filepath, verbose=verbose)
    
    def _load_or_initialize(self) -> Dict:
        """Load or create annotation structure."""
        if self.filepath.exists():
            return self.load_json()
            
        return {
            "file_info": {
                "creation_time": self.get_timestamp(),
                "last_updated": self.get_timestamp()
            },
            "images": {},
            "high_quality_annotations": {}
        }
    
    def _validate_schema(self, data: Dict) -> None:
        """Validate annotation structure."""
        required_keys = ["file_info", "images"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
    
    def add_annotation(self, image_id: str, prompt: str, 
                      detections: List[Dict], model_metadata: Dict,
                      inference_params: Optional[Dict] = None,
                      overwrite: bool = False) -> bool:
        """
        Add detection annotation for an image.
        
        Args:
            image_id: Image identifier
            prompt: Detection prompt used
            detections: List of detection dicts with box_xyxy, confidence, phrase
            model_metadata: Model configuration and weights info
            inference_params: Detection parameters (thresholds, etc.)
            overwrite: Whether to overwrite existing annotation
        """
        # Ensure image exists in structure
        if image_id not in self.data["images"]:
            self.data["images"][image_id] = {"annotations": []}
        
        image_data = self.data["images"][image_id]
        
        # Check for existing annotation with same prompt
        existing_idx = None
        for idx, ann in enumerate(image_data["annotations"]):
            if ann.get("prompt") == prompt:
                if not overwrite:
                    if self.verbose:
                        print(f"⚠️  Annotation exists for '{prompt}' on {image_id}")
                    return False
                existing_idx = idx
                break
        
        # Create annotation entry
        annotation = {
            "annotation_id": f"ann_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "prompt": prompt,
            "model_metadata": model_metadata,
            "inference_params": inference_params or {},
            "timestamp": self.get_timestamp(),
            "num_detections": len(detections),
            "detections": detections
        }
        
        # Add or replace annotation
        if existing_idx is not None:
            image_data["annotations"][existing_idx] = annotation
        else:
            image_data["annotations"].append(annotation)
        
        self.mark_changed()
        return True
    
    def get_detections(self, image_id: str, prompt: Optional[str] = None) -> List[Dict]:
        """Get detections for an image, optionally filtered by prompt."""
        image_data = self.data["images"].get(image_id, {})
        annotations = image_data.get("annotations", [])
        
        if prompt:
            for ann in annotations:
                if ann.get("prompt") == prompt:
                    return ann.get("detections", [])
            return []
        
        # Return all detections if no prompt specified
        all_detections = []
        for ann in annotations:
            all_detections.extend(ann.get("detections", []))
        return all_detections
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def generate_high_quality_annotations(self, 
                                        image_ids: List[str],
                                        prompt: str = "individual embryo",
                                        confidence_threshold: float = 0.5,
                                        iou_threshold: float = 0.5,
                                        save_to_self: bool = True) -> Dict:
        """
        Generate high-quality annotations by filtering detections.
        
        Process:
        1. Collect all detections for given prompt
        2. Filter by confidence threshold
        3. Apply NMS with IoU threshold
        4. Group by experiment
        """
        if self.verbose:
            print(f"🎯 Generating high-quality annotations for {len(image_ids)} images")
            print(f"   Prompt: '{prompt}', Confidence: {confidence_threshold}, IoU: {iou_threshold}")
        
        # Get experiment mapping if metadata available
        image_to_exp = {}
        if self._metadata:
            for exp_id, exp_data in self._metadata.get("experiments", {}).items():
                for video_data in exp_data.get("videos", {}).values():
                    for img_id in video_data.get("images", {}).keys():
                        image_to_exp[img_id] = exp_id
        
        # Collect and filter detections
        high_quality_by_exp = defaultdict(lambda: {
            "prompt": prompt,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "timestamp": self.get_timestamp(),
            "filtered": {}
        })
        
        for image_id in image_ids:
            detections = self.get_detections(image_id, prompt)
            
            # Filter by confidence
            confident_dets = [d for d in detections if d.get("confidence", 0) >= confidence_threshold]
            
            if not confident_dets:
                continue
            
            # Apply NMS
            confident_dets.sort(key=lambda x: x["confidence"], reverse=True)
            keep_dets = []
            
            for det in confident_dets:
                should_keep = True
                for kept in keep_dets:
                    iou = self.calculate_iou(det["box_xyxy"], kept["box_xyxy"])
                    if iou > iou_threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    keep_dets.append(det)
            
            # Add to results
            if keep_dets:
                exp_id = image_to_exp.get(image_id, "unknown")
                high_quality_by_exp[exp_id]["filtered"][image_id] = keep_dets
        
        results = dict(high_quality_by_exp)
        
        # Save to self if requested
        if save_to_self:
            self.data["high_quality_annotations"] = results
            self.mark_changed()
            
        return {
            "filtered": results,
            "statistics": {
                "total_images": sum(len(exp["filtered"]) for exp in results.values()),
                "total_detections": sum(
                    len(dets) for exp in results.values() 
                    for dets in exp["filtered"].values()
                )
            }
        }
    
    def get_missing_annotations(self, prompt: str, 
                               image_ids: Optional[List[str]] = None) -> List[str]:
        """Find images missing annotations for a given prompt."""
        if image_ids is None and self._metadata:
            # Get all image IDs from metadata
            image_ids = []
            for exp_data in self._metadata.get("experiments", {}).values():
                for video_data in exp_data.get("videos", {}).values():
                    image_ids.extend(video_data.get("images", {}).keys())
        
        if not image_ids:
            return []
        
        missing = []
        for image_id in image_ids:
            detections = self.get_detections(image_id, prompt)
            if not detections:
                missing.append(image_id)
        
        return missing
```

### Step 2: Create `utils/annotation/grounded_sam/gsam_pipeline.py`

```python
"""
GroundedSam: All-in-one detection + segmentation pipeline.
Handles the complete workflow from detection to tracking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
import numpy as np
from datetime import datetime

from ...core import (
    BaseAnnotationParser, parse_entity_id, get_parent_ids,
    get_timestamp, validate_path
)
from ..detection.gdino_annotations import GdinoAnnotations
from ..segmentation.sam2_annotations import Sam2Annotations

class GroundedSam:
    """
    Unified pipeline for detection + segmentation + tracking.
    
    Workflow:
    1. Load/run GDINO detection on seed frames
    2. Initialize SAM2 with detection boxes
    3. Propagate masks through video
    4. Assign and track embryo IDs
    5. Optional: Generate visualization
    """
    
    def __init__(self, config: Dict, 
                 device: str = "cuda",
                 enable_viz: bool = True,
                 verbose: bool = True):
        """
        Initialize GroundedSam pipeline.
        
        Args:
            config: Pipeline configuration dict
            device: Computation device
            enable_viz: Enable visualization capabilities
            verbose: Verbose output
        """
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        self.enable_viz = enable_viz
        
        # Model placeholders (lazy loading)
        self._gdino_model = None
        self._sam2_model = None
        
        # Annotation managers
        self.gdino_annotations = None
        self.sam2_annotations = None
        
        # Visualization manager (if enabled)
        if enable_viz:
            from ...visualization import PipelineVisualizer
            self.visualizer = PipelineVisualizer()
        else:
            self.visualizer = None
    
    def load_models(self):
        """Load GDINO and SAM2 models."""
        if self.verbose:
            print("🔧 Loading models...")
        
        # Load GDINO
        from ..detection.gdino_inference import load_gdino_model
        self._gdino_model = load_gdino_model(
            self.config["models"]["gdino"]["config"],
            self.config["models"]["gdino"]["weights"],
            device=self.device
        )
        
        # Load SAM2
        from ..segmentation.sam2_predictor import load_sam2_model
        self._sam2_model = load_sam2_model(
            self.config["models"]["sam2"]["config"],
            self.config["models"]["sam2"]["checkpoint"],
            device=self.device
        )
        
        if self.verbose:
            print("✅ Models loaded successfully")
    
    def process_video(self, video_id: str,
                     experiment_metadata_path: Path,
                     gdino_annotations_path: Optional[Path] = None,
                     sam2_annotations_path: Optional[Path] = None,
                     target_prompt: str = "individual embryo",
                     output_viz: bool = True) -> Dict:
        """
        Process a complete video through the pipeline.
        
        Args:
            video_id: Video identifier
            experiment_metadata_path: Path to experiment metadata
            gdino_annotations_path: Optional existing GDINO annotations
            sam2_annotations_path: Optional output path for SAM2 annotations
            target_prompt: Detection prompt
            output_viz: Generate visualization video
            
        Returns:
            Processing results dictionary
        """
        if self.verbose:
            print(f"\n🎬 Processing video: {video_id}")
        
        # Ensure models are loaded
        if self._gdino_model is None:
            self.load_models()
        
        # Load metadata
        from ...metadata.experiment import ExperimentMetadata
        exp_metadata = ExperimentMetadata(experiment_metadata_path, verbose=False)
        
        # Get video info
        level, components = parse_entity_id(video_id, "video")
        exp_id = components["experiment_id"]
        video_data = exp_metadata.data["experiments"][exp_id]["videos"].get(video_id)
        
        if not video_data:
            raise ValueError(f"Video {video_id} not found in metadata")
        
        # Step 1: Get or generate detections
        if self.verbose:
            print("📍 Step 1: Detection")
        
        detections = self._get_or_generate_detections(
            video_id, video_data, target_prompt, 
            gdino_annotations_path, exp_metadata
        )
        
        if not detections:
            if self.verbose:
                print("❌ No detections found")
            return {"status": "no_detections", "video_id": video_id}
        
        # Step 2: Select seed frame
        if self.verbose:
            print("🌱 Step 2: Seed frame selection")
        
        seed_frame_id, seed_detections = self._select_seed_frame(detections)
        if self.verbose:
            print(f"   Selected: {seed_frame_id} with {len(seed_detections)} detections")
        
        # Step 3: Initialize SAM2 and propagate
        if self.verbose:
            print("🔄 Step 3: Mask propagation")
        
        masks = self._propagate_masks(
            video_id, video_data, seed_frame_id, 
            seed_detections, sam2_annotations_path
        )
        
        # Step 4: Generate visualization if requested
        if output_viz and self.visualizer:
            if self.verbose:
                print("🎨 Step 4: Generating visualization")
            
            viz_path = self._generate_visualization(
                video_id, video_data, detections, masks
            )
        else:
            viz_path = None
        
        return {
            "status": "success",
            "video_id": video_id,
            "seed_frame": seed_frame_id,
            "num_embryos": len(seed_detections),
            "frames_processed": len(masks),
            "visualization": viz_path
        }
    
    def _get_or_generate_detections(self, video_id: str, video_data: Dict,
                                   target_prompt: str, 
                                   gdino_annotations_path: Optional[Path],
                                   exp_metadata) -> Dict[str, List[Dict]]:
        """Get existing detections or generate new ones."""
        # Check for existing annotations
        if gdino_annotations_path and gdino_annotations_path.exists():
            self.gdino_annotations = GdinoAnnotations(gdino_annotations_path)
            
            # Check if we have high-quality annotations for this video
            hq_annotations = self.gdino_annotations.data.get("high_quality_annotations", {})
            
            for exp_data in hq_annotations.values():
                if exp_data.get("prompt") == target_prompt:
                    # Collect detections for this video's images
                    detections = {}
                    for image_id in video_data.get("images", {}).keys():
                        if image_id in exp_data.get("filtered", {}):
                            detections[image_id] = exp_data["filtered"][image_id]
                    
                    if detections:
                        return detections
        
        # Generate new detections
        if self.verbose:
            print("   Generating new detections...")
        
        from ..detection.gdino_inference import run_batch_inference
        
        # Get image paths
        image_ids = list(video_data.get("images", {}).keys())
        image_paths = self._get_image_paths(image_ids, video_data)
        
        # Run inference
        results = run_batch_inference(
            self._gdino_model,
            image_paths,
            target_prompt,
            box_threshold=self.config.get("detection", {}).get("box_threshold", 0.35),
            text_threshold=self.config.get("detection", {}).get("text_threshold", 0.25),
            device=self.device
        )
        
        # Convert to detections dict
        detections = {}
        for image_id, (boxes, scores, phrases) in results.items():
            if len(boxes) > 0:
                detections[image_id] = [
                    {
                        "box_xyxy": box.tolist(),
                        "confidence": float(score),
                        "phrase": phrase
                    }
                    for box, score, phrase in zip(boxes, scores, phrases)
                ]
        
        return detections
    
    def _select_seed_frame(self, detections: Dict[str, List[Dict]]) -> Tuple[str, List[Dict]]:
        """
        Select optimal seed frame for propagation.
        
        Criteria:
        1. Maximum number of high-confidence detections
        2. Good spatial distribution
        3. Prefer earlier frames if equal
        """
        candidates = []
        
        for image_id, dets in detections.items():
            if not dets:
                continue
            
            # Calculate metrics
            num_detections = len(dets)
            avg_confidence = np.mean([d["confidence"] for d in dets])
            
            # Extract frame number for ordering
            level, components = parse_entity_id(image_id, "image")
            frame_num = int(components["frame_number"])
            
            candidates.append({
                "image_id": image_id,
                "detections": dets,
                "num_detections": num_detections,
                "avg_confidence": avg_confidence,
                "frame_number": frame_num
            })
        
        if not candidates:
            raise ValueError("No valid seed frames found")
        
        # Sort by: num_detections (desc), avg_confidence (desc), frame_number (asc)
        candidates.sort(
            key=lambda x: (-x["num_detections"], -x["avg_confidence"], x["frame_number"])
        )
        
        best = candidates[0]
        return best["image_id"], best["detections"]
    
    def _propagate_masks(self, video_id: str, video_data: Dict,
                        seed_frame_id: str, seed_detections: List[Dict],
                        output_path: Optional[Path]) -> Dict:
        """Propagate masks through video using SAM2."""
        from ..segmentation.sam2_predictor import Sam2VideoPredictor
        
        predictor = Sam2VideoPredictor(self._sam2_model, self.device)
        
        # Initialize video
        video_path = Path(video_data["mp4_path"])
        predictor.init_video(video_path)
        
        # Get seed frame index
        image_ids = sorted(video_data.get("images", {}).keys())
        seed_idx = image_ids.index(seed_frame_id)
        
        # Initialize