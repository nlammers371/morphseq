# Implementation Instructions: Embryo Mask Export and QC Pipeline

## Overview

This document provides step-by-step instructions for implementing:
1. **Priority 1**: Export embryo masks as labeled images (embryo number → pixel value)
2. **Priority 2**: GSAM QC class for automated quality control
3. **Priority 3**: Integration with EmbryoMetadata for tracking mask paths
## Approach

## 4. Minimize Compute and I/O  
- Avoid loading entire folders or huge files; parse only what’s needed.  
- Use intelligent file queries (e.g., `Path.glob`, search by function name) instead of brute-force reads.  

## 5. Focus on Core Functionality  
- Prioritize correctness of core APIs before adding bells and whistles.  
- Defer non-critical enhancements (e.g., performance tuning) until after basic features pass tests.  

## 6. Incremental Testing  
- After implementing or modifying a piece of functionality, run focused unit tests.  
- Only when unit tests pass, run

## 7. KEEP A LOG OF YOUR  PROGRESS NOTING WHAT YOUVE ACCOMPLISHED AND WHAT YOU WILL DO NEXT IMPORTANT!!!!!
---

## Phase 1: Embryo Mask Export Module (Priority 1)

### Module 1.1: Create `mask_export_utils.py`

**Purpose**: Export SAM2 segmentation masks as labeled embryo images

**Location**: `scripts/utils/mask_export_utils.py`

**Implementation Steps**:

```python
"""
Mask Export Utilities for SAM2 Annotations
Exports embryo masks as labeled images where pixel value = embryo number
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from pycocotools import mask as mask_utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class EmbryoMaskExporter:
    """
    Export SAM2 masks as labeled embryo images.
    
    Note: While PNG would be ideal for label masks (lossless compression),
    we use JPEG with quality 100 to match the pipeline's image format convention.
    """
    
    def __init__(self, sam2_annotations_path: Path, output_base_dir: Path, 
                 verbose: bool = True, output_format: str = "jpg"):
        """
        Initialize mask exporter.
        
        Args:
            sam2_annotations_path: Path to grounded_sam_annotations.json
            output_base_dir: Base directory for mask exports
            verbose: Enable verbose output
            output_format: Output format ('jpg' or 'png'). PNG recommended for label masks.
        """
        self.sam2_path = Path(sam2_annotations_path)
        self.output_base_dir = Path(output_base_dir)
        self.verbose = verbose
        self.output_format = output_format.lower()
        
        if self.output_format not in ['jpg', 'jpeg', 'png']:
            raise ValueError(f"Invalid output format: {output_format}. Use 'jpg' or 'png'")
        
        # Normalize jpeg/jpg
        if self.output_format == 'jpeg':
            self.output_format = 'jpg'
        
        # Load SAM2 annotations
        with open(self.sam2_path, 'r') as f:
            self.sam2_data = json.load(f)
        
        # Track export statistics
        self.export_stats = {
            "total_images": 0,
            "total_masks": 0,
            "overlapping_masks": 0,
            "export_paths": {}  # image_id -> export_path mapping
        }
    
    def decode_rle_mask(self, rle_data: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """Decode RLE mask to binary array."""
        if isinstance(rle_data, dict) and 'counts' in rle_data and 'size' in rle_data:
            # Standard COCO RLE format
            binary_mask = mask_utils.decode(rle_data)
            return binary_mask
        else:
            # Handle other formats
            raise ValueError(f"Unknown RLE format: {type(rle_data)}")
    
    def export_image_masks(self, image_id: str, embryo_data: Dict, 
                          image_shape: Tuple[int, int]) -> Path:
        """
        Export masks for a single image as labeled embryo image.
        
        Args:
            image_id: Image identifier
            embryo_data: Dict of embryo_id -> mask data
            image_shape: (height, width) of output image
            
        Returns:
            Path to exported mask image
        """
        # Initialize empty label image
        label_image = np.zeros(image_shape, dtype=np.uint8)
        
        # Track overlapping pixels
        overlap_count = 0
        
        # Process each embryo mask
        for embryo_id, mask_data in embryo_data.items():
            # Extract embryo number (e.g., "20240411_A01_e01" -> 1)
            embryo_num = int(embryo_id.split('_e')[-1])
            
            # Decode mask
            if mask_data['segmentation_format'] == 'rle':
                binary_mask = self.decode_rle_mask(mask_data['segmentation'], image_shape)
            else:
                raise NotImplementedError(f"Format {mask_data['segmentation_format']} not supported")
            
            # Check for overlaps
            overlap_pixels = np.sum((label_image > 0) & (binary_mask > 0))
            if overlap_pixels > 0:
                overlap_count += overlap_pixels
                if self.verbose:
                    print(f"   ⚠️  Overlap detected: {overlap_pixels} pixels for embryo {embryo_num}")
            
            # Apply embryo number to mask pixels (overwriting overlaps)
            label_image[binary_mask > 0] = embryo_num
        
        # Create output filename with embryo count
        num_embryos = len(embryo_data)
        output_filename = f"{image_id}_masks_emnum_{num_embryos}.{self.output_format}"
        
        # Determine output path (organized by experiment)
        experiment_id = image_id.split('_')[0]
        output_dir = self.output_base_dir / experiment_id / "masks"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / output_filename
        
        # Save based on format
        if self.output_format == 'jpg':
            # Save as JPEG with maximum quality for mask data
            # WARNING: JPEG is lossy and may introduce artifacts that change label values!
            # Consider using PNG for label masks to ensure exact pixel values are preserved.
            cv2.imwrite(str(output_path), label_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            # Verify the saved image maintains label integrity (optional check)
            if self.verbose and num_embryos > 0:
                saved_img = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                unique_values = np.unique(saved_img)
                expected_values = set(range(num_embryos + 1))  # 0 to num_embryos
                if not set(unique_values).issubset(expected_values):
                    print(f"   ⚠️  JPEG compression may have altered label values in {output_filename}")
        else:
            # Save as PNG (lossless, recommended for label masks)
            cv2.imwrite(str(output_path), label_image)
        
        # Update statistics
        self.export_stats["total_masks"] += num_embryos
        if overlap_count > 0:
            self.export_stats["overlapping_masks"] += 1
        
        return output_path
    
    def export_all_masks(self, max_workers: int = 4) -> Dict[str, Path]:
        """
        Export all masks from SAM2 annotations.
        
        Args:
            max_workers: Number of parallel workers
            
        Returns:
            Dict mapping image_id to export path
        """
        if self.verbose:
            print(f"🎯 Starting mask export...")
            print(f"   Output directory: {self.output_base_dir}")
        
        export_paths = {}
        
        # Collect all image export tasks
        export_tasks = []
        for exp_id, exp_data in self.sam2_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_data.get("embryos"):
                        # Get image dimensions from first embryo's RLE size
                        first_embryo = next(iter(image_data["embryos"].values()))
                        if first_embryo['segmentation_format'] == 'rle':
                            height, width = first_embryo['segmentation']['size']
                            image_shape = (height, width)
                        else:
                            # Default shape
                            image_shape = (512, 512)
                        
                        export_tasks.append((image_id, image_data["embryos"], image_shape))
        
        if self.verbose:
            print(f"📊 Found {len(export_tasks)} images to export")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(self.export_image_masks, task[0], task[1], task[2]): task[0]
                for task in export_tasks
            }
            
            for future in as_completed(future_to_image):
                image_id = future_to_image[future]
                try:
                    export_path = future.result()
                    export_paths[image_id] = export_path
                    self.export_stats["total_images"] += 1
                    
                    if self.verbose and self.export_stats["total_images"] % 100 == 0:
                        print(f"   Exported {self.export_stats['total_images']} images...")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"   ❌ Error exporting {image_id}: {e}")
        
        # Save export mapping
        self.export_stats["export_paths"] = {k: str(v) for k, v in export_paths.items()}
        self._save_export_manifest()
        
        if self.verbose:
            print(f"\n✅ Export complete!")
            print(f"   Total images: {self.export_stats['total_images']}")
            print(f"   Total masks: {self.export_stats['total_masks']}")
            print(f"   Images with overlaps: {self.export_stats['overlapping_masks']}")
        
        return export_paths
    
    def _save_export_manifest(self):
        """Save export manifest with paths and statistics."""
        manifest = {
            "export_timestamp": datetime.now().isoformat(),
            "source_file": str(self.sam2_path),
            "output_base_dir": str(self.output_base_dir),
            "statistics": self.export_stats,
            "format_info": {
                "description": "Labeled embryo masks where pixel value = embryo number",
                "background_value": 0,
                "embryo_values": "1 to N (embryo number)",
                "file_format": f"{self.output_format.upper()} ({'lossy, quality 100' if self.output_format == 'jpg' else 'lossless'})"
            }
        }
        
        manifest_path = self.output_base_dir / "mask_export_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        if self.verbose:
            print(f"📄 Saved export manifest: {manifest_path}")
```

### Module 1.2: Create Export Script

**Location**: `scripts/05_export_embryo_masks.py`

```python
#!/usr/bin/env python3
"""
Export embryo masks from SAM2 annotations as labeled images
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR.parent))

from scripts.utils.mask_export_utils import EmbryoMaskExporter

def main():
    parser = argparse.ArgumentParser(description="Export embryo masks as labeled images")
    parser.add_argument("--annotations", required=True, help="Path to grounded_sam_annotations.json")
    parser.add_argument("--output", required=True, help="Output directory for mask images")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], 
                       help="Output format (default: jpg, recommended: png for label masks)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🎭 Embryo Mask Export")
    print("=" * 40)
    
    if args.format == "jpg":
        print("⚠️  Warning: Using JPEG format for label masks may introduce compression artifacts.")
        print("   Consider using --format png for exact pixel value preservation.")
    
    # Initialize exporter
    exporter = EmbryoMaskExporter(
        sam2_annotations_path=args.annotations,
        output_base_dir=args.output,
        verbose=args.verbose,
        output_format=args.format
    )
    
    # Export all masks
    export_paths = exporter.export_all_masks(max_workers=args.workers)
    
    print(f"\n✅ Exported {len(export_paths)} mask images")
    print(f"📁 Output directory: {args.output}")

if __name__ == "__main__":
    main()
```

---

## Phase 2: GSAM QC Class Module

### Module 2.1: Create `gsam_qc_class.py`

**Location**: `scripts/utils/gsam_qc_class.py`

```python
"""
GSAM Quality Control Class
Analyzes SAM2 annotations for quality issues and flags
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import json
from collections import defaultdict

from base_annotation_parser import BaseAnnotationParser
from embryo_metadata_refactored import EmbryoMetadata

class GSAMQualityControl(BaseAnnotationParser):
    """
    Quality control analysis for GSAM annotations.
    Inherits from BaseAnnotationParser for ID parsing and file operations.
    """
    
    def __init__(self, 
                 gsam_annotations_path: Path,
                 embryo_metadata: EmbryoMetadata,
                 verbose: bool = True):
        """
        Initialize QC analyzer.
        
        Args:
            gsam_annotations_path: Path to grounded_sam_annotations.json
            embryo_metadata: EmbryoMetadata instance to add flags to
            verbose: Enable verbose output
        """
        super().__init__(gsam_annotations_path, verbose)
        
        self.embryo_metadata = embryo_metadata
        self.gsam_data = self.load_json()
        
        # QC parameters (configurable)
        self.params = {
            "area_variance_threshold": 0.10,  # 10% area change threshold
            "area_comparison_frames": 2,      # Compare with ±2 frames
            "edge_distance_threshold": 5,     # Pixels from image edge
            "large_mask_threshold": 0.40,     # 40% of image area
            "min_mask_area": 100             # Minimum valid mask area
        }
        
        # Track QC results
        self.qc_results = {
            "total_snips_analyzed": 0,
            "total_flags_added": 0,
            "flag_counts": defaultdict(int),
            "flagged_entities": defaultdict(set)
        }
    
    def validate_gsam_id_match(self) -> bool:
        """Validate that GSAM ID matches between annotations and metadata."""
        # Extract GSAM ID from annotations (if implemented)
        gsam_id_annotations = self.gsam_data.get("gsam_annotation_id")
        gsam_id_metadata = self.embryo_metadata.results.get("source", {}).get("gsam_annotation_id")
        
        if gsam_id_annotations and gsam_id_metadata:
            if gsam_id_annotations != gsam_id_metadata:
                raise ValueError(f"GSAM ID mismatch: {gsam_id_annotations} != {gsam_id_metadata}")
        
        return True
    
    def run_all_qc_checks(self, author: str = "gsam_qc_auto") -> Dict:
        """
        Run all QC checks and add flags to EmbryoMetadata.
        
        Args:
            author: Author name for flag attribution
            
        Returns:
            Dict with QC summary
        """
        if self.verbose:
            print("🔍 Starting GSAM Quality Control Analysis...")
        
        # Validate GSAM ID match
        self.validate_gsam_id_match()
        
        # Run each QC check
        self._check_segmentation_variability(author)
        self._check_missing_embryos(author)
        self._check_mask_at_border(author)
        self._check_overlapping_masks(author)
        self._check_non_continuous_masks(author)
        self._check_no_embryos_detected(author)
        self._check_large_masks(author)
        
        if self.verbose:
            print(f"\n✅ QC Analysis Complete!")
            print(f"   Total snips analyzed: {self.qc_results['total_snips_analyzed']}")
            print(f"   Total flags added: {self.qc_results['total_flags_added']}")
            print(f"\n📊 Flag Summary:")
            for flag_name, count in self.qc_results['flag_counts'].items():
                print(f"   {flag_name}: {count}")
        
        return self.qc_results
    
    def _check_segmentation_variability(self, author: str):
        """Check for high segmentation area variability between frames."""
        if self.verbose:
            print("\n🔍 Checking segmentation variability...")
        
        threshold = self.params["area_variance_threshold"]
        compare_frames = self.params["area_comparison_frames"]
        
        # Organize data by embryo_id
        embryo_areas = self._organize_by_embryo()
        
        for embryo_id, frame_data in embryo_areas.items():
            # Sort by frame number
            sorted_frames = sorted(frame_data.items(), key=lambda x: self.extract_frame_number(x[0]))
            
            for i, (snip_id, area) in enumerate(sorted_frames):
                # Get comparison areas
                prev_areas = []
                next_areas = []
                
                # Previous frames
                for j in range(max(0, i - compare_frames), i):
                    prev_areas.append(sorted_frames[j][1])
                
                # Next frames
                for j in range(i + 1, min(len(sorted_frames), i + compare_frames + 1)):
                    next_areas.append(sorted_frames[j][1])
                
                # Calculate variance
                if prev_areas or next_areas:
                    comparison_areas = prev_areas + next_areas
                    mean_area = np.mean(comparison_areas)
                    
                    if mean_area > 0:
                        variance = abs(area - mean_area) / mean_area
                        
                        if variance >= threshold:
                            self._add_flag(
                                snip_id, 
                                "HIGH_SEGMENTATION_VARIABILITY",
                                f"Area variance {variance:.2%} exceeds threshold",
                                author,
                                "snip"
                            )
    
    def _check_missing_embryos(self, author: str):
        """Check for embryos that disappear and reappear in videos."""
        if self.verbose:
            print("\n🔍 Checking for missing embryos...")
        
        # Organize by video and embryo
        video_embryo_frames = defaultdict(lambda: defaultdict(set))
        
        for exp_data in self.gsam_data.get("experiments", {}).values():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("images", {}).items():
                    frame_num = self.extract_frame_number(image_id)
                    
                    for embryo_id in image_data.get("embryos", {}):
                        video_embryo_frames[video_id][embryo_id].add(frame_num)
        
        # Check for gaps in frame sequences
        for video_id, embryo_frames in video_embryo_frames.items():
            for embryo_id, frame_set in embryo_frames.items():
                if len(frame_set) > 1:
                    sorted_frames = sorted(frame_set)
                    expected_frames = set(range(sorted_frames[0], sorted_frames[-1] + 1))
                    missing_frames = expected_frames - frame_set
                    
                    if missing_frames:
                        self._add_flag(
                            embryo_id,
                            "MISSING_EMBRYO_IN_VIDEO", 
                            f"Missing in frames: {sorted(missing_frames)}",
                            author,
                            "embryo"
                        )
    
    def _check_mask_at_border(self, author: str):
        """Check for masks too close to image border."""
        if self.verbose:
            print("\n🔍 Checking masks at image borders...")
        
        edge_threshold = self.params["edge_distance_threshold"]
        
        for exp_data in self.gsam_data.get("experiments", {}).values():
            for video_data in exp_data.get("videos", {}).values():
                for image_data in video_data.get("images", {}).values():
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        bbox = embryo_data.get("bbox", [])
                        if len(bbox) == 4:
                            # bbox is [x1, y1, x2, y2] normalized
                            # Get image dimensions from RLE
                            if embryo_data['segmentation_format'] == 'rle':
                                h, w = embryo_data['segmentation']['size']
                                
                                # Convert normalized bbox to pixel coordinates
                                x1 = bbox[0] * w
                                y1 = bbox[1] * h
                                x2 = bbox[2] * w
                                y2 = bbox[3] * h
                                
                                # Check distance to edges
                                if (x1 <= edge_threshold or 
                                    y1 <= edge_threshold or
                                    x2 >= w - edge_threshold or
                                    y2 >= h - edge_threshold):
                                    
                                    snip_id = embryo_data.get("snip_id")
                                    if snip_id:
                                        self._add_flag(
                                            snip_id,
                                            "MASK_AT_BORDER",
                                            f"Within {edge_threshold}px of image edge",
                                            author,
                                            "snip"
                                        )
    
    def _check_overlapping_masks(self, author: str):
        """Check for overlapping masks in same image."""
        if self.verbose:
            print("\n🔍 Checking for overlapping masks...")
        
        # This would require decoding masks and checking pixel overlap
        # For now, using bbox overlap as proxy
        
        for exp_data in self.gsam_data.get("experiments", {}).values():
            for video_data in exp_data.get("videos", {}).values():
                for image_id, image_data in video_data.get("images", {}).items():
                    embryos = image_data.get("embryos", {})
                    
                    if len(embryos) > 1:
                        # Check all pairs for overlap
                        embryo_list = list(embryos.items())
                        
                        for i in range(len(embryo_list)):
                            for j in range(i + 1, len(embryo_list)):
                                embryo1_id, embryo1_data = embryo_list[i]
                                embryo2_id, embryo2_data = embryo_list[j]
                                
                                if self._check_bbox_overlap(embryo1_data.get("bbox", []), 
                                                          embryo2_data.get("bbox", [])):
                                    # Add flag to both snips
                                    for snip_id in [embryo1_data.get("snip_id"), 
                                                   embryo2_data.get("snip_id")]:
                                        if snip_id:
                                            self._add_flag(
                                                snip_id,
                                                "OVERLAPPING_MASK",
                                                f"Overlaps with other embryo",
                                                author,
                                                "snip"
                                            )
    
    def _check_non_continuous_masks(self, author: str):
        """Check for non-continuous (fragmented) masks."""
        # This would require mask analysis - placeholder for now
        pass
    
    def _check_no_embryos_detected(self, author: str):
        """Check for videos with no embryo detections."""
        if self.verbose:
            print("\n🔍 Checking for videos with no embryos...")
        
        for exp_data in self.gsam_data.get("experiments", {}).values():
            for video_id, video_data in exp_data.get("videos", {}).items():
                has_embryos = False
                
                for image_data in video_data.get("images", {}).values():
                    if image_data.get("embryos"):
                        has_embryos = True
                        break
                
                if not has_embryos:
                    self._add_flag(
                        video_id,
                        "NO_EMBRYOS_DETECTED",
                        "No embryo masks in entire video",
                        author,
                        "video"
                    )
    
    def _check_large_masks(self, author: str):
        """Check for unusually large masks."""
        if self.verbose:
            print("\n🔍 Checking for large masks...")
        
        threshold = self.params["large_mask_threshold"]
        
        for exp_data in self.gsam_data.get("experiments", {}).values():
            for video_data in exp_data.get("videos", {}).values():
                for image_data in video_data.get("images", {}).values():
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        # Get image dimensions
                        if embryo_data['segmentation_format'] == 'rle':
                            h, w = embryo_data['segmentation']['size']
                            image_area = h * w
                            
                            mask_area = embryo_data.get("area", 0)
                            if mask_area > threshold * image_area:
                                snip_id = embryo_data.get("snip_id")
                                if snip_id:
                                    self._add_flag(
                                        snip_id,
                                        "LARGE_MASK",
                                        f"Mask covers {mask_area/image_area:.1%} of image",
                                        author,
                                        "snip"
                                    )
    
    # Helper methods
    
    def _organize_by_embryo(self) -> Dict[str, Dict[str, float]]:
        """Organize mask areas by embryo_id."""
        embryo_areas = defaultdict(dict)
        
        for exp_data in self.gsam_data.get("experiments", {}).values():
            for video_data in exp_data.get("videos", {}).values():
                for image_data in video_data.get("images", {}).values():
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        area = embryo_data.get("area", 0)
                        
                        if snip_id:
                            embryo_areas[embryo_id][snip_id] = area
                            self.qc_results["total_snips_analyzed"] += 1
        
        return embryo_areas
    
    def _check_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bboxes overlap (normalized coordinates)."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return False
        
        # Check for overlap
        return not (bbox1[2] < bbox2[0] or  # box1 right < box2 left
                   bbox2[2] < bbox1[0] or   # box2 right < box1 left
                   bbox1[3] < bbox2[1] or   # box1 bottom < box2 top
                   bbox2[3] < bbox1[1])     # box2 bottom < box1 top
    
    def _add_flag(self, entity_id: str, flag_name: str, notes: str, 
                  author: str, level: str):
        """Add flag to EmbryoMetadata."""
        try:
            if level == "snip":
                self.embryo_metadata.add_flag_to_snip(
                    snip_id=entity_id,
                    flag=flag_name,
                    author=author,
                    notes=notes
                )
            elif level == "embryo":
                self.embryo_metadata.add_flag_to_embryo(
                    embryo_id=entity_id,
                    flag=flag_name,
                    author=author,
                    notes=notes
                )
            elif level == "video":
                self.embryo_metadata.add_flag_to_video(
                    video_id=entity_id,
                    flag=flag_name,
                    author=author,
                    notes=notes
                )
            
            self.qc_results["total_flags_added"] += 1
            self.qc_results["flag_counts"][flag_name] += 1
            self.qc_results["flagged_entities"][level].add(entity_id)
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Failed to add flag {flag_name} to {entity_id}: {e}")
```

### Module 2.2: Update Permitted Values Schema

Add these QC flags to `config/permitted_values_schema.json`:

```json
{
  "flags": {
    "snip_level": {
      "HIGH_SEGMENTATION_VARIABILITY": {
        "description": "Mask area variance >10% compared to adjacent frames",
        "severity": "warning"
      },
      "MASK_AT_BORDER": {
        "description": "Mask within 5 pixels of image edge",
        "severity": "warning"
      },
      "OVERLAPPING_MASK": {
        "description": "Mask overlaps with another embryo mask",
        "severity": "warning"
      },
      "NON_CONTINUOUS_MASK": {
        "description": "Mask is fragmented or non-continuous",
        "severity": "warning"
      },
      "LARGE_MASK": {
        "description": "Mask covers >40% of image area",
        "severity": "warning"
      }
    },
    "embryo_level": {
      "MISSING_EMBRYO_IN_VIDEO": {
        "description": "Embryo missing in some frames of video",
        "severity": "error"
      }
    },
    "video_level": {
      "NO_EMBRYOS_DETECTED": {
        "description": "No embryo masks detected in entire video",
        "severity": "error"
      }
    }
  }
}
```

---

## Phase 3: Integration Updates

### Module 3.1: Update SAM2 Annotations to Track Mask Paths

Add to `sam2_utils.py` in the `GroundedSamAnnotations` class:

```python
def update_with_mask_paths(self, mask_export_paths: Dict[str, Path]):
    """
    Update SAM2 annotations with exported mask file paths.
    
    Args:
        mask_export_paths: Dict of image_id -> mask file path
    """
    for exp_data in self.results.get("experiments", {}).values():
        for video_data in exp_data.get("videos", {}).values():
            for image_id, image_data in video_data.get("images", {}).items():
                if image_id in mask_export_paths:
                    image_data["mask_export_path"] = str(mask_export_paths[image_id])
                    
                    # Also update each embryo's snip_id entry
                    for embryo_data in image_data.get("embryos", {}).values():
                        embryo_data["mask_export_path"] = str(mask_export_paths[image_id])
    
    self._unsaved_changes = True
    if self.verbose:
        print(f"✅ Updated {len(mask_export_paths)} images with mask export paths")
```

### Module 3.2: Update EmbryoMetadata to Track Mask Paths

Add to `embryo_metadata_refactored.py`:

```python
def add_mask_paths_from_sam(self, sam_annotations: Dict):
    """
    Add mask export paths from SAM annotations to snip metadata.
    
    Args:
        sam_annotations: SAM2 annotations dict with mask_export_path info
    """
    updated_count = 0
    
    for exp_data in sam_annotations.get("experiments", {}).values():
        for video_data in exp_data.get("videos", {}).values():
            for image_id, image_data in video_data.get("images", {}).items():
                mask_path = image_data.get("mask_export_path")
                
                if mask_path:
                    # Update each embryo's snip_id with mask path
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        
                        if snip_id and snip_id in self.results["snips"]:
                            if "metadata" not in self.results["snips"][snip_id]:
                                self.results["snips"][snip_id]["metadata"] = {}
                            
                            self.results["snips"][snip_id]["metadata"]["mask_export_path"] = mask_path
                            updated_count += 1
    
    self._unsaved_changes = True
    if self.verbose:
        print(f"✅ Updated {updated_count} snips with mask export paths")
```

### Module 3.3: Complete Pipeline Script

**Location**: `scripts/06_complete_pipeline.py`

```python
#!/usr/bin/env python3
"""
Complete Pipeline: Export Masks and Run QC
Combines mask export, QC analysis, and metadata updates
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR.parent))

from scripts.utils.mask_export_utils import EmbryoMaskExporter
from scripts.utils.gsam_qc_class import GSAMQualityControl
from scripts.utils.embryo_metadata_refactored import EmbryoMetadata
from scripts.utils.sam2_utils import GroundedSamAnnotations

def main():
    parser = argparse.ArgumentParser(description="Complete pipeline: export masks and run QC")
    parser.add_argument("--sam-annotations", required=True, help="Path to grounded_sam_annotations.json")
    parser.add_argument("--embryo-metadata", required=True, help="Path to embryo_metadata.json")
    parser.add_argument("--mask-output", required=True, help="Output directory for mask images")
    parser.add_argument("--mask-format", default="jpg", choices=["jpg", "png"],
                       help="Mask output format (default: jpg)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--run-qc", action="store_true", help="Run QC analysis after export")
    parser.add_argument("--qc-author", default="gsam_qc_auto", help="Author name for QC flags")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🎯 Complete Embryo Mask Pipeline")
    print("=" * 50)
    
    # Step 1: Export masks
    print("\n📦 Step 1: Exporting embryo masks...")
    exporter = EmbryoMaskExporter(
        sam2_annotations_path=args.sam_annotations,
        output_base_dir=args.mask_output,
        verbose=args.verbose,
        output_format=args.mask_format
    )
    
    export_paths = exporter.export_all_masks(max_workers=args.workers)
    print(f"✅ Exported {len(export_paths)} mask images")
    
    # Step 2: Update SAM annotations with mask paths
    print("\n📝 Step 2: Updating SAM annotations with mask paths...")
    sam_annotations = GroundedSamAnnotations(
        filepath=args.sam_annotations,
        seed_annotations_path=None,  # Not needed for updates
        experiment_metadata_path=None,
        verbose=args.verbose
    )
    
    sam_annotations.update_with_mask_paths(export_paths)
    sam_annotations.save()
    
    # Step 3: Load/Initialize EmbryoMetadata
    print("\n📊 Step 3: Loading EmbryoMetadata...")
    embryo_metadata = EmbryoMetadata(
        sam_annotation_path=args.sam_annotations,
        embryo_metadata_path=args.embryo_metadata,
        gen_if_no_file=True,
        verbose=args.verbose
    )
    
    # Step 4: Update EmbryoMetadata with mask paths
    print("\n📝 Step 4: Updating EmbryoMetadata with mask paths...")
    with open(args.sam_annotations, 'r') as f:
        import json
        sam_data = json.load(f)
    
    embryo_metadata.add_mask_paths_from_sam(sam_data)
    
    # Step 5: Run QC if requested
    if args.run_qc:
        print("\n🔍 Step 5: Running QC analysis...")
        qc_analyzer = GSAMQualityControl(
            gsam_annotations_path=args.sam_annotations,
            embryo_metadata=embryo_metadata,
            verbose=args.verbose
        )
        
        qc_results = qc_analyzer.run_all_qc_checks(author=args.qc_author)
        
        # Save updated metadata with QC flags
        embryo_metadata.save()
        print("✅ QC analysis complete and flags added to metadata")
    
    print("\n🎉 Pipeline complete!")
    print(f"📁 Mask images: {args.mask_output}")
    print(f"📊 Updated metadata: {args.embryo_metadata}")

if __name__ == "__main__":
    main()
```

---

## Implementation Order and Timeline

### Day 1: Priority 1 - Mask Export
1. **Hour 1-2**: Create `mask_export_utils.py`
   - Implement `EmbryoMaskExporter` class
   - Test RLE decoding functionality
   
2. **Hour 3**: Create `05_export_embryo_masks.py`
   - Test with small batch of images
   - Verify output format correct

3. **Hour 4**: Run full export
   - Monitor for overlapping masks
   - Verify file naming convention

### Day 2: Priority 2 - QC Implementation
1. **Hour 1-2**: Create `gsam_qc_class.py`
   - Implement base class structure
   - Add segmentation variability check
   
2. **Hour 3-4**: Implement remaining QC checks
   - Test each check individually
   - Verify flag creation

3. **Hour 5**: Update permitted values schema
   - Add new QC flags
   - Test with EmbryoMetadata

### Day 3: Integration
1. **Hour 1-2**: Update SAM2 and EmbryoMetadata classes
   - Add mask path tracking
   - Test path updates
   
2. **Hour 3-4**: Create complete pipeline script
   - Test end-to-end workflow
   - Verify all components work together

---

## Testing Strategy

### Unit Tests

1. **Test mask export**:
```bash
# Test single image with JPEG output (as requested)
python scripts/05_export_embryo_masks.py \
  --annotations test_data/sam_annotations.json \
  --output test_output/masks \
  --format jpg \
  --verbose

# Or use PNG for lossless label masks (recommended)
python scripts/05_export_embryo_masks.py \
  --annotations test_data/sam_annotations.json \
  --output test_output/masks \
  --format png \
  --verbose

# Verify output (checks both jpg and png)
ls test_output/masks/*/masks/*_masks_emnum_*.*
```

2. **Test QC checks**:
```python
# Test individual QC check
qc = GSAMQualityControl(sam_path, embryo_metadata)
qc._check_segmentation_variability("test_author")
```

### Integration Tests

```bash
# Full pipeline test
python scripts/06_complete_pipeline.py \
  --sam-annotations data/sam2_annotations.json \
  --embryo-metadata data/embryo_metadata.json \
  --mask-output data/embryo_masks \
  --run-qc \
  --verbose
```

---

## Expected Output Structure

```
data/
├── embryo_masks/
│   ├── 20240411/
│   │   └── masks/
│   │       ├── 20240411_A01_0000_masks_emnum_3.jpg
│   │       ├── 20240411_A01_0001_masks_emnum_3.jpg
│   │       └── ...
│   ├── 20240412/
│   │   └── masks/
│   │       └── ...
│   └── mask_export_manifest.json
├── sam2_annotations.json (updated with mask paths)
└── embryo_metadata.json (updated with QC flags and mask paths)
```

---

## Optimization Notes

1. **Parallel Processing**: Use ThreadPoolExecutor for mask export
2. **Memory Management**: Process images in batches to avoid memory issues
3. **Error Handling**: Continue processing even if individual images fail
4. **Progress Tracking**: Show progress every 100 images

---

## Future Enhancements

1. **Non-continuous mask detection**: Implement connected component analysis
2. **Mask overlap quantification**: Calculate exact pixel overlap percentage
3. **Temporal consistency**: Track mask shape changes over time
4. **Visualization tools**: Create overlay images showing QC issues
5. **GSAM ID integration**: Add unique GSAM annotation ID tracking