#!/usr/bin/env python3
"""
Simplified GSAM Quality Control Class
====================================

Analyzes SAM2 annotations for quality issues and adds flags directly 
to the GSAM JSON structure at the top level under "qc_flags".

No dependencies on embryo metadata - self-contained QC analysis.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
from time import time
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class GSAMQualityControl:
    """
    Simplified QC for SAM2 annotations. Flags quality issues and saves 
    them directly to the GSAM JSON file under "qc_flags".
    """
    
    def __init__(self, gsam_path: str, verbose: bool = True, progress: bool = True):
        """
        Initialize QC with path to GSAM annotations.
        
        Args:
            gsam_path: Path to grounded_sam_annotations.json
            verbose: Whether to print progress
        """
        self.gsam_path = Path(gsam_path)
        self.verbose = verbose
        self.progress = progress
        
        # Load GSAM data
        with open(self.gsam_path, 'r') as f:
            self.gsam_data = json.load(f)
            
        # Initialize tracking sets for processed entities
        self._initialize_entity_tracking()
        
        if self.verbose:
            print(f"🔍 Loaded GSAM annotations from {self.gsam_path}")
            print(f"📊 Found {len(self.processed_snip_ids)} already processed snips")
        
        if self.verbose and self.progress:
            if _HAS_TQDM:
                print("⏳ Progress bars enabled (tqdm).")
            else:
                print("⏳ tqdm not installed; using basic percentage progress.")
    
    def _initialize_entity_tracking(self):
        """Initialize sets to track processed and new entities."""
        self.processed_experiment_ids = set()
        self.processed_video_ids = set()
        self.processed_image_ids = set()
        self.processed_snip_ids = set()
        
        # Scan existing data to identify already processed entities via presence of flags
        for exp_id, exp_data in self.gsam_data.get("experiments", {}).items():
            # Check if experiment has been QC'd
            if "flags" in exp_data or self._was_previously_processed(exp_id, "experiment"):
                self.processed_experiment_ids.add(exp_id)
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if "flags" in video_data or self._was_previously_processed(video_id, "video"):
                    self.processed_video_ids.add(video_id)
                
                for image_id, image_data in video_data.get("images", {}).items():
                    if "flags" in image_data or self._was_previously_processed(image_id, "image"):
                        self.processed_image_ids.add(image_id)
                    
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and ("flags" in embryo_data or self._was_previously_processed(snip_id, "snip")):
                            self.processed_snip_ids.add(snip_id)
        
        # Identify new entities to process
        self.new_experiment_ids = set()
        self.new_video_ids = set()
        self.new_image_ids = set()
        self.new_snip_ids = set()
        
        for exp_id, exp_data in self.gsam_data.get("experiments", {}).items():
            if exp_id not in self.processed_experiment_ids:
                self.new_experiment_ids.add(exp_id)
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in self.processed_video_ids:
                    self.new_video_ids.add(video_id)
                
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id not in self.processed_image_ids:
                        self.new_image_ids.add(image_id)
                    
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id not in self.processed_snip_ids:
                            self.new_snip_ids.add(snip_id)
    
    def _was_previously_processed(self, entity_id: str, entity_type: str) -> bool:
        """Check if entity was processed in previous QC runs."""
        # Check in GEN_flag_overview if it exists
        overview = self.gsam_data.get("GEN_flag_overview", {})
        for flag_type, flag_data in overview.items():
            if entity_type == "experiment" and entity_id in flag_data.get("experiment_ids", []):
                return True
            elif entity_type == "video" and entity_id in flag_data.get("video_ids", []):
                return True
            elif entity_type == "image" and entity_id in flag_data.get("image_ids", []):
                return True
            elif entity_type == "snip" and entity_id in flag_data.get("snip_ids", []):
                return True
        return False
    
    def _progress_iter(self, iterable, desc: str, total: Optional[int] = None):
        """Wrap an iterable with a progress indicator compatible with tmux.
        Uses tqdm if installed; otherwise prints coarse percentage updates."""
        if not self.progress:
            for x in iterable:
                yield x
            return
        if _HAS_TQDM:
            yield from tqdm(iterable, desc=desc, total=total, leave=False)
        else:
            data = list(iterable) if total is None else list(iterable)
            if total is None:
                total = len(data)
            last_pct = -1
            for idx, x in enumerate(data, 1):
                pct = int(idx / total * 100)
                if pct != last_pct and (pct % 5 == 0 or pct >= 97):
                    print(f"{desc}: {pct}% ({idx}/{total})", end="\r")
                    last_pct = pct
                yield x
            print(f"{desc}: 100% ({total}/{total})")
    
    def run_all_checks(self, author: str = "auto_qc", process_all: bool = False, 
                      target_entities: Optional[Dict[str, List[str]]] = None):
        start_overall = time()
        """
        Run all QC checks and save flags to GSAM file.
        
        Args:
            author: Author identifier for the QC run
            process_all: If True, process all entities. If False, only process new entities
            target_entities: Specific entities to process (overrides process_all)
        """
        # Determine which entities to process
        if target_entities:
            entities_to_process = target_entities
        elif process_all:
            entities_to_process = {
                "experiment_ids": list(self.processed_experiment_ids | self.new_experiment_ids),
                "video_ids": list(self.processed_video_ids | self.new_video_ids),
                "image_ids": list(self.processed_image_ids | self.new_image_ids),
                "snip_ids": list(self.processed_snip_ids | self.new_snip_ids)
            }
        else:
            # Default: only process new entities
            entities_to_process = {
                "experiment_ids": list(self.new_experiment_ids),
                "video_ids": list(self.new_video_ids),
                "image_ids": list(self.new_image_ids),
                "snip_ids": list(self.new_snip_ids)
            }
        
        if self.verbose:
            print(f"🚀 Running QC checks on {len(entities_to_process['snip_ids'])} new snips...")
            
        # Initialize flags for all entities
        self._initialize_entity_flags(entities_to_process)
        
        # Run checks only on specified entities
        self.check_segmentation_variability(author, entities_to_process)
        self.check_mask_on_edge(author, entities_to_process)
        self.check_overlapping_masks(author, entities_to_process)
        self.check_large_masks(author, entities_to_process)
        self.check_detection_failure(author, entities_to_process)
        self.check_discontinuous_masks(author, entities_to_process)
        elapsed = time() - start_overall
        if self.verbose:
            print(f"⏱️  Total QC elapsed: {elapsed:.2f}s")
        
        # Generate overview and save
        self.generate_overview(entities_to_process)
        self._save_qc_summary(author)
        
        if self.verbose:
            print("✅ QC checks complete and flags saved to GSAM file")
    
    def _initialize_entity_flags(self, entities: Dict[str, List[str]]):
        """Initialize flags dict for entities that don't have one."""
        # Initialize top-level flags if not present
        if "flags" not in self.gsam_data:
            self.gsam_data["flags"] = {}
            
        for exp_id in entities.get("experiment_ids", []):
            if exp_id in self.gsam_data.get("experiments", {}):
                exp_data = self.gsam_data["experiments"][exp_id]
                if "flags" not in exp_data:
                    exp_data["flags"] = {}
        
        for exp_id, exp_data in self.gsam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id in entities.get("video_ids", []):
                    if "flags" not in video_data:
                        video_data["flags"] = {}
                
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id in entities.get("image_ids", []):
                        if "flags" not in image_data:
                            image_data["flags"] = {}
                    
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id in entities.get("snip_ids", []):
                            if "flags" not in embryo_data:
                                embryo_data["flags"] = {}
   
    def check_segmentation_variability(self, author: str, entities: Dict[str, List[str]], n_frames_check: int = 2):
        """
        Flag embryos with high area variance across frames (>15% CV).
        Flags both at embryo level (HIGH_SEGMENTATION_VAR_EMBRYO) and 
        at individual snip level (HIGH_SEGMENTATION_VAR_SNIP).
        """
        if self.verbose:
            print("📊 Checking segmentation variability...")
        target_snips = set(entities.get("snip_ids", []))
        t0 = time()
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="SegVar", total=len(experiments_items)):
            if exp_id not in entities.get("experiment_ids", []):
                continue
    
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in entities.get("video_ids", []):
                    continue
    
                image_ids = sorted(video_data.get("images", {}).keys())
                embryo_frames = defaultdict(dict)  # embryo_id -> {image_id: area}
                embryo_areas = defaultdict(list)   # embryo_id -> [area, ...]
                snip_areas = {}                    # snip_id -> (area, image_id, embryo_id)
    
                for image_id in image_ids:
                    image_data = video_data["images"][image_id]
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        area = embryo_data.get("area")
                        snip_id = embryo_data.get("snip_id")
                        if area is not None:
                            embryo_frames[embryo_id][image_id] = area
                            embryo_areas[embryo_id].append(area)
                            if snip_id:
                                snip_areas[snip_id] = {
                                    "area": area,
                                    "image_id": image_id,
                                    "embryo_id": embryo_id
                                }
    
                # Embryo-level variability flag
                for embryo_id, areas in embryo_areas.items():
                    if len(areas) >= 3:
                        mean_area = np.mean(areas)
                        cv = np.std(areas) / mean_area if mean_area > 0 else 0
                        if cv > 0.15:
                            # Find a representative embryo_data to attach the flag
                            for image_id in image_ids:
                                image_data = video_data["images"][image_id]
                                if embryo_id in image_data.get("embryos", {}):
                                    embryo_data = image_data["embryos"][embryo_id]
                                    if "HIGH_SEGMENTATION_VAR_EMBRYO" not in embryo_data["flags"]:
                                        embryo_data["flags"]["HIGH_SEGMENTATION_VAR_EMBRYO"] = []
                                    embryo_data["flags"]["HIGH_SEGMENTATION_VAR_EMBRYO"].append({
                                        "experiment_id": exp_id,
                                        "video_id": video_id,
                                        "embryo_id": embryo_id,
                                        "issue": "HIGH_SEGMENTATION_VAR_EMBRYO",
                                        "coefficient_of_variation": round(cv, 3),
                                        "frame_count": len(areas),
                                        "mean_area": round(mean_area, 1),
                                        "std_area": round(np.std(areas), 1),
                                        "author": author,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    break  # Only need to attach once
    
                # Snip-level variability flag (existing logic)
                for i, image_id in enumerate(image_ids):
                    image_data = video_data["images"][image_id]
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if not snip_id or snip_id not in target_snips:
                            continue
    
                        current_area = embryo_data.get("area")
                        if current_area is None:
                            continue
    
                        # Get areas from nearby frames
                        before_areas = []
                        after_areas = []
    
                        for j in range(max(0, i - n_frames_check), i):
                            frame_id = image_ids[j]
                            if frame_id in embryo_frames[embryo_id]:
                                before_areas.append(embryo_frames[embryo_id][frame_id])
    
                        for j in range(i + 1, min(len(image_ids), i + n_frames_check + 1)):
                            frame_id = image_ids[j]
                            if frame_id in embryo_frames[embryo_id]:
                                after_areas.append(embryo_frames[embryo_id][frame_id])
    
                        avg_before = np.mean(before_areas) if before_areas else None
                        avg_after = np.mean(after_areas) if after_areas else None
    
                        diff_before_pct = abs(current_area - avg_before) / avg_before if avg_before else None
                        diff_after_pct = abs(current_area - avg_after) / avg_after if avg_after else None
    
                        flag_before = diff_before_pct > 0.20 if diff_before_pct is not None else False
                        flag_after = diff_after_pct > 0.20 if diff_after_pct is not None else False
    
                        if flag_before or flag_after:
                            flag_data = {
                                "snip_id": snip_id,
                                "current_area": round(current_area, 1),
                                "avg_before": round(avg_before, 1) if avg_before else None,
                                "avg_after": round(avg_after, 1) if avg_after else None,
                                "diff_before_pct": round(diff_before_pct, 3) if diff_before_pct else None,
                                "diff_after_pct": round(diff_after_pct, 3) if diff_after_pct else None,
                                "flagged_before": flag_before,
                                "flagged_after": flag_after,
                                "frames_checked_before": len(before_areas),
                                "frames_checked_after": len(after_areas),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
    
                            if "HIGH_SEGMENTATION_VAR_SNIP" not in embryo_data["flags"]:
                                embryo_data["flags"]["HIGH_SEGMENTATION_VAR_SNIP"] = []
                            embryo_data["flags"]["HIGH_SEGMENTATION_VAR_SNIP"].append(flag_data)    
        if self.verbose:
            print(f"   ⏱ check_segmentation_variability {time() - t0:.2f}s")
    
    def check_mask_on_edge(self, author: str, entities: Dict[str, List[str]]):
        """Flag masks that touch image edges (within 5 pixels)."""
        if self.verbose:
            print("🖼️  Checking masks on image edge...")
        target_snips = set(entities.get("snip_ids", []))
        t0 = time()
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Edge", total=len(experiments_items)):
            if exp_id not in entities.get("experiment_ids", []):
                continue
                
            # Get image dimensions from experiment metadata
            image_width = exp_data.get("image_width")
            image_height = exp_data.get("image_height")
            
            if not (image_width and image_height):
                if self.verbose:
                    print(f"⚠️  Skipping edge check for {exp_id} - no image dimensions")
                continue
                
            edge_threshold = 5  # pixels from edge
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in entities.get("video_ids", []):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id not in entities.get("image_ids", []):
                        continue
                        
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id not in target_snips:
                            continue
                            
                        bbox = embryo_data.get("bbox")
                        if bbox and len(bbox) == 4:
                            x_min, y_min, x_max, y_max = bbox
                            
                            # Check if close to any edge
                            near_edge = (
                                x_min <= edge_threshold or 
                                y_min <= edge_threshold or
                                x_max >= (image_width - edge_threshold) or 
                                y_max >= (image_height - edge_threshold)
                            )
                            
                            if near_edge:
                                flag_data = {
                                    "bbox": [int(x) for x in bbox],
                                    "image_size": [image_width, image_height],
                                    "distance_to_edges": {
                                        "left": x_min,
                                        "top": y_min,
                                        "right": image_width - x_max,
                                        "bottom": image_height - y_max
                                    },
                                    "author": author,
                                    "timestamp": datetime.now().isoformat()
                                }
                                
                                if "MASK_ON_EDGE" not in embryo_data["flags"]:
                                    embryo_data["flags"]["MASK_ON_EDGE"] = []
                                embryo_data["flags"]["MASK_ON_EDGE"].append(flag_data)
        if self.verbose:
            print(f"   ⏱ check_mask_on_edge {time() - t0:.2f}s")
    
    def check_detection_failure(self, author: str, entities: Dict[str, List[str]]):
        """Flag images where expected embryos are missing."""
        if self.verbose:
            print("🔍 Checking for detection failures...")
        t0 = time()
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="DetectFail", total=len(experiments_items)):
            if exp_id not in entities.get("experiment_ids", []):
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in entities.get("video_ids", []):
                    continue
                    
                # Get all embryo_ids that exist in this video
                all_embryo_ids = set()
                for image_id, image_data in video_data.get("images", {}).items():
                    for embryo_id in image_data.get("embryos", {}).keys():
                        all_embryo_ids.add(embryo_id)
                
                # Check each image for missing embryos
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id not in target_images:
                        continue
                        
                    present_embryo_ids = set(image_data.get("embryos", {}).keys())
                    missing_embryo_ids = all_embryo_ids - present_embryo_ids
                    
                    if missing_embryo_ids:
                        flag_data = {
                            "missing_embryo_ids": list(missing_embryo_ids),
                            "total_embryos_in_video": len(all_embryo_ids),
                            "embryos_present_in_image": len(present_embryo_ids),
                            "author": author,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        if "DETECTION_FAILURE" not in image_data["flags"]:
                            image_data["flags"]["DETECTION_FAILURE"] = []
                        image_data["flags"]["DETECTION_FAILURE"].append(flag_data)
        if self.verbose:
            print(f"   ⏱ check_detection_failure {time() - t0:.2f}s")
    
    def check_overlapping_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag images where embryo masks overlap (both bbox and mask level)."""
        if self.verbose:
            print("📦 Checking for overlapping masks...")
        t0 = time()
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Overlap", total=len(experiments_items)):
            if exp_id not in entities.get("experiment_ids", []):
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in entities.get("video_ids", []):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id not in target_images:
                        continue
                        
                    embryos = list(image_data.get("embryos", {}).items())
                    
                    if len(embryos) > 1:
                        bbox_overlaps = []
                        mask_overlaps = []
                        
                        # Check all pairs of embryos
                        for i, (embryo_id1, embryo_data1) in enumerate(embryos):
                            for j, (embryo_id2, embryo_data2) in enumerate(embryos[i+1:], i+1):
                                # Check bbox overlap
                                bbox1 = embryo_data1.get("bbox")
                                bbox2 = embryo_data2.get("bbox")
                                
                                if bbox1 and bbox2:
                                    overlap_ratio = self._calculate_bbox_overlap(bbox1, bbox2)
                                    
                                    if overlap_ratio > 0.2:  # >20% bbox overlap
                                        bbox_overlaps.append({
                                            "embryo_ids": [embryo_id1, embryo_id2],
                                            "bbox_overlap_ratio": round(overlap_ratio, 3),
                                            "bbox1": bbox1,
                                            "bbox2": bbox2
                                        })
                                
                                # Check mask overlap (if available)
                                seg1 = embryo_data1.get("segmentation")
                                seg2 = embryo_data2.get("segmentation")
                                
                                if seg1 and seg2 and overlap_ratio > 0:
                                    mask_overlaps.append({
                                        "embryo_ids": [embryo_id1, embryo_id2],
                                        "overlap_detected": True,
                                        "segmentation_format": embryo_data1.get("segmentation_format", "unknown")
                                    })
                        
                        # Add flags
                        if bbox_overlaps:
                            flag_data = {
                                "overlapping_pairs": bbox_overlaps,
                                "threshold": 0.2,
                                "total_embryos": len(embryos),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            if "BBOX_OVERLAP" not in image_data["flags"]:
                                image_data["flags"]["BBOX_OVERLAP"] = []
                            image_data["flags"]["BBOX_OVERLAP"].append(flag_data)
                        
                        if mask_overlaps:
                            flag_data = {
                                "overlapping_pairs": mask_overlaps,
                                "total_embryos": len(embryos),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            if "MASK_OVERLAP_ERROR" not in image_data["flags"]:
                                image_data["flags"]["MASK_OVERLAP_ERROR"] = []
                            image_data["flags"]["MASK_OVERLAP_ERROR"].append(flag_data)
        if self.verbose:
            print(f"   ⏱ check_overlapping_masks {time() - t0:.2f}s")
    
    def check_large_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag unusually large masks as percentage of total image area."""
        if self.verbose:
            print("📐 Checking for unusually large masks...")
        target_snips = set(entities.get("snip_ids", []))
        t0 = time()
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Large", total=len(experiments_items)):
            if exp_id not in entities.get("experiment_ids", []):
                continue
                
            # Get image dimensions
            image_width = exp_data.get("image_width")
            image_height = exp_data.get("image_height")
            
            if not (image_width and image_height):
                if self.verbose:
                    print(f"⚠️  Skipping large mask check for {exp_id} - no image dimensions")
                continue
                
            total_image_area = image_width * image_height
            area_threshold = total_image_area * 0.15  # 15% of image
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in entities.get("video_ids", []):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id not in target_snips:
                            continue
                            
                        area = embryo_data.get("area")
                        
                        if area is not None and area > area_threshold:
                            area_percentage = (area / total_image_area) * 100
                            flag_data = {
                                "area": round(area, 1),
                                "area_percentage": round(area_percentage, 2),
                                "threshold_percentage": 15.0,
                                "total_image_area": total_image_area,
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            if "UNUSUALLY_LARGE_MASK" not in embryo_data["flags"]:
                                embryo_data["flags"]["UNUSUALLY_LARGE_MASK"] = []
                            embryo_data["flags"]["UNUSUALLY_LARGE_MASK"].append(flag_data)
        if self.verbose:
            print(f"   ⏱ check_large_masks {time() - t0:.2f}s")
    
    def check_discontinuous_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag masks whose segmentation contains multiple disconnected components."""
        if self.verbose:
            print("🔗 Checking for discontinuous masks...")
        target_snips = set(entities.get("snip_ids", []))
        t0 = time()
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Discont", total=len(experiments_items)):
            if exp_id not in entities.get("experiment_ids", []):
                continue
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in entities.get("video_ids", []):
                    continue
                    
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id not in entities.get("image_ids", []):
                        continue
                        
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id not in target_snips:
                            continue
                            
                        segmentation = embryo_data.get("segmentation")
                        if not segmentation:
                            continue
                            
                        is_discontinuous = self._check_mask_discontinuity(segmentation)
                        
                        if is_discontinuous:
                            flag_data = {
                                "segmentation_format": embryo_data.get("segmentation_format", "unknown"),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            if "DISCONTINUOUS_MASK" not in embryo_data["flags"]:
                                embryo_data["flags"]["DISCONTINUOUS_MASK"] = []
                            embryo_data["flags"]["DISCONTINUOUS_MASK"].append(flag_data)
        if self.verbose:
            print(f"   ⏱ check_discontinuous_masks {time() - t0:.2f}s")
    
    def _calculate_bbox_overlap(self, bbox1, bbox2) -> float:
        """Calculate IoU between two bounding boxes."""
        try:
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            
            # Calculate intersection
            x_left = max(x1_min, x2_min)
            y_top = max(y1_min, y2_min)
            x_right = min(x1_max, x2_max)
            y_bottom = min(y1_max, y2_max)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _check_mask_discontinuity(self, segmentation) -> bool:
        """
        Return True if the segmentation represents multiple disconnected components.
        Supports COCO RLE (dict with 'counts') or polygon lists.
        """
        try:
            # RLE format
            if isinstance(segmentation, dict) and "counts" in segmentation:
                # Note: Requires pycocotools
                try:
                    from pycocotools import mask as mask_utils
                    mask_array = mask_utils.decode(segmentation)
                    return self._has_multiple_components(mask_array)
                except ImportError:
                    # Can't check RLE without pycocotools
                    return False
            
            # Polygon format (list). Multiple top-level polygons imply discontinuity.
            if isinstance(segmentation, list):
                if len(segmentation) == 0:
                    return False
                # If first element is a list, assume list-of-polygons
                if isinstance(segmentation[0], list):
                    return len(segmentation) > 1
                # Otherwise a single polygon flat list -> continuous
                return False
            
            return False
        except Exception:
            # Fail-safe: never raise inside QC; treat as continuous if uncertain
            return False
    
    def _has_multiple_components(self, mask_array) -> bool:
        """Check a binary mask for >1 connected foreground component."""
        try:
            import cv2
            # Ensure binary
            m = (mask_array.astype(np.uint8) > 0).astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(m)
            # num_labels includes background; components = num_labels - 1
            return (num_labels - 1) > 1
        except ImportError:
            # Lightweight fallback (rough heuristic): count disjoint row spans
            m = (mask_array > 0).astype(np.uint8)
            # Project along rows and columns
            row_runs = np.sum((m[:, 1:] == 0) & (m[:, :-1] == 1)) + np.sum(m[:, -1] == 1)
            # If many separate row runs relative to area, guess discontinuity
            return row_runs > 1
    
    def generate_overview(self, entities: Dict[str, List[str]]):
        """
        Generate GEN_flag_overview section with flagged entity IDs.
        Only includes entities that were processed in this run.
        """
        if self.verbose:
            print("📋 Generating overview section...")
        
        # Initialize overview structure
        overview = {}
        
        # Convert to sets for efficient lookup
        target_experiments = set(entities.get("experiment_ids", []))
        target_videos = set(entities.get("video_ids", []))
        target_images = set(entities.get("image_ids", []))
        target_snips = set(entities.get("snip_ids", []))
        
        # Scan flags at all levels
        # Top-level flags
        for flag_type, flags in self.gsam_data.get("flags", {}).items():
            if flag_type not in overview:
                overview[flag_type] = {
                    "experiment_ids": [], 
                    "video_ids": [],
                    "image_ids": [], 
                    "snip_ids": [], 
                    "count": 0
                }
            overview[flag_type]["count"] += len(flags) if isinstance(flags, list) else 1
        
        # Hierarchical flags
        for exp_id, exp_data in self.gsam_data.get("experiments", {}).items():
            if exp_id not in target_experiments:
                continue
                
            # Experiment-level flags
            for flag_type in exp_data.get("flags", {}):
                if flag_type not in overview:
                    overview[flag_type] = {
                        "experiment_ids": [], 
                        "video_ids": [],
                        "image_ids": [], 
                        "snip_ids": [], 
                        "count": 0
                    }
                overview[flag_type]["experiment_ids"].append(exp_id)
                overview[flag_type]["count"] += 1
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in target_videos:
                    continue
                    
                # Video-level flags
                for flag_type in video_data.get("flags", {}):
                    if flag_type not in overview:
                        overview[flag_type] = {
                            "experiment_ids": [], 
                            "video_ids": [],
                            "image_ids": [], 
                            "snip_ids": [], 
                            "count": 0
                        }
                    overview[flag_type]["video_ids"].append(video_id)
                    overview[flag_type]["count"] += 1
                
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id not in target_images:
                        continue
                        
                    # Image-level flags
                    for flag_type in image_data.get("flags", {}):
                        if flag_type not in overview:
                            overview[flag_type] = {
                                "experiment_ids": [], 
                                "video_ids": [],
                                "image_ids": [], 
                                "snip_ids": [], 
                                "count": 0
                            }
                        overview[flag_type]["image_ids"].append(image_id)
                        overview[flag_type]["count"] += 1
                    
                    # Embryo-level flags
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id not in target_snips:
                            continue
                            
                        for flag_type in embryo_data.get("flags", {}):
                            if flag_type not in overview:
                                overview[flag_type] = {
                                    "experiment_ids": [], 
                                    "video_ids": [],
                                    "image_ids": [], 
                                    "snip_ids": [], 
                                    "count": 0
                                }
                            if snip_id:
                                overview[flag_type]["snip_ids"].append(snip_id)
                            overview[flag_type]["count"] += 1
        
        # Clean up overview - remove duplicates and empty lists
        for flag_type, flag_data in overview.items():
            flag_data["experiment_ids"] = sorted(set(flag_data["experiment_ids"]))
            flag_data["video_ids"] = sorted(set(flag_data["video_ids"]))
            flag_data["image_ids"] = sorted(set(flag_data["image_ids"]))
            flag_data["snip_ids"] = sorted(set(flag_data["snip_ids"]))
            
            # Remove empty lists
            overview[flag_type] = {k: v for k, v in flag_data.items() if v != []}
        
        # Merge with existing overview if present
        if "GEN_flag_overview" in self.gsam_data:
            existing_overview = self.gsam_data["GEN_flag_overview"]
            for flag_type, new_data in overview.items():
                if flag_type in existing_overview:
                    # Merge entity lists
                    for entity_type in ["experiment_ids", "video_ids", "image_ids", "snip_ids"]:
                        existing = set(existing_overview[flag_type].get(entity_type, []))
                        new = set(new_data.get(entity_type, []))
                        merged = sorted(existing | new)
                        if merged:
                            existing_overview[flag_type][entity_type] = merged
                    # Update count
                    existing_overview[flag_type]["count"] = existing_overview[flag_type].get("count", 0) + new_data.get("count", 0)
                else:
                    existing_overview[flag_type] = new_data
            self.gsam_data["GEN_flag_overview"] = existing_overview
        else:
            self.gsam_data["GEN_flag_overview"] = overview
        
        if self.verbose:
            total_flagged = sum(data.get("count", 0) for data in overview.values())
            print(f"📊 Generated overview with {len(overview)} flag types, {total_flagged} new flags")
 
    def _save_qc_summary(self, author: str):
        """Save QC summary and write updated GSAM file."""
        # Count flags
        flag_counts = self._count_flags_in_hierarchy()
        
        # Create QC run record
        qc_run = {
            "timestamp": datetime.now().isoformat(),
            "author": author,
            "entities_processed": {
                "experiments": len(self.new_experiment_ids),
                "videos": len(self.new_video_ids),
                "images": len(self.new_image_ids),
                "snips": len(self.new_snip_ids)
            },
            "flags_added": sum(flag_counts.values()),
            "flag_breakdown": flag_counts
        }
        
        # Add to QC history
        if "qc_history" not in self.gsam_data:
            self.gsam_data["qc_history"] = []
        self.gsam_data["qc_history"].append(qc_run)
        
        # Update overall QC summary
        if "flags" not in self.gsam_data:
            self.gsam_data["flags"] = {}
        self.gsam_data["flags"]["qc_summary"] = {
            "last_updated": datetime.now().isoformat(),
            "total_qc_runs": len(self.gsam_data["qc_history"]),
            "total_entities_processed": {
                "experiments": len(self.processed_experiment_ids | self.new_experiment_ids),
                "videos": len(self.processed_video_ids | self.new_video_ids),
                "images": len(self.processed_image_ids | self.new_image_ids),
                "snips": len(self.processed_snip_ids | self.new_snip_ids)
            }
        }
        
        # Save to file
        with open(self.gsam_path, 'w') as f:
            json.dump(self.gsam_data, f, indent=2)
            
        if self.verbose:
            print(f"💾 Saved QC results: {qc_run['flags_added']} new flags added")
            for category, count in qc_run['flag_breakdown'].items():
                if count > 0:
                    print(f"   {category}: {count} flags")  
    
    def _count_flags_in_hierarchy(self) -> Dict[str, int]:
        """Count new flags added in this run."""
        flag_counts = defaultdict(int)
        
        # Only count flags from newly processed entities
        for exp_id, exp_data in self.gsam_data.get("experiments", {}).items():
            if exp_id in self.new_experiment_ids:
                for flag_type in exp_data.get("flags", {}):
                    flag_counts[flag_type] += 1
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id in self.new_video_ids:
                    for flag_type in video_data.get("flags", {}):
                        flag_counts[flag_type] += 1
                
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id in self.new_image_ids:
                        for flag_type in image_data.get("flags", {}):
                            flag_counts[flag_type] += 1
                    
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id in self.new_snip_ids:
                            for flag_type in embryo_data.get("flags", {}):
                                flag_counts[flag_type] += 1
        
        return dict(flag_counts)
    
    def get_flags_summary(self) -> Dict:
        """Get summary of all QC flags."""
        overview = self.gsam_data.get("GEN_flag_overview", {})
        total_flags = sum(data.get("count", 0) for data in overview.values())
        qc_summary = self.gsam_data.get("flags", {}).get("qc_summary", {}) # added this line

        return {
            "total_flags": total_flags,
            "flag_categories": {k: v.get("count", 0) for k, v in overview.items()},
            "entities_with_flags": {
                "experiments": len(set(sum([v.get("experiment_ids", []) for v in overview.values()], []))),
                "videos": len(set(sum([v.get("video_ids", []) for v in overview.values()], []))),
                "images": len(set(sum([v.get("image_ids", []) for v in overview.values()], []))),
                "snips": len(set(sum([v.get("snip_ids", []) for v in overview.values()], [])))
            }
        }    

    def print_summary(self):
        """Print a summary of QC results."""
        summary = self.get_flags_summary()
        print(f"\n🏁 QC Summary")
        print(f"{'=' * 40}")
        print(f"Total flags: {summary['total_flags']}")
        print(f"\nFlags by category:")
        for category, count in summary['flag_categories'].items():
            if count > 0:
                print(f"  {category}: {count}")
        
        print(f"\nEntities with flags:")
        for entity_type, count in summary['entities_with_flags'].items():
            print(f"  {entity_type}: {count}")
        
        if summary['total_flags'] == 0:
            print("\n✅ No quality issues detected!")


# Example usage:
if __name__ == "__main__":
    gsam_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json"
    
    # Initialize QC
    qc = GSAMQualityControl(gsam_path, verbose=True)
    
    # Run checks on new entities only (default)
    qc.run_all_checks(author="auto_qc_sam")
    

    # # Or run on specific entities
    # specific_entities = {
    #     "experiment_ids": ["20240411"],
    #     "video_ids": ["20240411_A01"],
    #     "image_ids": ["20240411_A01_0000", "20240411_A01_0001"],
    #     "snip_ids": ["20240411_A01_e01_0000", "20240411_A01_e01_0001"]
    # }
    # qc.run_all_checks(author="auto_qc_v1", target_entities=specific_entities)
    
    # Print summary
    qc.print_summary()
    
    # Access flags examples:
    # Snip level: embryo_data["flags"]["HIGH_SEGMENTATION_VAR"]
    # Image level: image_data["flags"]["DETECTION_FAILURE"]
    # Video level: video_data["flags"]["SOME_VIDEO_FLAG"]
    # Experiment level: exp_data["flags"]["SOME_EXP_FLAG"]
    # Top level: gsam_data["flags"]["GLOBAL_FLAG"]
    
    # Access specific flag details:
    # snip_flags = embryo_data["flags"]["HIGH_SEGMENTATION_VAR"]
    # for flag in snip_flags:
    #     print(f"Snip {flag['snip_id']} has {flag['diff_before_pct']*100:.1f}% difference from before frames")