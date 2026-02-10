
"""
Simplified GSAM Quality Control Class
====================================

Analyzes SAM2 annotations for quality issues and adds flags directly 
to the GSAM JSON structure at the top level under "flags".

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


def ensure_json_serializable(obj):
    """
    Recursively convert numpy types and other non-serializable objects to JSON-safe types.
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


class GSAMQualityControl:
    """
    Simplified QC for SAM2 annotations. Flags quality issues and saves 
    them directly to the GSAM JSON file under top-level "flags".
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
            
        # Initialize top-level flags structure
        if "flags" not in self.gsam_data:
            self.gsam_data["flags"] = {
                "qc_meta": {},
                "qc_history": [],
                "flag_overview": {},
                "by_experiment": {},
                "by_video": {},
                "by_image": {},
                "by_snip": {}
            }
        
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
        
    def diagnose_data_structure(self, max_samples=3):
        """Diagnose the GSAM data structure to understand why no snips are found."""
        print("\n🔍 GSAM Data Structure Diagnosis")
        print("=" * 50)
        
        experiments = self.gsam_data.get("experiments", {})
        print(f"Total experiments: {len(experiments)}")
        
        if not experiments:
            print("❌ No experiments found in data!")
            print("Available top-level keys:", list(self.gsam_data.keys()))
            return
        
        sample_count = 0
        for exp_id, exp_data in experiments.items():
            if sample_count >= max_samples:
                break
                
            print(f"\n📁 Experiment: {exp_id}")
            videos = exp_data.get("videos", {})
            print(f"   Videos: {len(videos)}")
            
            if not videos:
                print("   ❌ No videos found in this experiment!")
                print(f"   Available keys in experiment: {list(exp_data.keys())}")
                continue
            
            video_sample_count = 0
            for video_id, video_data in videos.items():
                if video_sample_count >= 2:  # Limit video samples
                    break
                    
                print(f"   📹 Video: {video_id}")
                images = video_data.get("images", {})
                print(f"      Images: {len(images)}")
                
                if not images:
                    print("      ❌ No images found in this video!")
                    print(f"      Available keys in video: {list(video_data.keys())}")
                    continue
                
                image_sample_count = 0
                for image_id, image_data in images.items():
                    if image_sample_count >= 2:  # Limit image samples
                        break
                        
                    print(f"      🖼️  Image: {image_id}")
                    embryos = image_data.get("embryos", {})
                    print(f"         Embryos: {len(embryos)}")
                    
                    if not embryos:
                        print("         ❌ No embryos found in this image!")
                        print(f"         Available keys in image: {list(image_data.keys())}")
                        continue
                    
                    embryo_sample_count = 0
                    for embryo_id, embryo_data in embryos.items():
                        if embryo_sample_count >= 2:  # Limit embryo samples
                            break
                            
                        snip_id = embryo_data.get("snip_id")
                        print(f"         🧬 Embryo: {embryo_id}")
                        print(f"            snip_id: {snip_id}")
                        print(f"            Available keys: {list(embryo_data.keys())}")
                        
                        if not snip_id:
                            print("            ❌ No snip_id found!")
                        
                        embryo_sample_count += 1
                    
                    image_sample_count += 1
                
                video_sample_count += 1
            
            sample_count += 1
        
        print("\n" + "=" * 50)

    def _initialize_entity_tracking(self):
        """Simple and fast entity tracking using stored processed IDs."""
        
        if self.verbose:
            print("🔄 Initializing entity tracking...")
            start_time = time()
        
        # Get previously processed IDs from flags.qc_meta (or empty sets if first run)
        qc_meta = self.gsam_data["flags"].setdefault("qc_meta", {})
        self.processed_experiment_ids = set(qc_meta.get("processed_experiment_ids", []))
        self.processed_video_ids = set(qc_meta.get("processed_video_ids", []))
        self.processed_image_ids = set(qc_meta.get("processed_image_ids", []))
        self.processed_snip_ids = set(qc_meta.get("processed_snip_ids", []))
        
        # Find all current entities in the data
        all_experiment_ids = set()
        all_video_ids = set()
        all_image_ids = set()
        all_snip_ids = set()
        
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Scanning", total=len(experiments_items)):
            all_experiment_ids.add(exp_id)
            
            for video_id, video_data in exp_data.get("videos", {}).items():
                all_video_ids.add(video_id)
                
                for image_id, image_data in video_data.get("images", {}).items():
                    all_image_ids.add(image_id)
                    
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id:
                            all_snip_ids.add(snip_id)
        
        # New entities = all entities - processed entities
        self.new_experiment_ids = all_experiment_ids - self.processed_experiment_ids
        self.new_video_ids = all_video_ids - self.processed_video_ids
        self.new_image_ids = all_image_ids - self.processed_image_ids
        self.new_snip_ids = all_snip_ids - self.processed_snip_ids
        
        if self.verbose:
            elapsed = time() - start_time
            print(f"🔍 Loaded GSAM annotations from {self.gsam_path}")
            print(f"📊 Total entities found:")
            print(f"   Experiments: {len(all_experiment_ids)}")
            print(f"   Videos: {len(all_video_ids)}")
            print(f"   Images: {len(all_image_ids)}")
            print(f"   Snips: {len(all_snip_ids)}")
            print(f"📊 Previously processed: {len(self.processed_snip_ids)} snips")
            print(f"🆕 New to process: {len(self.new_snip_ids)} snips")
            if len(self.new_snip_ids) == 0 and len(all_snip_ids) > 0:
                print(f"⚠️  All {len(all_snip_ids)} snips were already processed!")
                print(f"   Use process_all=True to reprocess existing snips")
            elif len(all_snip_ids) == 0:
                print(f"❌ No snips found in the data structure!")
                print(f"   Running diagnosis...")
                self.diagnose_data_structure()
            print(f"⏱️  Initialization took: {elapsed:.2f}s")

    def _mark_entities_checked(self, entities):
        """Mark entities as processed by adding them to the stored sets."""
        ts = datetime.now().isoformat()
        
        # Update our tracking sets
        self.processed_experiment_ids.update(entities.get("experiment_ids", []))
        self.processed_video_ids.update(entities.get("video_ids", []))
        self.processed_image_ids.update(entities.get("image_ids", []))
        self.processed_snip_ids.update(entities.get("snip_ids", []))
        
        # Save to flags.qc_meta for persistence
        qc_meta = self.gsam_data["flags"]["qc_meta"]
        qc_meta["processed_experiment_ids"] = sorted(self.processed_experiment_ids)
        qc_meta["processed_video_ids"] = sorted(self.processed_video_ids)
        qc_meta["processed_image_ids"] = sorted(self.processed_image_ids)
        qc_meta["processed_snip_ids"] = sorted(self.processed_snip_ids)
        qc_meta["last_updated"] = ts

    def _add_flag(self, flag_type: str, flag_data: dict, entity_type: str, entity_id: str):
        """
        Add a flag to the top-level flags structure.
        
        Args:
            flag_type: Type of flag (e.g., 'HIGH_SEGMENTATION_VAR_SNIP')
            flag_data: Flag data dictionary
            entity_type: 'experiment', 'video', 'image', or 'snip'
            entity_id: ID of the entity being flagged
        """
        flags_section = self.gsam_data["flags"]
        
        # Add to appropriate entity section
        entity_key = f"by_{entity_type}"
        if entity_key not in flags_section:
            flags_section[entity_key] = {}
        
        if entity_id not in flags_section[entity_key]:
            flags_section[entity_key][entity_id] = {}
        
        if flag_type not in flags_section[entity_key][entity_id]:
            flags_section[entity_key][entity_id][flag_type] = []
        
        flags_section[entity_key][entity_id][flag_type].append(flag_data)

    def _progress_iter(self, iterable, desc: str, total: Optional[int] = None):
        """Wrap an iterable with a progress indicator compatible with tmux."""
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
    
    def run_all_checks(self,
                      author: str = "auto_qc",
                      process_all: bool = False,
                      target_entities: Optional[Dict[str, List[str]]] = None,
                      save_in_place: bool = True):
        start_overall = time()
        """
        Run all QC checks and save flags to GSAM file.

        Args:
            author: Author identifier for the QC run
            process_all: If True, process all entities. If False, only process new entities
            target_entities: Specific entities to process (overrides process_all)
            save_in_place: If True (default) QC modifications are written back to the
                           GSAM JSON on disk.  If False, the GSAM data is mutated in
                           memory only (caller can decide when / whether to save).
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
            
        # Flags are created lazily; mark entities checked before overview
        self._mark_entities_checked(entities_to_process)

        if self.verbose:
            print("✓ Marked entities as processed")

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
        
        # Always build in‑memory overview
        self.generate_overview(entities_to_process)

        if save_in_place:
            # Create backup before saving
            self._create_backup("pre_qc_save")
            # Persist to disk
            self._save_qc_summary(author)
            if self.verbose:
                print("✅ QC checks complete and flags saved to GSAM file")
        else:
            if self.verbose:
                print("ℹ️  QC checks complete (results NOT saved – save_in_place=False)")
    
    def _create_backup(self, suffix="backup"):
        """Create a timestamped backup of the JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.gsam_path.parent / f".{self.gsam_path.stem}_backups"
            backup_dir.mkdir(exist_ok=True)
            
            backup_name = f"{self.gsam_path.stem}_{timestamp}_{suffix}.json"
            backup_path = backup_dir / backup_name
            
            import shutil
            shutil.copy2(self.gsam_path, backup_path)
            
            if self.verbose:
                print(f"💾 Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to create backup: {e}")
            return None
   
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
                            flag_data = {
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "embryo_id": embryo_id,
                                "issue": "HIGH_SEGMENTATION_VAR_EMBRYO",
                                "coefficient_of_variation": float(round(cv, 3)),
                                "frame_count": int(len(areas)),
                                "mean_area": float(round(mean_area, 1)),
                                "std_area": float(round(np.std(areas), 1)),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            # Use first image_id where this embryo appears for reference
                            ref_image_id = next(img_id for img_id in image_ids 
                                              if embryo_id in video_data["images"][img_id].get("embryos", {}))
                            entity_ref = f"{exp_id}_{video_id}_{ref_image_id}_{embryo_id}"
                            self._add_flag("HIGH_SEGMENTATION_VAR_EMBRYO", flag_data, "embryo", entity_ref)
    
                # Snip-level variability flag
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
    
                        # FIX: Explicit boolean conversion to avoid numpy.bool_ serialization issues
                        flag_before = bool(diff_before_pct > 0.20) if diff_before_pct is not None else False
                        flag_after = bool(diff_after_pct > 0.20) if diff_after_pct is not None else False
    
                        if flag_before or flag_after:
                            flag_data = {
                                "snip_id": snip_id,
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "image_id": image_id,
                                "embryo_id": embryo_id,
                                "current_area": float(round(current_area, 1)),
                                "avg_before": float(round(avg_before, 1)) if avg_before else None,
                                "avg_after": float(round(avg_after, 1)) if avg_after else None,
                                "diff_before_pct": float(round(diff_before_pct, 3)) if diff_before_pct else None,
                                "diff_after_pct": float(round(diff_after_pct, 3)) if diff_after_pct else None,
                                "flagged_before": flag_before,
                                "flagged_after": flag_after,
                                "frames_checked_before": int(len(before_areas)),
                                "frames_checked_after": int(len(after_areas)),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            self._add_flag("HIGH_SEGMENTATION_VAR_SNIP", flag_data, "snip", snip_id)
        if self.verbose:
            print(f"   ⏱ check_segmentation_variability {time() - t0:.2f}s")
    
    def check_mask_on_edge(self, author: str, entities: Dict[str, List[str]]):
        """Flag masks that touch image edges (within 5 pixels)."""
        if self.verbose:
            print("🖼️  Checking masks on image edge...")

        target_snips = set(entities.get("snip_ids", []))
        t0 = time()
        margin_pixels = 5  # safety margin in *pixels*

        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Edge", total=len(experiments_items)):
            if exp_id not in entities.get("experiment_ids", []):
                continue

            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in entities.get("video_ids", []):
                    continue

                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id not in entities.get("image_ids", []):
                        continue

                    # ── infer width & height once per image from first embryo's segmentation.size
                    height = width = None
                    sample_embryo = next(iter(image_data.get("embryos", {}).values()), None)
                    if sample_embryo and isinstance(sample_embryo.get("segmentation"), dict):
                        sz = sample_embryo["segmentation"].get("size")  # [h, w]
                        if sz and len(sz) == 2:
                            height, width = sz

                    # If we somehow cannot deduce dims, fall back to a 1 % normalised margin
                    norm_margin = 0.01
                    px_margin_x = margin_pixels
                    px_margin_y = margin_pixels
                    if width and height:
                        norm_margin_x = margin_pixels / width
                        norm_margin_y = margin_pixels / height
                    else:
                        norm_margin_x = norm_margin_y = norm_margin

                    for embryo_id, embryo_data in image_data["embryos"].items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id not in target_snips:
                            continue

                        bbox = embryo_data.get("bbox")
                        if not (bbox and len(bbox) == 4):
                            continue

                        x_min, y_min, x_max, y_max = bbox

                        # Detect absolute bbox values (>1) and normalise if we have dims
                        if (x_max > 1 or y_max > 1) and width and height:
                            x_min /= width
                            x_max /= width
                            y_min /= height
                            y_max /= height

                        near_edge = (
                            x_min <= norm_margin_x or
                            y_min <= norm_margin_y or
                            (1 - x_max) <= norm_margin_x or
                            (1 - y_max) <= norm_margin_y
                        )

                        if near_edge:
                            flag_data = {
                                "snip_id": snip_id,
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "image_id": image_id,
                                "embryo_id": embryo_id,
                                "bbox_norm": [float(round(x_min, 4)), float(round(y_min, 4)),
                                             float(round(x_max, 4)), float(round(y_max, 4))],
                                "margin_norm": float(round(max(norm_margin_x, norm_margin_y), 4)),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            self._add_flag("MASK_ON_EDGE", flag_data, "snip", snip_id)
    def check_detection_failure(self, author: str, entities: Dict[str, List[str]]):
        """Flag images where expected embryos are missing."""
        if self.verbose:
            print("🔍 Checking for detection failures...")
        target_images = set(entities.get("image_ids", []))

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
                            "experiment_id": exp_id,
                            "video_id": video_id,
                            "image_id": image_id,
                            "missing_embryo_ids": list(missing_embryo_ids),
                            "total_embryos_in_video": int(len(all_embryo_ids)),
                            "embryos_present_in_image": int(len(present_embryo_ids)),
                            "author": author,
                            "timestamp": datetime.now().isoformat()
                        }
                        self._add_flag("DETECTION_FAILURE", flag_data, "image", image_id)
        if self.verbose:
            print(f"   ⏱ check_detection_failure {time() - t0:.2f}s")
    
    def check_overlapping_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag images where embryo masks overlap (both bbox and mask level)."""
        if self.verbose:
            print("📦 Checking for overlapping masks...")

        target_images = set(entities.get("image_ids", []))
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
                                            "bbox_overlap_ratio": float(round(overlap_ratio, 3)),
                                            "bbox1": [float(x) for x in bbox1],
                                            "bbox2": [float(x) for x in bbox2]
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
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "image_id": image_id,
                                "overlapping_pairs": bbox_overlaps,
                                "threshold": 0.2,
                                "total_embryos": int(len(embryos)),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            self._add_flag("BBOX_OVERLAP", flag_data, "image", image_id)
                        
                        if mask_overlaps:
                            flag_data = {
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "image_id": image_id,
                                "overlapping_pairs": mask_overlaps,
                                "total_embryos": int(len(embryos)),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            self._add_flag("MASK_OVERLAP_ERROR", flag_data, "image", image_id)
        if self.verbose:
            print(f"   ⏱ check_overlapping_masks {time() - t0:.2f}s")
    
    def check_large_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag unusually large masks as percentage of total image area."""
        if self.verbose:
            print("📐 Checking for unusually large masks...")
        target_snips = set(entities.get("snip_ids", []))
        t0 = time()
        pct_threshold = 0.15  # 15 % of frame area
        experiments_items = list(self.gsam_data.get("experiments", {}).items())

        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Large", total=len(experiments_items)):
            if exp_id not in entities.get("experiment_ids", []):
                continue

            for video_id, video_data in exp_data.get("videos", {}).items():
                if video_id not in entities.get("video_ids", []):
                    continue

                for image_id, image_data in video_data.get("images", {}).items():
                    if image_id not in entities.get("image_ids", []):
                        continue

                    # ── infer width & height from the first embryo's segmentation.size
                    height = width = None
                    first_embryo = next(iter(image_data.get("embryos", {}).values()), None)
                    if first_embryo and isinstance(first_embryo.get("segmentation"), dict):
                        sz = first_embryo["segmentation"].get("size")  # [h, w]
                        if sz and len(sz) == 2:
                            height, width = sz

                    if not (height and width):
                        # still no dims → skip this image gracefully
                        continue

                    frame_area = height * width
                    area_cutoff = frame_area * pct_threshold

                    for embryo_id, embryo_data in image_data["embryos"].items():
                        snip_id = embryo_data.get("snip_id")
                        if snip_id and snip_id not in target_snips:
                            continue

                        area = embryo_data.get("area")
                        if area is None:
                            # as a fallback compute from RLE if present
                            seg = embryo_data.get("segmentation")
                            if isinstance(seg, dict) and "counts" in seg:
                                try:
                                    from pycocotools import mask as mask_utils
                                    area = float(mask_utils.area(seg))
                                except Exception:
                                    area = None

                        if area and area > area_cutoff:
                            flag_data = {
                                "snip_id": snip_id,
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "image_id": image_id,
                                "embryo_id": embryo_id,
                                "area_px": float(round(area, 1)),
                                "area_pct": float(round(area / frame_area * 100, 2)),
                                "threshold_pct": float(pct_threshold * 100),
                                "author": author,
                                "timestamp": datetime.now().isoformat()
                            }
                            self._add_flag("UNUSUALLY_LARGE_MASK", flag_data, "snip", snip_id)
        if self.verbose:
            print(f"   ⏱ check_large_masks {time() - t0:.2f}s")
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
    
    def check_discontinuous_masks(self, author: str, entities: Dict[str, List[str]]):
        """Flag masks whose segmentation contains multiple disconnected components."""
        if self.verbose:
            print("🔗 Checking for discontinuous masks...")
        
        target_snips = set(entities.get("snip_ids", []))
        t0 = time()
        
        # Collect all masks to check in one pass
        masks_to_check = []
        mask_locations = {}
        
        experiments_items = list(self.gsam_data.get("experiments", {}).items())
        for exp_id, exp_data in self._progress_iter(experiments_items, desc="Collecting masks", total=len(experiments_items)):
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
                        
                        mask_idx = len(masks_to_check)
                        masks_to_check.append(segmentation)
                        mask_locations[mask_idx] = {
                            'snip_id': snip_id,
                            'experiment_id': exp_id,
                            'video_id': video_id,
                            'image_id': image_id,
                            'embryo_id': embryo_id,
                            'format': embryo_data.get("segmentation_format", "unknown")
                        }
        
        if self.verbose:
            print(f"   📦 Checking {len(masks_to_check)} masks...")
        
        # Batch check for discontinuity
        batch_size = 1000
        discontinuous_indices = []
        
        for i in range(0, len(masks_to_check), batch_size):
            batch_end = min(i + batch_size, len(masks_to_check))
            batch = masks_to_check[i:batch_end]
            
            if self.verbose and len(masks_to_check) > batch_size:
                print(f"   🔄 Processing batch {i//batch_size + 1}/{(len(masks_to_check)-1)//batch_size + 1}")
            
            # Check batch
            batch_results = self._check_mask_discontinuity(batch)
            
            # Collect discontinuous mask indices
            for j, is_disc in enumerate(batch_results):
                if is_disc:
                    discontinuous_indices.append(i + j)
        
        # Apply flags
        for idx in discontinuous_indices:
            location = mask_locations[idx]
            
            flag_data = {
                "snip_id": location['snip_id'],
                "experiment_id": location['experiment_id'],
                "video_id": location['video_id'],
                "image_id": location['image_id'],
                "embryo_id": location['embryo_id'],
                "segmentation_format": location['format'],
                "author": author,
                "timestamp": datetime.now().isoformat()
            }
            self._add_flag("DISCONTINUOUS_MASK", flag_data, "snip", location['snip_id'])
        
        if self.verbose:
            print(f"   ⏱ check_discontinuous_masks {time() - t0:.2f}s - found {len(discontinuous_indices)} discontinuous masks")

    def _check_mask_discontinuity(self, segmentation) -> bool:
        """
        Return True if the segmentation represents multiple disconnected components.
        Supports COCO RLE (dict with 'counts') or polygon lists.
        Now handles both single masks and batches.
        """
        # Handle batch input
        if isinstance(segmentation, list) and len(segmentation) > 0 and isinstance(segmentation[0], (dict, list)):
            # This is a batch of segmentations
            return [self._check_single_mask_discontinuity(seg) for seg in segmentation]
        else:
            # Single segmentation
            return self._check_single_mask_discontinuity(segmentation)

    def _check_single_mask_discontinuity(self, segmentation) -> bool:
        """Check a single mask for discontinuity."""
        try:
            # Polygon format (fast check)
            if isinstance(segmentation, list):
                if len(segmentation) == 0:
                    return False
                # Multiple polygons = discontinuous
                if isinstance(segmentation[0], list):
                    return len(segmentation) > 1
                return False
            
            # RLE format (expensive check - only if needed)
            if isinstance(segmentation, dict) and "counts" in segmentation:
                # Skip RLE checks if too many masks
                if hasattr(self, '_skip_rle_checks'):
                    return False
                    
                try:
                    from pycocotools import mask as mask_utils
                    mask_array = mask_utils.decode(segmentation)
                    return self._has_multiple_components(mask_array)
                except ImportError:
                    return False
            
            return False
        except Exception:
            return False

    def _has_multiple_components(self, mask_array) -> bool:
        """Check a binary mask for >1 connected foreground component."""
        try:
            # Quick pre-check: if mask is very small, likely continuous
            if mask_array.sum() < 100:  # Less than 100 pixels
                return False
                
            import cv2
            # Ensure binary
            m = (mask_array.astype(np.uint8) > 0).astype(np.uint8)
            
            # Downsample for faster processing if mask is large
            if m.shape[0] > 256 or m.shape[1] > 256:
                m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            num_labels, _ = cv2.connectedComponents(m)
            return (num_labels - 1) > 1
            
        except ImportError:
            # Lightweight fallback
            m = (mask_array > 0).astype(np.uint8)
            # Simple heuristic: check for gaps in rows
            row_sums = m.sum(axis=1)
            gaps = np.sum((row_sums[1:] == 0) & (row_sums[:-1] > 0))
            return gaps > 2  # More than 2 gaps suggests discontinuity  

    def generate_overview(self, entities: Dict[str, List[str]]):
        """
        Generate flag_overview section with ALL flagged entity IDs.
        Includes both existing flags and any new flags from this run.
        """
        if self.verbose:
            print("📋 Generating overview section...")
        
        # Initialize overview structure
        overview = {}
        flags_section = self.gsam_data["flags"]
        
        # Collect ALL flag types and count them (not just from this run)
        for entity_type in ["by_experiment", "by_video", "by_image", "by_snip", "by_embryo"]:
            if entity_type in flags_section:
                for entity_id, entity_flags in flags_section[entity_type].items():
                    for flag_type, flag_instances in entity_flags.items():
                        if flag_type not in overview:
                            overview[flag_type] = {
                                "experiment_ids": set(),
                                "video_ids": set(),
                                "image_ids": set(),
                                "snip_ids": set(),
                                "count": 0
                            }

                        # Add entity ID to appropriate set and extract related IDs from flag data
                        if entity_type == "by_experiment":
                            overview[flag_type]["experiment_ids"].add(entity_id)
                        elif entity_type == "by_video":
                            overview[flag_type]["video_ids"].add(entity_id)
                            # Extract experiment_id from flag data if available
                            for flag_instance in flag_instances:
                                if "experiment_id" in flag_instance:
                                    overview[flag_type]["experiment_ids"].add(flag_instance["experiment_id"])
                        elif entity_type == "by_image":
                            overview[flag_type]["image_ids"].add(entity_id)
                            # Extract experiment_id and video_id from flag data if available
                            for flag_instance in flag_instances:
                                if "experiment_id" in flag_instance:
                                    overview[flag_type]["experiment_ids"].add(flag_instance["experiment_id"])
                                if "video_id" in flag_instance:
                                    overview[flag_type]["video_ids"].add(flag_instance["video_id"])
                        elif entity_type == "by_snip":
                            overview[flag_type]["snip_ids"].add(entity_id)
                            # Extract experiment_id, video_id, and image_id from flag data if available
                            for flag_instance in flag_instances:
                                if "experiment_id" in flag_instance:
                                    overview[flag_type]["experiment_ids"].add(flag_instance["experiment_id"])
                                if "video_id" in flag_instance:
                                    overview[flag_type]["video_ids"].add(flag_instance["video_id"])
                                if "image_id" in flag_instance:
                                    overview[flag_type]["image_ids"].add(flag_instance["image_id"])
                        elif entity_type == "by_embryo":
                            # Properly aggregate embryo-level flags into overview
                            for flag_instance in flag_instances:
                                if "experiment_id" in flag_instance:
                                    overview[flag_type]["experiment_ids"].add(flag_instance["experiment_id"])
                                if "video_id" in flag_instance:
                                    overview[flag_type]["video_ids"].add(flag_instance["video_id"])
                                if "image_id" in flag_instance:
                                    overview[flag_type]["image_ids"].add(flag_instance["image_id"])
                                if "embryo_id" in flag_instance:
                                    overview[flag_type]["snip_ids"].add(flag_instance["embryo_id"])

                        # Count flag instances
                        overview[flag_type]["count"] += len(flag_instances)
        
        # Convert sets to sorted lists for JSON serialization, filtering out None
        for flag_type, flag_data in overview.items():
            flag_data["experiment_ids"] = sorted([eid for eid in flag_data.get("experiment_ids", []) if eid is not None])
            flag_data["video_ids"] = sorted([vid for vid in flag_data.get("video_ids", []) if vid is not None])
            flag_data["image_ids"] = sorted([iid for iid in flag_data.get("image_ids", []) if iid is not None])
            flag_data["snip_ids"] = sorted([sid for sid in flag_data.get("snip_ids", []) if sid is not None])
            # Remove empty lists to keep JSON clean
            overview[flag_type] = {k: v for k, v in flag_data.items() if v != []}
        
        # Store in flags section
        flags_section["flag_overview"] = overview
        
        if self.verbose:
            total_flagged = sum(data.get("count", 0) for data in overview.values())
            if total_flagged > 0:
                print(f"📊 Generated overview with {len(overview)} flag types, {total_flagged} total flags")
            else:
                print(f"📊 Generated overview - no flags found in the system")

    def _save_qc_summary(self, author: str):
        """Save QC summary and write updated GSAM file."""
        # Count flags
        flag_counts = self._count_flags_in_hierarchy()
        
        # Create QC run record
        qc_run = {
            "timestamp": datetime.now().isoformat(),
            "author": author,
            "entities_processed": {
                "experiments": int(len(self.new_experiment_ids)),
                "videos": int(len(self.new_video_ids)),
                "images": int(len(self.new_image_ids)),
                "snips": int(len(self.new_snip_ids))
            },
            "flags_added": int(sum(flag_counts.values())),
            "flag_breakdown": {k: int(v) for k, v in flag_counts.items()}
        }
        
        # Add to QC history
        self.gsam_data["flags"]["qc_history"].append(qc_run)
        
        # Update overall QC summary
        self.gsam_data["flags"]["qc_summary"] = {
            "last_updated": datetime.now().isoformat(),
            "total_qc_runs": int(len(self.gsam_data["flags"]["qc_history"])),
            "total_entities_processed": {
                "experiments": int(len(self.processed_experiment_ids | self.new_experiment_ids)),
                "videos": int(len(self.processed_video_ids | self.new_video_ids)),
                "images": int(len(self.processed_image_ids | self.new_image_ids)),
                "snips": int(len(self.processed_snip_ids | self.new_snip_ids))
            }
        }
        
        # Ensure JSON serializable before saving
        self.gsam_data = ensure_json_serializable(self.gsam_data)
        
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
        flags_section = self.gsam_data["flags"]
        
        # Count flags from all entity types, but only for newly processed entities
        for entity_type in ["by_experiment", "by_video", "by_image", "by_snip"]:
            if entity_type not in flags_section:
                continue
                
            for entity_id, entity_flags in flags_section[entity_type].items():
                # Check if this entity was processed in this run
                entity_is_new = False
                if entity_type == "by_experiment" and entity_id in self.new_experiment_ids:
                    entity_is_new = True
                elif entity_type == "by_video" and entity_id in self.new_video_ids:
                    entity_is_new = True
                elif entity_type == "by_image" and entity_id in self.new_image_ids:
                    entity_is_new = True
                elif entity_type == "by_snip" and entity_id in self.new_snip_ids:
                    entity_is_new = True
                
                # Only count flags for newly processed entities
                if entity_is_new:
                    for flag_type, flag_instances in entity_flags.items():
                        flag_counts[flag_type] += len(flag_instances)
        
        return dict(flag_counts)
    
    def get_flags_summary(self) -> Dict:
        """Get summary of all QC flags."""
        overview = self.gsam_data["flags"].get("flag_overview", {})
        total_flags = sum(data.get("count", 0) for data in overview.values())

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

    def get_flags_for_entity(self, entity_type: str, entity_id: str) -> Dict:
        """Get all flags for a specific entity."""
        entity_key = f"by_{entity_type}"
        flags_section = self.gsam_data["flags"]
        
        if entity_key in flags_section and entity_id in flags_section[entity_key]:
            return flags_section[entity_key][entity_id]
        return {}

    def get_flags_by_type(self, flag_type: str) -> List[Dict]:
        """Get all instances of a specific flag type across all entities."""
        all_flags = []
        flags_section = self.gsam_data["flags"]
        
        for entity_type in ["by_experiment", "by_video", "by_image", "by_snip"]:
            if entity_type in flags_section:
                for entity_id, entity_flags in flags_section[entity_type].items():
                    if flag_type in entity_flags:
                        all_flags.extend(entity_flags[flag_type])
        
        return all_flags
        
        

if __name__ == "__main__":
    gsam_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json"
    # gsam_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/archive/grounded_sam_annotations copy.json"
    # Initialize QC
    qc = GSAMQualityControl(gsam_path, verbose=True)
    
    # Run checks on new entities only (default)
    # qc.run_all_checks(author="auto_qc_sam", save_in_place=True)

    def rerun_segvar_only(qc: GSAMQualityControl, author="auto_qc"):
        ents = {
            "experiment_ids": list(qc.processed_experiment_ids | qc.new_experiment_ids),
            "video_ids":      list(qc.processed_video_ids      | qc.new_video_ids),
            "image_ids":      list(qc.processed_image_ids      | qc.new_image_ids),
            "snip_ids":       list(qc.processed_snip_ids       | qc.new_snip_ids),
        }
        qc._mark_entities_checked(ents)
        qc.check_segmentation_variability(author, ents)
        qc.generate_overview(ents)
        qc._create_backup("segvar_only")
        qc._save_qc_summary(author)

    # Usage:
    # qc = GSAMQualityControl(gsam_path)
    rerun_segvar_only(qc, author="auto_qc_segvar")
        

    # # Or run on specific entities
    # specific_entities = {
    #     # "experiment_ids": ["20240411"],
    #     # "video_ids": ["20240411_A01"],
    #     # "image_ids": ["20240411_A01_0000", "20240411_A01_0001"],
    #     "snip_ids": ["20240530_F12_0024", "20240411_A01_e01_0001"]
    # }
    # qc.run_all_checks(author="auto_qc_sam", target_entities=specific_entities, save_in_place=True)

    # # When satisfied, persist:
    # qc.save(author="approved_post_check")
    
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