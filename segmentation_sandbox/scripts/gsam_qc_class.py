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

from scripts.utils.base_annotation_parser import BaseAnnotationParser
from scripts.utils.embryo_metada_dev_instruction.embryo_metadata_refactored import EmbryoMetadata

class GSAMQualityControl(BaseAnnotationParser):
    """
    Quality control for SAM2 annotations. Flags segmentation variability, edge proximity, detection failures.
    """
    def __init__(self, sam_path: str, embryo_metadata_path: str, verbose: bool = True):
        super().__init__(sam_path, verbose=verbose)
        self.sam_path = Path(sam_path)
        self.embryo_metadata_path = Path(embryo_metadata_path)
        self.verbose = verbose
        self.sam_data = self.load_json(self.sam_path)
        self.embryo_metadata = EmbryoMetadata(sam_annotation_path=sam_path, embryo_metadata_path=embryo_metadata_path, verbose=verbose)
        self.flags = defaultdict(list)

    def check_segmentation_variability(self, author: str, threshold: float = 0.10):
        """
        Flag snips where mask area is an outlier compared to the average of previous or next 2 snips (>threshold).
        """
        for exp_id, exp_data in self.sam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for embryo_id, embryo_data in video_data.get("embryos", {}).items():
                    # Collect snip_ids and areas in frame order
                    snip_area_list = []
                    for image_id, image_data in sorted(video_data.get("images", {}).items()):
                        embryos = image_data.get("embryos", {})
                        if embryo_id in embryos:
                            mask_info = embryos[embryo_id]
                            area = mask_info.get("area")
                            snip_id = mask_info.get("snip_id", f"{embryo_id}_{image_id}")
                            if area is not None:
                                snip_area_list.append((snip_id, area))
                    n = len(snip_area_list)
                    for i, (snip_id, area) in enumerate(snip_area_list):
                        # Previous 2
                        prev_areas = [a for _, a in snip_area_list[max(0, i-2):i]]
                        # Next 2
                        next_areas = [a for _, a in snip_area_list[i+1:i+3]]
                        outlier = False
                        # Compare to previous 2
                        if len(prev_areas) == 2:
                            prev_mean = np.mean(prev_areas)
                            if prev_mean > 0 and abs(area - prev_mean) / prev_mean > threshold:
                                outlier = True
                        # Compare to next 2
                        if len(next_areas) == 2:
                            next_mean = np.mean(next_areas)
                            if next_mean > 0 and abs(area - next_mean) / next_mean > threshold:
                                outlier = True
                        if outlier:
                            self.flags[snip_id].append({
                                "flag": "HIGHLY_VAR_MASK",
                                "author": author,
                                "details": f"Mask area outlier (> {threshold*100:.1f}% diff from neighbors)"
                            })
                            if self.verbose:
                                print(f"Flagged {snip_id} for mask area outlier.")

    def check_mask_on_edge(self, author: str, edge_pixels: int = 5):
        """
        Flag snips where mask is within edge_pixels of image edge.
        """
        for exp_id, exp_data in self.sam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("images", {}).items():
                    for embryo_id, mask_info in image_data.get("embryos", {}).items():
                        snip_id = mask_info.get("snip_id", f"{embryo_id}_{image_id}")
                        seg_format = mask_info.get("segmentation_format")
                        seg = mask_info.get("segmentation")
                        shape = mask_info.get("segmentation", {}).get("size")
                        if seg_format == "rle" and seg and shape:
                            from pycocotools import mask as mask_utils
                            mask = mask_utils.decode(seg)
                            h, w = shape
                            rows, cols = np.where(mask > 0)
                            if (np.any(rows < edge_pixels) or np.any(rows >= h - edge_pixels) or
                                np.any(cols < edge_pixels) or np.any(cols >= w - edge_pixels)):
                                self.flags[snip_id].append({
                                    "flag": "MASK_ON_EDGE",
                                    "author": author,
                                    "details": f"Mask within {edge_pixels} pixels of image edge"
                                })
                                if self.verbose:
                                    print(f"Flagged {snip_id} for mask on edge.")

    def check_detection_failure(self, author: str):
        """
        Flag images where embryo is missing (MISSING_EMBRYO) and videos with no embryos at all (NO_EMBRYO).
        Organized by entity level.
        """
        for exp_id, exp_data in self.sam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                images = list(sorted(video_data.get("images", {}).items()))
                embryo_ids_seen = set()
                snip_ids_per_frame = []
                snip_id_map = []
                for image_id, image_data in images:
                    embryos = image_data.get("embryos", {})
                    snip_ids = set()
                    snip_id_lookup = dict()
                    for embryo_id, mask_info in embryos.items():
                        snip_id = mask_info.get("snip_id", f"{embryo_id}_{image_id}")
                        snip_ids.add(snip_id)
                        snip_id_lookup[embryo_id] = snip_id
                        embryo_ids_seen.add(embryo_id)
                    snip_ids_per_frame.append(snip_ids)
                    snip_id_map.append(snip_id_lookup)
                # Flag MISSING_EMBRYO for each snip_id missing in a frame but present in previous frame
                for i in range(1, len(images)):
                    prev_snip_ids = snip_ids_per_frame[i-1]
                    curr_snip_ids = snip_ids_per_frame[i]
                    missing_snip_ids = prev_snip_ids - curr_snip_ids
                    image_id = images[i][0]
                    for snip_id in missing_snip_ids:
                        self.flags[snip_id].append({
                            "flag": "MISSING_EMBRYO",
                            "author": author,
                            "details": f"Embryo instance {snip_id} present in previous frame but missing in this frame"
                        })
                        if self.verbose:
                            print(f"Flagged {snip_id} for missing embryo instance in {image_id}.")
                # Flag NO_EMBRYO at video level if no embryos detected in any frame
                if not embryo_ids_seen:
                    self.flags[video_id].append({
                        "flag": "NO_EMBRYO",
                        "author": author,
                        "details": "No embryos detected in any frame of video"
                    })
                    if self.verbose:
                        print(f"Flagged {video_id} for no embryos detected in video.")

    def save_flags(self, output_path: Optional[str] = None):
        """
        Save all flags to a JSON file.
        """
        if output_path is None:
            output_path = self.sam_path.parent / "gsam_qc_flags.json"
        with open(output_path, 'w') as f:
            json.dump(self.flags, f, indent=2)
        if self.verbose:
            print(f"QC flags saved to {output_path}")
