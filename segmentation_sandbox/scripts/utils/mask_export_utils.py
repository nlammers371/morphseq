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
        self.sam2_path = Path(sam2_annotations_path)
        self.output_base_dir = Path(output_base_dir)
        self.verbose = verbose
        self.output_format = output_format.lower()
        if self.output_format not in ['jpg', 'jpeg', 'png']:
            raise ValueError(f"Invalid output format: {output_format}. Use 'jpg' or 'png'")
        if self.output_format == 'jpeg':
            self.output_format = 'jpg'
        with open(self.sam2_path, 'r') as f:
            self.sam2_data = json.load(f)
        self.export_stats = {
            "total_images": 0,
            "total_masks": 0,
            "overlapping_masks": 0,
            "export_paths": {}
        }

    def decode_rle_mask(self, rle_data: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        if isinstance(rle_data, dict) and 'counts' in rle_data and 'size' in rle_data:
            binary_mask = mask_utils.decode(rle_data)
            return binary_mask
        else:
            raise ValueError(f"Unknown RLE format: {type(rle_data)}")

    def export_image_masks(self, image_id: str, embryo_data: Dict, image_shape: Tuple[int, int]) -> Path:
        label_image = np.zeros(image_shape, dtype=np.uint8)
        overlap_count = 0
        for embryo_id, mask_data in embryo_data.items():
            embryo_num = int(embryo_id.split('_e')[-1])
            if mask_data['segmentation_format'] == 'rle':
                binary_mask = self.decode_rle_mask(mask_data['segmentation'], image_shape)
            else:
                raise NotImplementedError(f"Format {mask_data['segmentation_format']} not supported")
            overlap_pixels = np.sum((label_image > 0) & (binary_mask > 0))
            if overlap_pixels > 0:
                overlap_count += overlap_pixels
                if self.verbose:
                    print(f"   ⚠️  Overlap detected: {overlap_pixels} pixels for embryo {embryo_num}")
            label_image[binary_mask > 0] = embryo_num
        num_embryos = len(embryo_data)
        output_filename = f"{image_id}_masks_emnum_{num_embryos}.{self.output_format}"
        experiment_id = image_id.split('_')[0]
        output_dir = self.output_base_dir / experiment_id / "masks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        if self.output_format == 'jpg':
            cv2.imwrite(str(output_path), label_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if self.verbose and num_embryos > 0:
                saved_img = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                unique_values = np.unique(saved_img)
                expected_values = set(range(num_embryos + 1))
                if not set(unique_values).issubset(expected_values):
                    print(f"   ⚠️  JPEG compression may have altered label values in {output_filename}")
        else:
            cv2.imwrite(str(output_path), label_image)
        self.export_stats["total_masks"] += num_embryos
        if overlap_count > 0:
            self.export_stats["overlapping_masks"] += 1
        return output_path

    def export_all_masks(self, max_workers: int = 4) -> Dict[str, Path]:
        if self.verbose:
            print(f"🎯 Starting mask export...")
            print(f"   Output directory: {self.output_base_dir}")
        export_paths = {}
        export_tasks = []
        for exp_id, exp_data in self.sam2_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_data.get("embryos"):
                        first_embryo = next(iter(image_data["embryos"].values()))
                        if first_embryo['segmentation_format'] == 'rle':
                            height, width = first_embryo['segmentation']['size']
                            image_shape = (height, width)
                        else:
                            image_shape = (512, 512)
                        export_tasks.append((image_id, image_data["embryos"], image_shape))
        if self.verbose:
            print(f"📊 Found {len(export_tasks)} images to export")
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
        self.export_stats["export_paths"] = {k: str(v) for k, v in export_paths.items()}
        self._save_export_manifest()
        if self.verbose:
            print(f"\n✅ Export complete!")
            print(f"   Total images: {self.export_stats['total_images']}")
            print(f"   Total masks: {self.export_stats['total_masks']}")
            print(f"   Images with overlaps: {self.export_stats['overlapping_masks']}")
        return export_paths

    def _save_export_manifest(self):
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
