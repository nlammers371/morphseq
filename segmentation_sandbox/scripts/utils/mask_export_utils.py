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
                 gsam_annotation_id: str = None, force_reexport: bool = False,
                 verbose: bool = True, output_format: str = "jpg"):
        self.sam2_path = Path(sam2_annotations_path)
        self.output_base_dir = Path(output_base_dir)
        self.verbose = verbose
        self.output_format = output_format.lower()
        self.gsam_annotation_id = gsam_annotation_id
        self.force_reexport = force_reexport
        if self.output_format not in ['jpg', 'jpeg', 'png']:
            raise ValueError(f"Invalid output format: {output_format}. Use 'jpg' or 'png'")
        if self.output_format == 'jpeg':
            self.output_format = 'jpg'
        with open(self.sam2_path, 'r') as f:
            self.sam2_data = json.load(f)
        self.manifest_path = self.output_base_dir / "mask_export_manifest.json"
        self.existing_manifest = self._load_existing_manifest()

    def _load_existing_manifest(self) -> Dict:
        """Load existing export manifest to track what's already been exported"""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {
            "exports": {},  # image_id -> export_info
            "gsam_mappings": {},  # gsam_id -> list of image_ids
            "last_updated": None,
            "source_files": []
        }

    def _needs_export(self, image_id: str, embryo_data: Dict) -> bool:
        """Check if image needs to be exported/re-exported"""
        if self.force_reexport:
            return True
        if image_id not in self.existing_manifest.get("exports", {}):
            return True
        export_info = self.existing_manifest["exports"][image_id]
        output_path = Path(export_info["output_path"])
        if not output_path.exists():
            if self.verbose:
                print(f"   🔄 Re-exporting {image_id}: output file missing")
            return True
        current_embryo_count = len(embryo_data)
        if export_info["embryo_count"] != current_embryo_count:
            if self.verbose:
                print(f"   🔄 Re-exporting {image_id}: embryo count changed ({export_info['embryo_count']} -> {current_embryo_count})")
            return True
        if self.gsam_annotation_id and export_info.get("gsam_id") != self.gsam_annotation_id:
            if self.verbose:
                print(f"   🔄 Re-exporting {image_id}: GSAM annotation updated")
            return True
        return False

    def _validate_existing_export(self, image_id: str, expected_embryo_count: int) -> bool:
        """Validate that existing export matches expected data"""
        if image_id not in self.existing_manifest.get("exports", {}):
            return False
        export_info = self.existing_manifest["exports"][image_id]
        output_path = Path(export_info["output_path"])
        if not output_path.exists():
            return False
        expected_filename_pattern = f"{image_id}_masks_emnum_{expected_embryo_count}.{self.output_format}"
        if output_path.name != expected_filename_pattern:
            return False
        return True

    def _export_and_track(self, image_id: str, embryo_data: Dict, image_shape: Tuple[int, int]) -> Path:
        """Export image masks and update tracking information"""
        output_path = self.export_image_masks(image_id, embryo_data, image_shape)
        export_info = {
            "image_id": image_id,
            "output_path": str(output_path),
            "embryo_count": len(embryo_data),
            "export_timestamp": datetime.now().isoformat(),
            "gsam_id": self.gsam_annotation_id,
            "source_file": str(self.sam2_path),
            "image_shape": image_shape
        }
        self.existing_manifest.setdefault("exports", {})[image_id] = export_info
        if self.gsam_annotation_id:
            gsam_mappings = self.existing_manifest.setdefault("gsam_mappings", {})
            gsam_mappings.setdefault(self.gsam_annotation_id, []).append(image_id)
        return output_path

    def _update_manifest(self, export_paths: Dict[str, Path]):
        """Update the export manifest with current state"""
        self.existing_manifest.update({
            "last_updated": datetime.now().isoformat(),
            "total_managed_images": len(export_paths),
            "gsam_annotation_id": self.gsam_annotation_id,
            "source_files": list(set(self.existing_manifest.get("source_files", []) + [str(self.sam2_path)]))
        })
        current_image_ids = set(export_paths.keys())
        existing_exports = self.existing_manifest.get("exports", {})
        orphaned_ids = set(existing_exports.keys()) - current_image_ids
        if orphaned_ids and self.verbose:
            print(f"   🧹 Cleaning up {len(orphaned_ids)} orphaned export entries")
        for orphaned_id in orphaned_ids:
            del existing_exports[orphaned_id]
        with open(self.manifest_path, 'w') as f:
            json.dump(self.existing_manifest, f, indent=2)
        if self.verbose:
            print(f"📄 Updated export manifest: {self.manifest_path}")

    def get_export_summary(self) -> Dict:
        """Get summary of current export state"""
        manifest = self.existing_manifest
        summary = {
            "total_exports": len(manifest.get("exports", {})),
            "gsam_sources": list(manifest.get("gsam_mappings", {}).keys()),
            "last_updated": manifest.get("last_updated"),
            "source_files": manifest.get("source_files", [])
        }
        exp_breakdown = {}
        for image_id, export_info in manifest.get("exports", {}).items():
            exp_id = image_id.split('_')[0]
            exp_breakdown.setdefault(exp_id, 0)
            exp_breakdown[exp_id] += 1
        summary["experiment_breakdown"] = exp_breakdown
        return summary

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
        """Export masks with incremental processing and GSAM_ID tracking"""
        if self.verbose:
            print(f"🎯 Starting incremental mask export...")
            print(f"   GSAM Annotation ID: {self.gsam_annotation_id}")
            print(f"   Force re-export: {self.force_reexport}")
        export_paths = {}
        export_tasks = []
        skipped_count = 0
        for exp_id, exp_data in self.sam2_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for image_id, image_data in video_data.get("images", {}).items():
                    if image_data.get("embryos"):
                        embryo_data = image_data["embryos"]
                        if self._needs_export(image_id, embryo_data):
                            first_embryo = next(iter(embryo_data.values()))
                            if first_embryo['segmentation_format'] == 'rle':
                                height, width = first_embryo['segmentation']['size']
                                image_shape = (height, width)
                            else:
                                image_shape = (512, 512)
                            export_tasks.append((image_id, embryo_data, image_shape))
                        else:
                            existing_info = self.existing_manifest["exports"][image_id]
                            export_paths[image_id] = Path(existing_info["output_path"])
                            skipped_count += 1
        if self.verbose:
            print(f"📊 Export analysis:")
            print(f"   Images to export: {len(export_tasks)}")
            print(f"   Images to skip: {skipped_count}")
            print(f"   Total images: {len(export_tasks) + skipped_count}")
        if export_tasks:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_image = {
                    executor.submit(self._export_and_track, task[0], task[1], task[2]): task[0]
                    for task in export_tasks
                }
                for future in as_completed(future_to_image):
                    image_id = future_to_image[future]
                    try:
                        export_path = future.result()
                        export_paths[image_id] = export_path
                        if self.verbose and len(export_paths) % 100 == 0:
                            print(f"   Exported {len(export_paths)} images...")
                    except Exception as e:
                        if self.verbose:
                            print(f"   ❌ Error exporting {image_id}: {e}")
        self._update_manifest(export_paths)
        if self.verbose:
            print(f"\n✅ Incremental export complete!")
            print(f"   New exports: {len(export_tasks)}")
            print(f"   Skipped (up-to-date): {skipped_count}")
            print(f"   Total managed: {len(export_paths)}")
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
