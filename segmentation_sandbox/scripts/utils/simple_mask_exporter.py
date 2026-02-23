"""
Simple Mask Exporter for SAM2 Annotations
CRUD-based implementation using EntityIDTracker
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from .mask_utils import decode_mask_rle
from .entity_id_tracker import EntityIDTracker
from .parsing_utils import extract_experiment_id


class SimpleMaskExporter:
    """Export SAM2 masks as labeled images where pixel value = embryo number"""
    
    def __init__(self, sam2_path: Path, output_dir: Path, format: str = "png"):
        self.sam2_path = Path(sam2_path)
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        # Default (monolithic) manifest path; may be overridden to per-experiment below
        self.manifest_path = self.output_dir / "mask_export_manifest.json"
        
        if self.format not in ['jpg', 'jpeg', 'png', 'tiff']:
            raise ValueError(f"Unsupported format: {format}")
        if self.format == 'jpeg':
            self.format = 'jpg'
        
        # Load SAM2 data
        with open(self.sam2_path) as f:
            self.sam2_data = json.load(f)

        # If the annotations file is per-experiment (single experiment present),
        # write the manifest under that experiment's directory to avoid a monolithic manifest.
        try:
            experiments = list(self.sam2_data.get("experiments", {}).keys())
        except Exception:
            experiments = []

        if len(experiments) == 1:
            exp_id = experiments[0]
            exp_dir = self.output_dir / exp_id
            self.manifest_path = exp_dir / f"mask_export_manifest_{exp_id}.json"
    
    def process_missing_masks(self, experiment_ids: Optional[List[str]] = None, 
                            overwrite: bool = False) -> Dict[str, Path]:
        """
        Export missing masks with optional experiment filtering and overwrite control.
        
        Args:
            experiment_ids: List of experiment IDs to process, None for all
            overwrite: If True, re-export existing files
            
        Returns:
            Dict mapping image_id to output_path for exported masks
        """
        if overwrite:
            # Get all images from SAM2
            current_entities = EntityIDTracker.extract_entities(self.sam2_data)
            target_images = current_entities["images"]
        else:
            # Get only missing images
            target_images = self._get_missing_images()
        
        # Filter by experiments if specified
        if experiment_ids:
            target_images = {img for img in target_images 
                           if extract_experiment_id(img) in experiment_ids}
        
        # Export images
        exported = self._export_images(target_images)
        
        # Update manifest (always), so per-experiment manifest is created/kept current
        self._update_manifest()
        
        return exported
    
    def get_export_status(self) -> Dict:
        """Get summary of export status"""
        current_entities = EntityIDTracker.extract_entities(self.sam2_data)
        manifest_entities = self._load_manifest_entities()
        
        missing_images = current_entities["images"] - manifest_entities["images"]
        
        return {
            "total_images": len(current_entities["images"]),
            "exported_images": len(manifest_entities["images"]),
            "missing_images": len(missing_images),
            "available_experiments": list({extract_experiment_id(img) 
                                         for img in current_entities["images"]}),
            "format": self.format
        }
    
    def _get_missing_images(self) -> Set[str]:
        """Get images that need to be exported"""
        current_entities = EntityIDTracker.extract_entities(self.sam2_data)
        manifest_entities = self._load_manifest_entities()
        
        return current_entities["images"] - manifest_entities["images"]
    
    def _export_images(self, image_ids: Set[str]) -> Dict[str, Path]:
        """Export set of images to labeled mask files"""
        exported = {}
        
        # Handle both list and dictionary formats for image_ids
        image_ids_list = sorted(image_ids.keys()) if isinstance(image_ids, dict) else image_ids
        for image_id in image_ids_list:
            embryo_data = self._get_embryo_data(image_id)
            if not embryo_data:
                continue
                
            try:
                output_path = self._export_single_image(image_id, embryo_data)
                exported[image_id] = output_path
            except Exception as e:
                print(f"Failed to export {image_id}: {e}")
        
        return exported
    
    def _export_single_image(self, image_id: str, embryo_data: Dict) -> Path:
        """Convert embryo masks to labeled image and save"""
        # Get image dimensions from first mask
        first_embryo = next(iter(embryo_data.values()))
        segmentation = first_embryo['segmentation']
        
        height, width = segmentation['size']
        label_image = np.zeros((height, width), dtype=np.uint8)
        
        # Convert each embryo mask
        for embryo_id, mask_data in embryo_data.items():
            embryo_num = int(embryo_id.split('_e')[-1])
            binary_mask = decode_mask_rle(mask_data['segmentation'])
            label_image[binary_mask > 0] = embryo_num
        
        # Save to file
        output_path = self._get_output_path(image_id, len(embryo_data))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.format == 'jpg':
            cv2.imwrite(str(output_path), label_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        elif self.format == 'png':
            cv2.imwrite(str(output_path), label_image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(str(output_path), label_image)
        
        return output_path
    
    def _get_embryo_data(self, image_id: str) -> Optional[Dict]:
        """Extract embryo data for image from SAM2 structure"""
        for exp_data in self.sam2_data.get("experiments", {}).values():
            for video_data in exp_data.get("videos", {}).values():
                if image_id in video_data.get("image_ids", {}):
                    return video_data["image_ids"][image_id].get("embryos", {})
        return None
    
    def _get_output_path(self, image_id: str, embryo_count: int) -> Path:
        """Generate output path with embryo count in filename"""
        experiment_id = extract_experiment_id(image_id)
        filename = f"{image_id}_masks_emnum_{embryo_count}.{self.format}"
        return self.output_dir / experiment_id / "masks" / filename
    
    
    def _load_manifest_entities(self) -> Dict[str, Set[str]]:
        """Load entity tracking from manifest file"""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                manifest = json.load(f)
                entity_data = manifest.get("entity_tracking", {})
                # Convert lists back to sets
                return {k: set(v) for k, v in entity_data.items()}
        
        # Return empty entity structure if no manifest
        return {
            "experiments": set(),
            "videos": set(),
            "images": set(),
            "embryos": set(),
            "snips": set()
        }
    
    def _update_manifest(self):
        """Update manifest with current export state"""
        # Get current entities
        current_entities = EntityIDTracker.extract_entities(self.sam2_data)
        
        # Load existing manifest or create new
        manifest = {}
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                manifest = json.load(f)
        
        # Update entity tracking (convert sets to lists for JSON)
        manifest["entity_tracking"] = {k: list(v) for k, v in current_entities.items()}
        
        # Update exports section
        if "exports" not in manifest:
            manifest["exports"] = {}
        
        # Add/update export info for current images
        for image_id in current_entities["images"]:
            embryo_data = self._get_embryo_data(image_id)
            if embryo_data:
                embryo_count = len(embryo_data)
                output_path = self._get_output_path(image_id, embryo_count)
                
                if output_path.exists():
                    # Get image shape from first embryo
                    first_embryo = next(iter(embryo_data.values()))
                    height, width = first_embryo['segmentation']['size']
                    
                    manifest["exports"][image_id] = {
                        "image_id": image_id,
                        "output_path": str(output_path),
                        "embryo_count": embryo_count,
                        "export_timestamp": datetime.now().isoformat(),
                        "source_file": str(self.sam2_path),
                        "image_shape": [height, width]
                    }
        
        # Update metadata
        manifest["last_updated"] = datetime.now().isoformat()
        manifest["total_exported"] = len([k for k, v in manifest["exports"].items() 
                                        if Path(v["output_path"]).exists()])
        
        # Save manifest
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


# Usage example
if __name__ == "__main__":
    exporter = SimpleMaskExporter("sam2_annotations.json", "mask_outputs/", format="png")
    
    # Export missing masks for all experiments
    exported = exporter.process_missing_masks()
    
    # Export specific experiment, overwriting existing
    exported = exporter.process_missing_masks(experiment_ids=["20240411"], overwrite=True)
    
    # Check export status
    status = exporter.get_export_status()
    print(f"Exported: {status['exported_images']}/{status['total_images']} images")
