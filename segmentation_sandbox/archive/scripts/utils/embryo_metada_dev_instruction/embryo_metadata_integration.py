"""
Module 7: Integration Layer for EmbryoMetadata

This module provides minimal but effective integration with SAM annotations
and GSAM ID management for bidirectional linking.

Author: EmbryoMetadata Development Team
Date: July 15, 2025
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union


class SamAnnotationIntegration:
    """Integration with GroundedSamAnnotation data."""
    
    @staticmethod
    def load_sam_annotations(sam_path: Path) -> Dict:
        """
        Load and validate SAM annotation file.
        
        Args:
            sam_path: Path to grounded_sam_annotations.json
        
        Returns:
            Loaded SAM annotations
            
        Raises:
            ValueError: If file invalid or missing required fields
        """
        if not sam_path.exists():
            raise FileNotFoundError(f"SAM annotation file not found: {sam_path}")
        
        try:
            with open(sam_path, 'r') as f:
                sam_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in SAM annotation file: {e}")
        
        # Basic validation - check for key structures
        if "experiments" not in sam_data:
            raise ValueError("SAM annotation missing 'experiments' field")
        
        if not isinstance(sam_data.get("experiments"), dict):
            raise ValueError("SAM annotation 'experiments' must be a dictionary")
        
        return sam_data
    
    @staticmethod
    def extract_embryo_structure(sam_data: Dict) -> Dict:
        """
        Extract embryo/snip structure from SAM annotations.
        
        Returns:
            Dict mapping embryo_id to metadata including snips
        """
        embryo_structure = {}
        
        for exp_id, exp_data in sam_data.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                # Process each image to find embryo/snip mappings
                for image_id, image_data in video_data.get("images", {}).items():
                    for embryo_id, embryo_info in image_data.get("embryos", {}).items():
                        if embryo_id not in embryo_structure:
                            embryo_structure[embryo_id] = {
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "snips": {}
                            }
                        
                        # Add snip
                        snip_id = embryo_info.get("snip_id")
                        if snip_id:
                            embryo_structure[embryo_id]["snips"][snip_id] = {
                                "image_id": image_id,
                                "frame_index": image_data.get("frame_index", -1),
                                "is_seed_frame": image_data.get("is_seed_frame", False),
                                "bbox": embryo_info.get("bbox", []),
                                "area": embryo_info.get("area", 0),
                                "mask_confidence": embryo_info.get("mask_confidence", 0.0)
                            }
        
        return embryo_structure


class GsamIdManager:
    """Manage GSAM annotation IDs for tracking."""
    
    @staticmethod
    def generate_gsam_id() -> int:
        """Generate a unique 4-digit GSAM annotation ID."""
        return random.randint(1000, 9999)
    
    @staticmethod
    def add_gsam_id_to_sam_annotation(sam_path: Path, 
                                     gsam_id: Optional[int] = None) -> int:
        """
        Add GSAM annotation ID to SAM annotation file.
        
        Args:
            sam_path: Path to SAM annotation file
            gsam_id: Specific ID to use (or generate new)
        
        Returns:
            The GSAM ID that was added
        """
        # Load SAM annotation
        with open(sam_path, 'r') as f:
            sam_data = json.load(f)
        
        # Generate ID if not provided
        if gsam_id is None:
            gsam_id = GsamIdManager.generate_gsam_id()
        
        # Add to file_info section
        if "file_info" not in sam_data:
            sam_data["file_info"] = {}
        
        sam_data["file_info"]["gsam_annotation_id"] = gsam_id
        sam_data["file_info"]["gsam_id_added"] = datetime.now().isoformat()
        
        # Save back
        with open(sam_path, 'w') as f:
            json.dump(sam_data, f, indent=2)
        
        return gsam_id
    
    @staticmethod
    def get_gsam_id_from_sam(sam_path: Path) -> Optional[int]:
        """Get existing GSAM ID from SAM annotation file."""
        try:
            with open(sam_path, 'r') as f:
                sam_data = json.load(f)
            return sam_data.get("file_info", {}).get("gsam_annotation_id")
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    @staticmethod
    def link_embryo_metadata_to_sam(metadata: 'EmbryoMetadata',
                                   sam_path: Path) -> int:
        """
        Create bidirectional link between metadata and SAM annotation.
        
        Args:
            metadata: EmbryoMetadata instance
            sam_path: Path to SAM annotation file
            
        Returns:
            The GSAM ID used for linking
        """
        # Check if SAM file has GSAM ID
        gsam_id = GsamIdManager.get_gsam_id_from_sam(sam_path)
        
        if not gsam_id:
            # Add GSAM ID to SAM file
            gsam_id = GsamIdManager.add_gsam_id_to_sam_annotation(sam_path)
            if metadata.verbose:
                print(f"✅ Added GSAM ID {gsam_id} to SAM annotation file")
        
        # Store in metadata
        if "file_info" not in metadata.data:
            metadata.data["file_info"] = {}
            
        metadata.data["file_info"]["gsam_annotation_id"] = gsam_id
        metadata.data["file_info"]["linked_sam_annotation"] = str(sam_path)
        metadata.data["file_info"]["link_created"] = datetime.now().isoformat()
        metadata._unsaved_changes = True
        
        if metadata.verbose:
            print(f"🔗 Linked EmbryoMetadata to SAM annotation with ID {gsam_id}")
        
        return gsam_id


class ConfigurationManager:
    """Manage configuration inheritance from source models."""
    
    @staticmethod
    def inherit_model_configs(metadata: 'EmbryoMetadata', 
                            sam_data: Dict) -> None:
        """
        Inherit model configurations from SAM annotations.
        
        Args:
            metadata: EmbryoMetadata instance
            sam_data: Loaded SAM annotation data
        """
        if "config" not in metadata.data:
            metadata.data["config"] = {}
        
        config = metadata.data["config"]
        
        # Detection model config (from GroundedDINO)
        seed_info = sam_data.get("seed_annotations_info", {})
        if seed_info:
            config["detection_model"] = {
                "config": seed_info.get("model_config", "unknown"),
                "weights": seed_info.get("model_weights", "unknown"),
                "architecture": seed_info.get("model_architecture", "GroundedDINO")
            }
        
        # Segmentation model config (from SAM2)
        sam2_info = sam_data.get("sam2_model_info", {})
        if sam2_info:
            config["segmentation_model"] = {
                "config": Path(sam2_info.get("config_path", "unknown")).name,
                "weights": Path(sam2_info.get("checkpoint_path", "unknown")).name,
                "architecture": sam2_info.get("model_architecture", "SAM2")
            }
        
        # Processing parameters
        config["processing_params"] = {
            "target_prompt": sam_data.get("target_prompt", "individual embryo"),
            "segmentation_format": sam_data.get("segmentation_format", "rle"),
            "sam_creation_time": sam_data.get("creation_time", "unknown"),
            "sam_last_updated": sam_data.get("last_updated", "unknown")
        }
        
        metadata._unsaved_changes = True


def create_embryo_metadata_from_sam(sam_path: Path, 
                                   metadata_path: Path,
                                   verbose: bool = True) -> 'EmbryoMetadata':
    """
    Create a new EmbryoMetadata file from SAM annotation.
    
    This is a convenience function that:
    1. Loads SAM annotation
    2. Extracts embryo structure  
    3. Creates EmbryoMetadata file
    4. Links them bidirectionally
    
    Args:
        sam_path: Path to SAM annotation file
        metadata_path: Path for new metadata file
        verbose: Enable verbose output
        
    Returns:
        New EmbryoMetadata instance
    """
    # Import here to avoid circular imports
    from embryo_metadata_refactored import EmbryoMetadata
    
    # Load SAM data
    sam_data = SamAnnotationIntegration.load_sam_annotations(sam_path)
    embryo_structure = SamAnnotationIntegration.extract_embryo_structure(sam_data)
    
    if verbose:
        print(f"📊 SAM annotation contains:")
        print(f"   Embryos: {len(embryo_structure)}")
        total_snips = sum(len(emb['snips']) for emb in embryo_structure.values())
        print(f"   Snips: {total_snips}")
    
    # Create EmbryoMetadata
    em = EmbryoMetadata(
        sam_annotation_path=sam_path,
        embryo_metadata_path=metadata_path,
        gen_if_no_file=True,
        verbose=verbose
    )
    
    # Initialize with SAM structure
    for embryo_id, embryo_info in embryo_structure.items():
        # Add embryo
        if embryo_id not in em.data["embryos"]:
            em.data["embryos"][embryo_id] = {
                "genotypes": {},
                "treatments": {},
                "flags": {},
                "source": {
                    "experiment_id": embryo_info["experiment_id"],
                    "video_id": embryo_info["video_id"],
                    "sam_annotation_source": str(sam_path)
                },
                "snips": {}
            }
        
        # Add snips
        for snip_id, snip_info in embryo_info["snips"].items():
            em.data["embryos"][embryo_id]["snips"][snip_id] = {
                "phenotype": {
                    "value": "NONE",
                    "author": "system",
                    "timestamp": datetime.now().isoformat()
                },
                "flags": []
            }
    
    # Link to SAM annotation
    gsam_id = GsamIdManager.link_embryo_metadata_to_sam(em, sam_path)
    
    # Inherit configurations
    ConfigurationManager.inherit_model_configs(em, sam_data)
    
    # Save
    em.save()
    
    if verbose:
        print(f"✅ Created EmbryoMetadata with GSAM ID {gsam_id}")
        print(f"💾 Saved to: {metadata_path}")
    
    return em


# Helper function to add to main EmbryoMetadata class
def _get_sam_features_for_snip(metadata: 'EmbryoMetadata', 
                              snip_id: str) -> Optional[Dict]:
    """
    Get SAM annotation features for a snip.
    
    This requires loading the linked SAM annotation file.
    """
    # Get linked SAM file
    sam_path = metadata.data.get("file_info", {}).get("linked_sam_annotation")
    if not sam_path or not Path(sam_path).exists():
        return None
    
    try:
        # Load SAM data
        sam_data = SamAnnotationIntegration.load_sam_annotations(Path(sam_path))
        
        # Find the snip
        for exp_data in sam_data.get("experiments", {}).values():
            for video_data in exp_data.get("videos", {}).values():
                for image_data in video_data.get("images", {}).values():
                    for embryo_id, embryo_info in image_data.get("embryos", {}).items():
                        if embryo_info.get("snip_id") == snip_id:
                            return {
                                "bbox": embryo_info.get("bbox", []),
                                "area": embryo_info.get("area", 0),
                                "mask_confidence": embryo_info.get("mask_confidence", 0.0),
                                "frame_index": image_data.get("frame_index", -1),
                                "is_seed_frame": image_data.get("is_seed_frame", False)
                            }
    except Exception:
        pass
    
    return None
