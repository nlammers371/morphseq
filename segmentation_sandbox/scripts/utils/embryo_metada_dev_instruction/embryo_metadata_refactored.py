"""
EmbryoMetadata Core Class Implementation (Refactored)
Module 1: Core class structure, initialization, and basic data management

This version uses mixin classes to keep the main class focused and manageable.
The original 1604-line file has been redistributed into focused modules.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Import our foundation modules
from base_annotation_parser import BaseAnnotationParser
from embryo_metadata_models import ValidationError
from embryo_metadata_utils import (
    validate_path, load_json, save_json, 
    validate_embryo_metadata_structure,
    DEFAULT_EMBRYO_METADATA_CONFIG
)
from permitted_values_manager import PermittedValuesManager

# Import our manager mixins
from embryo_phenotype_manager import EmbryoPhenotypeManager
from embryo_genotype_manager import EmbryoGenotypeManager
from embryo_flag_manager import EmbryoFlagManager
from embryo_treatment_manager import EmbryoTreatmentManager


class EmbryoMetadata(BaseAnnotationParser, 
                    EmbryoPhenotypeManager, 
                    EmbryoGenotypeManager, 
                    EmbryoFlagManager, 
                    EmbryoTreatmentManager):
    """
    Main class for managing embryo metadata including phenotypes, genotypes, and flags.
    
    This class provides:
    - Hierarchical data storage (experiment → video → image → snip)
    - Validation against permitted values schema
    - Change tracking and atomic saves
    - Integration with GroundedSamAnnotation data
    - Management of treatment annotations
    
    Inherits from BaseAnnotationParser for common functionality and uses
    manager mixins for specialized operations to keep code modular.
    """
    
    def __init__(self, sam_annotation_path: Union[str, Path], 
                 embryo_metadata_path: Optional[Union[str, Path]] = None,
                 gen_if_no_file: bool = False, 
                 auto_validate: bool = True, 
                 verbose: bool = True,
                 config: Optional[Dict] = None):
        """
        Initialize EmbryoMetadata instance.
        
        Args:
            sam_annotation_path: Path to GroundedSam annotation file
            embryo_metadata_path: Path to embryo metadata file (auto-generated if None)
            gen_if_no_file: Create new file if metadata doesn't exist
            auto_validate: Run consistency checks on initialization
            verbose: Enable verbose output
            config: Configuration overrides
        """
        # Step 1: Path validation
        self.sam_annotation_path = validate_path(sam_annotation_path, must_exist=True)
        
        if embryo_metadata_path is None:
            # Auto-generate metadata path from SAM annotation path
            embryo_metadata_path = self.sam_annotation_path.with_name(
                self.sam_annotation_path.stem + "_embryo_metadata.json"
            )
        
        # Initialize base class with embryo metadata path
        super().__init__(embryo_metadata_path, verbose=verbose)
        
        # Step 2: Load configuration
        self.config = DEFAULT_EMBRYO_METADATA_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Step 3: Initialize schema manager
        self.schema_manager = PermittedValuesManager()
        self.permitted_values = self.schema_manager.schema
        
        # Step 4: Load source data
        try:
            self.sam_annotations = self.load_json(self.sam_annotation_path)
            if not self.sam_annotations:
                raise ValueError("SAM annotations file is empty")
        except Exception as e:
            raise ValueError(f"Failed to load SAM annotations: {e}")
        
        # Step 5: Load or initialize embryo metadata
        if self.filepath.exists():
            try:
                self.data = self.load_json()
                self._validate_schema(self.data)
                if self.verbose:
                    print(f"📂 Loaded existing metadata: {len(self.data.get('embryos', {}))} embryos")
            except Exception as e:
                if gen_if_no_file:
                    if self.verbose:
                        print(f"⚠️  Failed to load existing metadata ({e}), creating new")
                    self.data = self._initialize_empty_metadata()
                else:
                    raise ValueError(f"Failed to load embryo metadata: {e}")
        elif gen_if_no_file:
            self.data = self._initialize_empty_metadata()
            if self.verbose:
                print(f"🆕 Created new metadata with {len(self.data['embryos'])} embryos")
        else:
            raise FileNotFoundError("Embryo metadata not found and gen_if_no_file=False")
        
        # Step 6: Initialize tracking
        self._initialize_caches()
        
        # Step 7: Inherit configurations from SAM annotations
        self._inherit_configurations()
        
        # Step 8: Consistency checks
        if auto_validate:
            self._run_consistency_checks()
    
    def _initialize_empty_metadata(self) -> Dict:
        """Create empty metadata structure with defaults."""
        metadata = {
            "file_info": {
                "version": "1.0",
                "creation_time": self.get_timestamp(),
                "last_updated": self.get_timestamp(),
                "source_sam_annotation": str(self.sam_annotation_path),
                "gsam_annotation_id": self._generate_gsam_id()
            },
            "permitted_values": self.permitted_values.copy(),
            "embryos": {},
            "flags": {"experiment": {}, "video": {}, "image": {}},
            "config": {}
        }
        
        # Import embryo structure from SAM annotations
        self._populate_from_sam_annotations(metadata)
        return metadata
    
    def _populate_from_sam_annotations(self, metadata: Dict) -> None:
        """Populate metadata structure from SAM annotations."""
        experiments = self.sam_annotations.get("experiments", {})
        
        for exp_id, exp_data in experiments.items():
            videos = exp_data.get("videos", {})
            
            for video_id, video_data in videos.items():
                embryo_ids = video_data.get("embryo_ids", [])
                images = video_data.get("images", {})
                
                for embryo_id in embryo_ids:
                    if embryo_id not in metadata["embryos"]:
                        metadata["embryos"][embryo_id] = {
                            "genotypes": {},
                            "treatments": {},
                            "phenotypes": {},
                            "flags": {},
                            "metadata": {
                                "created": self.get_timestamp(),
                                "last_updated": self.get_timestamp()
                            },
                            "source": {
                                "experiment_id": exp_id,
                                "video_id": video_id,
                                "sam_annotation_source": str(self.sam_annotation_path)
                            },
                            "snips": {}
                        }
                    
                    # Add snips for this embryo
                    for image_id, image_data in images.items():
                        embryos_in_image = image_data.get("embryos", {})
                        
                        if embryo_id in embryos_in_image:
                            embryo_data = embryos_in_image[embryo_id]
                            snip_id = embryo_data.get("snip_id")
                            
                            if snip_id:
                                metadata["embryos"][embryo_id]["snips"][snip_id] = {
                                    "phenotype": {
                                        "value": "NONE",
                                        "author": "system_init",
                                        "timestamp": self.get_timestamp()
                                    },
                                    "flags": []
                                }
    
    def _validate_schema(self, data: Dict) -> None:
        """Validate metadata structure against expected schema."""
        issues = validate_embryo_metadata_structure(data)
        if issues:
            raise ValidationError(f"Schema validation failed: {issues}")
        
        # Additional EmbryoMetadata specific validations
        required_keys = ["file_info", "embryos", "flags"]
        for key in required_keys:
            if key not in data:
                raise ValidationError(f"Missing required key: {key}")
    
    def _generate_gsam_id(self) -> int:
        """Generate a 4-digit random ID for GSAM annotation linking."""
        return random.randint(1000, 9999)
    
    def _initialize_caches(self) -> None:
        """Initialize internal caches for performance."""
        self._embryo_cache = {}
        self._snip_to_embryo_cache = {}
        self._embryo_to_snips_cache = {}
        
        # Pre-populate snip to embryo mapping
        for embryo_id, embryo_data in self.data.get("embryos", {}).items():
            snips = embryo_data.get("snips", {})
            for snip_id in snips.keys():
                self._snip_to_embryo_cache[snip_id] = embryo_id
            
            if snips:
                self._embryo_to_snips_cache[embryo_id] = list(snips.keys())
    
    def _inherit_configurations(self) -> None:
        """Inherit configuration from SAM annotations."""
        sam_config = self.sam_annotations.get("config", {})
        if sam_config:
            self.data["config"].update(sam_config)
            if self.verbose:
                print(f"🔧 Inherited configuration from SAM annotations")
    
    def _run_consistency_checks(self) -> None:
        """Run consistency checks between SAM annotations and metadata."""
        issues = []
        
        # Check embryo ID consistency
        sam_embryo_ids = set(self.sam_annotations.get("embryo_ids", []))
        metadata_embryo_ids = set(self.data["embryos"].keys())
        
        missing_in_metadata = sam_embryo_ids - metadata_embryo_ids
        extra_in_metadata = metadata_embryo_ids - sam_embryo_ids
        
        if missing_in_metadata:
            issues.append(f"Embryos in SAM but not in metadata: {missing_in_metadata}")
        
        if extra_in_metadata:
            issues.append(f"Embryos in metadata but not in SAM: {extra_in_metadata}")
        
        # Check snip ID consistency
        sam_snip_ids = set(self.sam_annotations.get("snip_ids", []))
        metadata_snip_ids = set()
        
        for embryo_data in self.data["embryos"].values():
            metadata_snip_ids.update(embryo_data.get("snips", {}).keys())
        
        missing_snips = sam_snip_ids - metadata_snip_ids
        if missing_snips and self.verbose:
            print(f"⚠️  {len(missing_snips)} snips in SAM but not in metadata")
        
        if issues:
            if self.config.get("validation", {}).get("strict_id_format", True):
                raise ValidationError(f"Consistency check failed: {issues}")
            elif self.verbose:
                print(f"⚠️  Consistency issues: {issues}")
    
    # -------------------------------------------------------------------------
    # Core Data Access Methods
    # -------------------------------------------------------------------------
    
    def get_embryo_ids(self) -> List[str]:
        """Get list of all embryo IDs."""
        return list(self.data["embryos"].keys())
    
    def get_snip_ids(self, embryo_id: Optional[str] = None) -> List[str]:
        """Get list of snip IDs."""
        if embryo_id:
            if embryo_id in self._embryo_to_snips_cache:
                return self._embryo_to_snips_cache[embryo_id].copy()
            
            embryo_data = self.data["embryos"].get(embryo_id, {})
            return list(embryo_data.get("snips", {}).keys())
        
        # Return all snip IDs
        return list(self._snip_to_embryo_cache.keys())
    
    def get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
        """Get embryo ID from snip ID using cache."""
        # Check cache first
        if snip_id in self._snip_to_embryo_cache:
            return self._snip_to_embryo_cache[snip_id]
        
        # Fallback to base parser method
        return super().get_embryo_id_from_snip(snip_id)
    
    def get_embryo_data(self, embryo_id: str) -> Optional[Dict]:
        """Get complete embryo data."""
        return self.data["embryos"].get(embryo_id)
    
    def get_snip_data(self, snip_id: str) -> Optional[Dict]:
        """Get snip data."""
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        if not embryo_id:
            return None
        
        embryo_data = self.data["embryos"].get(embryo_id, {})
        return embryo_data.get("snips", {}).get(snip_id)
    
    # -------------------------------------------------------------------------
    # Save and File Operations
    # -------------------------------------------------------------------------
    
    def save(self, backup: bool = True, force: bool = False) -> None:
        """Save metadata to file with atomic write."""
        if not self.has_unsaved_changes and not force:
            if self.verbose:
                print("💾 No changes to save")
            return
        
        # Update file info
        self.data["file_info"]["last_updated"] = self.get_timestamp()
        
        # Save using base class method (includes backup and atomic write)
        self.save_json(self.data, create_backup=backup)
        
        # Clear change tracking
        self.mark_saved()
        
        if self.verbose:
            embryo_count = len(self.data["embryos"])
            snip_count = sum(len(e.get("snips", {})) for e in self.data["embryos"].values())
            print(f"💾 Saved metadata: {embryo_count} embryos, {snip_count} snips")
    
    def backup(self) -> Path:
        """Create a backup of the current metadata file."""
        if not self.filepath.exists():
            raise FileNotFoundError("Cannot backup non-existent file")
        
        return self._create_backup(self.filepath)
    
    def reload(self) -> None:
        """Reload metadata from file, discarding unsaved changes."""
        if not self.filepath.exists():
            raise FileNotFoundError("Cannot reload non-existent file")
        
        old_changes = self.has_unsaved_changes
        
        self.data = self.load_json()
        self._validate_schema(self.data)
        self._initialize_caches()
        self.mark_saved()
        
        if self.verbose:
            if old_changes:
                print("🔄 Reloaded metadata (unsaved changes discarded)")
            else:
                print("🔄 Reloaded metadata")
    
    def __str__(self) -> str:
        """String representation of EmbryoMetadata."""
        embryo_count = len(self.data["embryos"])
        snip_count = sum(len(e.get("snips", {})) for e in self.data["embryos"].values())
        status = 'unsaved' if self.has_unsaved_changes else 'saved'
        return f"EmbryoMetadata({embryo_count} embryos, {snip_count} snips, {status})"
