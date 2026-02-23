"""
EmbryoMetadata Core Class Implementation (Refactored)
Module 1: Core class structure, initialization, and basic data management

This version uses mixin classes to keep the main class focused and manageable.
The original 1604-line file has been redistributed into focused modules.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Import our foundation modules
from base_annotation_parser import BaseAnnotationParser
from data_managers.embryo_metadata_models import ValidationError
from embryo_metadata_utils import (
    validate_path, load_json, save_json, 
    validate_embryo_metadata_structure,
    DEFAULT_EMBRYO_METADATA_CONFIG
)
from data_managers.permitted_values_manager import PermittedValuesManager

# Import our manager mixins
from data_managers.embryo_phenotype_manager import EmbryoPhenotypeManager
from data_managers.embryo_genotype_manager import EmbryoGenotypeManager
from data_managers.embryo_flag_manager import EmbryoFlagManager
from data_managers.embryo_treatment_manager import EmbryoTreatmentManager

# Import batch processing capabilities
from embryo_metadata_batch import (
    RangeParser, TemporalRangeParser, BatchProcessor, BatchOperations,
    create_progress_callback, estimate_batch_time
)


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
    
    @property
    def snip_ids(self) -> List[str]:
        """Get all snip IDs across all embryos."""
        return self.get_snip_ids()
    
    # ========================
    # BATCH PROCESSING METHODS
    # ========================
    
    def batch_add_phenotypes(self, assignments: List[Dict], author: str,
                           validate_ranges: bool = True, 
                           parallel: bool = False,
                           auto_save_interval: Optional[int] = None) -> Dict:
        """
        Batch assign phenotypes using advanced range syntax.
        
        Args:
            assignments: List of assignment dictionaries with format:
                {
                    "embryo_id": "20240411_A01_e01",
                    "phenotype": "EDEMA", 
                    "frames": "[10:20]",  # or "all", "death:", etc.
                    "confidence": 0.95,   # optional
                    "notes": "Manual annotation"  # optional
                }
            author: Author of annotations
            validate_ranges: Validate frame ranges exist
            parallel: Use parallel processing
            auto_save_interval: Auto-save after N operations
            
        Returns:
            Processing results dictionary
        """
        if self.verbose:
            print(f"🔄 Starting batch phenotype assignment: {len(assignments)} assignments")
            if parallel:
                print("⚡ Using parallel processing")
        
        results = BatchOperations.batch_phenotype_assignment(
            self, assignments, author, validate_ranges, parallel
        )
        
        if auto_save_interval and results["assigned"] > 0:
            self.save()
            if self.verbose:
                print("💾 Auto-saved after batch operation")
        
        if self.verbose:
            print(f"✅ Batch complete: {results['assigned']} phenotypes assigned")
            if results["failed"]:
                print(f"❌ {len(results['failed'])} assignments failed")
        
        return results
    
    def batch_add_genotypes(self, assignments: List[Dict], author: str,
                          overwrite: bool = False,
                          parallel: bool = False,
                          auto_save_interval: Optional[int] = None) -> Dict:
        """
        Batch assign genotypes to embryos.
        
        Args:
            assignments: List of assignment dictionaries with format:
                {
                    "embryo_id": "20240411_A01_e01",
                    "genotype": "WT",
                    "gene": "lmx1b",  # optional
                    "notes": "PCR confirmed"  # optional
                }
            author: Author of annotations
            overwrite: Allow overwriting existing genotypes
            parallel: Use parallel processing  
            auto_save_interval: Auto-save after N operations
            
        Returns:
            Processing results dictionary
        """
        if self.verbose:
            print(f"🔄 Starting batch genotype assignment: {len(assignments)} assignments")
        
        results = BatchOperations.batch_genotype_assignment(
            self, assignments, author, overwrite, parallel
        )
        
        if auto_save_interval and results["assigned"] > 0:
            self.save()
            if self.verbose:
                print("💾 Auto-saved after batch operation")
        
        if self.verbose:
            print(f"✅ Batch complete: {results['assigned']} genotypes assigned")
        
        return results
    
    def batch_detect_flags(self, detectors: List[Dict],
                          entities: Optional[List[str]] = None,
                          parallel: bool = True,
                          auto_save_interval: Optional[int] = None) -> Dict:
        """
        Run batch flag detection with custom detectors.
        
        Args:
            detectors: List of detector configurations with format:
                {
                    "name": "motion_blur_detector",
                    "level": "snip",
                    "function": detect_function,
                    "params": {"threshold": 0.1},
                    "severity": "warning"
                }
            entities: Specific entities to check (None = all)
            parallel: Use parallel processing
            auto_save_interval: Auto-save after N operations
            
        Returns:
            Detection results dictionary
        """
        if self.verbose:
            print(f"🔍 Starting batch flag detection: {len(detectors)} detectors")
        
        results = BatchOperations.batch_flag_detection(
            self, detectors, entities, parallel
        )
        
        if auto_save_interval:
            self.save()
            if self.verbose:
                print("💾 Auto-saved after batch detection")
        
        if self.verbose:
            total_flags = sum(r["flags_added"] for r in results.values())
            print(f"🚩 Batch detection complete: {total_flags} flags added")
        
        return results
    
    def parse_range(self, range_spec: Union[str, List], 
                   embryo_id: Optional[str] = None) -> List[str]:
        """
        Parse range specification into list of IDs.
        
        Args:
            range_spec: Range specification (e.g., "[10:20]", "all", ["id1", "id2"])
            embryo_id: Embryo ID for temporal ranges (required for temporal syntax)
            
        Returns:
            List of resolved IDs
        """
        if embryo_id and isinstance(range_spec, str):
            # Use temporal parser for embryo-specific ranges
            return TemporalRangeParser.parse_frame_range(embryo_id, range_spec, self)
        else:
            # Use general range parser
            if embryo_id:
                # Get all snips for embryo
                embryo_data = self.data["embryos"].get(embryo_id, {})
                available_items = sorted(embryo_data.get("snips", {}).keys())
            else:
                # Use all snip IDs
                available_items = self.snip_ids
            
            return RangeParser.parse_range(range_spec, available_items)
    
    def estimate_processing_time(self, operation_count: int,
                               operation_type: str = "annotation") -> str:
        """
        Estimate time for batch operations.
        
        Args:
            operation_count: Number of operations
            operation_type: Type of operation (affects speed estimate)
            
        Returns:
            Human-readable time estimate
        """
        # Speed estimates based on operation type
        speeds = {
            "annotation": 50.0,    # annotations per second
            "phenotype": 100.0,    # phenotype additions per second  
            "genotype": 200.0,     # genotype additions per second
            "flag": 300.0,         # flag operations per second
            "validation": 500.0    # validations per second
        }
        
        speed = speeds.get(operation_type, 50.0)
        return estimate_batch_time(operation_count, speed)
    
    def create_batch_processor(self, parallel: bool = False, 
                             num_workers: int = 4,
                             progress_callback: Optional[callable] = None) -> BatchProcessor:
        """
        Create a batch processor instance for custom operations.
        
        Args:
            parallel: Enable parallel processing
            num_workers: Number of parallel workers
            progress_callback: Custom progress callback
            
        Returns:
            Configured BatchProcessor instance
        """
        if progress_callback is None and self.verbose:
            progress_callback = create_progress_callback(verbose=True)
        
        return BatchProcessor(
            self, 
            parallel=parallel,
            num_workers=num_workers, 
            progress_callback=progress_callback,
            verbose=self.verbose
        )
    
    # ========================
    # HELPER METHODS FOR BATCH PROCESSING
    # ========================
    
    def _get_all_entities_at_level(self, level: str) -> List[str]:
        """Get all entity IDs at specified level."""
        if level == "experiment":
            return list(self.data.get("experiments", {}).keys())
        elif level == "video":
            videos = []
            for exp_data in self.data.get("experiments", {}).values():
                videos.extend(exp_data.get("videos", {}).keys())
            return videos
        elif level == "image":
            images = []
            for exp_data in self.data.get("experiments", {}).values():
                for video_data in exp_data.get("videos", {}).values():
                    images.extend(video_data.get("images", {}).keys())
            return images
        elif level == "embryo":
            return list(self.data.get("embryos", {}).keys())
        elif level == "snip":
            return self.snip_ids
        else:
            raise ValueError(f"Unknown entity level: {level}")
    
    def _get_entity_data(self, entity_id: str, level: str) -> Optional[Dict]:
        """Get data for entity at specified level."""
        try:
            if level == "snip":
                # Find snip in embryo data
                for embryo_data in self.data["embryos"].values():
                    if entity_id in embryo_data.get("snips", {}):
                        return embryo_data["snips"][entity_id]
                return None
            
            elif level == "embryo":
                return self.data["embryos"].get(entity_id)
            
            elif level == "image":
                # Parse image ID to find in hierarchy
                exp_id = self.get_experiment_id_from_entity(entity_id)
                video_id = self.get_video_id_from_entity(entity_id)
                exp_data = self.data.get("experiments", {}).get(exp_id, {})
                video_data = exp_data.get("videos", {}).get(video_id, {})
                return video_data.get("images", {}).get(entity_id)
            
            # Add other levels as needed
            return None
            
        except Exception:
            return None
    
    # Integration Layer Methods (Module 7)
    
    def link_to_sam_annotation(self, sam_path: Union[str, Path]) -> int:
        """
        Link this metadata to a SAM annotation file.
        
        Args:
            sam_path: Path to SAM annotation file
            
        Returns:
            GSAM ID used for linking
        """
        from embryo_metadata_integration import GsamIdManager
        return GsamIdManager.link_embryo_metadata_to_sam(self, Path(sam_path))
    
    def get_sam_features_for_snip(self, snip_id: str) -> Optional[Dict]:
        """Get SAM annotation features for a snip."""
        from embryo_metadata_integration import _get_sam_features_for_snip
        return _get_sam_features_for_snip(self, snip_id)
    
    def inherit_sam_configs(self) -> None:
        """Inherit model configurations from linked SAM annotation."""
        from embryo_metadata_integration import SamAnnotationIntegration, ConfigurationManager
        
        sam_path = self.data.get("file_info", {}).get("linked_sam_annotation")
        if sam_path and Path(sam_path).exists():
            sam_data = SamAnnotationIntegration.load_sam_annotations(Path(sam_path))
            ConfigurationManager.inherit_model_configs(self, sam_data)
            if self.verbose:
                print("🔧 Inherited model configurations from SAM annotation")
        else:
            if self.verbose:
                print("⚠️ No linked SAM annotation found")
