"""
EmbryoMetadata Core Class Implementation
Module 1: Core class structure, initialization, and basic data management
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Import our foundation modules
from base_annotation_parser import BaseAnnotationParser
from embryo_metadata_models import (
    Phenotype, Genotype, Flag, Treatment, 
    Validator, Serializer, ValidationError
)
from embryo_metadata_utils import (
    validate_path, load_json, save_json, 
    validate_embryo_metadata_structure,
    DEFAULT_EMBRYO_METADATA_CONFIG
)
from permitted_values_manager import PermittedValuesManager

class EmbryoMetadata(BaseAnnotationParser):
    """
    Main class for managing embryo metadata including phenotypes, genotypes, and flags.
    
    This class provides:
    - Hierarchical data storage (experiment → video → image → snip)
    - Validation against permitted values schema
    - Change tracking and atomic saves
    - Integration with GroundedSamAnnotation data
    - Management of treatment annotations
    
    Inherits from BaseAnnotationParser for common functionality.
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
        """
        Create empty metadata structure with defaults.
        
        Returns:
            Empty metadata structure populated with embryos from SAM annotations
        """
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
            "flags": {
                "experiment": {},
                "video": {},
                "image": {}
            },
            "config": {}
        }
        
        # Import embryo structure from SAM annotations
        self._populate_from_sam_annotations(metadata)
        
        return metadata
    
    def _populate_from_sam_annotations(self, metadata: Dict) -> None:
        """
        Populate metadata structure from SAM annotations.
        
        Args:
            metadata: Metadata dictionary to populate
        """
        experiments = self.sam_annotations.get("experiments", {})
        
        for exp_id, exp_data in experiments.items():
            videos = exp_data.get("videos", {})
            
            for video_id, video_data in videos.items():
                embryo_ids = video_data.get("embryo_ids", [])
                images = video_data.get("images", {})
                
                for embryo_id in embryo_ids:
                    if embryo_id not in metadata["embryos"]:
                        metadata["embryos"][embryo_id] = {
                            "genotype": None,
                            "treatments": {},  # Support multiple treatments
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
        """
        Validate metadata structure against expected schema.
        
        Args:
            data: Metadata to validate
            
        Raises:
            ValidationError: If schema validation fails
        """
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
        """
        Get list of snip IDs.
        
        Args:
            embryo_id: If provided, return snips for this embryo only
            
        Returns:
            List of snip IDs
        """
        if embryo_id:
            if embryo_id in self._embryo_to_snips_cache:
                return self._embryo_to_snips_cache[embryo_id].copy()
            
            embryo_data = self.data["embryos"].get(embryo_id, {})
            return list(embryo_data.get("snips", {}).keys())
        
        # Return all snip IDs
        return list(self._snip_to_embryo_cache.keys())
    
    def get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
        """
        Get embryo ID from snip ID using cache.
        
        Args:
            snip_id: Snip ID to look up
            
        Returns:
            Embryo ID or None if not found
        """
        # Check cache first
        if snip_id in self._snip_to_embryo_cache:
            return self._snip_to_embryo_cache[snip_id]
        
        # Fallback to base parser method
        return super().get_embryo_id_from_snip(snip_id)
    
    def get_embryo_data(self, embryo_id: str) -> Optional[Dict]:
        """
        Get complete embryo data.
        
        Args:
            embryo_id: Embryo ID
            
        Returns:
            Embryo data dictionary or None if not found
        """
        return self.data["embryos"].get(embryo_id)
    
    def get_snip_data(self, snip_id: str) -> Optional[Dict]:
        """
        Get snip data.
        
        Args:
            snip_id: Snip ID
            
        Returns:
            Snip data dictionary or None if not found
        """
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        if not embryo_id:
            return None
        
        embryo_data = self.data["embryos"].get(embryo_id, {})
        return embryo_data.get("snips", {}).get(snip_id)
    
    # -------------------------------------------------------------------------
    # Save and File Operations
    # -------------------------------------------------------------------------
    
    def save(self, backup: bool = True, force: bool = False) -> None:
        """
        Save metadata to file with atomic write.
        
        Args:
            backup: Create backup before saving
            force: Save even if no changes detected
        """
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
        """
        Create a backup of the current metadata file.
        
        Returns:
            Path to created backup file
        """
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
    
    # -------------------------------------------------------------------------
    # Summary and Statistics
    # -------------------------------------------------------------------------
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the metadata.
        
        Returns:
            Dictionary with summary statistics
        """
        embryos = self.data["embryos"]
        
        # Count statistics
        total_embryos = len(embryos)
        total_snips = sum(len(e.get("snips", {})) for e in embryos.values())
        
        # Genotype statistics
        genotyped_embryos = sum(1 for e in embryos.values() if e.get("genotype"))
        missing_genotypes = total_embryos - genotyped_embryos
        
        # Phenotype statistics
        phenotyped_snips = 0
        phenotype_counts = {}
        
        for embryo_data in embryos.values():
            for snip_data in embryo_data.get("snips", {}).values():
                phenotype = snip_data.get("phenotype", {})
                if phenotype and phenotype.get("value") != "NONE":
                    phenotyped_snips += 1
                    pheno_val = phenotype.get("value", "UNKNOWN")
                    phenotype_counts[pheno_val] = phenotype_counts.get(pheno_val, 0) + 1
        
        # Flag statistics
        total_flags = 0
        flag_counts = {}
        
        for level_data in self.data["flags"].values():
            for entity_flags in level_data.values():
                if isinstance(entity_flags, list):
                    total_flags += len(entity_flags)
                    for flag in entity_flags:
                        flag_val = flag.get("value", "UNKNOWN") if isinstance(flag, dict) else str(flag)
                        flag_counts[flag_val] = flag_counts.get(flag_val, 0) + 1
        
        return {
            "file_info": self.data["file_info"],
            "totals": {
                "embryos": total_embryos,
                "snips": total_snips,
                "flags": total_flags
            },
            "genotypes": {
                "genotyped": genotyped_embryos,
                "missing": missing_genotypes,
                "completion_rate": genotyped_embryos / total_embryos if total_embryos > 0 else 0
            },
            "phenotypes": {
                "phenotyped_snips": phenotyped_snips,
                "completion_rate": phenotyped_snips / total_snips if total_snips > 0 else 0,
                "counts": phenotype_counts
            },
            "flags": {
                "total": total_flags,
                "counts": flag_counts
            },
            "has_unsaved_changes": self.has_unsaved_changes
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of the metadata."""
        summary = self.get_summary()
        
        print("📊 EmbryoMetadata Summary")
        print("=" * 50)
        
        # File info
        print(f"📂 File: {self.filepath.name}")
        print(f"🔗 Source: {Path(summary['file_info']['source_sam_annotation']).name}")
        print(f"📅 Last updated: {summary['file_info']['last_updated']}")
        print()
        
        # Totals
        totals = summary["totals"]
        print(f"🧬 Embryos: {totals['embryos']}")
        print(f"🔬 Snips: {totals['snips']}")
        print(f"🚩 Flags: {totals['flags']}")
        print()
        
        # Genotypes
        geno = summary["genotypes"]
        print(f"🧪 Genotypes: {geno['genotyped']}/{totals['embryos']} ({geno['completion_rate']:.1%})")
        if geno["missing"] > 0:
            print(f"   ⚠️  Missing: {geno['missing']} embryos")
        print()
        
        # Phenotypes
        pheno = summary["phenotypes"]
        print(f"🎯 Phenotypes: {pheno['phenotyped_snips']}/{totals['snips']} ({pheno['completion_rate']:.1%})")
        if pheno["counts"]:
            print("   Top phenotypes:")
            for phenotype, count in sorted(pheno["counts"].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     • {phenotype}: {count}")
        print()
        
        # Status
        if summary["has_unsaved_changes"]:
            print("💾 Status: Unsaved changes")
        else:
            print("✅ Status: All changes saved")
    
    # =============================================================================
    # PHENOTYPE MANAGEMENT METHODS
    # =============================================================================
    
    def add_phenotype(self, snip_id: str, phenotype: str, author: str,
                     notes: str = None, confidence: float = None,
                     force_dead: bool = False) -> bool:
        """
        Add a phenotype to a snip.
        
        Args:
            snip_id: Valid snip ID (e.g., "20240411_A01_e01_0001")
            phenotype: Phenotype value (must be in permitted values)
            author: Author of the annotation
            notes: Optional notes
            confidence: Confidence score (0.0-1.0) for ML predictions
            force_dead: Allow adding DEAD even with existing phenotypes
            
        Returns:
            bool: True if added successfully
            
        Raises:
            ValueError: If validation fails
        """
        # Validate snip ID format
        if not self.validate_id_format(snip_id, "snip"):
            raise ValueError(f"Invalid snip ID format: {snip_id}")
        
        # Get embryo ID from snip
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        if not embryo_id:
            raise ValueError(f"Cannot find embryo for snip: {snip_id}")
        
        # Validate phenotype against schema
        phenotypes = self.schema_manager.get_phenotypes()
        if not self.schema_manager.is_valid_phenotype(phenotype):
            available = list(phenotypes.keys())
            raise ValueError(f"Invalid phenotype '{phenotype}'. Available: {available}")
        
        # Get existing phenotypes for validation
        snip_data = self.get_snip_data(snip_id)
        existing_phenotypes = []
        if snip_data and "phenotype" in snip_data:
            existing_phenotypes.append(snip_data["phenotype"]["value"])
        
        # Validate using our validator
        try:
            Validator.validate_phenotype(phenotype, phenotypes, existing_phenotypes)
        except ValidationError as e:
            if not force_dead:
                raise ValueError(str(e))
        
        # Initialize embryo and snip data if needed
        if embryo_id not in self.data["embryos"]:
            self.data["embryos"][embryo_id] = {
                "genotypes": {},
                "treatments": {},  # Support multiple treatments
                "phenotypes": {},
                "flags": {},
                "metadata": {
                    "created": self.get_timestamp(),
                    "last_updated": self.get_timestamp()
                },
                "source": {"sam_annotation_source": str(self.sam_annotation_path)},
                "snips": {}
            }
        
        if snip_id not in self.data["embryos"][embryo_id]["snips"]:
            self.data["embryos"][embryo_id]["snips"][snip_id] = {"flags": []}
        
        # Create phenotype using our model
        phenotype_obj = Phenotype(
            value=phenotype,
            author=author,
            notes=notes,
            confidence=confidence
        )
        
        # Add to snip data
        self.data["embryos"][embryo_id]["snips"][snip_id]["phenotype"] = phenotype_obj.to_dict()
        
        # Log operation
        self.log_operation("add_phenotype", snip_id,
                         phenotype=phenotype, author=author, confidence=confidence)
        
        if self.verbose:
            print(f"✅ Added phenotype '{phenotype}' to {snip_id}")
        
        return True
    
    def edit_phenotype(self, embryo_id: str, phenotype_name: str,
                      severity: str = None, confidence: float = None,
                      notes: str = None) -> bool:
        """
        Edit an existing phenotype.
        
        Args:
            embryo_id: Valid embryo ID
            phenotype_name: Name of existing phenotype
            severity: New severity (optional)
            confidence: New confidence (optional)
            notes: New notes (optional)
            
        Returns:
            bool: True if edited successfully
            
        Raises:
            ValueError: If phenotype doesn't exist or validation fails
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if phenotype exists
        if phenotype_name not in self.data["embryos"][embryo_id]["phenotypes"]:
            available = list(self.data["embryos"][embryo_id]["phenotypes"].keys())
            raise ValueError(f"Phenotype '{phenotype_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Get current phenotype
        current = self.data["embryos"][embryo_id]["phenotypes"][phenotype_name]
        
        # Update fields if provided
        if severity is not None:
            if not self.schema_manager.validate_value("severity_levels", severity):
                available = self.schema_manager.get_values("severity_levels")
                raise ValueError(f"Invalid severity '{severity}'. Available: {available}")
            current["severity"] = severity
        
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            current["confidence"] = confidence
        
        if notes is not None:
            current["notes"] = notes
        
        # Update timestamp
        current["last_updated"] = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        changes = {}
        if severity is not None:
            changes["severity"] = severity
        if confidence is not None:
            changes["confidence"] = confidence
        if notes is not None:
            changes["notes"] = notes
        
        self.log_operation("edit_phenotype", embryo_id, 
                         phenotype=phenotype_name, changes=changes)
        
        if self.verbose:
            print(f"✅ Updated phenotype '{phenotype_name}' for {embryo_id}")
        
        return True
    
    def remove_phenotype(self, embryo_id: str, phenotype_name: str) -> bool:
        """
        Remove a phenotype from an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            phenotype_name: Name of phenotype to remove
            
        Returns:
            bool: True if removed successfully
            
        Raises:
            ValueError: If embryo or phenotype doesn't exist
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if phenotype exists
        if phenotype_name not in self.data["embryos"][embryo_id]["phenotypes"]:
            available = list(self.data["embryos"][embryo_id]["phenotypes"].keys())
            raise ValueError(f"Phenotype '{phenotype_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Remove phenotype
        removed = self.data["embryos"][embryo_id]["phenotypes"].pop(phenotype_name)
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Update summary stats
        self._update_summary_stats()
        
        # Log operation
        self.log_operation("remove_phenotype", embryo_id, 
                         phenotype=phenotype_name, removed_data=removed)
        
        if self.verbose:
            print(f"🗑️ Removed phenotype '{phenotype_name}' from {embryo_id}")
        
        return True
    
    def get_phenotypes(self, embryo_id: str) -> dict:
        """
        Get all phenotypes for an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            
        Returns:
            dict: Phenotypes data
        """
        if embryo_id not in self.data["embryos"]:
            return {}
        
        return self.data["embryos"][embryo_id]["phenotypes"].copy()
    
    def list_phenotypes_by_name(self, phenotype_name: str) -> list:
        """
        Find all embryos with a specific phenotype.
        
        Args:
            phenotype_name: Name of phenotype to search for
            
        Returns:
            list: List of embryo IDs with this phenotype
        """
        embryos_with_phenotype = []
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if phenotype_name in embryo_data["phenotypes"]:
                embryos_with_phenotype.append(embryo_id)
        
        return embryos_with_phenotype
    
    # =============================================================================
    # GENOTYPE MANAGEMENT METHODS
    # =============================================================================
    
    def add_genotype(self, embryo_id: str, gene_name: str, allele: str,
                    zygosity: str = "heterozygous", confidence: float = 1.0,
                    notes: str = "", overwrite: bool = False) -> bool:
        """
        Add a genotype to an embryo. 
        
        CRITICAL: Only ONE genotype per embryo is allowed (experimental design requirement).
        If embryo already has a genotype, this will fail unless overwrite=True.
        
        Args:
            embryo_id: Valid embryo ID
            gene_name: Gene name (must be in permitted values)
            allele: Allele designation
            zygosity: Zygosity (homozygous, heterozygous, hemizygous)
            confidence: Confidence score (0.0-1.0)
            notes: Optional notes
            overwrite: Whether to overwrite existing genotype
            
        Returns:
            bool: True if added successfully
            
        Raises:
            ValueError: If validation fails or multiple genotypes attempted
        """
        # Validate embryo ID
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Validate gene name
        if not self.schema_manager.validate_value("gene_names", gene_name):
            available = self.schema_manager.get_values("gene_names")
            raise ValueError(f"Invalid gene '{gene_name}'. Available: {available}")
        
        # Validate zygosity
        if not self.schema_manager.validate_value("zygosity_types", zygosity):
            available = self.schema_manager.get_values("zygosity_types")
            raise ValueError(f"Invalid zygosity '{zygosity}'. Available: {available}")
        
        # Validate confidence
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        # Initialize embryo data if needed
        if embryo_id not in self.data["embryos"]:
            self.data["embryos"][embryo_id] = {
                "phenotypes": {},
                "genotypes": {},
                "flags": {},
                "treatments": {},
                "metadata": {
                    "created": self.get_timestamp(),
                    "last_updated": self.get_timestamp()
                }
            }
        
        # 🚨 CRITICAL: ENFORCE SINGLE GENOTYPE PER EMBRYO 🚨
        existing_genotypes = self.data["embryos"][embryo_id]["genotypes"]
        
        # Case 1: No existing genotypes - OK to add
        if not existing_genotypes:
            pass  # Good to go
            
        # Case 2: Overwriting the same gene - OK
        elif gene_name in existing_genotypes and overwrite:
            if self.verbose:
                print(f"🔄 Overwriting existing genotype for gene '{gene_name}' in {embryo_id}")
            
        # Case 3: Adding same gene without overwrite - Error
        elif gene_name in existing_genotypes and not overwrite:
            existing = existing_genotypes[gene_name]
            raise ValueError(f"Genotype for '{gene_name}' already exists for {embryo_id}. "
                           f"Current: {existing}. Use overwrite=True to replace.")
            
        # Case 4: Different gene when genotype already exists - FORBIDDEN
        elif existing_genotypes and gene_name not in existing_genotypes:
            existing_genes = list(existing_genotypes.keys())
            raise ValueError(f"❌ SINGLE GENOTYPE RULE VIOLATION: Embryo {embryo_id} already has "
                           f"genotype for {existing_genes}. Only ONE genotype per embryo is allowed. "
                           f"Cannot add additional gene '{gene_name}'. This is an experimental design constraint.")
        
        # Create genotype using our model
        genotype = Genotype(
            gene=gene_name,
            allele=allele,
            zygosity=zygosity,
            confidence=confidence,
            notes=notes
        )
        
        # Add to data (clear existing genotypes to ensure single genotype)
        if not overwrite and existing_genotypes:
            # This should never happen due to above checks, but safety net
            raise ValueError(f"Internal error: attempting to add genotype when {len(existing_genotypes)} already exist")
        
        # If overwriting, clear all genotypes and add the new one (single genotype rule)
        self.data["embryos"][embryo_id]["genotypes"] = {gene_name: genotype.to_dict()}
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Update summary stats
        self._update_summary_stats()
        
        # Log operation
        self.log_operation("add_genotype", embryo_id,
                         gene=gene_name, allele=allele, zygosity=zygosity, confidence=confidence,
                         enforced_single_genotype=True)
        
        if self.verbose:
            print(f"🧬 Added genotype '{gene_name}:{allele}' to {embryo_id} (single genotype enforced)")
        
        return True
    
    def edit_genotype(self, embryo_id: str, gene_name: str,
                     allele: str = None, zygosity: str = None,
                     confidence: float = None, notes: str = None) -> bool:
        """
        Edit an existing genotype.
        
        Args:
            embryo_id: Valid embryo ID
            gene_name: Name of existing gene
            allele: New allele (optional)
            zygosity: New zygosity (optional)
            confidence: New confidence (optional)
            notes: New notes (optional)
            
        Returns:
            bool: True if edited successfully
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if genotype exists
        if gene_name not in self.data["embryos"][embryo_id]["genotypes"]:
            available = list(self.data["embryos"][embryo_id]["genotypes"].keys())
            raise ValueError(f"Genotype for '{gene_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Get current genotype
        current = self.data["embryos"][embryo_id]["genotypes"][gene_name]
        
        # Update fields if provided
        if allele is not None:
            current["allele"] = allele
        
        if zygosity is not None:
            if not self.schema_manager.validate_value("zygosity_types", zygosity):
                available = self.schema_manager.get_values("zygosity_types")
                raise ValueError(f"Invalid zygosity '{zygosity}'. Available: {available}")
            current["zygosity"] = zygosity
        
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            current["confidence"] = confidence
        
        if notes is not None:
            current["notes"] = notes
        
        # Update timestamp
        current["last_updated"] = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        changes = {}
        if allele is not None:
            changes["allele"] = allele
        if zygosity is not None:
            changes["zygosity"] = zygosity
        if confidence is not None:
            changes["confidence"] = confidence
        if notes is not None:
            changes["notes"] = notes
        
        self.log_operation("edit_genotype", embryo_id,
                         gene=gene_name, changes=changes)
        
        if self.verbose:
            print(f"🧬 Updated genotype for '{gene_name}' in {embryo_id}")
        
        return True
    
    def remove_genotype(self, embryo_id: str, gene_name: str) -> bool:
        """
        Remove a genotype from an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            gene_name: Name of gene to remove
            
        Returns:
            bool: True if removed successfully
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if genotype exists
        if gene_name not in self.data["embryos"][embryo_id]["genotypes"]:
            available = list(self.data["embryos"][embryo_id]["genotypes"].keys())
            raise ValueError(f"Genotype for '{gene_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Remove genotype
        removed = self.data["embryos"][embryo_id]["genotypes"].pop(gene_name)
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Update summary stats
        self._update_summary_stats()
        
        # Log operation
        self.log_operation("remove_genotype", embryo_id,
                         gene=gene_name, removed_data=removed)
        
        if self.verbose:
            print(f"🗑️ Removed genotype for '{gene_name}' from {embryo_id}")
        
        return True
    
    def get_genotypes(self, embryo_id: str) -> dict:
        """
        Get all genotypes for an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            
        Returns:
            dict: Genotypes data
        """
        if embryo_id not in self.data["embryos"]:
            return {}
        
        return self.data["embryos"][embryo_id]["genotypes"].copy()
    
    def list_genotypes_by_gene(self, gene_name: str) -> list:
        """
        Find all embryos with genotypes for a specific gene.
        
        Args:
            gene_name: Name of gene to search for
            
        Returns:
            list: List of embryo IDs with this gene
        """
        embryos_with_gene = []
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if gene_name in embryo_data["genotypes"]:
                embryos_with_gene.append(embryo_id)
        
        return embryos_with_gene
    
    # =============================================================================
    # FLAG MANAGEMENT METHODS
    # =============================================================================
    
    def add_flag(self, embryo_id: str, flag_type: str, description: str = "",
                priority: str = "medium", confidence: float = 1.0,
                notes: str = "", overwrite: bool = False) -> bool:
        """
        Add a flag to an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            flag_type: Type of flag (must be in permitted values)
            description: Flag description
            priority: Priority level (low, medium, high, critical)
            confidence: Confidence score (0.0-1.0)
            notes: Optional notes
            overwrite: Whether to overwrite existing flag
            
        Returns:
            bool: True if added successfully
        """
        # Validate embryo ID
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Validate flag type
        if not self.schema_manager.validate_value("flag_types", flag_type):
            available = self.schema_manager.get_values("flag_types")
            raise ValueError(f"Invalid flag type '{flag_type}'. Available: {available}")
        
        # Validate priority
        if not self.schema_manager.validate_value("priority_levels", priority):
            available = self.schema_manager.get_values("priority_levels")
            raise ValueError(f"Invalid priority '{priority}'. Available: {available}")
        
        # Validate confidence
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        # Initialize embryo data if needed
        if embryo_id not in self.data["embryos"]:
            self.data["embryos"][embryo_id] = {
                "phenotypes": {},
                "genotypes": {},
                "flags": {},
                "treatments": {},
                "metadata": {
                    "created": self.get_timestamp(),
                    "last_updated": self.get_timestamp()
                }
            }
        
        # Check for existing flag
        if flag_type in self.data["embryos"][embryo_id]["flags"] and not overwrite:
            existing = self.data["embryos"][embryo_id]["flags"][flag_type]
            raise ValueError(f"Flag '{flag_type}' already exists for {embryo_id}. "
                           f"Current: {existing}. Use overwrite=True to replace.")
        
        # Create flag using our model
        flag = Flag(
            flag_type=flag_type,
            description=description,
            priority=priority,
            confidence=confidence,
            notes=notes
        )
        
        # Add to data
        self.data["embryos"][embryo_id]["flags"][flag_type] = flag.to_dict()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Update summary stats
        self._update_summary_stats()
        
        # Log operation
        self.log_operation("add_flag", embryo_id,
                         flag_type=flag_type, priority=priority, confidence=confidence)
        
        if self.verbose:
            print(f"🚩 Added flag '{flag_type}' to {embryo_id}")
        
        return True
    
    def edit_flag(self, embryo_id: str, flag_type: str,
                 description: str = None, priority: str = None,
                 confidence: float = None, notes: str = None) -> bool:
        """
        Edit an existing flag.
        
        Args:
            embryo_id: Valid embryo ID
            flag_type: Type of existing flag
            description: New description (optional)
            priority: New priority (optional)
            confidence: New confidence (optional)
            notes: New notes (optional)
            
        Returns:
            bool: True if edited successfully
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if flag exists
        if flag_type not in self.data["embryos"][embryo_id]["flags"]:
            available = list(self.data["embryos"][embryo_id]["flags"].keys())
            raise ValueError(f"Flag '{flag_type}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Get current flag
        current = self.data["embryos"][embryo_id]["flags"][flag_type]
        
        # Update fields if provided
        if description is not None:
            current["description"] = description
        
        if priority is not None:
            if not self.schema_manager.validate_value("priority_levels", priority):
                available = self.schema_manager.get_values("priority_levels")
                raise ValueError(f"Invalid priority '{priority}'. Available: {available}")
            current["priority"] = priority
        
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            current["confidence"] = confidence
        
        if notes is not None:
            current["notes"] = notes
        
        # Update timestamp
        current["last_updated"] = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        changes = {}
        if description is not None:
            changes["description"] = description
        if priority is not None:
            changes["priority"] = priority
        if confidence is not None:
            changes["confidence"] = confidence
        if notes is not None:
            changes["notes"] = notes
        
        self.log_operation("edit_flag", embryo_id,
                         flag_type=flag_type, changes=changes)
        
        if self.verbose:
            print(f"🚩 Updated flag '{flag_type}' for {embryo_id}")
        
        return True
    
    def remove_flag(self, embryo_id: str, flag_type: str) -> bool:
        """
        Remove a flag from an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            flag_type: Type of flag to remove
            
        Returns:
            bool: True if removed successfully
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if flag exists
        if flag_type not in self.data["embryos"][embryo_id]["flags"]:
            available = list(self.data["embryos"][embryo_id]["flags"].keys())
            raise ValueError(f"Flag '{flag_type}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Remove flag
        removed = self.data["embryos"][embryo_id]["flags"].pop(flag_type)
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Update summary stats
        self._update_summary_stats()
        
        # Log operation
        self.log_operation("remove_flag", embryo_id,
                         flag_type=flag_type, removed_data=removed)
        
        if self.verbose:
            print(f"🗑️ Removed flag '{flag_type}' from {embryo_id}")
        
        return True
    
    def get_flags(self, embryo_id: str) -> dict:
        """
        Get all flags for an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            
        Returns:
            dict: Flags data
        """
        if embryo_id not in self.data["embryos"]:
            return {}
        
        return self.data["embryos"][embryo_id]["flags"].copy()
    
    def list_flags_by_type(self, flag_type: str) -> list:
        """
        Find all embryos with a specific flag type.
        
        Args:
            flag_type: Type of flag to search for
            
        Returns:
            list: List of embryo IDs with this flag
        """
        embryos_with_flag = []
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if flag_type in embryo_data["flags"]:
                embryos_with_flag.append(embryo_id)
        
        return embryos_with_flag
    
    def get_high_priority_flags(self) -> dict:
        """
        Get all high priority and critical flags across all embryos.
        
        Returns:
            dict: {embryo_id: {flag_type: flag_data}} for high/critical flags
        """
        high_priority_flags = {}
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            for flag_type, flag_data in embryo_data["flags"].items():
                if flag_data["priority"] in ["high", "critical"]:
                    if embryo_id not in high_priority_flags:
                        high_priority_flags[embryo_id] = {}
                    high_priority_flags[embryo_id][flag_type] = flag_data
        
        return high_priority_flags
    
    # =============================================================================
    # TREATMENT MANAGEMENT METHODS
    # =============================================================================
    
    def add_treatment(self, embryo_id: str, treatment_name: str, 
                     concentration: Optional[str] = None, 
                     duration: Optional[str] = None,
                     temperature: Optional[str] = None,
                     confidence: float = 1.0, notes: str = "", 
                     overwrite: bool = False) -> bool:
        """
        Add a treatment to an embryo. Supports multiple treatments per embryo.
        
        Note: Multiple treatments per embryo are allowed (e.g., chemical + temperature).
        A warning will be issued if more than one treatment is detected.
        
        Args:
            embryo_id: Valid embryo ID
            treatment_name: Name of treatment (must be in permitted values)
            concentration: Treatment concentration (optional)
            duration: Treatment duration (optional)
            temperature: Treatment temperature (optional)
            confidence: Confidence score (0.0-1.0)
            notes: Optional notes
            overwrite: Whether to overwrite existing treatment
            
        Returns:
            bool: True if added successfully
            
        Raises:
            ValueError: If embryo or treatment validation fails
        """
        # Validate embryo ID
        if not self.validate_id_format(embryo_id, "embryo"):
            raise ValueError(f"Invalid embryo ID format: {embryo_id}")
        
        # Validate treatment name
        if not self.schema_manager.validate_value("treatment_types", treatment_name):
            available = self.schema_manager.get_values("treatment_types")
            raise ValueError(f"Invalid treatment '{treatment_name}'. Available: {available}")
        
        # Validate confidence
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        # Check if embryo exists in data
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Initialize treatments dict if not exists
        if "treatments" not in self.data["embryos"][embryo_id]:
            self.data["embryos"][embryo_id]["treatments"] = {}
        
        # Check for existing treatment
        existing_treatments = self.data["embryos"][embryo_id]["treatments"]
        if treatment_name in existing_treatments and not overwrite:
            raise ValueError(f"Treatment '{treatment_name}' already exists for {embryo_id}. "
                           f"Use overwrite=True to replace")
        
        # Create treatment data
        treatment_data = {
            "concentration": concentration,
            "duration": duration,
            "temperature": temperature,
            "confidence": confidence,
            "notes": notes,
            "author": "manual",
            "timestamp": self.get_timestamp(),
            "last_updated": self.get_timestamp()
        }
        
        # Add treatment
        self.data["embryos"][embryo_id]["treatments"][treatment_name] = treatment_data
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Check for multiple treatments and warn
        treatment_count = len(existing_treatments) + (0 if treatment_name in existing_treatments else 1)
        if treatment_count > 1:
            treatment_list = list(existing_treatments.keys())
            if treatment_name not in treatment_list:
                treatment_list.append(treatment_name)
            if self.verbose:
                print(f"⚠️  WARNING: Embryo {embryo_id} now has {treatment_count} treatments: {treatment_list}")
                print("   Multiple treatments may indicate complex experimental design.")
        
        # Update summary stats
        self._update_summary_stats()
        
        # Log operation
        self.log_operation("add_treatment", embryo_id, 
                         treatment=treatment_name, 
                         data=treatment_data,
                         total_treatments=treatment_count)
        
        if self.verbose:
            print(f"✅ Added treatment '{treatment_name}' to {embryo_id}")
        
        return True
    
    def edit_treatment(self, embryo_id: str, treatment_name: str,
                      concentration: Optional[str] = None,
                      duration: Optional[str] = None,
                      temperature: Optional[str] = None,
                      confidence: Optional[float] = None,
                      notes: Optional[str] = None) -> bool:
        """
        Edit an existing treatment.
        
        Args:
            embryo_id: Valid embryo ID
            treatment_name: Name of existing treatment
            concentration: New concentration (optional)
            duration: New duration (optional)
            temperature: New temperature (optional)
            confidence: New confidence (optional)
            notes: New notes (optional)
            
        Returns:
            bool: True if edited successfully
            
        Raises:
            ValueError: If treatment doesn't exist or validation fails
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if treatment exists
        treatments = self.data["embryos"][embryo_id].get("treatments", {})
        if treatment_name not in treatments:
            available = list(treatments.keys())
            raise ValueError(f"Treatment '{treatment_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Get current treatment
        current = treatments[treatment_name]
        
        # Update fields if provided
        if concentration is not None:
            current["concentration"] = concentration
        
        if duration is not None:
            current["duration"] = duration
        
        if temperature is not None:
            current["temperature"] = temperature
        
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            current["confidence"] = confidence
        
        if notes is not None:
            current["notes"] = notes
        
        # Update timestamp
        current["last_updated"] = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Log operation
        changes = {}
        if concentration is not None:
            changes["concentration"] = concentration
        if duration is not None:
            changes["duration"] = duration
        if temperature is not None:
            changes["temperature"] = temperature
        if confidence is not None:
            changes["confidence"] = confidence
        if notes is not None:
            changes["notes"] = notes
        
        self.log_operation("edit_treatment", embryo_id, 
                         treatment=treatment_name, changes=changes)
        
        if self.verbose:
            print(f"✅ Updated treatment '{treatment_name}' for {embryo_id}")
        
        return True
    
    def remove_treatment(self, embryo_id: str, treatment_name: str) -> bool:
        """
        Remove a treatment from an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            treatment_name: Name of treatment to remove
            
        Returns:
            bool: True if removed successfully
            
        Raises:
            ValueError: If embryo or treatment doesn't exist
        """
        # Check if embryo exists
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        # Check if treatment exists
        treatments = self.data["embryos"][embryo_id].get("treatments", {})
        if treatment_name not in treatments:
            available = list(treatments.keys())
            raise ValueError(f"Treatment '{treatment_name}' not found for {embryo_id}. "
                           f"Available: {available}")
        
        # Remove treatment
        removed = treatments.pop(treatment_name)
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = self.get_timestamp()
        
        # Update summary stats
        self._update_summary_stats()
        
        # Log operation
        self.log_operation("remove_treatment", embryo_id,
                         treatment=treatment_name, removed_data=removed)
        
        if self.verbose:
            print(f"🗑️ Removed treatment '{treatment_name}' from {embryo_id}")
        
        return True
    
    def get_treatments(self, embryo_id: str) -> dict:
        """
        Get all treatments for an embryo.
        
        Args:
            embryo_id: Valid embryo ID
            
        Returns:
            dict: Treatments data
        """
        if embryo_id not in self.data["embryos"]:
            return {}
        
        return self.data["embryos"][embryo_id].get("treatments", {}).copy()
    
    def list_treatments_by_type(self, treatment_name: str) -> list:
        """
        Find all embryos with a specific treatment.
        
        Args:
            treatment_name: Name of treatment to search for
            
        Returns:
            list: List of embryo IDs with this treatment
        """
        embryos_with_treatment = []
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            treatments = embryo_data.get("treatments", {})
            if treatment_name in treatments:
                embryos_with_treatment.append(embryo_id)
        
        return embryos_with_treatment
    
    def get_multi_treatment_embryos(self) -> dict:
        """
        Find all embryos with multiple treatments (warning cases).
        
        Returns:
            dict: {embryo_id: [treatment_names]} for embryos with >1 treatment
        """
        multi_treatment_embryos = {}
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            treatments = embryo_data.get("treatments", {})
            if len(treatments) > 1:
                multi_treatment_embryos[embryo_id] = list(treatments.keys())
        
        return multi_treatment_embryos
    
    def get_treatment_combinations(self) -> dict:
        """
        Get summary of all treatment combinations in the dataset.
        
        Returns:
            dict: {combination_str: count} of treatment combinations
        """
        combinations = {}
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            treatments = embryo_data.get("treatments", {})
            if treatments:
                # Create sorted combination string
                combo = "+".join(sorted(treatments.keys()))
                combinations[combo] = combinations.get(combo, 0) + 1
        
        return combinations
    
    def validate_treatment_overlays(self) -> dict:
        """
        Validate treatment overlays for experimental design consistency.
        
        Returns:
            dict: Validation report with warnings and recommendations
        """
        report = {
            "multi_treatment_count": 0,
            "combinations": {},
            "warnings": [],
            "recommendations": []
        }
        
        multi_treatments = self.get_multi_treatment_embryos()
        report["multi_treatment_count"] = len(multi_treatments)
        report["combinations"] = self.get_treatment_combinations()
        
        # Analyze for common patterns
        combo_counts = report["combinations"]
        total_treated = sum(combo_counts.values())
        
        if report["multi_treatment_count"] > 0:
            pct_multi = (report["multi_treatment_count"] / total_treated) * 100
            report["warnings"].append(
                f"{report['multi_treatment_count']} embryos ({pct_multi:.1f}%) have multiple treatments"
            )
        
        # Check for unusual combinations
        for combo, count in combo_counts.items():
            if "+" in combo and count < 3:  # Multi-treatment with low count
                report["warnings"].append(
                    f"Low-frequency treatment combination '{combo}' ({count} embryos)"
                )
        
        # Recommendations
        if report["multi_treatment_count"] > total_treated * 0.1:  # >10% multi-treatment
            report["recommendations"].append(
                "Consider validating experimental design - high frequency of multi-treatment embryos"
            )
        
        return report

    def _update_summary_stats(self) -> None:
        """Update summary statistics and mark as having unsaved changes."""
        # Mark that we have unsaved changes
        self._unsaved_changes = True
        
        # Add to change log for tracking
        self._add_change_log("data_modified", {
            "embryo_count": len(self.data["embryos"]),
            "timestamp": self.get_timestamp()
        })

    def __str__(self) -> str:
        """String representation of EmbryoMetadata."""
        summary = self.get_summary()
        return (f"EmbryoMetadata({summary['totals']['embryos']} embryos, "
                f"{summary['totals']['snips']} snips, "
                f"{'unsaved' if summary['has_unsaved_changes'] else 'saved'})")
