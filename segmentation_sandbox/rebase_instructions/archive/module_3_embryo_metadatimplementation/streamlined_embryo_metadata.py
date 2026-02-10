"""
EmbryoMetadata Core Class - Simple and Focused
Inherits business logic from UnifiedEmbryoManager, handles file I/O and initialization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from base_annotation_parser import BaseAnnotationParser
from data_managers.permitted_values_manager import PermittedValuesManager
from data_managers.unified_embryo_manager import UnifiedEmbryoManager
from data_managers.entity_id_tracker import EntityIDTracker
from embryo_metadata_utils import validate_path, DEFAULT_EMBRYO_METADATA_CONFIG


class EmbryoMetadata(BaseAnnotationParser, UnifiedEmbryoManager):
    """
    Main embryo metadata class.
    
    Core functionality:
    - File I/O and persistence
    - SAM annotation integration
    - Data validation and consistency
    - Embryo/snip hierarchy management
    - Batch processing with validation and error handling
    
    Business logic inherited from UnifiedEmbryoManager.
    """
    
    def __init__(self, sam_annotation_path: Union[str, Path], 
                 embryo_metadata_path: Optional[Union[str, Path]] = None,
                 gen_if_no_file: bool = False, 
                 verbose: bool = True,
                 schema_path=None):
        
        # Setup paths
        self.sam_annotation_path = validate_path(sam_annotation_path, must_exist=True)
        if embryo_metadata_path is None:
            embryo_metadata_path = self.sam_annotation_path.with_name(
                self.sam_annotation_path.stem + "_embryo_metadata.json"
            )
        
        # Initialize base
        super().__init__(embryo_metadata_path, verbose=verbose)
        
        # Configuration and schema
        self.config = DEFAULT_EMBRYO_METADATA_CONFIG.copy()
        self.schema_manager = SchemaManager(schema_path) if schema_path else SchemaManager()
        self._auto_validate = True  # Default validation enabled
        
        # Load SAM annotations
        self.sam_annotations = self.load_json(self.sam_annotation_path)
        
        # Load or create metadata
        if self.filepath.exists():
            self.data = self.load_json()
            self._validate_entity_tracking()
        elif gen_if_no_file:
            self.data = self._create_from_sam()
            self._initialize_entity_tracking()
        else:
            raise FileNotFoundError(f"Metadata not found: {embryo_metadata_path}")
        
        # Initialize lookup caches
        self._build_caches()
    
    def _validate_entity_tracking(self):
        """Validate entity hierarchy and update tracking section."""
        # Extract current entities
        current_entities = EntityIDTracker.extract_entities(self.data)
        
        # Validate hierarchy (raises error on violations)
        EntityIDTracker.validate_hierarchy(current_entities, raise_on_violations=True)
        
        # Update entity_tracking section
        self.data["entity_tracking"] = {
            entity_type: list(ids) for entity_type, ids in current_entities.items()
        }
        
        if self.verbose:
            counts = EntityIDTracker.get_counts(current_entities)
            print(f"📊 Entities: {counts}")
    
    def _initialize_entity_tracking(self):
        """Initialize entity tracking for new metadata."""
        # Extract from SAM annotations
        sam_entities = EntityIDTracker.extract_entities(self.sam_annotations)
        
        # Extract from created metadata
        metadata_entities = EntityIDTracker.extract_entities(self.data)
        
        # Find missing entities
        missing = EntityIDTracker.compare_entities(sam_entities, metadata_entities)
        
        # Store tracking info
        self.data["entity_tracking"] = {
            "sam_source": {entity_type: list(ids) for entity_type, ids in sam_entities.items()},
            "metadata": {entity_type: list(ids) for entity_type, ids in metadata_entities.items()},
            "missing": {entity_type: list(ids) for entity_type, ids in missing.items()}
        }
        
        if self.verbose and any(missing.values()):
            print(f"⚠️ Missing entities from SAM: {EntityIDTracker.get_counts(missing)}")
    
    def check_sam_consistency(self) -> Dict:
        """Check consistency with original SAM file."""
        expected_sam = self.data.get("file_info", {}).get("source_sam_filename")
        current_sam = Path(self.sam_annotation_path).name
        
        if expected_sam and expected_sam != current_sam:
            raise ValueError(f"SAM file mismatch: expected {expected_sam}, got {current_sam}")
        
        # Compare entities
        sam_entities = EntityIDTracker.extract_entities(self.sam_annotations)
        metadata_entities = EntityIDTracker.extract_entities(self.data)
        missing = EntityIDTracker.compare_entities(sam_entities, metadata_entities)
        
        return {"missing_from_metadata": missing, "consistent": not any(missing.values())}
    
    def _create_from_sam(self) -> Dict:
        """Create metadata structure from SAM annotations."""
        embryos = {}
        
        # Extract embryo structure from SAM
        for exp_id, exp_data in self.sam_annotations.get("experiments", {}).items():
            for video_id, video_data in exp_data.get("videos", {}).items():
                for embryo_id in video_data.get("embryo_ids", []):
                    if embryo_id not in embryos:
                        embryos[embryo_id] = {
                            "genotype": None,
                            "treatments": {},
                            "flags": {},
                            "notes": "",
                            "metadata": {"created": self.get_timestamp()},
                            "snips": {}
                        }
                    
                    # Add snips for this embryo
                    for image_id, image_data in video_data.get("images", {}).items():
                        if embryo_id in image_data.get("embryos", {}):
                            snip_id = image_data["embryos"][embryo_id].get("snip_id")
                            if snip_id:
                                embryos[embryo_id]["snips"][snip_id] = {"flags": []}
        
        return {
            "file_info": {
                "version": "1.0",
                "created": self.get_timestamp(),
                "source_sam": str(self.sam_annotation_path)
            },
            "embryos": embryos
        }
    
    def _build_caches(self):
        """Build lookup caches for performance."""
        self._snip_to_embryo = {}
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            for snip_id in embryo_data.get("snips", {}):
                self._snip_to_embryo[snip_id] = embryo_id
    
    def get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
        """Get embryo ID from snip ID."""
        return self._snip_to_embryo.get(snip_id)
    
    def get_available_snips(self, embryo_id: Optional[str] = None) -> List[str]:
        """Get available snip IDs."""
        if embryo_id:
            return list(self.data["embryos"].get(embryo_id, {}).get("snips", {}).keys())
        return list(self._snip_to_embryo.keys())
    
    def get_snip_data(self, snip_id: str) -> Optional[Dict]:
        """Get snip data."""
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        if embryo_id:
            return self.data["embryos"][embryo_id]["snips"].get(snip_id)
        return None
    
    def validate_id_format(self, entity_id: str, entity_type: str) -> bool:
        """Validate ID format."""
        # Simple validation - extend as needed
        if entity_type == "snip":
            return "_" in entity_id and len(entity_id) > 10
        elif entity_type == "embryo":
            return "_e" in entity_id
        return True
    
    def save(self, backup: bool = True):
        """Save metadata with optional backup and entity validation."""
        # Validate entity hierarchy before saving
        current_entities = EntityIDTracker.extract_entities(self.data)
        EntityIDTracker.validate_hierarchy(current_entities, raise_on_violations=True)
        
        # Update tracking section and timestamps
        self.data["entity_tracking"]["metadata"] = {
            entity_type: list(ids) for entity_type, ids in current_entities.items()
        }
        self.data["file_info"]["last_updated"] = self.get_timestamp()
        
        # Save with base class method
        self.save_json(self.data, create_backup=backup)
        
        if self.verbose:
            embryo_count = len(self.data["embryos"])
            snip_count = len(self._snip_to_embryo)
            print(f"💾 Saved: {embryo_count} embryos, {snip_count} snips (validated)")
    
    def reload(self):
        """Reload from file."""
        self.data = self.load_json()
        self._build_caches()
        if self.verbose:
            print("🔄 Reloaded metadata")
    
    def get_entity_counts(self) -> Dict[str, int]:
        """Get counts of all entity types by parsing IDs."""
        # Extract all entities and get counts
        current_entities = EntityIDTracker.extract_entities(self.data)
        return EntityIDTracker.get_counts(current_entities)
    
    @property
    def embryo_count(self) -> int:
        """Number of embryos."""
        return self.get_entity_counts()["embryos"]
    
    @property
    def snip_count(self) -> int:
        """Number of snips."""
        return self.get_entity_counts()["snips"]
    
    def apply_batch(self, batch, dry_run: bool = False, interactive: bool = True) -> Dict:
        """
        Apply annotation batch with validation and error handling.
        
        Args:
            batch: AnnotationBatch instance
            dry_run: If True, validate but don't apply changes
            interactive: If True, prompt user for error resolution
        
        Returns:
            Dict with results and any errors
        """
        results = {
            "applied": 0,
            "errors": [],
            "skipped": 0,
            "frame_resolutions": {}
        }
        
        if self.verbose:
            print(f"🔄 {'Validating' if dry_run else 'Applying'} batch: {len(batch.data)} embryos")
        
        # Step 1: Resolve frames for all phenotype operations
        for embryo_id, embryo_data in batch.data.items():
            if "phenotype_operations" in embryo_data:
                for i, op in enumerate(embryo_data["phenotype_operations"]):
                    try:
                        resolved_snips = self._resolve_frames(embryo_id, op["frames"])
                        op["resolved_snips"] = resolved_snips
                        results["frame_resolutions"][f"{embryo_id}_op_{i}"] = resolved_snips
                    except Exception as e:
                        results["errors"].append({
                            "embryo": embryo_id,
                            "operation": "phenotype",
                            "error": f"Frame resolution failed: {e}",
                            "suggestion": "Check frame specification or embryo snips"
                        })
        
        # Step 2: Validate each operation
        for embryo_id, embryo_data in batch.data.items():
            
            # Validate genotype operations
            if "genotype" in embryo_data:
                try:
                    existing_genotype = self.get_genotype(embryo_id)
                    if existing_genotype and not dry_run:
                        results["errors"].append({
                            "embryo": embryo_id,
                            "operation": "genotype",
                            "error": f"Genotype already exists: {existing_genotype['value']}",
                            "suggestion": "Use batch.edit_annotation() or remove existing first"
                        })
                except Exception as e:
                    results["errors"].append({
                        "embryo": embryo_id,
                        "operation": "genotype",
                        "error": str(e),
                        "suggestion": "Check genotype data format"
                    })
            
            # Validate phenotype operations
            if "phenotype_operations" in embryo_data:
                for op in embryo_data["phenotype_operations"]:
                    if "resolved_snips" not in op:
                        continue  # Already errored in frame resolution
                    
                    for snip_id in op["resolved_snips"]:
                        try:
                            # Check if snip exists
                            if snip_id not in self.get_available_snips(embryo_id):
                                results["errors"].append({
                                    "embryo": embryo_id,
                                    "operation": "phenotype",
                                    "snip": snip_id,
                                    "error": f"Snip {snip_id} not found",
                                    "suggestion": "Check snip ID or frame range"
                                })
                                continue
                            
                            # Check DEAD logic using existing validation
                            existing_phenotypes = self.get_phenotypes(snip_id)
                            existing_values = [p["value"] for p in existing_phenotypes]

                            if "DEAD" in existing_values and op["value"] != "DEAD":
                                if not op.get("overwrite_dead", False):
                                    results["errors"].append({
                                        "embryo": embryo_id,
                                        "operation": "phenotype",
                                        "snip": snip_id,
                                        "error": f"Cannot add {op['value']} - snip already DEAD",
                                        "suggestion": "Use overwrite_dead=True or remove from batch"
                                    })
                        
                        except Exception as e:
                            results["errors"].append({
                                "embryo": embryo_id,
                                "operation": "phenotype",
                                "snip": snip_id,
                                "error": str(e),
                                "suggestion": "Check phenotype data"
                            })
        
        # Step 3: Handle errors if found
        if results["errors"]:
            if interactive and not dry_run:
                self._prompt_batch_fixes(batch, results["errors"])
                return results  # Let user fix and retry
            elif not interactive:
                raise ValueError(f"Batch validation failed with {len(results['errors'])} errors")
        
        # Step 4: Apply operations if not dry run
        if not dry_run and not results["errors"]:
            for embryo_id, embryo_data in batch.data.items():
                try:
                    # Apply genotype
                    if "genotype" in embryo_data:
                        g = embryo_data["genotype"]
                        self.add_genotype(
                            embryo_id, g["value"], g.get("allele"), 
                            g.get("zygosity", "heterozygous"), g.get("author", batch.author)
                        )
                        results["applied"] += 1
                    
                    # Apply phenotypes
                    if "phenotype_operations" in embryo_data:
                        for op in embryo_data["phenotype_operations"]:
                            if "resolved_snips" in op:
                                for snip_id in op["resolved_snips"]:
                                    self.add_phenotype(
                                        snip_id, op["value"], op.get("author", batch.author),
                                        op.get("notes"), op.get("confidence")
                                    )
                                    results["applied"] += 1
                    
                    # Apply treatments
                    if "treatments" in embryo_data:
                        for treatment_data in embryo_data["treatments"].values():
                            self.add_treatment(
                                embryo_id, treatment_data["value"],
                                treatment_data.get("dosage"),
                                treatment_data.get("timing"),
                                treatment_data.get("author", batch.author),
                                treatment_data.get("notes")
                            )
                            results["applied"] += 1
                    
                    # Apply flags
                    if "flags" in embryo_data:
                        for flag_data in embryo_data["flags"]:
                            self.add_flag(
                                embryo_id, flag_data["flag_type"],
                                flag_data.get("level", "auto"),
                                flag_data.get("description", ""),
                                flag_data.get("priority", "medium"),
                                flag_data.get("notes", "")
                            )
                            results["applied"] += 1
                
                except Exception as e:
                    results["errors"].append({
                        "embryo": embryo_id,
                        "operation": "apply",
                        "error": str(e),
                        "suggestion": "Check data consistency"
                    })
        
        if self.verbose:
            if dry_run:
                print(f"✅ Validation complete: {len(results['errors'])} errors found")
            else:
                print(f"✅ Applied {results['applied']} annotations, {len(results['errors'])} errors")
        
        return results
    
    def _resolve_frames(self, embryo_id: str, frames: Union[str, List[str]]) -> List[str]:
        """Resolve frame specification to actual snip IDs."""
        available_snips = self.get_available_snips(embryo_id)
        
        if frames == "all":
            return available_snips
        elif isinstance(frames, list):
            return [s for s in frames if s in available_snips]
        elif isinstance(frames, str):
            # Use FrameAwareRangeParser for range specifications
            from parsing_utils import FrameAwareRangeParser
            return FrameAwareRangeParser.parse_snip_range(frames, available_snips)
        else:
            raise ValueError(f"Invalid frames specification: {frames}")
    
    def _prompt_batch_fixes(self, batch, errors):
        """Interactive error resolution prompt."""
        print(f"\n❌ Found {len(errors)} errors in batch:")
        
        for i, error in enumerate(errors[:5]):  # Show first 5 errors
            print(f"\n{i+1}. Embryo: {error['embryo']}")
            print(f"   Operation: {error['operation']}")
            print(f"   Error: {error['error']}")
            print(f"   Suggestion: {error['suggestion']}")
        
        if len(errors) > 5:
            print(f"\n... and {len(errors) - 5} more errors")
        
        print(f"\nRecommended fixes:")
        print(f"- batch.remove_annotation(embryo_id, annotation_type)")
        print(f"- batch.edit_annotation(embryo_id, annotation_type, **updates)")
        print(f"- batch.clear_embryo(embryo_id)")
        print(f"- Fix frame specifications or check embryo IDs")
    
    def add_phenotype(self, snip_id: str, phenotype: str, author: str,
                     notes: str = None, confidence: float = None) -> bool:
        """Add phenotype directly to metadata (in-memory only)."""
        result = super().add_phenotype(snip_id, phenotype, author, notes, confidence)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def add_genotype(self, embryo_id: str, gene_name: str, allele: str,
                    zygosity: str = "heterozygous", author: str = None,
                    overwrite_genotype: bool = False) -> bool:
        """Add genotype directly to metadata (in-memory only)."""
        result = super().add_genotype(embryo_id, gene_name, allele, zygosity, author, overwrite_genotype)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def add_flag(self, entity_id: str, flag_type: str, level: str = "auto",
                description: str = "", priority: str = "medium", 
                notes: str = "", overwrite: bool = False) -> bool:
        """Add flag directly to metadata (in-memory only)."""
        result = super().add_flag(entity_id, flag_type, level, description, priority, notes, overwrite)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def add_treatment(self, embryo_id: str, treatment_name: str, dosage: str = None,
                     timing: str = None, author: str = None, notes: str = None) -> bool:
        """Add treatment directly to metadata (in-memory only)."""
        result = super().add_treatment(embryo_id, treatment_name, dosage, timing, author, notes)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def remove_phenotype(self, snip_id: str) -> bool:
        """Remove phenotype from snip (in-memory only)."""
        result = super().remove_phenotype(snip_id)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def remove_genotype(self, embryo_id: str) -> bool:
        """Remove genotype from embryo (in-memory only)."""
        result = super().remove_genotype(embryo_id)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def edit_phenotype(self, snip_id: str, confidence: float = None, notes: str = None) -> bool:
        """Edit phenotype (in-memory only)."""
        result = super().edit_phenotype(snip_id, confidence, notes)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def edit_genotype(self, embryo_id: str, allele: str = None, 
                     zygosity: str = None, confidence: float = None, 
                     notes: str = None) -> bool:
        """Edit genotype (in-memory only)."""
        result = super().edit_genotype(embryo_id, allele, zygosity, confidence, notes)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def auto_validate(self, enabled: bool = True):
        """Enable/disable automatic validation after changes."""
        self._auto_validate = enabled
        if self.verbose:
            status = "enabled" if enabled else "disabled"
            print(f"🔧 Auto-validation {status}")
    
    def clear_embryo_annotations(self, embryo_id: str, annotation_types: Union[str, List[str]] = "all") -> bool:
        """Clear specific annotation types for embryo."""
        result = super().clear_embryo_data(embryo_id, annotation_types)
        if result and self._auto_validate:
            self.validate_data_integrity()
        return result
    
    def find_embryos_with_phenotype(self, phenotype: str) -> List[str]:
        """Find embryos with specific phenotype."""
        return super().list_snips_by_phenotype(phenotype)
    
    def find_embryos_with_genotype(self, genotype: str) -> List[str]:
        """Find embryos with specific genotype."""
        return super().list_embryos_by_gene(genotype)
    
    def find_embryos_with_flag(self, flag_type: str, level: str = "both") -> Dict[str, List[str]]:
        """Find embryos with specific flag."""
        return super().list_flags_by_type(flag_type, level)
    
    def find_embryos_with_treatment(self, treatment: str) -> List[str]:
        """Find embryos with specific treatment."""
        found = []
        for embryo_id, embryo_data in self.data["embryos"].items():
            for treatment_data in embryo_data.get("treatments", {}).values():
                if treatment_data.get("value") == treatment:
                    found.append(embryo_id)
                    break
        return found
    
    def mark_dead(self, embryo_id: str, start_frame: int = None, author: str = None):
        """Mark embryo as dead from specified frame onward."""
        snips = self.get_available_snips(embryo_id)
        
        if start_frame is not None:
            # Filter snips from start_frame onward
            snips = [s for s in snips if extract_frame_number(s) >= start_frame]
        
        # Create batch for atomic operation
        batch = AnnotationBatch(author or self.config.get("default_author"))
        batch.add_phenotype(embryo_id, "DEAD", frames=snips)
        
        # Apply with force_dead since we're intentionally marking dead
        results = self.apply_batch(batch, interactive=False)
        
        if self.verbose and not results["errors"]:
            print(f"☠️ Marked {embryo_id} as DEAD from frame {start_frame or 'all'}")
        
        return results

    def get_summary(self) -> Dict:
        """Get overall metadata summary."""
        entity_counts = self.get_entity_counts()
        stats = super().get_phenotype_statistics()
        genotype_stats = super().get_genotype_statistics()
        
        return {
            "entity_counts": entity_counts,
            "phenotype_stats": stats,
            "genotype_stats": genotype_stats,
            "auto_validate": self._auto_validate,
            "file_info": self.data.get("file_info", {})
        }
    
    def validate_batch(self, batch) -> List[Dict]:
        """Pre-flight batch validation."""
        result = self.apply_batch(batch, dry_run=True, interactive=False)
        return result["errors"]

