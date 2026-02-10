"""
Embryo Metadata Managers
Comprehensive implementation with validation rules and multi-level support.
"""

from typing import Dict, List, Optional, Union
from datetime import datetime

# Module imports
from utils.parsing_utils import extract_frame_number, parse_entity_id, get_entity_type
from metadata.schema_manager import SchemaManager


class Phenotype:
    """Phenotype data model."""
    def __init__(self, value: str, author: str, notes: str = None, confidence: float = None):
        self.value = value
        self.author = author
        self.notes = notes
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()
        self.last_updated = self.timestamp
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "author": self.author,
            "notes": self.notes,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated
        }


class Flag:
    """Flag data model."""
    def __init__(self, value: str, author: str, flag_type: str, priority: str = "medium", notes: str = ""):
        self.value = value
        self.author = author
        self.flag_type = flag_type
        self.priority = priority
        self.notes = notes
        self.timestamp = datetime.now().isoformat()
        self.last_updated = self.timestamp
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "author": self.author,
            "flag_type": self.flag_type,
            "priority": self.priority,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated
        }


class Genotype:
    """Genotype data model."""
    def __init__(self, value: str, allele: str, zygosity: str = "heterozygous", author: str = "unknown"):
        self.value = value
        self.allele = allele
        self.zygosity = zygosity
        self.author = author
        self.timestamp = datetime.now().isoformat()
        self.last_updated = self.timestamp
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "allele": self.allele,
            "zygosity": self.zygosity,
            "author": self.author,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated
        }


class Treatment:
    """Treatment data model."""
    def __init__(self, value: str, author: str, dosage: str = None, timing: str = None, notes: str = None):
        self.value = value
        self.author = author
        self.dosage = dosage
        self.timing = timing
        self.notes = notes
        self.timestamp = datetime.now().isoformat()
        self.last_updated = self.timestamp
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "author": self.author,
            "dosage": self.dosage,
            "timing": self.timing,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated
        }


class EmbryoPhenotypeManager:
    """Phenotype management with DEAD exclusivity enforcement."""
    def _get_valid_phenotypes(self) -> Dict[str, Dict]:
        return self.schema_manager.get_phenotypes()

    def add_phenotype(self, snip_id: str, phenotype: str, author: str,
                     notes: str = None, confidence: float = None,
                     force_dead: bool = False) -> bool:
        """Add phenotype to snip with DEAD exclusivity validation."""
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        self._ensure_structures(embryo_id, snip_id)
        
        # Validate phenotype value
        valid_phenotypes = self._get_valid_phenotypes()
        if phenotype not in valid_phenotypes:
            available = list(valid_phenotypes.keys())
            raise ValueError(f"Invalid phenotype '{phenotype}'. Available: {available}")
        
        # Get existing phenotypes for this embryo
        existing_phenotypes = self._get_embryo_phenotypes(embryo_id)
        
        # DEAD exclusivity enforcement
        if phenotype == "DEAD":
            if existing_phenotypes and not force_dead:
                raise ValueError(f"DEAD phenotype cannot coexist with {existing_phenotypes}. Use force_dead=True to override.")
        elif "DEAD" in existing_phenotypes and not force_dead:
            raise ValueError(f"Cannot add '{phenotype}' - embryo already marked DEAD. Use force_dead=True to override.")
        
        # Add phenotype
        phenotype_obj = Phenotype(value=phenotype, author=author, notes=notes, confidence=confidence)
        self.data["embryos"][embryo_id]["snips"][snip_id]["phenotype"] = phenotype_obj.to_dict()
        
        # Validate DEAD consistency after adding
        if phenotype == "DEAD":
            self._validate_embryo_dead_logic(embryo_id)
        
        self._update_timestamps(embryo_id)
        return True
    
    def _validate_embryo_dead_logic(self, embryo_id: str):
        """
        Validate DEAD phenotype consistency across all snips for an embryo.
        Issues warnings if DEAD logic appears inconsistent.
        """
        embryo_phenotypes = self._get_embryo_phenotypes(embryo_id)
        dead_snips = [snip_id for snip_id, pheno in embryo_phenotypes.items() if pheno == "DEAD"]
        
        if not dead_snips:
            return  # No DEAD phenotypes, nothing to validate
        
        # Check for temporal inconsistencies
        alive_after_dead = []
        for snip_id, phenotype in embryo_phenotypes.items():
            if phenotype != "DEAD" and phenotype != "NONE":
                # TODO: Add proper frame number comparison here
                # For now, just warn about non-DEAD phenotypes when DEAD exists
                alive_after_dead.append(f"{snip_id}:{phenotype}")
        
        if alive_after_dead and self.verbose:
            print(f"⚠️ WARNING: Embryo {embryo_id} has DEAD phenotype but also has non-DEAD phenotypes:")
            print(f"   DEAD snips: {dead_snips}")
            print(f"   Alive phenotypes: {alive_after_dead}")
            print(f"   Consider reconciling temporal logic or use force_dead=True if intentional")
    
    def _is_embryo_dead_before_timepoint(self, embryo_id: str, current_snip_id: str) -> bool:
        """
        Check if embryo has any DEAD phenotypes (simplified check).
        TODO: Implement proper temporal ordering using frame extraction.
        Refer to id_parsing_conventions.md for snip_id frame number format.
        """
        embryo_phenotypes = self._get_embryo_phenotypes(embryo_id)
        return any(pheno == "DEAD" for snip_id, pheno in embryo_phenotypes.items() 
                  if snip_id != current_snip_id)
    
    def _get_embryo_phenotypes(self, embryo_id: str) -> Dict[str, str]:
        """Get all phenotypes for an embryo across snips (snip_id -> phenotype)."""
        phenotypes = {}
        for snip_id, snip_data in self.data["embryos"][embryo_id]["snips"].items():
            if "phenotype" in snip_data:
                phenotypes[snip_id] = snip_data["phenotype"]["value"]
        return phenotypes


class EmbryoGenotypeManager:
    """Genotype management with single genotype enforcement."""
    
    def _get_valid_genotypes(self) -> Dict[str, Dict]:
        return self.schema_manager.get_genotypes()
    
    def _get_valid_zygosity_types(self) -> List[str]:
        return self.schema_manager.schema.get("zygosity_types", ["homozygous", "heterozygous", "crispant", "morpholino"])

    
    def add_genotype(self, embryo_id: str, gene_name: str, allele: str,
                    zygosity: str = "heterozygous", author: str = None,
                    overwrite_genotype: bool = False) -> bool:
        """Add single genotype to embryo with strict enforcement."""
        self._ensure_embryo_structure(embryo_id)
        
        # Single genotype enforcement
        existing = self.data["embryos"][embryo_id].get("genotype")
        if existing and not overwrite_genotype:
            current_gene = existing.get("value", "unknown")
            raise ValueError(f"Embryo {embryo_id} already has genotype '{current_gene}'. "
                           f"Use overwrite_genotype=True to change.")
        
        # Validate gene
        valid_genotypes = self._get_valid_genotypes()
        if gene_name not in valid_genotypes:
            available = list(valid_genotypes.keys())
            raise ValueError(f"Invalid gene '{gene_name}'. Available: {available}")
        
        # Validate zygosity
        valid_zygosity = self._get_valid_zygosity_types()
        if zygosity not in valid_zygosity:
            raise ValueError(f"Invalid zygosity '{zygosity}'. Available: {valid_zygosity}")
        
        # Create genotype
        genotype = Genotype(
            value=gene_name, 
            allele=allele, 
            zygosity=zygosity,
            author=author or self.config.get("default_author", "unknown")
        )
        
        self.data["embryos"][embryo_id]["genotype"] = genotype.to_dict()
        self._update_timestamps(embryo_id)
        
        if self.verbose:
            action = "Updated" if existing else "Added"
            print(f"🧬 {action} genotype {gene_name}:{allele} for {embryo_id}")
        
        return True
    
    def validate_genotype_coverage(self) -> Dict:
        """Check for missing genotypes and warn."""
        missing = []
        experiments_missing = set()
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            if not embryo_data.get("genotype"):
                missing.append(embryo_id)
                # Extract experiment ID - refer to id_parsing_conventions.md for details
                exp_id = "_".join(embryo_id.split("_")[:2]) if "_" in embryo_id else "unknown"
                experiments_missing.add(exp_id)
        
        if missing and self.verbose:
            print(f"⚠️ WARNING: {len(missing)} embryos missing genotype data")
            print(f"   Affected experiments: {sorted(experiments_missing)}")
        
        return {
            "missing_count": len(missing),
            "missing_embryos": missing,
            "affected_experiments": sorted(experiments_missing)
        }


class EmbryoFlagManager:
    """Multi-level flag management (snip, video, image, experiment)."""
    
    def _get_valid_treatments(self) -> Dict[str, Dict]:
        return self.schema_manager.get_treatments()

    def _get_valid_priority_levels(self) -> List[str]:
        """
        Get valid priority levels.
        
        NOTE: Currently hardcoded defaults. In future, this should load from
        JSON config via a helper class that manages permitted values across
        all annotation manager types.
        """
        return ["low", "medium", "high", "critical"]
    
    def add_flag(self, entity_id: str, flag_type: str, level: str = "auto",
                description: str = "", priority: str = "medium", 
                notes: str = "", overwrite: bool = False) -> bool:
        """Add flag with level detection and validation."""
        
        # Auto-detect level
        if level == "auto":
            level = self._detect_level(entity_id)
        
        # Validate flag type
        valid_flags = self._get_valid_flags_for_level(level)
        if flag_type not in valid_flags:
            raise ValueError(f"Invalid flag '{flag_type}' for {level} level. Available: {valid_flags}")
        
        return getattr(self, f"_add_{level}_flag")(entity_id, flag_type, description, priority, notes, overwrite)
    
    def _detect_level(self, entity_id: str) -> str:
        """
        Detect entity level from ID format.
        Refer to id_parsing_conventions.md for ID format specifications.
        """
        # Use parsing utils for entity type detection
        return get_entity_type(entity_id)
    
    def _get_valid_flags_for_level(self, level: str) -> List[str]:
        return list(self.schema_manager.get_flags_for_level(level).keys())

    
    def _add_snip_flag(self, snip_id: str, flag_type: str, description: str, 
                      priority: str, notes: str, overwrite: bool) -> bool:
        """Add snip-level flag."""
        embryo_id = self.get_embryo_id_from_snip(snip_id)
        self._ensure_structures(embryo_id, snip_id)
        
        snip_flags = self.data["embryos"][embryo_id]["snips"][snip_id]["flags"]
        
        # Check existing
        for i, flag in enumerate(snip_flags):
            if flag.get("flag_type") == flag_type:
                if not overwrite:
                    raise ValueError(f"Flag '{flag_type}' exists for {snip_id}. Use overwrite=True.")
                snip_flags[i] = self._create_flag_dict(flag_type, description, priority, notes)
                return True
        
        # Add new
        snip_flags.append(self._create_flag_dict(flag_type, description, priority, notes))
        return True
    
    def _create_flag_dict(self, flag_type: str, description: str, priority: str, notes: str) -> Dict:
        """Create standardized flag dictionary."""
        return Flag(
            value=flag_type,
            author=self.config.get("default_author", "unknown"),
            flag_type=flag_type,
            priority=priority,
            notes=f"{description}" + (f" | {notes}" if notes else "")
        ).to_dict()


class EmbryoTreatmentManager:
    """Treatment management - multiple treatments per embryo allowed."""
    
    def _get_valid_treatments(self) -> Dict[str, Dict]:
        """
        Get valid treatment types and their properties.
        
        NOTE: Currently hardcoded defaults. In future, this should load from
        JSON config via a helper class that manages permitted values across
        all annotation manager types.
        """
        return {
            "DMSO": {"type": "vehicle", "description": "Dimethyl sulfoxide vehicle control"},
            "PTU": {"type": "chemical", "description": "1-phenyl-2-thiourea (pigment inhibitor)"},
            "BIO": {"type": "chemical", "description": "BIO GSK-3 inhibitor"},
            "SB431542": {"type": "chemical", "description": "TGF-β inhibitor"},
            "DAPT": {"type": "chemical", "description": "Notch signaling inhibitor"},
            "heat_shock": {"type": "physical", "description": "Heat shock treatment"}
        }
    
    def add_treatment(self, embryo_id: str, treatment_name: str, dosage: str = None,
                     timing: str = None, author: str = None, notes: str = None) -> bool:
        """Add treatment to embryo (multiple allowed)."""
        self._ensure_embryo_structure(embryo_id)
        
        # Validate treatment
        valid_treatments = self._get_valid_treatments()
        if treatment_name not in valid_treatments:
            available = list(valid_treatments.keys())
            raise ValueError(f"Invalid treatment '{treatment_name}'. Available: {available}")
        
        # Create treatment
        treatment = Treatment(
            value=treatment_name,
            author=author or self.config.get("default_author", "unknown"),
            dosage=dosage,
            timing=timing,
            notes=notes
        )
        
        # Generate unique treatment ID
        treatment_id = f"{treatment_name}_{len(self.data['embryos'][embryo_id]['treatments']) + 1}"
        self.data["embryos"][embryo_id]["treatments"][treatment_id] = treatment.to_dict()
        
        self._update_timestamps(embryo_id)
        return True
    
    def get_treatments(self, embryo_id: str) -> Dict:
        """Get all treatments for embryo."""
        if embryo_id in self.data["embryos"]:
            return self.data["embryos"][embryo_id]["treatments"].copy()
        return {}


class EmbryoManagerBase:
    """Base utilities and structure management."""
    
    def _ensure_embryo_structure(self, embryo_id: str):
        """Initialize embryo with complete structure."""
        if embryo_id not in self.data["embryos"]:
            self.data["embryos"][embryo_id] = {
                "genotype": None,
                "treatments": {},
                "flags": {},
                "notes": "",
                "metadata": {
                    "created": self.get_timestamp(),
                    "last_updated": self.get_timestamp()
                },
                "snips": {}
            }
    
    def _ensure_structures(self, embryo_id: str, snip_id: str = None):
        """Ensure both embryo and snip structures."""
        self._ensure_embryo_structure(embryo_id)
        if snip_id and snip_id not in self.data["embryos"][embryo_id]["snips"]:
            self.data["embryos"][embryo_id]["snips"][snip_id] = {"flags": []}
    
    def _update_timestamps(self, embryo_id: str):
        """Update timestamps."""
        timestamp = self.get_timestamp()
        self.data["embryos"][embryo_id]["metadata"]["last_updated"] = timestamp
    
    def add_embryo_note(self, embryo_id: str, note: str, append: bool = True) -> bool:
        """Add note to embryo."""
        self._ensure_embryo_structure(embryo_id)
        
        if append and self.data["embryos"][embryo_id]["notes"]:
            self.data["embryos"][embryo_id]["notes"] += f" | {note}"
        else:
            self.data["embryos"][embryo_id]["notes"] = note
        
        self._update_timestamps(embryo_id)
        return True

    def get_embryo_id_from_snip(self, snip_id: str) -> str:
        """Extract embryo ID from snip ID using parsing utils."""
        try:
            parsed = parse_entity_id(snip_id)
            return parsed.get("embryo_id")
        except Exception:
            return None

    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()

    def validate_id_format(self, entity_id: str, entity_type: str) -> bool:
        """Use parsing utils for validation."""
        detected_type = get_entity_type(entity_id)
        return detected_type == entity_type


class UnifiedEmbryoManager(EmbryoPhenotypeManager, EmbryoGenotypeManager, 
                          EmbryoFlagManager, EmbryoTreatmentManager, EmbryoManagerBase):
    """
    Unified embryo metadata manager with comprehensive validation.
    
    SCOPE: This class handles operations at the EMBRYO level only. A separate 
    higher-level class will handle experiment-wide operations, cross-embryo 
    analysis, and batch processing. This manager focuses on:
    - Single embryo data consistency
    - Internal validation rules (DEAD logic, single genotype, etc.)
    - Data organization within embryo structure
    - Simple CRUD operations per embryo
    
    Key Features:
    - DEAD phenotype exclusivity enforcement
    - Single genotype per embryo with warnings for missing data
    - Multi-level flags (snip/video/image/experiment)
    - Multiple treatments per embryo
    - Comprehensive validation and error handling
    """
    
    def validate_data_integrity(self) -> Dict:
        """Comprehensive data validation."""
        results = {
            "genotype_coverage": self.validate_genotype_coverage(),
            "dead_conflicts": self._check_dead_conflicts(),
            "flag_distribution": self._analyze_flag_distribution()
        }
        return results
    
    def _check_dead_conflicts(self) -> Dict:
        """Check for DEAD phenotype conflicts."""
        conflicts = []
        temporal_violations = []
        
        for embryo_id, embryo_data in self.data["embryos"].items():
            # Get all phenotypes across snips
            phenotypes = self._get_embryo_phenotypes(embryo_id)
            
            for snip_id, snip_data in embryo_data["snips"].items():
                phenotype_data = snip_data.get("phenotype", {})
                if phenotype_data:
                    phenotype_value = phenotype_data.get("value")
                    
                    # Check for DEAD coexistence in same snip (simplified check)
                    if phenotype_value == "DEAD":
                        # Check if other phenotypes exist for this embryo
                        other_phenotypes = [p for sid, p in phenotypes.items() 
                                          if sid != snip_id and p != "DEAD"]
                        if other_phenotypes:
                            conflicts.append({
                                "embryo_id": embryo_id,
                                "snip_id": snip_id,
                                "conflicting_phenotypes": other_phenotypes
                            })
            
            # Check temporal DEAD logic - no return from DEAD
            if "DEAD" in phenotypes.values():
                # Get snips with frame numbers for temporal ordering
                snip_frames = []
                for snip_id, phenotype in phenotypes.items():
                    try:
                        frame_num = extract_frame_number(snip_id)
                        snip_frames.append((frame_num, snip_id, phenotype))
                    except ValueError:
                        continue  # Skip snips without valid frame numbers
                
                # Sort by frame number to check temporal order
                snip_frames.sort(key=lambda x: x[0])
                
                found_dead = False
                for frame_num, snip_id, phenotype in snip_frames:
                    if phenotype == "DEAD":
                        found_dead = True
                    elif found_dead and phenotype != "DEAD":
                        # Found non-DEAD phenotype after DEAD - temporal violation
                        temporal_violations.append({
                            "embryo_id": embryo_id,
                            "violation_snip": snip_id,
                            "violation_frame": frame_num,
                            "phenotype": phenotype,
                            "message": f"Non-DEAD phenotype '{phenotype}' found after embryo marked DEAD"
                        })
        
        return {
            "conflicts": conflicts, 
            "conflict_count": len(conflicts),
            "temporal_violations": temporal_violations,
            "temporal_violation_count": len(temporal_violations)
        }
    
    def _analyze_flag_distribution(self) -> Dict:
        """Analyze flag distribution across levels."""
        distribution = {"snip": 0, "embryo": 0, "video": 0, "image": 0, "experiment": 0}
        
        for embryo_data in self.data["embryos"].values():
            distribution["embryo"] += len(embryo_data.get("flags", {}))
            for snip_data in embryo_data.get("snips", {}).values():
                distribution["snip"] += len(snip_data.get("flags", []))
        
        return distribution
    
    def get_embryo_summary(self, embryo_id: str) -> Dict:
        """Get concise summary of embryo's current state."""
        phenotypes = self._get_embryo_phenotypes(embryo_id)
        
        # Find earliest DEAD snip
        death_snip_id = None
        if phenotypes:
            dead_snips = [(snip_id, phenotype) for snip_id, phenotype in phenotypes.items() if phenotype == "DEAD"]
            if dead_snips:
                try:
                    # Sort by frame number to find earliest
                    dead_with_frames = [(extract_frame_number(snip_id), snip_id) for snip_id, _ in dead_snips]
                    death_snip_id = min(dead_with_frames, key=lambda x: x[0])[1]
                except ValueError:
                    # Fallback if frame extraction fails
                    death_snip_id = dead_snips[0][0]
        
        return {
            "genotype": self.data["embryos"].get(embryo_id, {}).get("genotype"),
            "phenotype_count": len(phenotypes),
            "treatment_count": len(self.get_treatments(embryo_id)),
            "flag_count": len(self.data["embryos"].get(embryo_id, {}).get("flags", {})),
            "snip_count": len(self.data["embryos"].get(embryo_id, {}).get("snips", {})),
            "death_snip_id": death_snip_id,
            "is_dead": death_snip_id is not None
        }
    
    def clear_embryo_data(self, embryo_id: str, data_type: str = "all"):
        """Clear specific data types for embryo."""
        if embryo_id not in self.data["embryos"]:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        embryo = self.data["embryos"][embryo_id]
        
        if data_type in ["all", "phenotypes"]:
            for snip_data in embryo["snips"].values():
                if "phenotype" in snip_data:
                    del snip_data["phenotype"]
        
        if data_type in ["all", "flags"]:
            embryo["flags"].clear()
            for snip_data in embryo["snips"].values():
                snip_data["flags"].clear()
        
        if data_type in ["all", "treatments"]:
            embryo["treatments"].clear()
        
        if data_type in ["all", "genotype"]:
            embryo["genotype"] = None
        
        self._update_timestamps(embryo_id)
        return True
    
    def copy_embryo_structure(self, source_id: str, target_id: str):
        """Copy embryo structure (snips/metadata) without annotations."""
        if source_id not in self.data["embryos"]:
            raise ValueError(f"Source embryo {source_id} not found")
        
        source = self.data["embryos"][source_id]
        
        # Copy structure without annotations
        self.data["embryos"][target_id] = {
            "genotype": None,
            "treatments": {},
            "flags": {},
            "notes": "",
            "metadata": {"created": self.get_timestamp()},
            "snips": {snip_id: {"flags": []} for snip_id in source["snips"]}
        }
        
        return True
