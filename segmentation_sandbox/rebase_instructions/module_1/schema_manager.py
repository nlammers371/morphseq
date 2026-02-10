"""
Schema Manager - Dynamic schema management for MorphSeq pipeline.
Handles permitted values for phenotypes, genotypes, flags, treatments.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class SchemaManager:
    """Manages permitted values schema with live updates and persistence."""
    
    def __init__(self, schema_path: str = "morphseq_schema.json"):
        self.schema_path = Path(schema_path)
        self.schema = self._load_or_create_schema()
    
    def _load_or_create_schema(self) -> Dict:
        """Load existing schema or create default."""
        if self.schema_path.exists():
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        
        # Default schema
        return {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "phenotypes": {
                "EDEMA": {"description": "Fluid accumulation"},
                "BODY_AXIS_DEFECT": {"description": "Body axis abnormalities"},
                "CONVERGENCE_EXTENSION": {"description": "C&E movement defects"},
                "DEAD": {"exclusive": True, "terminal": True, "description": "Embryo death"}
            },
            "genotypes": {
                "TBD": {"default": True, "description": "To be determined"},
                "WT": {"description": "Wild type"},
                "tmem67": {"gene": "tmem67", "description": "Transmembrane protein 67"},
                "b9d2": {"gene": "b9d2", "description": "B9 domain containing 2"},
                "lmx1b": {"gene": "lmx1b", "description": "LIM homeobox transcription factor 1 beta"}
            },
            "flags": {
                "snip_level": {
                    "MOTION_BLUR": {"priority": "medium", "description": "Motion artifacts"},
                    "MASK_ON_EDGE": {"priority": "low", "description": "Mask near image edge"},
                    "HIGHLY_VAR_MASK": {"priority": "medium", "description": "Variable mask size"}
                },
                "video_level": {
                    "NONZERO_SEED_FRAME": {"priority": "high", "description": "Invalid seed frame"},
                    "NO_EMBRYO_DETECTED": {"priority": "critical", "description": "No embryos found"}
                },
                "image_level": {
                    "DETECTION_FAILURE": {"priority": "high", "description": "Detection algorithm failed"}
                },
                "experiment_level": {
                    "BATCH_CONTAMINATION": {"priority": "critical", "description": "Batch contaminated"},
                    "PROTOCOL_DEVIATION": {"priority": "high", "description": "Protocol not followed"}
                }
            },
            "treatments": {
                "DMSO": {"type": "vehicle", "description": "Vehicle control"},
                "PTU": {"type": "chemical", "description": "Pigment inhibitor"},
                "heat_shock": {"type": "physical", "description": "Heat shock treatment"}
            },
            "zygosity_types": ["homozygous", "heterozygous", "crispant", "morpholino"]
        }
    
    # ========== PHENOTYPE MANAGEMENT ==========
    
    def add_phenotype(self, name: str, description: str, exclusive: bool = False, 
                     terminal: bool = False, default: bool = False):
        """Add new phenotype type."""
        self.schema["phenotypes"][name] = {
            "description": description,
            "exclusive": exclusive,
            "terminal": terminal,
            "default": default
        }
        self._update_timestamp()
    
    def remove_phenotype(self, name: str):
        """Remove phenotype type."""
        if name in self.schema["phenotypes"]:
            del self.schema["phenotypes"][name]
            self._update_timestamp()
    
    def get_phenotypes(self) -> Dict:
        """Get all phenotype definitions."""
        return self.schema["phenotypes"].copy()
    
    # ========== GENOTYPE MANAGEMENT ==========
    
    def add_genotype(self, name: str, description: str, gene: str = None, default: bool = False):
        """Add new genotype type."""
        genotype_data = {"description": description, "default": default}
        if gene:
            genotype_data["gene"] = gene
        
        self.schema["genotypes"][name] = genotype_data
        self._update_timestamp()
    
    def remove_genotype(self, name: str):
        """Remove genotype type."""
        if name in self.schema["genotypes"]:
            del self.schema["genotypes"][name]
            self._update_timestamp()
    
    def get_genotypes(self) -> Dict:
        """Get all genotype definitions."""
        return self.schema["genotypes"].copy()
    
    # ========== FLAG MANAGEMENT ==========
    
    def add_flag(self, name: str, level: str, description: str, priority: str = "medium"):
        """Add new flag type."""
        level_key = f"{level}_level"
        if level_key not in self.schema["flags"]:
            self.schema["flags"][level_key] = {}
        
        self.schema["flags"][level_key][name] = {
            "priority": priority,
            "description": description
        }
        self._update_timestamp()
    
    def remove_flag(self, name: str, level: str):
        """Remove flag type."""
        level_key = f"{level}_level"
        if level_key in self.schema["flags"] and name in self.schema["flags"][level_key]:
            del self.schema["flags"][level_key][name]
            self._update_timestamp()
    
    def get_flags_for_level(self, level: str) -> Dict:
        """Get flags for specific level."""
        level_key = f"{level}_level"
        return self.schema["flags"].get(level_key, {}).copy()
    
    def get_all_flags(self) -> Dict:
        """Get all flag definitions."""
        return self.schema["flags"].copy()
    
    # ========== TREATMENT MANAGEMENT ==========
    
    def add_treatment(self, name: str, description: str, treatment_type: str = "chemical"):
        """Add new treatment type."""
        self.schema["treatments"][name] = {
            "type": treatment_type,
            "description": description
        }
        self._update_timestamp()
    
    def remove_treatment(self, name: str):
        """Remove treatment type."""
        if name in self.schema["treatments"]:
            del self.schema["treatments"][name]
            self._update_timestamp()
    
    def get_treatments(self) -> Dict:
        """Get all treatment definitions."""
        return self.schema["treatments"].copy()
    
    # ========== VALIDATION ==========
    
    # method to get validation methods for managers
    def get_validators(self):
        """Return validation callables for managers."""
        return {
            "phenotypes": self.get_phenotypes,
            "genotypes": self.get_genotypes,
            "treatments": self.get_treatments,
            "flags": self.get_flags_for_level,
            "zygosity": lambda: self.schema["zygosity_types"],
            "priority": lambda: self.schema.get("priority_levels", ["low", "medium", "high", "critical"])
        }
        
    def validate_phenotype(self, value: str) -> bool:
        """Check if phenotype is valid."""
        return value in self.schema["phenotypes"]
    
    def validate_genotype(self, value: str) -> bool:
        """Check if genotype is valid."""
        return value in self.schema["genotypes"]
    
    def validate_flag(self, value: str, level: str) -> bool:
        """Check if flag is valid for level."""
        level_key = f"{level}_level"
        return value in self.schema["flags"].get(level_key, {})
    
    def validate_treatment(self, value: str) -> bool:
        """Check if treatment is valid."""
        return value in self.schema["treatments"]
    
    def validate_zygosity(self, value: str) -> bool:
        """Check if zygosity is valid."""
        return value in self.schema["zygosity_types"]
    
    def validate_priority(self, value: str) -> bool:
        """Check if priority level is valid."""
        return value in self.schema["priority_levels"]
    
    # ========== BULK OPERATIONS ==========
    
    def add_multiple_phenotypes(self, phenotypes: List[Dict]):
        """Add multiple phenotypes at once."""
        for pheno in phenotypes:
            self.add_phenotype(**pheno)
    
    def add_multiple_genotypes(self, genotypes: List[Dict]):
        """Add multiple genotypes at once."""
        for geno in genotypes:
            self.add_genotype(**geno)
    
    def get_schema_summary(self) -> Dict:
        """Get count summary of schema."""
        return {
            "phenotypes": len(self.schema["phenotypes"]),
            "genotypes": len(self.schema["genotypes"]),
            "flags": sum(len(flags) for flags in self.schema["flags"].values()),
            "treatments": len(self.schema["treatments"]),
            "last_updated": self.schema["last_updated"]
        }
    
    # ========== PERSISTENCE ==========
    
    def save_schema(self):
        """Save schema to file."""
        with open(self.schema_path, 'w') as f:
            json.dump(self.schema, f, indent=2)
    
    def export_schema(self, path: Union[str, Path]):
        """Export schema to different file."""
        with open(path, 'w') as f:
            json.dump(self.schema, f, indent=2)
    
    def import_schema(self, path: Union[str, Path], merge: bool = True):
        """Import schema from file."""
        with open(path, 'r') as f:
            imported = json.load(f)
        
        if merge:
            # Merge with existing schema
            for category in ["phenotypes", "genotypes", "treatments"]:
                if category in imported:
                    self.schema[category].update(imported[category])
            
            # Merge flags by level
            if "flags" in imported:
                for level, flags in imported["flags"].items():
                    if level not in self.schema["flags"]:
                        self.schema["flags"][level] = {}
                    self.schema["flags"][level].update(flags)
        else:
            # Replace entire schema
            self.schema = imported
        
        self._update_timestamp()
    
    def _update_timestamp(self):
        """Update last modified timestamp."""
        self.schema["last_updated"] = datetime.now().isoformat()
    
    def __str__(self) -> str:
        summary = self.get_schema_summary()
        return f"SchemaManager({summary['phenotypes']} phenotypes, {summary['genotypes']} genotypes, {summary['flags']} flags, {summary['treatments']} treatments)"