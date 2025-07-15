"""
Permitted Values Manager for EmbryoMetadata System
Manages the JSON-based schema for permitted values
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class PermittedValuesManager:
    """Manage permitted values schema for EmbryoMetadata system."""
    
    def __init__(self, schema_path: Path = None):
        """
        Initialize schema manager.
        
        Args:
            schema_path: Path to schema JSON file (default: config/permitted_values_schema.json)
        """
        if schema_path is None:
            # Default location relative to this file
            schema_path = Path(__file__).parent / "config" / "permitted_values_schema.json"
        
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict:
        """Load schema from file or create default."""
        if self.schema_path.exists():
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        else:
            # Create default schema
            default = self._get_default_schema()
            self._save_schema(default)
            return default
    
    def _save_schema(self, schema: Dict) -> None:
        """Save schema to file."""
        self.schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema["last_updated"] = datetime.now().isoformat()
        
        with open(self.schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
    
    def _get_default_schema(self) -> Dict:
        """Get default schema structure."""
        return {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "phenotypes": {
                "NONE": {"description": "No phenotype observed", "is_default": True},
                "TBD": {"description": "To be determined", "is_default": False},
                "DEAD": {"description": "Embryo death", "exclusive": True, "terminal": True}
            },
            "genotypes": {
                "TBD": {"description": "To be determined", "is_default": True},
                "WT": {"description": "Wild-type, no genetic modifications", "is_default": False},
                "lmx1b": {"description": "LIM homeobox transcription factor 1-beta gene mutation", "is_default": False}
            },
            "treatments": {
                "NONE": {"description": "No treatment", "is_default": True},
                "shh-i": {"description": "Sonic hedgehog inhibition using cyclopamine", "type": "chemical", "is_default": False, "concentration_unit": "μM"}
            },
            "flags": {},
            "severity_levels": ["low", "medium", "high", "critical"],
            "zygosity_types": ["homozygous", "heterozygous", "compound_heterozygous", "hemizygous"],
            "priority_levels": ["low", "medium", "high", "critical"],
            "flag_types": ["quality", "analysis", "morphology", "tracking", "manual"],
            "treatment_types": ["chemical", "temperature", "mechanical", "genetic", "environmental"]
        }
    
    # -------------------------------------------------------------------------
    # Adding New Values
    # -------------------------------------------------------------------------
    
    def add_phenotype(self, name: str, description: str, 
                     exclusive: bool = False, terminal: bool = False) -> None:
        """Add new phenotype to schema."""
        if name in self.schema["phenotypes"]:
            raise ValueError(f"Phenotype '{name}' already exists")
        
        self.schema["phenotypes"][name] = {
            "description": description,
            "is_default": False,
            "exclusive": exclusive,
            "terminal": terminal
        }
        
        self._save_schema(self.schema)
        print(f"✅ Added phenotype '{name}'")
    
    def add_genotype(self, name: str, description: str, 
                    aliases: List[str] = None) -> None:
        """Add new genotype to schema."""
        if name in self.schema["genotypes"]:
            raise ValueError(f"Genotype '{name}' already exists")
        
        genotype_data = {"description": description, "is_default": False}
        if aliases:
            genotype_data["aliases"] = aliases
        
        self.schema["genotypes"][name] = genotype_data
        self._save_schema(self.schema)
        print(f"✅ Added genotype '{name}'")
    
    def add_treatment(self, name: str, description: str, 
                     treatment_type: str = "chemical",
                     concentration_unit: str = None) -> None:
        """Add new treatment to schema."""
        if name in self.schema["treatments"]:
            raise ValueError(f"Treatment '{name}' already exists")
        
        treatment_data = {
            "description": description,
            "type": treatment_type,
            "is_default": False
        }
        
        if concentration_unit:
            treatment_data["concentration_unit"] = concentration_unit
        
        self.schema["treatments"][name] = treatment_data
        self._save_schema(self.schema)
        print(f"✅ Added treatment '{name}'")
    
    def add_flag(self, level: str, name: str, description: str,
                severity: str = "warning") -> None:
        """Add new flag to schema."""
        level_key = f"{level}_level"
        
        if level_key not in self.schema["flags"]:
            self.schema["flags"][level_key] = {}
        
        if name in self.schema["flags"][level_key]:
            raise ValueError(f"Flag '{name}' already exists at {level} level")
        
        self.schema["flags"][level_key][name] = {
            "description": description,
            "severity": severity
        }
        
        self._save_schema(self.schema)
        print(f"✅ Added flag '{name}' at {level} level")
    
    # -------------------------------------------------------------------------
    # Getting Values
    # -------------------------------------------------------------------------
    
    def get_phenotypes(self) -> Dict:
        """Get all phenotypes."""
        return self.schema.get("phenotypes", {})
    
    def get_genotypes(self) -> Dict:
        """Get all genotypes."""
        return self.schema.get("genotypes", {})
    
    def get_treatments(self) -> Dict:
        """Get all treatments."""
        return self.schema.get("treatments", {})
    
    def get_flags(self, level: Optional[str] = None) -> Dict:
        """Get flags for specific level or all."""
        if level:
            return self.schema.get("flags", {}).get(f"{level}_level", {})
        return self.schema.get("flags", {})
    
    def get_default_value(self, category: str) -> Optional[str]:
        """Get default value for a category."""
        values = self.schema.get(category, {})
        for name, info in values.items():
            if info.get("is_default", False):
                return name
        return None
    
    def get_values(self, category: str) -> List:
        """Get list of values for a category."""
        schema_values = self.schema.get(category, [])
        if isinstance(schema_values, list):
            return schema_values
        elif isinstance(schema_values, dict):
            return list(schema_values.keys())
        return []

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    def validate_value(self, category: str, value: str) -> bool:
        """
        General validation method for any schema category.
        
        Args:
            category: Schema category (e.g., 'severity_levels', 'zygosity_types')
            value: Value to validate
            
        Returns:
            bool: True if value is valid for category
        """
        schema_values = self.schema.get(category, [])
        if isinstance(schema_values, list):
            return value in schema_values
        elif isinstance(schema_values, dict):
            return value in schema_values
        return False
    
    def is_valid_phenotype(self, phenotype: str) -> bool:
        """Check if phenotype is valid."""
        return phenotype in self.schema.get("phenotypes", {})
    
    def is_valid_genotype(self, genotype: str) -> bool:
        """Check if genotype is valid."""
        genotypes = self.schema.get("genotypes", {})
        
        # Check direct match
        if genotype in genotypes:
            return True
        
        # Check aliases
        for g_data in genotypes.values():
            if genotype in g_data.get("aliases", []):
                return True
        
        return False
    
    def is_valid_treatment(self, treatment: str) -> bool:
        """Check if treatment is valid."""
        return treatment in self.schema.get("treatments", {})
    
    def is_valid_flag(self, flag: str, level: str) -> bool:
        """Check if flag is valid for level."""
        level_flags = self.schema.get("flags", {}).get(f"{level}_level", {})
        return flag in level_flags
    
    def reload_schema(self) -> None:
        """Reload schema from file."""
        self.schema = self._load_schema()
