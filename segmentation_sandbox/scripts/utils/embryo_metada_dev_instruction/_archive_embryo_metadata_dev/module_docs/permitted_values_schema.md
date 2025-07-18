# Permitted Values Schema Management

## Overview
A JSON-based schema system for managing permitted values across the EmbryoMetadata system. This provides a single source of truth for valid phenotypes, genotypes, treatments, and flags.

## File Location
The permitted values should be stored in a JSON file at:
```
config/permitted_values_schema.json
```

This location is:
- Easy to find and modify
- Version controlled
- Separate from code but close to it
- Can be packaged with the module

## Schema Structure

```json
{
  "version": "1.0",
  "last_updated": "2024-12-15T10:00:00",
  "phenotypes": {
    "NONE": {
      "description": "No phenotype observed",
      "is_default": true,
      "color": "#CCCCCC"
    },
    "TBD": {
      "description": "To be determined",
      "is_default": false,
      "color": "#999999"
    },
    "EDEMA": {
      "description": "Fluid accumulation/swelling",
      "is_default": false,
      "color": "#3498DB"
    },
    "BODY_AXIS": {
      "description": "Body axis formation defect",
      "is_default": false,
      "color": "#E74C3C"
    },
    "CONVERGENCE_EXTENSION": {
      "description": "Convergent extension defect",
      "is_default": false,
      "color": "#F39C12"
    },
    "DEAD": {
      "description": "Embryo death",
      "is_default": false,
      "exclusive": true,
      "terminal": true,
      "color": "#2C3E50"
    }
  },
  "genotypes": {
    "TBD": {
      "description": "To be determined",
      "is_default": true
    },
    "wildtype": {
      "description": "Wild type",
      "aliases": ["wt", "WT", "+/+"]
    },
    "heterozygous": {
      "description": "Heterozygous mutant",
      "aliases": ["het", "+/-"]
    },
    "homozygous": {
      "description": "Homozygous mutant",
      "aliases": ["homo", "-/-"]
    }
  },
  "treatments": {
    "NONE": {
      "description": "No treatment",
      "is_default": true,
      "type": "control"
    },
    "DMSO": {
      "description": "DMSO vehicle control",
      "type": "control",
      "concentration_unit": "%"
    },
    "SU5402": {
      "description": "FGFR inhibitor",
      "type": "chemical",
      "concentration_unit": "μM",
      "target": "FGFR"
    },
    "heat_shock": {
      "description": "Heat shock treatment",
      "type": "temperature",
      "temperature_unit": "°C"
    }
  },
  "flags": {
    "snip_level": {
      "MOTION_BLUR": {
        "description": "Motion blur detected in frame",
        "severity": "warning"
      },
      "HIGHLY_VAR_MASK": {
        "description": "Mask area variance >10% between frames",
        "severity": "warning"
      },
      "MASK_ON_EDGE": {
        "description": "Mask within 5 pixels of image edge",
        "severity": "info"
      }
    },
    "image_level": {
      "DETECTION_FAILURE": {
        "description": "Embryo detection failed",
        "severity": "error"
      }
    },
    "video_level": {
      "NONZERO_SEED_FRAME": {
        "description": "Seed frame is not the first frame",
        "severity": "info"
      },
      "NO_EMBRYO": {
        "description": "No embryo detected in entire video",
        "severity": "error"
      }
    },
    "experiment_level": {
      "INCOMPLETE": {
        "description": "Experiment data incomplete",
        "severity": "warning"
      },
      "PROTOCOL_DEVIATION": {
        "description": "Deviation from standard protocol",
        "severity": "warning"
      }
    }
  }
}
```

## Schema Manager Utility

```python
# permitted_values_manager.py

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
            schema_path = Path(__file__).parent.parent / "config" / "permitted_values_schema.json"
        
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
                "TBD": {"description": "To be determined", "is_default": True}
            },
            "treatments": {
                "NONE": {"description": "No treatment", "is_default": True}
            },
            "flags": {}
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
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
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
```

## Usage in EmbryoMetadata

```python
# In embryo_metadata.py

class EmbryoMetadata(BaseAnnotationParser):
    def __init__(self, ...):
        super().__init__(...)
        
        # Load permitted values
        self.schema_manager = PermittedValuesManager()
        self.permitted_values = self.schema_manager.schema
    
    def add_phenotype(self, snip_id: str, phenotype: str, ...):
        # Validate against schema
        if not self.schema_manager.is_valid_phenotype(phenotype):
            valid = list(self.schema_manager.get_phenotypes().keys())
            raise ValueError(f"Invalid phenotype '{phenotype}'. Valid: {valid}")
        
        # Continue with adding phenotype...
    
    def add_new_permitted_phenotype(self, name: str, description: str):
        """Add new phenotype to permitted values."""
        self.schema_manager.add_phenotype(name, description)
        # Reload schema
        self.permitted_values = self.schema_manager.schema
```

## Command Line Utility

```python
# add_permitted_value.py

import argparse
from permitted_values_manager import PermittedValuesManager

def main():
    parser = argparse.ArgumentParser(description="Add permitted values to schema")
    parser.add_argument("category", choices=["phenotype", "genotype", "treatment", "flag"])
    parser.add_argument("name", help="Name of the value")
    parser.add_argument("description", help="Description of the value")
    parser.add_argument("--level", help="Flag level (for flags only)")
    parser.add_argument("--severity", default="warning", help="Flag severity")
    
    args = parser.parse_args()
    
    manager = PermittedValuesManager()
    
    if args.category == "phenotype":
        manager.add_phenotype(args.name, args.description)
    elif args.category == "genotype":
        manager.add_genotype(args.name, args.description)
    elif args.category == "treatment":
        manager.add_treatment(args.name, args.description)
    elif args.category == "flag":
        if not args.level:
            print("Error: --level required for flags")
            return
        manager.add_flag(args.level, args.name, args.description, args.severity)

if __name__ == "__main__":
    main()
```

Usage:
```bash
# Add new phenotype
python add_permitted_value.py phenotype HEART_DEFECT "Heart development defect"

# Add new treatment
python add_permitted_value.py treatment BMP4 "Bone morphogenetic protein 4"

# Add new flag
python add_permitted_value.py flag OUT_OF_FOCUS "Image out of focus" --level image --severity warning
```

## Benefits

1. **Single Source of Truth**: All permitted values in one file
2. **Easy to Modify**: JSON format is human-readable
3. **Version Controlled**: Changes tracked in git
4. **Validation Built-in**: Schema manager validates all inputs
5. **Extensible**: Easy to add new categories or values
6. **Command Line Tools**: Quick additions without code changes