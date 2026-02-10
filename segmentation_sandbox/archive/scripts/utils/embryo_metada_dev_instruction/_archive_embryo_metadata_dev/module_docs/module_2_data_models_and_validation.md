# Module 2: Data Models and Validation

## Overview
This module defines the data models, validation schemas, and permitted values management for the EmbryoMetadata system.

## Type Definitions

```python
from typing import TypedDict, Literal, Optional, Dict, List, Union
from datetime import datetime
from dataclasses import dataclass, field

# Phenotype types
PhenotypeValue = Literal["NONE", "EDEMA", "BODY_AXIS", "CONVERGENCE_EXTENSION", "DEAD"]

# Flag level types
FlagLevel = Literal["snip", "image", "video", "experiment"]

# Genotype can be any string (user-defined)
GenotypeValue = str
```

## Data Classes

### Annotation Base Class

```python
@dataclass
class AnnotationBase:
    """Base class for all annotations with common fields."""
    value: str
    author: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        data = {
            "value": self.value,
            "author": self.author,
            "timestamp": self.timestamp
        }
        if self.notes:
            data["notes"] = self.notes
        return data
    
    def validate(self, permitted_values: Optional[List[str]] = None):
        """Validate annotation against permitted values."""
        if not self.author:
            raise ValidationError("Author is required")
        
        if permitted_values and self.value not in permitted_values:
            raise ValidationError(
                f"Value '{self.value}' not in permitted values: {permitted_values}"
            )
```

### Phenotype Model

```python
@dataclass
class Phenotype(AnnotationBase):
    """Phenotype annotation with special handling for DEAD."""
    value: PhenotypeValue
    confidence: Optional[float] = None  # For ML predictions
    
    def to_dict(self) -> Dict:
        data = super().to_dict()
        if self.confidence is not None:
            data["confidence"] = self.confidence
        return data
    
    def validate(self, permitted_values: Optional[List[str]] = None):
        """Validate with phenotype-specific rules."""
        super().validate(permitted_values)
        
        if self.confidence is not None:
            if not 0 <= self.confidence <= 1:
                raise ValidationError("Confidence must be between 0 and 1")
```

### Genotype Model

```python
@dataclass
class Genotype(AnnotationBase):
    """Genotype annotation with overwrite protection."""
    value: GenotypeValue
    confirmed: bool = False  # Lab confirmation status
    method: Optional[str] = None  # Genotyping method (PCR, sequencing, etc.)
    
    def to_dict(self) -> Dict:
        data = super().to_dict()
        data["confirmed"] = self.confirmed
        if self.method:
            data["method"] = self.method
        return data
```

### Flag Model


```python
@dataclass
class Flag(AnnotationBase):
    """Quality control flag."""
    severity: Literal["info", "warning", "error"] = "warning"
    auto_generated: bool = False
    
    def to_dict(self) -> Dict:
        data = super().to_dict()
        data["severity"] = self.severity
        data["auto_generated"] = self.auto_generated
        return data
```

### Treatment Model
```python



@dataclass
class Treatment(AnnotationBase):
    """Treatment annotation for chemical or temperature treatments."""
    value: TreatmentValue
    details: Optional[str] = None  # Additional information, e.g., concentration or temperature specifics

    def to_dict(self) -> Dict:
        data = super().to_dict()
        if self.details is not None:
            data["details"] = self.details
        return data
```

## Permitted Values Schema

```python
class PermittedValuesSchema:
    """Schema for managing permitted values."""
    
    @staticmethod
    def get_default_schema():
        """Get default permitted values structure."""
        return {
            "phenotypes": {
                "NONE": {
                    "description": "No phenotype observed",
                    "is_default": True,
                    "color": "#CCCCCC"  # For visualization
                },
                "EDEMA": {
                    "description": "Fluid accumulation/swelling",
                    "is_default": False,
                    "color": "#3498DB"
                },
                "BODY_AXIS": {
                    "description": "Body axis formation defect",
                    "is_default": False,
                    "color": "#E74C3C"
                },
                "CONVERGENCE_EXTENSION": {
                    "description": "Convergent extension defect",
                    "is_default": False,
                    "color": "#F39C12"
                },
                "DEAD": {
                    "description": "Embryo death",
                    "is_default": False,
                    "exclusive": True,  # Cannot coexist with others
                    "terminal": True,   # No phenotypes after this
                    "color": "#2C3E50"
                }
            },
            "genotypes": {
                # User-defined, no defaults
            },
            "treatmeents": {
                # User-defined, no defaults
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

## Validation Functions

### Core Validation

```python
class Validator:
    """Validation utilities for EmbryoMetadata."""
    
    @staticmethod
    def validate_phenotype(phenotype: Union[str, Phenotype], 
                          permitted_values: Dict,
                          existing_phenotypes: List[str] = None) -> None:
        """
        Validate phenotype against rules.
        
        Checks:
        1. Value in permitted list
        2. DEAD exclusivity rule
        3. Terminal phenotype rules
        """
        if isinstance(phenotype, str):
            phenotype_value = phenotype
        else:
            phenotype_value = phenotype.value
        
        # Check permitted values
        if phenotype_value not in permitted_values:
            raise ValidationError(
                f"Phenotype '{phenotype_value}' not permitted. "
                f"Allowed: {list(permitted_values.keys())}"
            )
        
        # Check DEAD exclusivity
        if existing_phenotypes:
            phenotype_info = permitted_values.get(phenotype_value, {})
            
            # If adding DEAD, check no other phenotypes exist
            if phenotype_info.get("exclusive") and any(p != "NONE" for p in existing_phenotypes):
                raise ValidationError(
                    "DEAD phenotype cannot coexist with other phenotypes"
                )
            
            # If DEAD exists, prevent adding other phenotypes
            if "DEAD" in existing_phenotypes and phenotype_value not in ["NONE", "DEAD"]:
                raise ValidationError(
                    "Cannot add phenotypes after DEAD"
                )
    
    @staticmethod
    def validate_genotype(genotype: Union[str, Genotype],
                         existing_genotype: Optional[Genotype],
                         overwrite_genotype: bool = False) -> None:
        """
        Validate genotype with overwrite protection.
        
        Checks:
        1. Not overwriting without permission
        2. Valid genotype format
        """
        if existing_genotype and not overwrite_genotype:
            raise ValidationError(
                f"Genotype already set to '{existing_genotype.value}'. "
                "Use overwrite_genotype=True to change."
            )
        
        # Could add format validation here (e.g., nomenclature rules)
        if isinstance(genotype, str):
            if not genotype.strip():
                raise ValidationError("Genotype cannot be empty")
    
    @staticmethod
    def validate_flag(flag: Union[str, Flag],
                     level: FlagLevel,
                     permitted_values: Dict) -> None:
        """Validate flag against permitted values for level."""
        if isinstance(flag, str):
            flag_value = flag
        else:
            flag_value = flag.value
        
        level_key = f"{level}_level"
        if level_key not in permitted_values:
            raise ValidationError(f"Invalid flag level: {level}")
        
        level_flags = permitted_values[level_key]
        if flag_value not in level_flags:
            raise ValidationError(
                f"Flag '{flag_value}' not permitted for {level} level. "
                f"Allowed: {list(level_flags.keys())}"
            )
    
    @staticmethod
    def validate_id_format(id_value: str, id_type: str) -> None:
        """
        Validate ID formats.
        
        Expected formats:
        - experiment_id: YYYYMMDD
        - video_id: YYYYMMDD_WELL
        - image_id: YYYYMMDD_WELL_FRAME
        - embryo_id: YYYYMMDD_WELL_eNN
        - snip_id: YYYYMMDD_WELL_eNN_FRAME
        """
        patterns = {
            "experiment": r'^\d{8}$',
            "video": r'^\d{8}_[A-H]\d{2}$',
            "image": r'^\d{8}_[A-H]\d{2}_\d{4}$',
            "embryo": r'^\d{8}_[A-H]\d{2}_e\d{2}$',
            "snip": r'^\d{8}_[A-H]\d{2}_e\d{2}_\d{4}$'
        }
        
        import re
        pattern = patterns.get(id_type)
        if not pattern:
            raise ValidationError(f"Unknown ID type: {id_type}")
        
        if not re.match(pattern, id_value):
            raise ValidationError(
                f"Invalid {id_type} ID format: '{id_value}'. "
                f"Expected pattern: {pattern}"
            )
```

### Batch Validation

```python
class BatchValidator:
    """Validation for batch operations."""
    
    @staticmethod
    def validate_range_syntax(range_str: str, max_value: int) -> List[int]:
        """
        Validate and parse range syntax.
        
        Supported formats:
        - "[5]" -> [5]
        - "[5:10]" -> [5, 6, 7, 8, 9]
        - "[5::]" -> [5, 6, ..., max_value-1]
        - "[::3]" -> [0, 3, 6, 9, ...]
        - "[5:10:2]" -> [5, 7, 9]
        """
        import re
        
        # Remove brackets
        range_str = range_str.strip("[]")
        
        # Single value
        if ":" not in range_str:
            try:
                value = int(range_str)
                if 0 <= value < max_value:
                    return [value]
                else:
                    raise ValidationError(f"Index {value} out of range [0, {max_value})")
            except ValueError:
                raise ValidationError(f"Invalid range syntax: {range_str}")
        
        # Range syntax
        parts = range_str.split(":")
        if len(parts) > 3:
            raise ValidationError(f"Invalid range syntax: too many colons")
        
        # Parse start, stop, step
        start = 0 if not parts[0] else int(parts[0])
        stop = max_value if len(parts) < 2 or not parts[1] else int(parts[1])
        step = 1 if len(parts) < 3 or not parts[2] else int(parts[2])
        
        # Validate values
        if step == 0:
            raise ValidationError("Step cannot be zero")
        
        # Generate range
        indices = list(range(start, stop, step))
        
        # Filter valid indices
        valid_indices = [i for i in indices if 0 <= i < max_value]
        
        if not valid_indices:
            raise ValidationError(f"Range produces no valid indices")
        
        return valid_indices
    
    @staticmethod
    def validate_batch_data(batch_data: Dict, 
                           expected_type: type,
                           permitted_values: Optional[Dict] = None) -> None:
        """
        Validate batch data structure.
        
        Expected format:
        {
            "id1": value_or_list,
            "id2": value_or_list,
            ...
        }
        """
        if not isinstance(batch_data, dict):
            raise ValidationError("Batch data must be a dictionary")
        
        for key, value in batch_data.items():
            # Validate key format (should be an ID)
            if not isinstance(key, str):
                raise ValidationError(f"Key must be string, got {type(key)}")
            
            # Validate value
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, expected_type):
                        raise ValidationError(
                            f"List item must be {expected_type}, got {type(item)}"
                        )
            else:
                if not isinstance(value, expected_type):
                    raise ValidationError(
                        f"Value must be {expected_type} or list, got {type(value)}"
                    )
```

## Serialization Helpers

```python
class Serializer:
    """Serialization utilities for data models."""
    
    @staticmethod
    def serialize_annotation(annotation: AnnotationBase) -> Dict:
        """Serialize annotation to JSON-compatible dict."""
        return annotation.to_dict()
    
    @staticmethod
    def deserialize_phenotype(data: Dict) -> Phenotype:
        """Deserialize phenotype from dict."""
        return Phenotype(
            value=data["value"],
            author=data["author"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            notes=data.get("notes"),
            confidence=data.get("confidence")
        )
    
    @staticmethod
    def deserialize_genotype(data: Dict) -> Genotype:
        """Deserialize genotype from dict."""
        return Genotype(
            value=data["value"],
            author=data["author"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            notes=data.get("notes"),
            confirmed=data.get("confirmed", False),
            method=data.get("method")
        )
    
    @staticmethod
    def deserialize_flag(data: Dict) -> Flag:
        """Deserialize flag from dict."""
        return Flag(
            value=data["value"],
            author=data["author"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            notes=data.get("notes"),
            severity=data.get("severity", "warning"),
            auto_generated=data.get("auto_generated", False)
        )
```

## Custom Exceptions

```python
class ValidationError(Exception):
    """Raised when validation fails."""
    pass

class PermittedValueError(ValidationError):
    """Raised when value not in permitted list."""
    pass

class ExclusivityError(ValidationError):
    """Raised when exclusivity rules violated."""
    pass

class OverwriteProtectionError(ValidationError):
    """Raised when attempting unauthorized overwrite."""
    pass
```

## Usage Examples

```python
# Example: Validate phenotype with exclusivity
validator = Validator()
existing_phenotypes = ["EDEMA", "BODY_AXIS"]

# This should fail
try:
    validator.validate_phenotype("DEAD", permitted_values, existing_phenotypes)
except ValidationError as e:
    print(f"Validation failed: {e}")

# Example: Parse range syntax
range_validator = BatchValidator()
indices = range_validator.validate_range_syntax("[10::]", max_value=100)
# Returns: [10, 11, 12, ..., 99]

# Example: Serialize/deserialize
phenotype = Phenotype(
    value="EDEMA",
    author="researcher1",
    confidence=0.95
)
serialized = Serializer.serialize_annotation(phenotype)
deserialized = Serializer.deserialize_phenotype(serialized)
```