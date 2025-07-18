"""
Embryo Metadata Data Models and Validation Implementation
Auto-generated from module_2_data_models_and_validation.md
"""

from typing import TypedDict, Optional, Dict, List, Union
from datetime import datetime
from dataclasses import dataclass, field
import re

# Custom exceptions
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

# Type aliases (dynamic strings instead of rigid Literal types)
PhenotypeValue = str
FlagLevel = str
GenotypeValue = str
TreatmentValue = str  # User-defined values for treatments

@dataclass
class AnnotationBase:
    """Base class for all annotations with common fields."""
    value: str
    author: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        data = {
            "value": self.value,
            "author": self.author,
            "timestamp": self.timestamp
        }
        if self.notes:
            data["notes"] = self.notes
        return data

    def validate(self, permitted_values: Optional[List[str]] = None):
        if not self.author:
            raise ValidationError("Author is required")
        if permitted_values and self.value not in permitted_values:
            raise PermittedValueError(
                f"Value '{self.value}' not in permitted values: {permitted_values}"
            )

@dataclass
class Phenotype(AnnotationBase):
    """Phenotype annotation."""
    value: str
    confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        data = super().to_dict()
        if self.confidence is not None:
            data["confidence"] = self.confidence
        return data

    def validate(self, permitted_values: Optional[List[str]] = None):
        super().validate(permitted_values)
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValidationError("Confidence must be between 0 and 1")

@dataclass
class Genotype(AnnotationBase):
    """Genotype annotation with overwrite protection."""
    allele: Optional[str] = None
    zygosity: Optional[str] = None
    confidence: float = 1.0
    confirmed: bool = False
    method: Optional[str] = None

    def to_dict(self) -> Dict:
        data = super().to_dict()
        if self.allele:
            data["allele"] = self.allele
        if self.zygosity:
            data["zygosity"] = self.zygosity
        data["confidence"] = self.confidence
        data["confirmed"] = self.confirmed
        if self.method:
            data["method"] = self.method
        return data

@dataclass
class Flag(AnnotationBase):
    """Quality control flag."""
    flag_type: str = "quality"
    priority: str = "medium"
    confidence: float = 1.0
    severity: str = "warning"
    auto_generated: bool = False

    def to_dict(self) -> Dict:
        data = super().to_dict()
        data["flag_type"] = self.flag_type
        data["priority"] = self.priority
        data["confidence"] = self.confidence
        data["severity"] = self.severity
        data["auto_generated"] = self.auto_generated
        return data

@dataclass
class Treatment(AnnotationBase):
    """Treatment annotation for chemical or temperature treatments."""
    value: TreatmentValue
    details: Optional[str] = None

    def to_dict(self) -> Dict:
        data = super().to_dict()
        if self.details is not None:
            data["details"] = self.details
        return data

class PermittedValuesSchema:
    """Schema for managing permitted values - DEPRECATED: Use PermittedValuesManager instead."""

    @staticmethod
    def get_default_schema() -> Dict:
        """DEPRECATED: Use PermittedValuesManager._get_default_schema() instead."""
        # This is kept for backward compatibility
        return {
            "phenotypes": {
                "NONE": {"description": "No phenotype observed", "is_default": True, "color": "#CCCCCC"},
                "EDEMA": {"description": "Fluid accumulation/swelling", "is_default": False, "color": "#3498DB"},
                "BODY_AXIS": {"description": "Body axis formation defect", "is_default": False, "color": "#E74C3C"},
                "CONVERGENCE_EXTENSION": {"description": "Convergent extension defect", "is_default": False, "color": "#F39C12"},
                "DEAD": {"description": "Embryo death", "is_default": False, "exclusive": True, "terminal": True, "color": "#2C3E50"}
            },
            "genotypes": {},
            "treatments": {},
            "flags": {
                "snip_level": {
                    "MOTION_BLUR": {"description": "Motion blur detected in frame", "severity": "warning"},
                    "HIGHLY_VAR_MASK": {"description": "Mask area variance >10% between frames", "severity": "warning"},
                    "MASK_ON_EDGE": {"description": "Mask within 5 pixels of image edge", "severity": "info"}
                },
                "image_level": {
                    "DETECTION_FAILURE": {"description": "Embryo detection failed", "severity": "error"}
                },
                "video_level": {
                    "NONZERO_SEED_FRAME": {"description": "Seed frame is not the first frame", "severity": "info"},
                    "NO_EMBRYO": {"description": "No embryo detected in entire video", "severity": "error"}
                },
                "experiment_level": {
                    "INCOMPLETE": {"description": "Experiment data incomplete", "severity": "warning"},
                    "PROTOCOL_DEVIATION": {"description": "Deviation from standard protocol", "severity": "warning"}
                }
            }
        }

class Validator:
    """Validation utilities for EmbryoMetadata."""

    @staticmethod
    def validate_phenotype(phenotype: Union[str, Phenotype], permitted_values: Dict, existing_phenotypes: List[str] = None) -> None:
        """
        Validate phenotype against schema rules.
        
        Args:
            phenotype: Phenotype string or Phenotype object
            permitted_values: Dict from schema manager (phenotypes section)
            existing_phenotypes: List of existing phenotype values for this entity
        """
        if isinstance(phenotype, Phenotype):
            phenotype_value = phenotype.value
        else:
            phenotype_value = phenotype

        if phenotype_value not in permitted_values:
            raise PermittedValueError(
                f"Phenotype '{phenotype_value}' not permitted. Allowed: {list(permitted_values.keys())}"
            )
        
        if existing_phenotypes:
            info = permitted_values.get(phenotype_value, {})
            
            # Check exclusive phenotypes (like DEAD)
            if info.get("exclusive") and any(p != "NONE" for p in existing_phenotypes):
                raise ExclusivityError("DEAD phenotype cannot coexist with other phenotypes")
            
            # Check terminal phenotypes (can't add after DEAD)
            if "DEAD" in existing_phenotypes and phenotype_value not in ["NONE", "DEAD"]:
                raise ExclusivityError("Cannot add phenotypes after DEAD")

    @staticmethod
    def validate_genotype(genotype: Union[str, Genotype], existing_genotype: Optional[Genotype], overwrite_genotype: bool = False) -> None:
        """
        Validate genotype with overwrite protection.
        
        Args:
            genotype: Genotype string or Genotype object
            existing_genotype: Existing genotype if any
            overwrite_genotype: Whether to allow overwriting
        """
        if existing_genotype and not overwrite_genotype:
            raise OverwriteProtectionError(
                f"Genotype already set to '{existing_genotype.value}'. Use overwrite_genotype=True to change."
            )
        if isinstance(genotype, str) and not genotype.strip():
            raise ValidationError("Genotype cannot be empty")

    @staticmethod
    def validate_flag(flag: Union[str, Flag], level: FlagLevel, permitted_values: Dict) -> None:
        """
        Validate flag against permitted values for level.
        
        Args:
            flag: Flag string or Flag object
            level: Flag level (snip, image, video, experiment)
            permitted_values: Dict from schema manager (flags section)
        """
        flag_value = flag.value if isinstance(flag, Flag) else flag
        level_key = f"{level}_level"
        
        if level_key not in permitted_values:
            raise ValidationError(f"Invalid flag level: {level}")
        
        level_flags = permitted_values[level_key]
        if flag_value not in level_flags:
            raise ValidationError(
                f"Flag '{flag_value}' not permitted for {level} level. Allowed: {list(level_flags.keys())}"
            )

    @staticmethod
    def validate_treatment(treatment: Union[str, Treatment], permitted_values: Dict) -> None:
        """
        Validate treatment against permitted values.
        
        Args:
            treatment: Treatment string or Treatment object
            permitted_values: Dict from schema manager (treatments section)
        """
        treatment_value = treatment.value if isinstance(treatment, Treatment) else treatment
        
        if treatment_value not in permitted_values:
            raise PermittedValueError(
                f"Treatment '{treatment_value}' not permitted. Allowed: {list(permitted_values.keys())}"
            )

class BatchValidator:
    """Validation for batch operations."""

    @staticmethod
    def validate_range_syntax(range_str: str, max_value: int) -> List[int]:
        s = range_str.strip("[]")
        if ":" not in s:
            try:
                v = int(s)
                if 0 <= v < max_value:
                    return [v]
                raise ValidationError(f"Index {v} out of range [0, {max_value})")
            except ValueError:
                raise ValidationError(f"Invalid range syntax: {s}")
        parts = s.split(":")
        if len(parts) > 3:
            raise ValidationError("Invalid range syntax: too many colons")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else max_value
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        if step == 0:
            raise ValidationError("Step cannot be zero")
        indices = list(range(start, stop, step))
        valid = [i for i in indices if 0 <= i < max_value]
        if not valid:
            raise ValidationError(f"Range produces no valid indices")
        return valid

    @staticmethod
    def validate_batch_data(batch_data: Dict, expected_type: type, permitted_values: Optional[Dict] = None) -> None:
        if not isinstance(batch_data, dict):
            raise ValidationError("Batch data must be a dictionary")
        for key, value in batch_data.items():
            if not isinstance(key, str):
                raise ValidationError(f"Key must be string, got {type(key)}")
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

class Serializer:
    """Serialization utilities for data models."""

    @staticmethod
    def serialize_annotation(annotation: AnnotationBase) -> Dict:
        return annotation.to_dict()

    @staticmethod
    def deserialize_phenotype(data: Dict) -> Phenotype:
        return Phenotype(
            value=data["value"], author=data["author"], timestamp=data.get("timestamp", datetime.now().isoformat()),
            notes=data.get("notes"), confidence=data.get("confidence")
        )

    @staticmethod
    def deserialize_genotype(data: Dict) -> Genotype:
        return Genotype(
            value=data["value"], author=data["author"], timestamp=data.get("timestamp", datetime.now().isoformat()),
            notes=data.get("notes"), confirmed=data.get("confirmed", False), method=data.get("method")
        )

    @staticmethod
    def deserialize_flag(data: Dict) -> Flag:
        return Flag(
            value=data["value"], author=data["author"], timestamp=data.get("timestamp", datetime.now().isoformat()),
            notes=data.get("notes"), severity=data.get("severity", "warning"), auto_generated=data.get("auto_generated", False)
        )

    @staticmethod
    def deserialize_treatment(data: Dict) -> Treatment:
        return Treatment(
            value=data["value"], author=data["author"], timestamp=data.get("timestamp", datetime.now().isoformat()),
            notes=data.get("notes"), details=data.get("details")
        )
