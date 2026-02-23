"""
MorphSeq • Base Annotation Models (v0.2‑rebased)
=================================================
should be called embryo_annotation_fields.py
=================================================
This **single file** serves as the *minimal reference implementation* for
annotation dataclasses used throughout the MorphSeq pipeline.  It is deliberately
kept **small & intuitive** so downstream modules (e.g. Module 0.1 ID parsing,


> **NOTE** – All validators accept an optional `permitted_values` arg.  In the
> upcoming `PermittedValuesManager`, this will be wired automatically.  Until
> that manager lands you can simply pass `None` or a stub dict in tests.
>
> This file should be viewed as a *starting point* that will evolve together
> with the ExperimentMetadata class and the *_manager.py* helper scripts.  When
> you add new annotation kinds, keep the philosophy: **one dataclass ≈ one JSON
> blob; round‑tripable with `.to_dict()` and `deserialize_annotation()`.**
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when a general validation rule fails."""

class PermittedValueError(ValidationError):
    """Raised when a value is not in the permitted list."""

class ExclusivityError(ValidationError):
    """Raised when mutually‑exclusive phenotype rules are violated."""

class OverwriteProtectionError(ValidationError):
    """Raised when attempting to overwrite a protected field without opt‑in."""


# ──────────────────────────────────────────────────────────────────────────────
# Core helper – AnnotationBase
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class AnnotationBase:
    """Fields common to **all** annotations (value / author / timestamp / notes)."""

    value: str
    author: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    notes: Optional[str] = None

    # Round‑trip helpers -------------------------------------------------------
    def to_dict(self) -> Dict[str, Union[str, float, bool]]:
        """Serialize to a flat, JSON‑friendly dict."""
        d: Dict[str, Union[str, float, bool]] = {
            "kind": self.__class__.__name__.lower(),  # e.g. "phenotype"
            "value": self.value,
            "author": self.author,
            "timestamp": self.timestamp,
        }
        if self.notes:
            d["notes"] = self.notes
        return d

    # Generic validation -------------------------------------------------------
    def validate(self, permitted_values: Optional[List[str]] = None):  # noqa: D401
        """Validate presence of author & optional membership in *permitted_values*."""
        if not self.author:
            raise ValidationError("Author is required")
        if permitted_values is not None and self.value not in permitted_values:
            raise PermittedValueError(
                f"Value '{self.value}' not in permitted values: {permitted_values}"
            )
        return self  # fluent style


# ──────────────────────────────────────────────────────────────────────────────
# Phenotype – snip‑level or embryo‑level annotation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Phenotype(AnnotationBase):
    """Embryo phenotype call plus optional confidence (0–1)."""

    confidence: Optional[float] = None

    # ---------- serialization ----------
    def to_dict(self) -> Dict[str, Union[str, float, bool]]:  # type: ignore[override]
        d = super().to_dict()
        if self.confidence is not None:
            d["confidence"] = self.confidence
        return d

    # ---------- validation -------------
    def validate(
        self,
        permitted_values: Optional[Dict[str, Dict]] = None,
        *,
        existing: Optional[List[str]] = None,
    ):
        # Validate value & author via base class (only keys list needed)
        super().validate(permitted_values=list(permitted_values) if permitted_values else None)

        # Confidence range check
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValidationError("Confidence must be between 0 and 1")

        # Exclusive / terminal logic
        if permitted_values and existing:
            meta = permitted_values.get(self.value, {})
            if meta.get("exclusive") and any(p != "NONE" for p in existing):
                raise ExclusivityError("Exclusive phenotype cannot coexist with others")
            if "DEAD" in existing and self.value not in {"NONE", "DEAD"}:
                raise ExclusivityError("Cannot add phenotypes after DEAD")
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Genotype – embryo‑level annotation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Genotype(AnnotationBase):
    """Genotype call with allele, zygosity, method, etc."""

    allele: Optional[str] = None
    zygosity: Optional[str] = None  # homozygous / heterozygous / crispant
    confidence: float = 1.0
    confirmed: bool = False
    method: Optional[str] = None  # sequencing / PCR
    gtype: Optional[str] = None   # additional *type* requested by user

    # ---------- serialization ----------
    def to_dict(self) -> Dict[str, Union[str, float, bool]]:  # type: ignore[override]
        d = super().to_dict()
        if self.allele is not None:
            d["allele"] = self.allele
        if self.zygosity is not None:
            d["zygosity"] = self.zygosity
        d["confidence"] = self.confidence
        d["confirmed"] = self.confirmed
        if self.method is not None:
            d["method"] = self.method
        if self.gtype is not None:
            d["type"] = self.gtype
        return d

    # ---------- validation -------------
    def validate(self, *, overwrite: bool = False, existing: Optional["Genotype"] = None):
        super().validate()
        if existing and not overwrite:
            # Compare *stable* fields only (ignore timestamp & author)
            def stable(g: "Genotype") -> Dict[str, Union[str, float, bool]]:
                d = g.to_dict().copy()
                d.pop("timestamp", None)
                d.pop("author", None)
                return d
            if stable(existing) != stable(self):
                raise OverwriteProtectionError(
                    "Genotype already set; pass overwrite=True to change it."
                )
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Treatment – embryo‑ or experiment‑level annotation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Treatment(AnnotationBase):
    details: Optional[str] = None  # e.g. concentration / duration

    def to_dict(self) -> Dict[str, Union[str, float, bool]]:  # type: ignore[override]
        d = super().to_dict()
        if self.details is not None:
            d["details"] = self.details
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Flag – quality‑control indicator (snip / image / video / experiment levels)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Flag(AnnotationBase):
    flag_type: str = "quality"   # arbitrary category string
    priority: str = "medium"     # free‑form, but should match severity list later
    confidence: float = 1.0
    auto_generated: bool = False

    def to_dict(self) -> Dict[str, Union[str, float, bool]]:  # type: ignore[override]
        d = super().to_dict()
        d.update(
            {
                "flag_type": self.flag_type,
                "priority": self.priority,
                "confidence": self.confidence,
                "auto_generated": self.auto_generated,
            }
        )
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory – deserialize_annotation
# ──────────────────────────────────────────────────────────────────────────────

def deserialize_annotation(data: Dict[str, Union[str, float, bool]]) -> AnnotationBase:
    """Return the correct *Annotation* subclass given a previously serialized dict."""
    kind = (data.get("kind") or data.get("_kind") or "").lower()
    if kind == "phenotype":
        return Phenotype(
            value=data["value"],
            author=data["author"],
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            notes=data.get("notes"),
            confidence=data.get("confidence"),
        )
    if kind == "genotype":
        return Genotype(
            value=data["value"],
            author=data["author"],
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            notes=data.get("notes"),
            allele=data.get("allele"),
            zygosity=data.get("zygosity"),
            confidence=data.get("confidence", 1.0),
            confirmed=data.get("confirmed", False),
            method=data.get("method"),
            gtype=data.get("type"),
        )
    if kind == "treatment":
        return Treatment(
            value=data["value"],
            author=data["author"],
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            notes=data.get("notes"),
            details=data.get("details"),
        )
    if kind == "flag":
        return Flag(
            value=data["value"],
            author=data["author"],
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            notes=data.get("notes"),
            flag_type=data.get("flag_type", "quality"),
            priority=data.get("priority", "medium"),
            confidence=data.get("confidence", 1.0),
            auto_generated=data.get("auto_generated", False),
        )
    raise ValidationError("Cannot infer annotation kind from data dictionary")


# ──────────────────────────────────────────────────────────────────────────────
# © 2025 MorphSeq Development Team – Simple, intuitive, and easy to evolve.     
# ──────────────────────────────────────────────────────────────────────────────
