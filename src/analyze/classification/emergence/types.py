"""Public dataclasses for the emergence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ReferenceValidation:
    """Result of Step 1: reference coherence check."""

    reference: list[str]
    status: Literal["valid", "ambiguous", "invalid"]
    coherence_score: float
    offending_pairs: list[tuple[str, str, float]]
    n_internal_pairs: int


@dataclass
class EmergenceScore:
    """Per-class emergence timing relative to the reference set."""

    class_name: str
    emergence_time: float
    emergence_min: float
    emergence_max: float
    n_resolved_refs: int
    n_total_refs: int
    per_ref_onsets: dict[str, float | None]


@dataclass
class EmergenceBlock:
    """A group of non-reference classes that emerge at the same time bin."""

    block_id: int
    members: list[str]
    bin_key: float
    emergence_time: float
    emergence_min: float
    emergence_max: float


@dataclass
class ResolutionNode:
    """Recursive tree node for within-block partitioning."""

    members: list[str]
    split_time: float | None
    children: list["ResolutionNode"]
    unresolved: bool


@dataclass
class EmergenceTimeline:
    """Full emergence timeline for a set of classes given a reference."""

    reference_validation: ReferenceValidation
    scores: list[EmergenceScore]
    blocks: list[EmergenceBlock]
    block_resolutions: dict[int, ResolutionNode]
    all_classes: list[str]
    reference: list[str]


@dataclass
class TransitivityViolation:
    """A single non-transitive triple at a time bin."""

    time_bin: float
    a: str
    b: str
    c: str
    state_ab: str
    state_bc: str
    state_ac: str
    violation_type: str
    auroc_ab: float
    auroc_bc: float
    auroc_ac: float
    pval_ab: float
    pval_bc: float
    pval_ac: float


@dataclass
class OnsetConsistencySummary:
    """Summary statistics for ultrametric gap diagnostic."""

    n_triples_total: int
    n_triples_evaluable: int
    n_gap_zero: int
    frac_gap_zero: float
    mean_gap: float
    median_gap: float
    max_gap: float


@dataclass
class TransitivityReport:
    params: object
    classified_df: object
    onset_df: object
    onset_matrix: object
    timebin_summary: object
    triple_violations: object
    onset_summary: OnsetConsistencySummary
    onset_triple_df: object
