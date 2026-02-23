"""
Resampling specification dataclasses.

Defines the configuration objects that describe *what* kind of resampling
to perform: index-based (bootstrap/subsample), label permutation, or
group permutation.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class IndicesParams:
    """Parameters for index-based resampling (bootstrap or subsample).

    Exactly one of ``size`` / ``frac`` may be set, or neither (defaults to N).
    """
    replacement: bool
    size: Optional[int] = None
    frac: Optional[float] = None

    def __post_init__(self):
        if self.size is not None and self.frac is not None:
            raise ValueError("Specify at most one of size or frac, not both.")
        if self.size is not None and self.size < 1:
            raise ValueError(f"size must be >= 1, got {self.size}")
        if self.frac is not None and not (0.0 < self.frac <= 1.0):
            raise ValueError(f"frac must be in (0, 1], got {self.frac}")


@dataclass(frozen=True)
class LabelsParams:
    """Parameters for label permutation. Labels are read from data['labels']."""
    pass


@dataclass(frozen=True)
class GroupsParams:
    """Parameters for group (pool-and-redistribute) permutation.

    PR1: ``within_key`` and ``unit_key`` are NOT supported for groups.
    Preflight rejects both.
    """
    a_key: str = "X1"
    b_key: str = "X2"


@dataclass(frozen=True)
class ResampleSpec:
    """Full specification for a resampling procedure.

    Parameters
    ----------
    kind : {"indices", "labels", "groups"}
        The resampling strategy.
    params : IndicesParams | LabelsParams | GroupsParams
        Kind-specific parameters.
    unit_key : str, optional
        Perturb at unit level (e.g., ``"embryo_id"``). Each unit's rows
        are kept together. Not valid for ``kind="groups"`` in PR1.
    within_key : str, optional
        Stratify perturbations within groups (e.g., ``"time_bin"``).
        Not valid for ``kind="groups"`` in PR1.
    """
    kind: Literal["indices", "labels", "groups"]
    params: Union[IndicesParams, LabelsParams, GroupsParams]
    unit_key: Optional[str] = None
    within_key: Optional[str] = None
