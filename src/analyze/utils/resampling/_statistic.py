"""
Statistic wrapper for resampling.

Wraps a callable scorer function with metadata (name, default tail, etc.)
so the engine and aggregator can make informed decisions.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np


@dataclass(frozen=True)
class Statistic:
    """Named wrapper around a scorer function.

    Parameters
    ----------
    name : str
        Human-readable name (e.g., ``"energy_distance"``).
    fn : callable
        Signature: ``fn(data: dict, rng: np.random.Generator | None) -> scalar | array | dict``.
    description : str, optional
        Longer description for reports / diagnostics.
    outputs : list of str, optional
        Names of dict keys returned (for multi-output scorers).
    default_alternative : str, optional
        Default tail for permutation p-values: ``"greater"`` | ``"less"`` | ``"two-sided"``.
    is_nonnegative : bool, optional
        If True, the statistic is >= 0 by construction (e.g., distances).
        Used to infer ``default_alternative="greater"`` when none is specified.
    """
    name: str
    fn: Callable[[dict, Optional[np.random.Generator]], Any]
    description: Optional[str] = None
    outputs: Optional[List[str]] = None
    default_alternative: Optional[str] = None
    is_nonnegative: Optional[bool] = None
