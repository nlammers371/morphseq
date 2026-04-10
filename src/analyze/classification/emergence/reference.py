"""Reference validation for the emergence pipeline."""

from __future__ import annotations

from itertools import combinations
from typing import Literal, Sequence

import math
import pandas as pd

from .types import ReferenceValidation
from .utils import symmetric_onset


def validate_reference(
    onset_matrix: pd.DataFrame,
    reference: Sequence[str],
) -> ReferenceValidation:
    """Check internal coherence of the reference set."""

    ref = list(reference)
    pairs = list(combinations(ref, 2))
    n_total = len(pairs)

    if n_total == 0:
        return ReferenceValidation(
            reference=ref,
            status="valid",
            coherence_score=1.0,
            offending_pairs=[],
            n_internal_pairs=0,
        )

    offending: list[tuple[str, str, float]] = []
    for a, b in pairs:
        v = symmetric_onset(onset_matrix, a, b)
        if math.isfinite(v):
            offending.append((a, b, v))

    n_nan = n_total - len(offending)
    coherence = n_nan / n_total

    if len(offending) == 0:
        status: Literal["valid", "ambiguous", "invalid"] = "valid"
    elif coherence >= 0.5:
        status = "ambiguous"
    else:
        status = "invalid"

    return ReferenceValidation(
        reference=ref,
        status=status,
        coherence_score=coherence,
        offending_pairs=offending,
        n_internal_pairs=n_total,
    )
