"""Emergence scoring relative to a reference set."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from .types import EmergenceBlock, EmergenceScore
from .utils import nanmax, nanmedian, nanmin, symmetric_onset


def compute_emergence_scores(
    onset_matrix: pd.DataFrame,
    reference: Sequence[str],
) -> list[EmergenceScore]:
    """Score each non-reference class by emergence time relative to the reference."""

    ref = list(reference)
    ref_set = set(ref)
    all_classes = list(onset_matrix.index)
    non_ref = [c for c in all_classes if c not in ref_set]

    scores: list[EmergenceScore] = []
    for c in non_ref:
        per_ref: dict[str, float | None] = {}
        for r in ref:
            v = symmetric_onset(onset_matrix, c, r)
            per_ref[r] = v if math.isfinite(v) else None

        finite_onsets = [v for v in per_ref.values() if v is not None]
        scores.append(
            EmergenceScore(
                class_name=c,
                emergence_time=nanmedian(finite_onsets) if finite_onsets else float("nan"),
                emergence_min=nanmin(finite_onsets) if finite_onsets else float("nan"),
                emergence_max=nanmax(finite_onsets) if finite_onsets else float("nan"),
                n_resolved_refs=len(finite_onsets),
                n_total_refs=len(ref),
                per_ref_onsets=per_ref,
            )
        )

    scores.sort(key=lambda s: (math.isnan(s.emergence_time), s.emergence_time))
    return scores


def form_emergence_blocks(
    scores: Sequence[EmergenceScore],
    *,
    bin_width: float = 4.0,
) -> list[EmergenceBlock]:
    """Group emergence scores into time bins."""

    finite_scores = [s for s in scores if math.isfinite(s.emergence_time)]
    nan_scores = [s for s in scores if not math.isfinite(s.emergence_time)]

    bin_groups: dict[float, list[EmergenceScore]] = {}
    for s in finite_scores:
        bin_key = math.floor(s.emergence_time / bin_width) * bin_width
        bin_groups.setdefault(bin_key, []).append(s)

    blocks: list[EmergenceBlock] = []
    block_id = 0

    for bin_key in sorted(bin_groups.keys()):
        group = bin_groups[bin_key]
        raw_times = [s.emergence_time for s in group]
        blocks.append(
            EmergenceBlock(
                block_id=block_id,
                members=[s.class_name for s in group],
                bin_key=bin_key,
                emergence_time=float(np.median(raw_times)),
                emergence_min=min(raw_times),
                emergence_max=max(raw_times),
            )
        )
        block_id += 1

    if nan_scores:
        blocks.append(
            EmergenceBlock(
                block_id=block_id,
                members=[s.class_name for s in nan_scores],
                bin_key=float("nan"),
                emergence_time=float("nan"),
                emergence_min=float("nan"),
                emergence_max=float("nan"),
            )
        )

    return blocks
