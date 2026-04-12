"""Top-level emergence tree assembly."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .emergence import compute_emergence_scores, form_emergence_blocks
from .partition import resolve_block
from .reference import validate_reference
from .types import EmergenceTimeline


def build_emergence_timeline(
    onset_matrix: pd.DataFrame,
    reference: Sequence[str],
    *,
    bin_width: float = 4.0,
    min_cross_support: float = 0.5,
) -> EmergenceTimeline:
    """Build a reference-rooted emergence timeline."""

    ref = list(reference)
    ref_validation = validate_reference(onset_matrix, ref)
    scores = compute_emergence_scores(onset_matrix, ref)
    blocks = form_emergence_blocks(scores, bin_width=bin_width)
    block_resolutions = {}
    for block in blocks:
        block_resolutions[block.block_id] = resolve_block(
            block.members,
            onset_matrix,
            min_cross_support=min_cross_support,
        )

    return EmergenceTimeline(
        reference_validation=ref_validation,
        scores=scores,
        blocks=blocks,
        block_resolutions=block_resolutions,
        all_classes=list(onset_matrix.index),
        reference=ref,
    )
