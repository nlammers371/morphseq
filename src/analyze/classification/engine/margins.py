"""Signed margin utilities for binary classification predictions.

Canonical range is [-1, 1]:

    class_signed_margin  — sign determined by the positive class axis.
                           +1 = strongest support for positive_label.
                           -1 = strongest support for negative_label.
                           Formula: 2 * p_pos - 1

    truth_signed_margin  — sign determined by the true label.
                           +1 = most correct.
                           -1 = most wrong.
                           Formula: class_signed_margin * (+1 if y_true==1 else -1)

Legacy note: older saved files store margins in [-0.5, 0.5] (= p_pos - 0.5).
Use coerce_margin_range() when loading from disk to normalise to [-1, 1].
"""

from __future__ import annotations

import numpy as np


def class_signed_margin(p_pos: float | np.ndarray) -> float | np.ndarray:
    """Signed margin relative to the positive-class axis.

    Sign is determined by which side of the decision boundary the prediction
    falls on, regardless of the true label:
      +1  →  strongest support for positive_label
       0  →  decision boundary
      -1  →  strongest support for negative_label

    Range: [-1, 1].
    """
    return 2.0 * np.asarray(p_pos, dtype=float) - 1.0


def truth_signed_margin(
    p_pos: float | np.ndarray,
    y_true: int | np.ndarray,
) -> float | np.ndarray:
    """Signed margin relative to the true label.

    Sign is determined by whether the prediction agrees with the true label:
      +1  →  most correct (strongest support for the right class)
       0  →  decision boundary
      -1  →  most wrong (strongest support for the wrong class)

    Computed as class_signed_margin flipped for true negatives (y_true == 0).

    Range: [-1, 1].
    """
    csm = class_signed_margin(p_pos)
    y_true = np.asarray(y_true, dtype=int)
    return np.where(y_true == 1, csm, -csm)


_LEGACY_THRESHOLD = 0.5 + 1e-6


def coerce_margin_range(margin: np.ndarray) -> np.ndarray:
    """Normalise a signed margin array to the canonical [-1, 1] range.

    If the array looks like it was stored in the legacy [-0.5, 0.5] convention
    (i.e. max(|margin|) <= 0.5 + eps), it is rescaled by 2. Otherwise it is
    returned unchanged.

    All loading and plotting code should pass margins through this function
    rather than detecting the convention inline.
    """
    margin = np.asarray(margin, dtype=float)
    if margin.size == 0:
        return margin
    if np.nanmax(np.abs(margin)) <= _LEGACY_THRESHOLD:
        return margin * 2.0
    return margin
