"""Group-aware resampling helpers shared by ROI bootstrap routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class BootstrapGroupSample:
    """One bootstrap replicate at embryo/group granularity."""

    inbag_group_ids: np.ndarray
    oob_group_ids: np.ndarray
    oob_is_empty: bool
    oob_single_class: bool


def iter_bootstrap_groups(
    groups: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    stratify: bool = True,
    random_seed: int = 42,
) -> Iterator[BootstrapGroupSample]:
    """Yield in-bag and out-of-bag group IDs for each bootstrap replicate."""
    unique_groups = np.unique(groups)
    if unique_groups.size == 0:
        return

    rng = np.random.default_rng(random_seed)

    group_to_label = {}
    for group_id in unique_groups:
        labels = np.unique(y[groups == group_id])
        if labels.size != 1:
            raise ValueError(
                f"Group '{group_id}' has mixed labels {labels}; expected one label per group."
            )
        group_to_label[group_id] = int(labels[0])

    if stratify:
        class0 = np.array([g for g in unique_groups if group_to_label[g] == 0])
        class1 = np.array([g for g in unique_groups if group_to_label[g] == 1])

    for _ in range(n_boot):
        if stratify:
            inbag0 = rng.choice(class0, size=len(class0), replace=True)
            inbag1 = rng.choice(class1, size=len(class1), replace=True)
            inbag = np.concatenate([inbag0, inbag1])
        else:
            inbag = rng.choice(unique_groups, size=len(unique_groups), replace=True)

        inbag_unique = np.unique(inbag)
        oob = np.array([g for g in unique_groups if g not in inbag_unique])

        oob_is_empty = oob.size == 0
        oob_single_class = False
        if not oob_is_empty:
            oob_labels = np.array([group_to_label[g] for g in oob])
            oob_single_class = np.unique(oob_labels).size < 2

        yield BootstrapGroupSample(
            inbag_group_ids=inbag,
            oob_group_ids=oob,
            oob_is_empty=oob_is_empty,
            oob_single_class=oob_single_class,
        )


__all__ = ["BootstrapGroupSample", "iter_bootstrap_groups"]
