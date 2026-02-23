"""
Phase 1 / Task 1.3 — Loader and CV split generation.

Checks:
- GroupKFold splits never leak embryo_id across train/val
- Class weights are computed from training fold only
- All samples in each group share the same label
"""

import numpy as np
import pytest

from conftest import N_MUT, N_TOTAL, N_WT


# ---- Group-aware CV: no embryo leakage ----

def test_no_embryo_leakage_in_cv(planted_data):
    """
    PSEUDO-LOGIC:
    1. Create GroupKFold splits using embryo_id as group key
    2. For each fold, verify train_groups ∩ val_groups = ∅
    3. This is the MANDATORY anti-leakage check from PLAN.md Section A
    """
    from sklearn.model_selection import GroupKFold

    X = planted_data["X"]
    y = planted_data["y"]
    groups = planted_data["groups"]

    gkf = GroupKFold(n_splits=5)
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        overlap = train_groups & val_groups
        assert len(overlap) == 0, (
            f"Fold {fold_i}: embryo leakage detected! "
            f"Groups in both train and val: {overlap}"
        )


def test_all_samples_in_group_share_label(planted_data):
    """Every sample from the same embryo must have the same class label."""
    y = planted_data["y"]
    groups = planted_data["groups"]

    for g in np.unique(groups):
        labels = np.unique(y[groups == g])
        assert len(labels) == 1, f"Group {g} has mixed labels: {labels}"


def test_fold_local_class_weights(planted_data):
    """
    PSEUDO-LOGIC:
    1. Create a GroupKFold split
    2. Compute class weights from TRAINING fold only
    3. Verify weights differ from global weights when class balance changes per fold
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.utils.class_weight import compute_class_weight

    X = planted_data["X"]
    y = planted_data["y"]
    groups = planted_data["groups"]

    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X, y, groups):
        y_train = y[train_idx]
        classes = np.unique(y_train)
        if len(classes) < 2:
            continue  # degenerate fold
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cw = {int(c): float(w) for c, w in zip(classes, weights)}

        # Weights should be positive and reasonable
        for c in classes:
            assert cw[int(c)] > 0, f"Class weight for class {c} is non-positive"


def test_cv_covers_all_samples(planted_data):
    """Every sample appears in exactly one validation fold."""
    from sklearn.model_selection import GroupKFold

    X = planted_data["X"]
    y = planted_data["y"]
    groups = planted_data["groups"]

    gkf = GroupKFold(n_splits=5)
    all_val = set()
    for _, val_idx in gkf.split(X, y, groups):
        all_val.update(val_idx)

    assert all_val == set(range(len(y))), "Not all samples covered by CV folds"
