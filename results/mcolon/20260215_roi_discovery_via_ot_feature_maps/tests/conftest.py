"""
Shared fixtures for ROI discovery test suite.

All fixtures use tiny grids (16x16 or 32x32) with 1-2 channels so tests
run fast on CPU without real data.

Biological prior: cep290 signal is planted in the BOTTOM rows ("tail")
of the grid. WT samples have uniform/random noise everywhere.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the parent ROI discovery module importable
ROI_DIR = Path(__file__).resolve().parent.parent
if str(ROI_DIR) not in sys.path:
    sys.path.insert(0, str(ROI_DIR))


# ---------------------------------------------------------------------------
# Grid / mask helpers
# ---------------------------------------------------------------------------

GRID_H, GRID_W = 32, 32
N_CHANNELS = 1
N_WT = 10
N_MUT = 10
N_TOTAL = N_WT + N_MUT
TAIL_START_ROW = 22  # bottom ~30% of the 32-row grid = "tail"


@pytest.fixture
def mask_ref():
    """A simple elliptical embryo mask on a 32x32 grid."""
    m = np.zeros((GRID_H, GRID_W), dtype=bool)
    cy, cx = GRID_H // 2, GRID_W // 2
    ry, rx = GRID_H // 2 - 2, GRID_W // 2 - 2
    for i in range(GRID_H):
        for j in range(GRID_W):
            if ((i - cy) / ry) ** 2 + ((j - cx) / rx) ** 2 <= 1.0:
                m[i, j] = True
    return m


@pytest.fixture
def tail_roi_mask(mask_ref):
    """Ground-truth ROI: bottom rows inside the embryo mask (tail region)."""
    roi = np.zeros_like(mask_ref)
    roi[TAIL_START_ROW:, :] = True
    roi = roi & mask_ref
    return roi


@pytest.fixture
def planted_data(mask_ref, tail_roi_mask):
    """
    Synthetic dataset with signal planted in the tail.

    WT (y=0): uniform low noise everywhere.
    MUT (y=1): same low noise + a positive bump in the tail ROI.

    This mimics the cep290 phenotype where OT cost maps show
    elevated signal in the tail.
    """
    rng = np.random.default_rng(42)

    X = rng.normal(0.0, 0.1, size=(N_TOTAL, GRID_H, GRID_W, N_CHANNELS)).astype(np.float32)
    y = np.array([0] * N_WT + [1] * N_MUT, dtype=np.int32)
    groups = np.arange(N_TOTAL)  # each sample is its own embryo

    # Plant signal in tail for mutants only
    signal_strength = 10.0
    for i in range(N_WT, N_TOTAL):
        X[i][tail_roi_mask, :] += signal_strength

    # Zero out anything outside the embryo mask
    X[:, ~mask_ref, :] = 0.0

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "mask_ref": mask_ref,
        "tail_roi": tail_roi_mask,
        "channel_names": ("total_cost",),
    }


@pytest.fixture
def class_weights():
    """Balanced class weights for equal-sized WT/MUT groups."""
    return {0: 1.0, 1: 1.0}


@pytest.fixture
def tiny_trainer_config():
    """Fast trainer config for tests: small grid, few steps."""
    from roi_config import TrainerConfig
    return TrainerConfig(
        learn_res=16,
        output_res=GRID_H,
        learning_rate=5e-2,
        max_steps=1000,
        convergence_tol=1e-8,
        log_every=10,
        random_seed=42,
    )
