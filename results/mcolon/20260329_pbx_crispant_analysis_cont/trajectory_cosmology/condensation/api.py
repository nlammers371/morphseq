"""
api.py
------
Public entry point for the condensation engine.

External callers only need:
    from trajectory_cosmology.condensation import api
    result = api.run_condensation(x0, mask, config)
"""
from __future__ import annotations

import numpy as np

from .dynamics import run_dynamics
from .state import CondensationConfig, CondensationResult


def run_condensation(
    x0: np.ndarray,
    mask: np.ndarray,
    config: CondensationConfig | None = None,
    log_every: int = 10,
    save_every: int | None = None,
    verbose: bool = True,
) -> CondensationResult:
    """Run trajectory condensation.

    Parameters
    ----------
    x0 : (N_e, T, 2)
        Initial 2D positions from init_embedding.
    mask : (N_e, T) bool
        True where embryo i is observed at time t.
    config : CondensationConfig, optional
        Hyperparameters. Uses defaults if None.
    log_every : int
        Print loss every N iterations.
    save_every : int or None
        Save a position snapshot every N iterations.
        Required to produce animation with animation.animate_iterations().
        None disables snapshot saving (default, saves memory).
    verbose : bool

    Returns
    -------
    CondensationResult
        result.position_history is (n_saved, N_e, T, 2) if save_every was set,
        else None.
    """
    if config is None:
        config = CondensationConfig()

    return run_dynamics(
        x0=x0, mask=mask, config=config,
        log_every=log_every, save_every=save_every, verbose=verbose,
    )
