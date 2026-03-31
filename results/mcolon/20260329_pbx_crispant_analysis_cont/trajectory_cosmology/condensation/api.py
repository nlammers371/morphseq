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

from .engine.run import run_dynamics
from .state import CondensationConfig, CondensationResult
from .engine.stopping import StoppingConfig


def run_condensation(
    x0: np.ndarray,
    mask: np.ndarray,
    config: CondensationConfig | None = None,
    stopping: StoppingConfig | None = None,
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
        Dynamical system hyperparameters. Uses defaults if None.
    stopping : StoppingConfig, optional
        Multi-metric stopping heuristics. Uses defaults if None.
        To disable a criterion, set its threshold to None, e.g.:
          StoppingConfig(disp_max_rel_threshold=None)
    log_every : int
        Print diagnostics every N iterations.
    save_every : int or None
        Save a position snapshot every N iterations for animation.
        Required to produce animation with animation.animate_iterations().
        None disables snapshot saving (default, saves memory).
    verbose : bool

    Returns
    -------
    CondensationResult
        result.metrics_history — full per-iteration diagnostic log (list of dicts)
        result.position_history — (n_saved, N_e, T, 2) if save_every set, else None
    """
    if config is None:
        config = CondensationConfig()
    if stopping is None:
        stopping = StoppingConfig()

    return run_dynamics(
        x0=x0, mask=mask, config=config, stopping=stopping,
        log_every=log_every, save_every=save_every, verbose=verbose,
    )
