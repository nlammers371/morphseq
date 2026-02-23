"""
Visualization Utilities

Generic visualization utilities for data analysis. These functions are
domain-agnostic and can be used across different analysis contexts.

Subpackages
===========
- plotting : Time series plotting utilities

For domain-specific trajectory visualizations (genotype styling, phenotype colors),
see: src.analyze.trajectory_analysis.viz
"""

import importlib
from .hpf_coverage import (
    plot_experiment_time_coverage,
    experiment_hpf_coverage,
    longest_interval_where,
    plot_hpf_overlap_quick,
)

__all__ = [
    'plotting',
    'plot_experiment_time_coverage',
    'experiment_hpf_coverage',
    'longest_interval_where',
    'plot_hpf_overlap_quick',
]


def __getattr__(name: str):
    if name == "plotting":
        try:
            module = importlib.import_module(f"{__name__}.plotting")
        except Exception as exc:
            raise ImportError(
                "Optional plotting utilities require extra dependencies (e.g., seaborn). "
                "Install the viz extras or import specific modules that avoid seaborn."
            ) from exc
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
