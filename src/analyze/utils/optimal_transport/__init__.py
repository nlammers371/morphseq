"""Reusable optimal transport utilities for mask-based analyses."""

from analyze.utils.optimal_transport.config import (
    UOTConfig,
    UOTFrame,
    UOTFramePair,
    UOTSupport,
    UOTProblem,
    SamplingMode,
    MassMode,
    Coupling,
)
from analyze.utils.optimal_transport.results import UOTResultWork, UOTResultCanonical
from analyze.utils.optimal_transport.solve import run_uot_on_working_grid, lift_work_result_to_canonical
from analyze.utils.optimal_transport.working_grid import (
    OutputFrame,
    WorkingGridConfig,
    WorkingGridPair,
    prepare_working_grid_pair,
)
from analyze.utils.optimal_transport.backends.base import UOTBackend, BackendResult
from analyze.utils.optimal_transport.density_transforms import (
    mask_to_density,
    mask_to_density_uniform,
    mask_to_density_boundary_band,
    mask_to_density_distance_transform,
    enforce_min_mass,
)
from analyze.utils.optimal_transport.multiscale_sampling import (
    pad_to_divisible,
    downsample_density,
    build_support,
)
from analyze.utils.optimal_transport.transport_maps import compute_transport_maps, compute_cost_maps
from analyze.utils.optimal_transport.metrics import summarize_metrics, compute_transport_metrics
from analyze.utils.optimal_transport.batch import BatchItem, solve_working_grid_batch

__all__ = [
    # Config and data structures
    "UOTConfig",
    "UOTFrame",
    "UOTFramePair",
    "UOTSupport",
    "UOTProblem",
    "UOTResultWork",
    "UOTResultCanonical",
    "SamplingMode",
    "MassMode",
    "Coupling",
    # Working-grid seam + solve
    "OutputFrame",
    "WorkingGridConfig",
    "WorkingGridPair",
    "prepare_working_grid_pair",
    "run_uot_on_working_grid",
    "lift_work_result_to_canonical",
    "BatchItem",
    "solve_working_grid_batch",
    # Backends
    "UOTBackend",
    "BackendResult",
    "POTBackend",
    "OTTBackend",
    # Density transforms
    "mask_to_density",
    "mask_to_density_uniform",
    "mask_to_density_boundary_band",
    "mask_to_density_distance_transform",
    "enforce_min_mass",
    # Multiscale and sampling
    "pad_to_divisible",
    "downsample_density",
    "build_support",
    # Transport maps
    "compute_transport_maps",
    "compute_cost_maps",
    # Metrics
    "summarize_metrics",
    "compute_transport_metrics",
]


def __getattr__(name: str):
    if name == "POTBackend":
        from analyze.utils.optimal_transport.backends.pot_backend import POTBackend as _POTBackend

        return _POTBackend
    if name == "OTTBackend":
        try:
            from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend as _OTTBackend
        except ImportError:
            return None
        return _OTTBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
