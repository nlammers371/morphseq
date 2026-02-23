"""Analysis-specific code for optimal transport morphometrics on embryo masks.

This module provides embryo-specific I/O, preprocessing, visualization, and
pipeline orchestration for morphological dynamics analysis using optimal transport.

Reusable optimal transport utilities are in src.analyze.utils.optimal_transport.
"""

__all__ = [
    # Re-exported from utils for convenience
    "UOTConfig",
    "UOTFrame",
    "UOTFramePair",
    "UOTSupport",
    "UOTProblem",
    "UOTResultWork",
    "UOTResultCanonical",
    "SamplingMode",
    "MassMode",
    "POTBackend",
    # Embryo-specific I/O
    "load_mask_from_csv",
    "load_mask_pair_from_csv",
    "load_mask_series_from_csv",
    "load_mask_from_png",
    # Embryo-specific preprocessing
    # Pipeline orchestration
    "run_uot_pair",
    "run_timeseries_from_csv",
    # Phase 2: Contract-compliant visualization
    "plot_uot_quiver",
    "plot_uot_cost_field",
    "plot_uot_creation_destruction",
    "plot_uot_overlay_with_transport",
    "plot_uot_diagnostic_suite",
    "UOTVizConfig",
    "DEFAULT_UOT_VIZ_CONFIG",
]


def __getattr__(name: str):
    if name in {
        "UOTConfig",
        "UOTFrame",
        "UOTFramePair",
        "UOTSupport",
        "UOTProblem",
        "UOTResultWork",
        "UOTResultCanonical",
        "SamplingMode",
        "MassMode",
        "POTBackend",
    }:
        import analyze.utils.optimal_transport as _ot

        return getattr(_ot, name)

    if name in {
        "load_mask_from_csv",
        "load_mask_pair_from_csv",
        "load_mask_series_from_csv",
        "load_mask_from_png",
    }:
        from . import frame_mask_io as _io

        return getattr(_io, name)

    if name in {"run_uot_pair"}:
        from . import run_transport as _rt

        return getattr(_rt, name)

    if name == "run_timeseries_from_csv":
        from . import run_timeseries as _ts

        return _ts.run_timeseries_from_csv

    if name in {
        "plot_uot_quiver",
        "plot_uot_cost_field",
        "plot_uot_creation_destruction",
        "plot_uot_overlay_with_transport",
        "plot_uot_diagnostic_suite",
        "UOTVizConfig",
        "DEFAULT_UOT_VIZ_CONFIG",
    }:
        from . import viz as _viz

        return getattr(_viz, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
