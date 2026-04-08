"""Core trajectory condensation exports."""

from . import (
    force_diagnostics,
    init_embedding,
    iteration_ranking,
    schema,
    space_density_metrics,
    viz,
)
from .viz import animation, plotting
from .viz import (
    VizConfig,
    RunDescriptor,
    render_run,
    render_feature_over_time_facets,
    compare_runs,
    compare_run_grid,
    load_run,
)
from .schema import CondensationData, validate, from_multiclass_csv, from_pairwise_margin_csv
from .force_diagnostics import ForceSnapshot, force_snapshot, force_target_table
from .condensation import (
    CondensationConfig,
    CondensationResult,
    CondensationState,
    ForceBalanceSummary,
    GeometryRefs,
    StoppingConfig,
    StoppingMonitor,
    describe_force_balance,
    estimate_geometry_refs,
    resolve_force_balance,
    run_condensation,
)

__all__ = [
    "CondensationData",
    "validate",
    "from_multiclass_csv",
    "from_pairwise_margin_csv",
    "force_diagnostics",
    "init_embedding",
    "iteration_ranking",
    "schema",
    "space_density_metrics",
    "viz",
    "animation",
    "plotting",
    "VizConfig",
    "ForceSnapshot",
    "RunDescriptor",
    "force_snapshot",
    "force_target_table",
    "render_run",
    "render_feature_over_time_facets",
    "compare_runs",
    "compare_run_grid",
    "load_run",
    "CondensationConfig",
    "CondensationResult",
    "CondensationState",
    "ForceBalanceSummary",
    "GeometryRefs",
    "StoppingConfig",
    "StoppingMonitor",
    "describe_force_balance",
    "estimate_geometry_refs",
    "resolve_force_balance",
    "run_condensation",
]
