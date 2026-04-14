"""Model-layer exports for particle prediction."""

from .kernels import (
    KernelSampleResult,
    compute_effective_sample_size,
    compute_weighted_covariance_diag,
    compute_weighted_mean,
    construct_tangent_aligned_covariance,
    sample_empirical_next_states,
)
from .local_transition_pf import LocalPredictionResult, LocalTransitionPredictor
from .matching import (
    MatchingConfig,
    compare_matching_modes,
    compute_fast_summary_distance_sq,
    compute_history_distance_sq,
    match_query_to_bank,
    retrieve_state_candidates,
)

__all__ = [
    "KernelSampleResult",
    "LocalPredictionResult",
    "LocalTransitionPredictor",
    "MatchingConfig",
    "compare_matching_modes",
    "compute_effective_sample_size",
    "compute_fast_summary_distance_sq",
    "compute_history_distance_sq",
    "compute_weighted_covariance_diag",
    "compute_weighted_mean",
    "construct_tangent_aligned_covariance",
    "match_query_to_bank",
    "retrieve_state_candidates",
    "sample_empirical_next_states",
]
