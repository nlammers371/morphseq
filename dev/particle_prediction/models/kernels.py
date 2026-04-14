"""Empirical local transition-kernel helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KernelSampleResult:
    """Sampled one-step transitions plus summary diagnostics."""

    sampled_next_states: np.ndarray
    sampled_increments: np.ndarray
    sampled_candidate_indices: np.ndarray
    mean_next_state: np.ndarray
    cov_diag: np.ndarray
    effective_sample_size: float


def _as_probability_vector(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return weights / total


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """Return the standard importance-sampling ESS."""

    weights = _as_probability_vector(weights)
    return float(1.0 / np.sum(weights ** 2))


def compute_weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute a weighted mean over rows of a 2D array."""

    values = np.asarray(values, dtype=np.float64)
    weights = _as_probability_vector(weights)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    if values.shape[0] != weights.shape[0]:
        raise ValueError("values and weights must agree on the row dimension")
    return np.sum(weights[:, None] * values, axis=0)


def compute_weighted_covariance_diag(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute the diagonal of the weighted covariance matrix."""

    values = np.asarray(values, dtype=np.float64)
    mean = compute_weighted_mean(values, weights)
    centered = values - mean[None, :]
    weights = _as_probability_vector(weights)
    return np.sum(weights[:, None] * (centered ** 2), axis=0)


def construct_tangent_aligned_covariance(
    reference_increment: np.ndarray,
    sigma_parallel: float,
    sigma_perp: float,
    min_variance: float = 1.0e-8,
) -> np.ndarray:
    """Construct a PSD covariance aligned to the increment tangent."""

    reference_increment = np.asarray(reference_increment, dtype=np.float64)
    if reference_increment.ndim != 1:
        raise ValueError("reference_increment must be a 1D array")
    if sigma_parallel < 0 or sigma_perp < 0:
        raise ValueError("jitter scales must be non-negative")
    if min_variance <= 0:
        raise ValueError("min_variance must be positive")

    dim = reference_increment.shape[0]
    norm = float(np.linalg.norm(reference_increment))
    if norm <= 1.0e-12:
        variance = max(sigma_perp ** 2, min_variance)
        return np.eye(dim, dtype=np.float64) * variance

    tangent = reference_increment / norm
    parallel_outer = np.outer(tangent, tangent)
    perp_outer = np.eye(dim, dtype=np.float64) - parallel_outer
    covariance = (sigma_parallel ** 2) * parallel_outer + (sigma_perp ** 2) * perp_outer
    covariance = covariance + np.eye(dim, dtype=np.float64) * min_variance
    return covariance


def sample_empirical_next_states(
    query_state: np.ndarray,
    candidate_increments: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_weights: np.ndarray,
    n_samples: int,
    sigma_parallel: float,
    sigma_perp: float,
    jitter_mode: str = "tangent",
    rng: np.random.Generator | None = None,
) -> KernelSampleResult:
    """Sample one-step next states from weighted empirical increments plus jitter."""

    query_state = np.asarray(query_state, dtype=np.float64)
    candidate_increments = np.asarray(candidate_increments, dtype=np.float64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_weights = _as_probability_vector(candidate_weights)
    rng = np.random.default_rng() if rng is None else rng

    if query_state.ndim != 1:
        raise ValueError("query_state must be a 1D array")
    if candidate_increments.ndim != 2:
        raise ValueError("candidate_increments must be a 2D array")
    if candidate_increments.shape[0] != candidate_weights.shape[0]:
        raise ValueError("candidate_increments and candidate_weights must agree on the row dimension")
    if candidate_indices.shape[0] != candidate_weights.shape[0]:
        raise ValueError("candidate_indices and candidate_weights must agree on the row dimension")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    sampled_row_indices = rng.choice(candidate_increments.shape[0], size=n_samples, replace=True, p=candidate_weights)
    sampled_increments = candidate_increments[sampled_row_indices].copy()

    jitter = np.zeros_like(sampled_increments)
    if jitter_mode not in {"none", "tangent", "isotropic"}:
        raise ValueError("jitter_mode must be 'none', 'tangent', or 'isotropic'")

    if jitter_mode == "tangent":
        for index, increment in enumerate(sampled_increments):
            covariance = construct_tangent_aligned_covariance(
                reference_increment=increment,
                sigma_parallel=sigma_parallel,
                sigma_perp=sigma_perp,
            )
            jitter[index] = rng.multivariate_normal(mean=np.zeros(query_state.shape[0], dtype=np.float64), cov=covariance)
    elif jitter_mode == "isotropic":
        scale = max(sigma_parallel, sigma_perp)
        jitter = rng.normal(loc=0.0, scale=scale, size=sampled_increments.shape)

    sampled_next_states = query_state[None, :] + sampled_increments + jitter
    mean_next_state = compute_weighted_mean(query_state[None, :] + candidate_increments, candidate_weights)
    cov_diag = compute_weighted_covariance_diag(query_state[None, :] + candidate_increments, candidate_weights)

    return KernelSampleResult(
        sampled_next_states=sampled_next_states,
        sampled_increments=sampled_increments + jitter,
        sampled_candidate_indices=candidate_indices[sampled_row_indices],
        mean_next_state=mean_next_state,
        cov_diag=cov_diag,
        effective_sample_size=compute_effective_sample_size(candidate_weights),
    )


__all__ = [
    "KernelSampleResult",
    "compute_effective_sample_size",
    "compute_weighted_covariance_diag",
    "compute_weighted_mean",
    "construct_tangent_aligned_covariance",
    "sample_empirical_next_states",
]
