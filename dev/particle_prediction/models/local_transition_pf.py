"""Local empirical one-step predictor built from a transition bank."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from dev.particle_prediction.data.dataset import PredictionQuery
from dev.particle_prediction.data.transition_bank import MatchResult, TransitionBank
from dev.particle_prediction.eval.predictions import RolloutPredictionResult, RolloutStepDiagnostics
from .kernels import KernelSampleResult, sample_empirical_next_states
from .matching import MatchingConfig, match_query_to_bank


@dataclass(frozen=True)
class LocalPredictionResult:
    """One-step local prediction output with diagnostics."""

    predicted_mean: np.ndarray
    predicted_cov_diag: np.ndarray
    forward_samples: np.ndarray
    match_result: MatchResult
    candidate_count: int
    effective_sample_size: float
    history_mismatch: float
    search_radius: float
    selected_class_weights: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class LocalTransitionPredictor:
    """One-step weighted empirical transition predictor plus multi-step rollout."""

    def __init__(
        self,
        bank: TransitionBank,
        matching_config: MatchingConfig | None = None,
        sigma_parallel: float = 0.05,
        sigma_perp: float = 0.1,
        jitter_mode: str = "tangent",
    ) -> None:
        self.bank = bank
        self.matching_config = MatchingConfig() if matching_config is None else matching_config
        self.sigma_parallel = float(sigma_parallel)
        self.sigma_perp = float(sigma_perp)
        self.jitter_mode = jitter_mode

    def _match(self, query: PredictionQuery) -> MatchResult:
        lambda_h = self.matching_config.lambda_h if query.mode == "history" else 0.0
        query_history = query.history_segments if query.mode == "history" else None
        history_mode = self.matching_config.history_mode if query.mode == "history" else "ordered_segments"
        return match_query_to_bank(
            bank=self.bank,
            query_state=query.current_state,
            query_history_segments=query_history,
            k_state=self.matching_config.k_state,
            offset_radius=self.matching_config.offset_radius,
            alpha=self.matching_config.alpha,
            sigma_z=self.matching_config.sigma_z,
            sigma_h=self.matching_config.sigma_h,
            lambda_h=lambda_h,
            class_priors=query.class_prior,
            retrieval_method=self.matching_config.retrieval_method,
            history_mode=history_mode,
        )

    def predict_query(
        self,
        query: PredictionQuery,
        n_samples: int = 128,
        rng: np.random.Generator | None = None,
    ) -> LocalPredictionResult:
        """Predict one next-step state from a snapshot or history query."""

        match_result = self._match(query)
        kernel_samples = sample_empirical_next_states(
            query_state=query.current_state,
            candidate_increments=np.vstack([window.increment for window in match_result.candidate_windows]),
            candidate_indices=match_result.candidate_indices,
            candidate_weights=match_result.normalized_weights,
            n_samples=n_samples,
            sigma_parallel=self.sigma_parallel,
            sigma_perp=self.sigma_perp,
            jitter_mode=self.jitter_mode,
            rng=rng,
        )
        return self._assemble_result(query=query, match_result=match_result, kernel_samples=kernel_samples)

    def _assemble_result(
        self,
        query: PredictionQuery,
        match_result: MatchResult,
        kernel_samples: KernelSampleResult,
    ) -> LocalPredictionResult:
        selected_class_weights: Dict[str, float] = {}
        for window, weight in zip(match_result.candidate_windows, match_result.normalized_weights):
            selected_class_weights[window.perturbation_class] = (
                selected_class_weights.get(window.perturbation_class, 0.0) + float(weight)
            )

        history_mismatch = float(np.average(match_result.d_hist_sq, weights=match_result.normalized_weights))
        search_radius = float(np.sqrt(np.max(match_result.d_state_sq))) if len(match_result.d_state_sq) else 0.0
        diagnostics = {
            "query_mode": query.mode,
            "jitter_mode": self.jitter_mode,
            "sigma_parallel": self.sigma_parallel,
            "sigma_perp": self.sigma_perp,
            "mean_score": float(np.mean(match_result.scores)),
        }
        return LocalPredictionResult(
            predicted_mean=kernel_samples.mean_next_state,
            predicted_cov_diag=kernel_samples.cov_diag,
            forward_samples=kernel_samples.sampled_next_states,
            match_result=match_result,
            candidate_count=len(match_result.candidate_indices),
            effective_sample_size=kernel_samples.effective_sample_size,
            history_mismatch=history_mismatch,
            search_radius=search_radius,
            selected_class_weights=selected_class_weights,
            diagnostics=diagnostics,
        )

    def _append_history(
        self,
        history_segments: np.ndarray | None,
        increment: np.ndarray,
    ) -> np.ndarray:
        increment = np.asarray(increment, dtype=np.float64).reshape(1, -1)
        if history_segments is None or history_segments.size == 0:
            return increment.copy()

        history_segments = np.asarray(history_segments, dtype=np.float64)
        updated = np.concatenate([history_segments, increment], axis=0)
        max_history = max(1, int(self.bank.history_length) - 2 * int(self.matching_config.offset_radius))
        if updated.shape[0] > max_history:
            updated = updated[-max_history:]
        return updated

    def _query_from_particle(
        self,
        current_state: np.ndarray,
        history_segments: np.ndarray | None,
        class_prior: Dict[str, float] | None,
    ) -> PredictionQuery:
        mode = "history" if history_segments is not None and history_segments.size > 0 else "snapshot"
        return PredictionQuery(
            mode=mode,
            current_state=np.asarray(current_state, dtype=np.float64).copy(),
            history_segments=None if mode == "snapshot" else np.asarray(history_segments, dtype=np.float64).copy(),
            class_prior=class_prior,
        )

    def rollout_query(
        self,
        query: PredictionQuery,
        n_steps: int,
        n_particles: int = 128,
        rng: np.random.Generator | None = None,
    ) -> RolloutPredictionResult:
        """Roll forward particles for `n_steps` empirical transitions."""

        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if n_particles < 1:
            raise ValueError("n_particles must be at least 1")

        rng = np.random.default_rng() if rng is None else rng
        current_particles = np.repeat(np.asarray(query.current_state, dtype=np.float64)[None, :], n_particles, axis=0)
        particle_histories: List[np.ndarray | None] = []
        base_history = None if query.mode == "snapshot" else np.asarray(query.history_segments, dtype=np.float64)
        for _ in range(n_particles):
            particle_histories.append(None if base_history is None else base_history.copy())

        n_dims = current_particles.shape[1]
        forward_samples = np.empty((n_steps, n_particles, n_dims), dtype=np.float64)
        predicted_mean = np.empty((n_steps, n_dims), dtype=np.float64)
        predicted_cov_diag = np.empty((n_steps, n_dims), dtype=np.float64)
        step_diagnostics: List[RolloutStepDiagnostics] = []

        for step in range(n_steps):
            next_particles = np.empty_like(current_particles)
            candidate_counts = np.empty(n_particles, dtype=np.float64)
            effective_sample_sizes = np.empty(n_particles, dtype=np.float64)
            history_mismatches = np.empty(n_particles, dtype=np.float64)
            search_radii = np.empty(n_particles, dtype=np.float64)
            class_weight_sums: Dict[str, float] = {}

            for particle_index in range(n_particles):
                particle_query = self._query_from_particle(
                    current_state=current_particles[particle_index],
                    history_segments=particle_histories[particle_index],
                    class_prior=query.class_prior,
                )
                particle_result = self.predict_query(particle_query, n_samples=1, rng=rng)
                next_state = particle_result.forward_samples[0]
                increment = next_state - current_particles[particle_index]

                next_particles[particle_index] = next_state
                particle_histories[particle_index] = self._append_history(particle_histories[particle_index], increment)
                candidate_counts[particle_index] = float(particle_result.candidate_count)
                effective_sample_sizes[particle_index] = float(particle_result.effective_sample_size)
                history_mismatches[particle_index] = float(particle_result.history_mismatch)
                search_radii[particle_index] = float(particle_result.search_radius)
                for class_name, weight in particle_result.selected_class_weights.items():
                    class_weight_sums[class_name] = class_weight_sums.get(class_name, 0.0) + float(weight)

            current_particles = next_particles
            forward_samples[step] = current_particles
            predicted_mean[step] = np.mean(current_particles, axis=0)
            predicted_cov_diag[step] = np.var(current_particles, axis=0)

            averaged_class_weights = {
                class_name: weight / float(n_particles) for class_name, weight in sorted(class_weight_sums.items())
            }
            step_diagnostics.append(
                RolloutStepDiagnostics(
                    candidate_count=float(np.mean(candidate_counts)),
                    effective_sample_size=float(np.mean(effective_sample_sizes)),
                    history_mismatch=float(np.mean(history_mismatches)),
                    search_radius=float(np.mean(search_radii)),
                    selected_class_weights=averaged_class_weights,
                    diagnostics={
                        "step_index": int(step),
                        "min_candidate_count": float(np.min(candidate_counts)),
                        "max_candidate_count": float(np.max(candidate_counts)),
                    },
                )
            )

        diagnostics = {
            "query_mode": query.mode,
            "n_steps": int(n_steps),
            "n_particles": int(n_particles),
            "jitter_mode": self.jitter_mode,
        }
        return RolloutPredictionResult(
            predicted_mean=predicted_mean,
            predicted_cov_diag=predicted_cov_diag,
            forward_samples=forward_samples,
            step_diagnostics=step_diagnostics,
            diagnostics=diagnostics,
        )


__all__ = ["LocalPredictionResult", "LocalTransitionPredictor"]
