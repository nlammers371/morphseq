"""
Post-hoc aggregation of resampling results.

``aggregate()`` dispatches on the spec kind:
- ``indices`` -> ``BootstrapSummary`` (percentile CI)
- ``labels`` or ``groups`` -> ``PermutationSummary`` (p-value + null stats)
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ._engine import ResampleRun


@dataclass
class BootstrapSummary:
    """Summary of bootstrap / subsample results.

    Attributes
    ----------
    observed : Any
        Observed statistic value.
    mean : Any
        Mean across bootstrap replicates.
    se : Any
        Standard error across bootstrap replicates.
    ci_low : Any
        Lower confidence interval bound.
    ci_high : Any
        Upper confidence interval bound.
    ci_method : str
        CI method used (``"percentile"`` in PR1).
    ci_is_exact : bool
        Whether CI is computed from stored samples (always True in PR1).
    n_success : int
        Number of successful iterations.
    n_failed : int
        Number of failed iterations.
    """
    observed: Any
    mean: Any
    se: Any
    ci_low: Any
    ci_high: Any
    ci_method: str
    ci_is_exact: bool
    n_success: int
    n_failed: int


@dataclass
class PermutationSummary:
    """Summary of permutation test results.

    Attributes
    ----------
    observed : float
        Observed test statistic.
    pvalue : float
        Permutation p-value with +1 smoothing: ``(1 + count_extreme) / (B + 1)``.
    null_distribution : ndarray or None
        Full null distribution if stored; ``None`` if reducer-only.
    null_mean : float
        Mean of the null distribution.
    null_std : float
        Standard deviation of the null distribution.
    alternative : str
        Tail direction used for p-value computation.
    n_permutations : int
        Number of successful permutation iterations.
    n_failed : int
        Number of failed iterations.
    statistic_name : str
        Name of the test statistic.
    """
    observed: float
    pvalue: float
    null_distribution: Optional[np.ndarray]
    null_mean: float
    null_std: float
    alternative: str
    n_permutations: int
    n_failed: int
    statistic_name: str

    def to_permutation_result(self):
        """Convert to legacy ``PermutationResult`` for backward compatibility.

        Returns
        -------
        PermutationResult
            From ``analyze.difference_detection.permutation_utils``.
        """
        from analyze.difference_detection.permutation_utils import PermutationResult

        null_dist = (self.null_distribution
                     if self.null_distribution is not None
                     else np.array([]))
        return PermutationResult(
            statistic_name=self.statistic_name,
            observed=self.observed,
            pvalue=self.pvalue,
            null_distribution=null_dist,
            alternative=self.alternative,
        )


def aggregate(
    out: ResampleRun,
    *,
    alpha: float = 0.05,
    ci_method: str = "percentile",
    alternative: Optional[str] = None,
):
    """Aggregate resampling results into a summary.

    Parameters
    ----------
    out : ResampleRun
        Output from ``run()``.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    ci_method : str, default="percentile"
        CI method. Only ``"percentile"`` is supported in PR1.
    alternative : str, optional
        Override the tail direction. Only allowed if ``out.samples`` is
        not ``None`` (full null stored), since the extremeness counts
        must be recomputable. Defaults to ``out.resolved_alternative``.

    Returns
    -------
    BootstrapSummary or PermutationSummary
        Depends on ``out.spec.kind``.
    """
    if out.spec.kind == "indices":
        return _aggregate_bootstrap(out, alpha=alpha, ci_method=ci_method)
    else:
        return _aggregate_permutation(out, alternative=alternative)


# ── bootstrap ────────────────────────────────────────────────────────

def _aggregate_bootstrap(
    out: ResampleRun, *, alpha: float, ci_method: str,
) -> BootstrapSummary:
    if ci_method != "percentile":
        raise ValueError(
            f"Only ci_method='percentile' is supported in PR1, got {ci_method!r}."
        )
    if out.samples is None:
        raise ValueError(
            "BootstrapSummary requires stored samples (store='all')."
        )

    arr = np.array(out.samples)
    mean = np.mean(arr, axis=0)
    se = np.std(arr, axis=0, ddof=1)
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    ci_low = np.percentile(arr, lo, axis=0)
    ci_high = np.percentile(arr, hi, axis=0)

    return BootstrapSummary(
        observed=out.observed,
        mean=mean,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method=ci_method,
        ci_is_exact=True,
        n_success=out.n_success,
        n_failed=out.n_failed,
    )


# ── permutation ──────────────────────────────────────────────────────

def _aggregate_permutation(
    out: ResampleRun, *, alternative: Optional[str],
) -> PermutationSummary:
    # Resolve alternative.
    if alternative is not None:
        if out.samples is None:
            raise ValueError(
                "Cannot override alternative when full null is not stored. "
                "The extremeness counts were already committed to "
                f"{out.resolved_alternative!r}."
            )
        alt = alternative
    else:
        alt = out.resolved_alternative or "two-sided"

    # Compute from stored samples or reducer state.
    if out.samples is not None:
        null = np.array(out.samples, dtype=float)
        observed = float(out.observed)
        pvalue = _compute_pvalue(observed, null, alt)
        null_mean = float(np.mean(null))
        null_std = float(np.std(null))
        null_dist = null
    elif out.reducer_state is not None:
        rs = out.reducer_state
        pvalue = rs["pvalue"]
        null_mean = rs["null_mean"]
        null_std = rs["null_std"]
        null_dist = None
    else:
        raise ValueError(
            "No samples or reducer state available for aggregation."
        )

    return PermutationSummary(
        observed=float(out.observed),
        pvalue=pvalue,
        null_distribution=null_dist,
        null_mean=null_mean,
        null_std=null_std,
        alternative=alt,
        n_permutations=out.n_success,
        n_failed=out.n_failed,
        statistic_name=out.statistic.name,
    )


def _compute_pvalue(observed: float, null: np.ndarray, alternative: str) -> float:
    """Permutation p-value with +1 smoothing."""
    B = len(null)
    if B == 0:
        return 1.0

    if alternative == "greater":
        k = int(np.sum(null >= observed))
    elif alternative == "less":
        k = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        k = int(np.sum(np.abs(null) >= abs(observed)))
    else:
        raise ValueError(f"Unknown alternative: {alternative!r}")

    return (1 + k) / (B + 1)
