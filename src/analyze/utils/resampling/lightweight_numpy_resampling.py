from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

REQUIRED_SUMMARY_KEYS = {
    "stat_name",
    "exceed_count",
    "null_mean",
    "null_std",
    "obs_stat",
    "pval",
    "z_score",
}


@dataclass(frozen=True)
class LiteRun:
    """Lightweight provenance container for Stage 2 vectorized null tests."""

    test_name: str
    n_iters: int
    seed: int
    spec: dict
    summary: dict
    samples: Any = None
    metadata: dict | None = None


def _validate_summary(summary: dict) -> None:
    missing = REQUIRED_SUMMARY_KEYS - set(summary.keys())
    if missing:
        raise ValueError(f"LiteRun.summary missing required keys: {sorted(missing)}")

    if not isinstance(summary["stat_name"], str):
        raise TypeError("LiteRun.summary['stat_name'] must be str")

    for key in ("exceed_count", "null_mean", "null_std", "obs_stat", "pval", "z_score"):
        if not isinstance(summary[key], np.ndarray):
            raise TypeError(f"LiteRun.summary['{key}'] must be np.ndarray, got {type(summary[key])}")

    n = len(summary["obs_stat"])
    for key in ("exceed_count", "null_mean", "null_std", "pval", "z_score"):
        if len(summary[key]) != n:
            raise ValueError(
                f"LiteRun.summary['{key}'] length {len(summary[key])} != obs_stat length {n}"
            )


def run_lite(
    *,
    test_name: str,
    n_iters: int,
    seed: int,
    spec: dict,
    kernel: Callable[[np.random.Generator, int], dict | tuple[dict, Any]],
    collect_samples: bool = False,
    metadata: dict | None = None,
) -> LiteRun:
    """Run a lightweight vectorized null-test kernel with standardized metadata."""
    if n_iters <= 0:
        raise ValueError(f"n_iters must be positive; got {n_iters}")

    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)

    out = kernel(rng, n_iters)
    if isinstance(out, tuple):
        summary, samples = out
    else:
        summary, samples = out, None

    if collect_samples and samples is None:
        raise ValueError("collect_samples=True requires kernel to return (summary, samples)")

    _validate_summary(summary)

    return LiteRun(
        test_name=test_name,
        n_iters=n_iters,
        seed=seed,
        spec=spec,
        summary=summary,
        samples=samples if collect_samples else None,
        metadata=metadata or {},
    )
