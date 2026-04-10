"""ClassificationAnalysis result object and lazy-loading layer registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .null import NullDistributions

# ClassifierDirections has moved to classification/directions/artifact.py.
# This re-export keeps existing imports working.
# New code should import from: analyze.classification.directions.artifact
from analyze.classification.directions.artifact import ClassifierDirections  # noqa: F401

# ---------------------------------------------------------------------------
# Scores validation
# ---------------------------------------------------------------------------

_SCORES_REQUIRED = frozenset({
    "feature_set",
    "comparison_id",
    "positive_label",
    "negative_label",
    "time_bin_center",
    "auroc_obs",
})

_SCORES_UNIQUE_KEY = ("feature_set", "comparison_id", "time_bin_center")


def _validate_scores(df: pd.DataFrame) -> None:
    missing = _SCORES_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Scores table missing required columns: {sorted(missing)}")
    if df.empty:
        return
    dupes = df.duplicated(subset=list(_SCORES_UNIQUE_KEY), keep=False)
    if dupes.any():
        n = int(dupes.sum())
        raise ValueError(
            f"Scores table has {n} rows with duplicate keys "
            f"{_SCORES_UNIQUE_KEY}. Each (feature_set, comparison_id, "
            f"time_bin_center) must be unique."
        )


# ---------------------------------------------------------------------------
# _LazyLayers
# ---------------------------------------------------------------------------


class _LazyLayers:
    """Lazy-loading artifact registry backed by an optional directory."""

    _REGISTRY: dict[str, tuple[str, str]] = {
        "predictions": ("predictions.parquet", "parquet"),
        "multiclass_predictions": ("multiclass_predictions.parquet", "parquet"),
        "confusion": ("confusion.parquet", "parquet"),
        "null_full": ("null_distributions.npz", "nulls"),
        "classifier_directions": ("classifier_directions.parquet", "directions"),
        "raw_contrast_scores_long": ("raw_contrast_scores_long.parquet", "parquet"),
        "contrast_support_long": ("contrast_support_long.parquet", "parquet"),
        "contrast_specificity_by_timebin": ("contrast_specificity_by_timebin.parquet", "parquet"),
        "raw_coordinates": ("raw_coordinates.parquet", "parquet"),
        "shrunk_coordinates": ("shrunk_coordinates.parquet", "parquet"),
        "residual_coordinates": ("residual_coordinates.parquet", "parquet"),
        "probe_index": ("probe_index.parquet", "parquet"),
    }

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else None
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        if key not in self._REGISTRY:
            raise KeyError(f"Unknown layer: {key!r}. Known: {sorted(self._REGISTRY)}")
        if self._base_dir is None:
            raise KeyError(
                f"Layer {key!r} was not computed during this run. "
                f"Re-run with the appropriate save_* flag enabled."
            )
        filename, kind = self._REGISTRY[key]
        path = self._base_dir / filename
        if not path.exists():
            if key == "multiclass_predictions":
                raise KeyError(
                    f"Layer {key!r} not found at {path}. "
                    "Misclassification pipeline requires multiclass_predictions. "
                    "Re-run run_classification(..., save_multiclass_predictions=True)."
                )
            raise KeyError(f"Layer {key!r} not found at {path}")
        data = self._load(kind, path)
        self._cache[key] = data
        return data

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        if key in self._cache:
            return True
        if key not in self._REGISTRY:
            return False
        if self._base_dir is None:
            return False
        filename, _ = self._REGISTRY[key]
        if key == "classifier_directions":
            return (self._base_dir / filename).exists() and (
                self._base_dir / "classifier_directions_vectors.npz"
            ).exists()
        return (self._base_dir / filename).exists()

    def available(self) -> list[str]:
        result = list(self._cache.keys())
        if self._base_dir is not None:
            for key, (filename, _) in self._REGISTRY.items():
                if key not in result and (self._base_dir / filename).exists():
                    result.append(key)
        return sorted(result)

    def cached(self) -> list[str]:
        return sorted(self._cache.keys())

    def store(self, key: str, data: Any) -> None:
        self._cache[key] = data

    def _fork(self) -> _LazyLayers:
        return _LazyLayers(self._base_dir)

    def _save_to_dir(self, path: Path, overwrite: bool = False) -> None:
        path = Path(path)
        for key, data in self._cache.items():
            if key not in self._REGISTRY:
                continue
            filename, kind = self._REGISTRY[key]
            target = path / filename
            if target.exists() and not overwrite:
                raise FileExistsError(f"{target} already exists (overwrite=False)")
            if kind == "parquet":
                data.to_parquet(target, index=False)
            elif kind == "nulls":
                data.save(target)
            elif kind == "directions":
                data.metadata.to_parquet(target, index=False)
                data.save(path / "classifier_directions_vectors.npz")

    @staticmethod
    def _load(kind: str, path: Path) -> Any:
        if kind == "parquet":
            return pd.read_parquet(path)
        if kind == "nulls":
            return NullDistributions.load(path)
        if kind == "directions":
            return ClassifierDirections.load(
                path,
                path.with_name("classifier_directions_vectors.npz"),
            )
        raise ValueError(f"Unknown layer kind: {kind!r}")


# ---------------------------------------------------------------------------
# ClassificationAnalysis
# ---------------------------------------------------------------------------


@dataclass
class ClassificationAnalysis:
    """Result of ``run_classification()``."""

    scores: pd.DataFrame
    uns: dict = field(default_factory=dict)
    layers: _LazyLayers = field(default_factory=_LazyLayers)

    def __post_init__(self) -> None:
        _validate_scores(self.scores)

    @property
    def feature_sets(self) -> list[str]:
        return sorted(self.scores["feature_set"].unique().tolist())

    @property
    def comparison_ids(self) -> list[str]:
        return sorted(self.scores["comparison_id"].unique().tolist())

    def subset(
        self,
        feature_set: str | list[str] | None = None,
        comparison_id: str | list[str] | None = None,
        positive_label: str | list[str] | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> ClassificationAnalysis:
        df = self.scores
        if feature_set is not None:
            vals = [feature_set] if isinstance(feature_set, str) else feature_set
            df = df[df["feature_set"].isin(vals)]
        if comparison_id is not None:
            vals = [comparison_id] if isinstance(comparison_id, str) else comparison_id
            df = df[df["comparison_id"].isin(vals)]
        if positive_label is not None:
            vals = [positive_label] if isinstance(positive_label, str) else positive_label
            df = df[df["positive_label"].isin(vals)]
        if time_range is not None:
            lo, hi = time_range
            df = df[(df["time_bin_center"] >= lo) & (df["time_bin_center"] <= hi)]
        return ClassificationAnalysis(
            scores=df.reset_index(drop=True),
            uns=self.uns,
            layers=self.layers._fork(),
        )

    def stack(
        self,
        other: ClassificationAnalysis,
        on_conflict: str = "error",
    ) -> ClassificationAnalysis:
        """Merge two analyses. Layers are NOT merged."""
        merged_scores = pd.concat([self.scores, other.scores], ignore_index=True)

        merged_comps = dict(self.uns.get("comparisons", {}))
        other_comps = other.uns.get("comparisons", {})
        for key, val in other_comps.items():
            if key in merged_comps:
                if on_conflict == "error" and merged_comps[key] != val:
                    raise ValueError(
                        f"Conflict in uns['comparisons'] key {key!r}. "
                        f"Use on_conflict='overwrite' to force."
                    )
            merged_comps[key] = val

        merged_uns = {**self.uns, "comparisons": merged_comps}
        return ClassificationAnalysis(
            scores=merged_scores,
            uns=merged_uns,
            layers=_LazyLayers(),
        )

    def plot_aurocs(self, *, curve_col: str | None = None, facet_col: str | None = None, **kwargs: Any) -> Any:
        from analyze.classification.viz.auroc_over_time import plot_aurocs_over_time
        return plot_aurocs_over_time(
            self.scores, curve_col=curve_col, facet_col=facet_col, **kwargs,
        )

    def plot_confusion(self, **kwargs: Any) -> Any:
        confusion = self.layers["confusion"]
        from analyze.classification.viz.confusion import plot_confusion
        return plot_confusion(self.scores, confusion, **kwargs)

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        scores_path = path / "scores.parquet"
        meta_path = path / "metadata.json"
        if not overwrite:
            for p in (scores_path, meta_path):
                if p.exists():
                    raise FileExistsError(f"{p} already exists (overwrite=False)")
        self.scores.to_parquet(scores_path, index=False)
        with open(meta_path, "w") as f:
            json.dump(self.uns, f, indent=2, default=str)
        self.layers._save_to_dir(path, overwrite=overwrite)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ClassificationAnalysis:
        path = Path(path)
        scores = pd.read_parquet(path / "scores.parquet")
        with open(path / "metadata.json") as f:
            uns = json.load(f)
        layers = _LazyLayers(path)
        return cls(scores=scores, uns=uns, layers=layers)

    @classmethod
    def from_legacy(cls, path: str | Path) -> ClassificationAnalysis:
        raise NotImplementedError(
            "from_legacy() is not yet implemented. "
            "Use the existing ClassificationResults.load() for legacy data."
        )
