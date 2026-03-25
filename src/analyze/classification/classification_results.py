"""
ClassificationResults: a thin multi-metric accumulator around MulticlassOVRResults.

Design goals
------------
- User-facing unit of work is a single object with a single comparisons table.
- Cross-metric plotting is natural (facet by the 'metric' column).
- Folder-based persistence is a convenience, not the primary interface.

Persistence layout (directory)
------------------------------
- comparisons.parquet        (required; includes a 'metric' column)
- null_summary.parquet       (optional)
- metadata.json              (required; human-readable run metadata)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union, Mapping

import pandas as pd

from .results import MulticlassOVRResults

PathLike = Union[str, Path]


class ClassificationResults:
    """Accumulate multiple classification runs into a single long-form table."""

    # NOTE: negative is intentionally optional; plots can facet by it when present,
    # but cross-metric accumulation should not require it.
    _REQUIRED_COLUMNS = ("metric", "tag", "positive", "time_bin_center", "auroc_obs")

    _TIME_CENTER_ALIASES = (
        "time_bin_center",
        "pred_hpf_bin_center",
        "bin_center",
        "predicted_stage_hpf_bin_center",
    )
    _AUROC_ALIASES = ("auroc_obs", "auroc_observed", "auroc")
    _POSITIVE_ALIASES = ("positive", "class", "positive_class")
    _NEGATIVE_ALIASES = ("negative", "negative_class")
    _PVAL_ALIASES = ("pval", "p_value")

    _TAG_DEFAULT = "default"

    def __init__(
        self,
        comparisons: Optional[pd.DataFrame] = None,
        *,
        null_summary: Optional[pd.DataFrame] = None,
        run_metadata: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        source_dir: Optional[PathLike] = None,
    ) -> None:
        import warnings
        warnings.warn(
            "ClassificationResults is deprecated. "
            "Use ClassificationAnalysis from analyze.classification instead.",
            FutureWarning,
            stacklevel=2,
        )
        self._source_dir: Optional[Path] = Path(source_dir).resolve() if source_dir is not None else None
        self._run_metadata: Dict[str, Dict[str, Dict[str, Any]]] = dict(run_metadata or {})

        self.comparisons = pd.DataFrame() if comparisons is None else comparisons.copy()
        self.null_summary = None if null_summary is None else null_summary.copy()

        if not self.comparisons.empty:
            self._ensure_metric_and_tag(self.comparisons)
            self._coerce_canonical_columns(self.comparisons)
            self._validate_required_columns(self.comparisons)
            self._sort_inplace(self.comparisons)

        if self.null_summary is not None and not self.null_summary.empty:
            self._ensure_metric_and_tag(self.null_summary)
            self.null_summary["metric"] = self.null_summary["metric"].astype(str)
            self.null_summary["tag"] = self.null_summary["tag"].astype(str)
            self._sort_inplace(self.null_summary)

    @property
    def source_dir(self) -> Optional[Path]:
        return self._source_dir

    @property
    def run_metadata(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return json.loads(json.dumps(self._run_metadata, default=str))

    @property
    def metrics(self) -> list[str]:
        if self.comparisons.empty:
            return []
        return sorted(self.comparisons["metric"].dropna().astype(str).unique().tolist())

    @property
    def tags(self) -> list[str]:
        if self.comparisons.empty:
            return []
        return sorted(self.comparisons["tag"].dropna().astype(str).unique().tolist())

    def __repr__(self) -> str:  # pragma: no cover (pretty-print)
        if self.comparisons.empty:
            base = "ClassificationResults(empty)"
        else:
            n_metrics = len(self.metrics)
            n_rows = len(self.comparisons)
            n_bins = self.comparisons[["metric", "time_bin_center"]].drop_duplicates().shape[0]
            if "negative" in self.comparisons.columns:
                n_pairs = self.comparisons[["metric", "positive", "negative"]].drop_duplicates().shape[0]
                pairs_label = "metric×comparisons"
            else:
                n_pairs = self.comparisons[["metric", "positive"]].drop_duplicates().shape[0]
                pairs_label = "metric×curves"
            n_metric_tags = self.comparisons[["metric", "tag"]].drop_duplicates().shape[0]
            base = (
                f"ClassificationResults(metrics={n_metrics}, metric×tags={n_metric_tags}, rows={n_rows}, "
                f"metric×bins={n_bins}, {pairs_label}={n_pairs})"
            )
        if self._source_dir is not None:
            base += f"\n  Source: {self._source_dir}"
        return base

    # ---------------------------------------------------------------------
    # Core ingestion
    # ---------------------------------------------------------------------

    def add(
        self,
        metric: str,
        results: Union[MulticlassOVRResults, pd.DataFrame],
        *,
        tag: Optional[str] = _TAG_DEFAULT,
        overwrite: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        allow_empty: bool = True,
    ) -> None:
        """Add a metric run (default: overwrite metric/tag).

        Parameters
        ----------
        metric : str
            Metric label to tag all appended rows.
        results : MulticlassOVRResults | pd.DataFrame
            Either the structured results object or a comparisons DataFrame.
        tag : str | None
            Run tag to disambiguate multiple runs of the same metric. Defaults to "default".
            If tag is None and overwrite=False, a new tag is auto-generated and appended.
        overwrite : bool
            If True (default), replace existing rows for (metric, tag). If False, append.
        metadata : dict, optional
            If provided, stored as run_metadata[metric][tag]. If results is MulticlassOVRResults,
            its metadata is used unless overridden.
        allow_empty : bool
            If True (default), empty results are skipped with no error.
        """
        metric = str(metric)
        if tag is not None:
            tag = str(tag)

        if isinstance(results, MulticlassOVRResults):
            df = results.comparisons.copy()
            meta = dict(results.metadata or {})
            if metadata is not None:
                meta.update(dict(metadata))
            incoming_null = results.null_summary
        else:
            df = results.copy()
            incoming_null = None

        if df is None or df.empty:
            if allow_empty:
                return
            raise ValueError(f"Empty results for metric={metric!r}")

        df = df.copy()
        df["metric"] = metric
        df = self._canonicalize_incoming(df)

        final_tag = self._resolve_tag_for_add(metric, tag=tag, overwrite=overwrite, incoming=df)
        df["tag"] = str(final_tag)
        self._coerce_canonical_columns(df)
        self._validate_required_columns(df)
        self._sort_inplace(df)

        if metadata is None and isinstance(results, MulticlassOVRResults):
            metadata = dict(results.metadata or {})
        if metadata is not None:
            self._run_metadata.setdefault(metric, {})[str(final_tag)] = dict(metadata)

        self._ensure_metric_and_tag(self.comparisons)
        self._upsert_table("comparisons", df, metric=metric, tag=str(final_tag), overwrite=bool(overwrite))

        if incoming_null is not None and not incoming_null.empty:
            ns = incoming_null.copy()
            ns["metric"] = metric
            ns = self._canonicalize_incoming(ns)
            ns["tag"] = str(final_tag)
            self._ensure_metric_and_tag(ns)
            ns["metric"] = ns["metric"].astype(str)
            ns["tag"] = ns["tag"].astype(str)
            self._sort_inplace(ns)
            self._upsert_table("null_summary", ns, metric=metric, tag=str(final_tag), overwrite=bool(overwrite))

    # ---------------------------------------------------------------------
    # Plotting convenience
    # ---------------------------------------------------------------------

    def plot_aurocs_over_time(self, *, tag: Union[str, None] = _TAG_DEFAULT, **kwargs):
        """Convenience wrapper around analyze.classification.viz.plot_aurocs_over_time()."""
        from .viz.auroc_over_time import plot_aurocs_over_time

        df = self.comparisons
        if not df.empty:
            df = df.copy()
            if "tag" not in df.columns:
                df["tag"] = self._TAG_DEFAULT
            if tag is None or str(tag).lower() == "all":
                pass
            else:
                df = df[df["tag"].astype(str) == str(tag)].copy()

            # Container-only "smart default": facet by metric when multiple metrics exist.
            facet_row = kwargs.get("facet_row", None)
            facet_col = kwargs.get("facet_col", None)
            if facet_row is None and facet_col is None and ("metric" in df.columns) and df["metric"].nunique() > 1:
                kwargs["facet_col"] = "metric"

        return plot_aurocs_over_time(df, **kwargs)

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------

    def save(self, path: Optional[PathLike] = None, *, overwrite: bool = True) -> Path:
        """Save to a directory containing comparisons.parquet + metadata.json (+ optional null_summary.parquet)."""
        if path is None:
            if self._source_dir is None:
                raise ValueError("No default save path set. Pass path=... to save().")
            out_dir = self._source_dir
        else:
            out_dir = Path(path).resolve()

        out_dir.mkdir(parents=True, exist_ok=True)

        comp_path = out_dir / "comparisons.parquet"
        if comp_path.exists() and not overwrite:
            raise FileExistsError(f"{comp_path} exists. Use overwrite=True to replace.")
        if self.comparisons.empty:
            raise ValueError("Cannot save: comparisons is empty.")
        self._ensure_metric_and_tag(self.comparisons)
        self.comparisons.to_parquet(comp_path, index=False)

        if self.null_summary is not None and not self.null_summary.empty:
            null_path = out_dir / "null_summary.parquet"
            if null_path.exists() and not overwrite:
                raise FileExistsError(f"{null_path} exists. Use overwrite=True to replace.")
            self._ensure_metric_and_tag(self.null_summary)
            self.null_summary.to_parquet(null_path, index=False)

        meta_path = out_dir / "metadata.json"
        if meta_path.exists() and not overwrite:
            raise FileExistsError(f"{meta_path} exists. Use overwrite=True to replace.")
        metrics_tags: Dict[str, list[str]] = {}
        if not self.comparisons.empty:
            for metric, sub in self.comparisons.groupby("metric", dropna=True):
                metrics_tags[str(metric)] = sorted(sub["tag"].dropna().astype(str).unique().tolist())
        payload = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "python_version": sys.version,
            "metrics": self.metrics,
            "metric_tags": metrics_tags,
            "comparisons_columns": list(self.comparisons.columns),
            "null_summary_columns": None if self.null_summary is None else list(self.null_summary.columns),
            "run_metadata": self._run_metadata,
        }
        meta_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

        self._source_dir = out_dir
        return out_dir

    @classmethod
    def load(cls, path: PathLike) -> "ClassificationResults":
        """Load from a directory created by save()."""
        in_dir = Path(path).resolve()
        comp_path = in_dir / "comparisons.parquet"
        meta_path = in_dir / "metadata.json"

        if not comp_path.exists():
            raise FileNotFoundError(f"Missing {comp_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")

        comparisons = pd.read_parquet(comp_path)
        meta = json.loads(meta_path.read_text())
        run_metadata_raw = meta.get("run_metadata", {}) or {}
        run_metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for metric, entry in dict(run_metadata_raw).items():
            if isinstance(entry, Mapping) and all(isinstance(v, Mapping) for v in entry.values()):
                run_metadata[str(metric)] = {str(t): dict(v) for t, v in entry.items()}
            else:
                # v1: metric -> metadata dict (no tag); nest under default.
                run_metadata[str(metric)] = {cls._TAG_DEFAULT: dict(entry) if isinstance(entry, Mapping) else {}}

        null_path = in_dir / "null_summary.parquet"
        null_summary = pd.read_parquet(null_path) if null_path.exists() else None

        # v1 compatibility: inject missing tag
        if "tag" not in comparisons.columns:
            comparisons = comparisons.copy()
            comparisons["tag"] = cls._TAG_DEFAULT
        if null_summary is not None and not null_summary.empty and "tag" not in null_summary.columns:
            null_summary = null_summary.copy()
            null_summary["tag"] = cls._TAG_DEFAULT

        return cls(
            comparisons=comparisons,
            null_summary=null_summary,
            run_metadata=run_metadata,
            source_dir=in_dir,
        )

    @classmethod
    def from_dirs(cls, metric_dirs: Dict[str, Union[PathLike, Dict[str, PathLike]]]) -> "ClassificationResults":
        """Migration bridge: load legacy folder-based MulticlassOVRResults bundles and stack.

        Prefer `.add(metric, results)` for new work.
        """
        obj = cls()
        for metric, d in metric_dirs.items():
            if isinstance(d, dict):
                for tag, p in d.items():
                    res = MulticlassOVRResults.from_dir(Path(p))
                    obj.add(metric, res, tag=str(tag), overwrite=True)
            else:
                res = MulticlassOVRResults.from_dir(Path(d))
                obj.add(metric, res, tag=cls._TAG_DEFAULT, overwrite=True)
        return obj

    # ---------------------------------------------------------------------
    # Subsetting convenience
    # ---------------------------------------------------------------------

    def subset(
        self,
        *,
        metric: Optional[Union[str, list[str]]] = None,
        tag: Optional[Union[str, list[str]]] = None,
        positive: Optional[Union[str, list[str]]] = None,
        negative: Optional[Union[str, list[str]]] = None,
        time_range: Optional[tuple[float, float]] = None,
    ) -> "ClassificationResults":
        """Return a filtered copy as a new ClassificationResults."""
        if self.comparisons.empty:
            return ClassificationResults()

        df = self.comparisons.copy()
        self._ensure_metric_and_tag(df)

        def _as_list(v: Optional[Union[str, list[str]]]) -> Optional[list[str]]:
            if v is None:
                return None
            if isinstance(v, list):
                return [str(x) for x in v]
            return [str(v)]

        metric_vals = _as_list(metric)
        tag_vals = _as_list(tag)
        pos_vals = _as_list(positive)
        neg_vals = _as_list(negative)

        if metric_vals is not None:
            df = df[df["metric"].astype(str).isin(metric_vals)].copy()
        if tag_vals is not None:
            df = df[df["tag"].astype(str).isin(tag_vals)].copy()
        if pos_vals is not None and "positive" in df.columns:
            df = df[df["positive"].astype(str).isin(pos_vals)].copy()
        if neg_vals is not None and "negative" in df.columns:
            df = df[df["negative"].astype(str).isin(neg_vals)].copy()
        if time_range is not None:
            lo, hi = float(time_range[0]), float(time_range[1])
            df = df[df["time_bin_center"].astype(float).between(lo, hi, inclusive="both")].copy()

        null_summary = None
        if self.null_summary is not None and not self.null_summary.empty:
            ns = self.null_summary.copy()
            self._ensure_metric_and_tag(ns)
            ns = ns[ns["metric"].astype(str).isin(df["metric"].astype(str).unique())].copy()
            ns = ns[ns["tag"].astype(str).isin(df["tag"].astype(str).unique())].copy()
            null_summary = ns

        # Prune run_metadata
        keep_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
        keep_pairs = set(df[["metric", "tag"]].drop_duplicates().astype(str).itertuples(index=False, name=None))
        for m, tags in self._run_metadata.items():
            for t, meta in tags.items():
                if (str(m), str(t)) in keep_pairs:
                    keep_meta.setdefault(str(m), {})[str(t)] = dict(meta)

        return ClassificationResults(comparisons=df, null_summary=null_summary, run_metadata=keep_meta)

    # ---------------------------------------------------------------------
    # Validation / coercion
    # ---------------------------------------------------------------------

    @classmethod
    def _validate_required_columns(cls, df: pd.DataFrame) -> None:
        missing = [c for c in cls._REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Present: {list(df.columns)}")

    @staticmethod
    def _ensure_metric_and_tag(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        if "metric" not in df.columns:
            raise ValueError("Missing required column 'metric'")
        if "tag" not in df.columns:
            df["tag"] = ClassificationResults._TAG_DEFAULT

    @classmethod
    def _canonicalize_incoming(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Rename common aliases to canonical columns (best-effort, non-destructive)."""
        work = df.copy()

        def _first_present(cands: tuple[str, ...]) -> Optional[str]:
            for c in cands:
                if c in work.columns:
                    return c
            return None

        # Positive
        pos_col = _first_present(cls._POSITIVE_ALIASES)
        if pos_col is not None and pos_col != "positive":
            work = work.rename(columns={pos_col: "positive"})

        # Negative (optional)
        neg_col = _first_present(cls._NEGATIVE_ALIASES)
        if neg_col is not None and neg_col != "negative":
            work = work.rename(columns={neg_col: "negative"})

        # AUROC
        auroc_col = _first_present(cls._AUROC_ALIASES)
        if auroc_col is not None and auroc_col != "auroc_obs":
            work = work.rename(columns={auroc_col: "auroc_obs"})

        # P-value
        pval_col = _first_present(cls._PVAL_ALIASES)
        if pval_col is not None and pval_col != "pval":
            work = work.rename(columns={pval_col: "pval"})

        # Time center: rename alias if present
        if "time_bin_center" not in work.columns:
            for c in cls._TIME_CENTER_ALIASES:
                if c in work.columns:
                    work = work.rename(columns={c: "time_bin_center"})
                    break

        # Derive time center if start/end columns exist
        if "time_bin_center" not in work.columns:
            if {"time_bin_start", "time_bin_end"}.issubset(work.columns):
                work["time_bin_center"] = (pd.to_numeric(work["time_bin_start"], errors="coerce") + pd.to_numeric(
                    work["time_bin_end"], errors="coerce"
                )) / 2.0
            elif {"time_bin_left", "time_bin_right"}.issubset(work.columns):
                work["time_bin_center"] = (pd.to_numeric(work["time_bin_left"], errors="coerce") + pd.to_numeric(
                    work["time_bin_right"], errors="coerce"
                )) / 2.0

        return work

    @classmethod
    def _coerce_canonical_columns(cls, df: pd.DataFrame) -> None:
        df["metric"] = df["metric"].astype(str)
        if "tag" not in df.columns:
            df["tag"] = cls._TAG_DEFAULT
        df["tag"] = df["tag"].astype(str)
        df["positive"] = df["positive"].astype(str)
        if "negative" in df.columns:
            df["negative"] = df["negative"].astype(str)
        df["time_bin_center"] = pd.to_numeric(df["time_bin_center"], errors="coerce")
        df["auroc_obs"] = pd.to_numeric(df["auroc_obs"], errors="coerce")

    @classmethod
    def _sort_inplace(cls, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        preferred = [
            "metric",
            "tag",
            "positive",
            "negative",
            "class",
            "time_bin_center",
            "time_bin",
        ]
        by = [c for c in preferred if c in df.columns]
        if not by:
            return
        # Stable sort keeps deterministic overlays when ties exist.
        df.sort_values(by=by, kind="mergesort", inplace=True, ignore_index=True)

    def _resolve_tag_for_add(
        self,
        metric: str,
        *,
        tag: Optional[str],
        overwrite: bool,
        incoming: pd.DataFrame,
    ) -> str:
        """Resolve the effective tag for this add() call."""
        if "tag" in incoming.columns:
            unique_incoming = sorted(incoming["tag"].dropna().astype(str).unique().tolist())
            if tag is None:
                if len(unique_incoming) == 1:
                    return unique_incoming[0]
                raise ValueError("Incoming results contain multiple tags; pass tag=... explicitly.")
            if len(unique_incoming) == 0:
                return str(tag)
            if unique_incoming != [str(tag)]:
                raise ValueError(
                    f"Incoming tag column {unique_incoming} does not match tag={tag!r}. "
                    "Pass tag=None to accept incoming tag, or pass a matching tag."
                )
            return str(tag)

        if tag is not None:
            return str(tag)

        if overwrite:
            return self._TAG_DEFAULT

        # Auto-tag for append mode
        existing = []
        if not self.comparisons.empty and "metric" in self.comparisons.columns and "tag" in self.comparisons.columns:
            existing = (
                self.comparisons.loc[self.comparisons["metric"].astype(str) == str(metric), "tag"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        n = 1
        while True:
            cand = f"run{n:03d}"
            if cand not in existing:
                return cand
            n += 1

    def _upsert_table(self, which: str, incoming: pd.DataFrame, *, metric: str, tag: str, overwrite: bool) -> None:
        if which not in {"comparisons", "null_summary"}:
            raise ValueError(f"Unknown table: {which}")

        base: Optional[pd.DataFrame]
        if which == "comparisons":
            base = self.comparisons
        else:
            base = self.null_summary

        if base is None or base.empty:
            if which == "comparisons":
                self.comparisons = incoming.copy()
            else:
                self.null_summary = incoming.copy()
            return

        base = base.copy()
        self._ensure_metric_and_tag(base)
        if overwrite:
            keep = ~((base["metric"].astype(str) == str(metric)) & (base["tag"].astype(str) == str(tag)))
            base = base.loc[keep].copy()

        base = pd.concat([base, incoming], ignore_index=True, sort=False)
        self._ensure_metric_and_tag(base)
        if which == "comparisons":
            self._coerce_canonical_columns(base)
        else:
            base["metric"] = base["metric"].astype(str)
            base["tag"] = base["tag"].astype(str)
        self._sort_inplace(base)

        if which == "comparisons":
            self.comparisons = base
        else:
            self.null_summary = base


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return ""
