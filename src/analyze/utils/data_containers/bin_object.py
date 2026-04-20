from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..binning import bins_in_time_window
from .reducers import (
    get_reducer,
    make_centered_reducer,
    make_group_centered_reducer,
    make_group_difference_reducer,
)
from .reports import SupportReport
from .specs import FeatureSpec, InputRef, LevelName, ReducerSpec


_RESERVED_COLUMNS = {
    "embryo_id",
    "bin_id",
    "bin_start",
    "bin_end",
    "bin_center_time",
    "n_frames",
    "bin_width_seconds",
}


def _as_tuple(value: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _infer_feature_specs(df: pd.DataFrame, embryo_id_col: str, time_col: str) -> list[FeatureSpec]:
    ignore = {embryo_id_col, time_col}
    numeric_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    feature_specs: list[FeatureSpec] = []
    vae_cols = [c for c in numeric_cols if "z_mu_b" in c]
    if vae_cols:
        for col in vae_cols:
            suffix = col.split("z_mu_b", 1)[-1].strip("_")
            feature_id = f"vae__{suffix}" if suffix else "vae"
            feature_specs.append(FeatureSpec(feature_id=feature_id, source_columns=(col,), within_bin_reducer="mean"))
    for col in numeric_cols:
        if col in vae_cols:
            continue
        feature_specs.append(FeatureSpec(feature_id=col, source_columns=(col,), within_bin_reducer="mean"))
    return feature_specs


def _aggregate_group(group: pd.DataFrame, spec: FeatureSpec) -> float:
    values = group.loc[:, list(spec.source_columns)]
    if spec.within_bin_reducer == "mean":
        return float(values.mean(numeric_only=True).mean()) if len(spec.source_columns) > 1 else float(values.iloc[:, 0].mean())
    if spec.within_bin_reducer == "median":
        return float(values.median(numeric_only=True).mean()) if len(spec.source_columns) > 1 else float(values.iloc[:, 0].median())
    if spec.within_bin_reducer == "sum":
        return float(values.sum(numeric_only=True).sum()) if len(spec.source_columns) > 1 else float(values.iloc[:, 0].sum())
    raise ValueError(f"Unsupported within-bin reducer {spec.within_bin_reducer!r} for {spec.feature_id!r}")


def _agg_name(within_bin_reducer: str) -> str:
    if within_bin_reducer in {"mean", "median", "sum", "min", "max"}:
        return within_bin_reducer
    raise ValueError(f"Unsupported within-bin reducer {within_bin_reducer!r}")


@dataclass
class LevelCollection:
    binned: pd.DataFrame
    raw: pd.DataFrame
    bin_meta: pd.DataFrame
    embryo_meta: pd.DataFrame
    cross_bin: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _level_columns(self, level: LevelName) -> list[str]:
        df = getattr(self, level)
        return [c for c in df.columns if c not in _RESERVED_COLUMNS]

    def inspect(self) -> str:
        parts = []
        for level in ["raw", "binned", "bin_meta", "embryo_meta", "cross_bin"]:
            df = getattr(self, level)
            grain = "(embryo_id, bin_id)" if level in {"raw", "binned", "bin_meta"} else "(embryo_id,)"
            parts.append(
                f"{level}: grain={grain}, rows={len(df)}, keys={self._level_columns(level)}"
            )
        text = "\n".join(parts)
        print(text)
        return text


@dataclass
class BinObject:
    levels: LevelCollection
    embryo_id_col: str = "embryo_id"
    time_col: str = "predicted_stage_hpf"
    bin_width: float = 2.0
    _feature_specs: list[FeatureSpec] = field(default_factory=list, repr=False)

    @classmethod
    def from_raw(
        cls,
        raw_df: pd.DataFrame,
        *,
        embryo_id_col: str = "embryo_id",
        time_col: str = "predicted_stage_hpf",
        bin_width: float = 2.0,
        feature_specs: list[FeatureSpec] | None = None,
        embryo_meta_cols: list[str] | None = None,
    ) -> "BinObject":
        if embryo_id_col not in raw_df.columns:
            raise ValueError(f"Missing embryo id column {embryo_id_col!r}")
        if time_col not in raw_df.columns:
            raise ValueError(f"Missing time column {time_col!r}")

        raw = raw_df.copy()
        raw["bin_id"] = np.floor(raw[time_col].astype(float) / float(bin_width)) * float(bin_width)
        raw["bin_start"] = raw["bin_id"]
        raw["bin_end"] = raw["bin_id"] + float(bin_width)
        raw["bin_center_time"] = raw["bin_id"] + float(bin_width) / 2.0
        raw["bin_width_seconds"] = float(bin_width)

        if feature_specs is None:
            feature_specs = _infer_feature_specs(raw_df, embryo_id_col=embryo_id_col, time_col=time_col)

        group_keys = [embryo_id_col, "bin_id"]
        binned = raw[group_keys].drop_duplicates().sort_values(group_keys).reset_index(drop=True)

        for spec in feature_specs:
            new_col = f"bin__{spec.feature_id}__{spec.within_bin_reducer}"
            agg_name = _agg_name(spec.within_bin_reducer)
            feature_group = raw.groupby(group_keys, as_index=False, sort=True)[list(spec.source_columns)].agg(agg_name)
            if len(spec.source_columns) == 1:
                feature_group = feature_group.rename(columns={spec.source_columns[0]: new_col})
            else:
                if spec.within_bin_reducer == "mean":
                    feature_group[new_col] = feature_group[list(spec.source_columns)].mean(axis=1)
                elif spec.within_bin_reducer == "median":
                    feature_group[new_col] = feature_group[list(spec.source_columns)].median(axis=1)
                elif spec.within_bin_reducer == "sum":
                    feature_group[new_col] = feature_group[list(spec.source_columns)].sum(axis=1)
                elif spec.within_bin_reducer == "min":
                    feature_group[new_col] = feature_group[list(spec.source_columns)].min(axis=1)
                elif spec.within_bin_reducer == "max":
                    feature_group[new_col] = feature_group[list(spec.source_columns)].max(axis=1)
                feature_group = feature_group[[embryo_id_col, "bin_id", new_col]]
            binned = binned.merge(feature_group[[embryo_id_col, "bin_id", new_col]], on=group_keys, how="left")

        bin_meta = (
            raw.groupby(group_keys, as_index=False, sort=True)
            .agg(
                n_frames=(time_col, "size"),
                bin_width_seconds=("bin_width_seconds", "first"),
                bin_center_time=("bin_center_time", "first"),
                bin_start=("bin_start", "first"),
                bin_end=("bin_end", "first"),
            )
            .sort_values(group_keys)
            .reset_index(drop=True)
        )

        binned = binned.merge(bin_meta, on=[embryo_id_col, "bin_id"], how="left")

        if embryo_meta_cols is None:
            ignore = {embryo_id_col, time_col, "bin_id", "bin_start", "bin_end", "bin_center_time", "bin_width_seconds"}
            embryo_meta_cols = [c for c in raw_df.columns if c not in ignore and c not in set().union(*(spec.source_columns for spec in feature_specs))]
        embryo_meta = raw[[embryo_id_col] + embryo_meta_cols].drop_duplicates(subset=[embryo_id_col]).copy()

        levels = LevelCollection(binned=binned, raw=raw, bin_meta=bin_meta, embryo_meta=embryo_meta, cross_bin=pd.DataFrame())
        return cls(levels=levels, embryo_id_col=embryo_id_col, time_col=time_col, bin_width=bin_width, _feature_specs=feature_specs)

    def _check_key_collision(self, key: str, overwrite: bool = False) -> None:
        if overwrite:
            return
        for level in ["raw", "binned", "bin_meta", "embryo_meta", "cross_bin"]:
            if key in self.levels._level_columns(level):
                raise ValueError(f"Key {key!r} already exists in level {level!r}")

    def _find_key_level(self, key: str) -> str | None:
        for level in ["raw", "binned", "bin_meta", "embryo_meta", "cross_bin"]:
            if key in self.levels._level_columns(level):
                return level
        return None

    def _level_grain_columns(self, level: str) -> list[str]:
        if level in {"raw", "binned", "bin_meta"}:
            return [self.embryo_id_col, "bin_id"]
        if level in {"embryo_meta", "cross_bin"}:
            return [self.embryo_id_col]
        raise ValueError(f"Unknown level {level!r}")

    def _validate_write_contract(self, *, level: str, key: str, overwrite: bool) -> None:
        owner = self._find_key_level(key)
        if owner is None:
            return
        if owner != level:
            raise ValueError(
                f"Key {key!r} already exists in level {owner!r}; cross-level overwrite is not allowed"
            )
        if not overwrite:
            raise ValueError(
                f"Key {key!r} already exists in level {level!r}. "
                "Set overwrite=True to replace it in-place."
            )

    def _upsert_level_features(
        self,
        *,
        level: str,
        values_df: pd.DataFrame,
        feature_cols: list[str],
        overwrite: bool,
    ) -> None:
        if values_df.empty:
            return
        grain_cols = self._level_grain_columns(level)
        missing_grain = [c for c in grain_cols if c not in values_df.columns]
        if missing_grain:
            raise KeyError(f"Incoming values are missing grain columns for level {level!r}: {missing_grain}")
        duplicates = values_df.duplicated(subset=grain_cols, keep=False)
        if duplicates.any():
            raise ValueError(f"Incoming values contain duplicate rows at level grain {grain_cols}")

        for key in feature_cols:
            self._validate_write_contract(level=level, key=key, overwrite=overwrite)

        incoming = values_df[grain_cols + feature_cols].copy()
        current = getattr(self.levels, level)

        if current.empty:
            setattr(self.levels, level, incoming)
            return

        base = current.copy()
        to_drop = [c for c in feature_cols if c in base.columns]
        if to_drop:
            base = base.drop(columns=to_drop)

        merged = base.merge(incoming, on=grain_cols, how="outer")
        setattr(self.levels, level, merged)

    def _resolve_reducer(self, reducer: str | ReducerSpec) -> ReducerSpec:
        return get_reducer(reducer)

    def _select_bins(self, time_window: tuple[float, float]) -> pd.DataFrame:
        meta = self.levels.bin_meta.drop_duplicates(subset=["bin_id"]).sort_values("bin_center_time").copy()
        mask = bins_in_time_window(meta["bin_start"].to_numpy(), meta["bin_end"].to_numpy(), time_window)
        return meta.loc[mask].copy()

    def _required_bins(self, *, bins_in_scope: int, reducer: ReducerSpec, bin_fract: float, min_bins: int | None) -> int:
        required_bins = max(reducer.math_min_bins, ceil(bin_fract * bins_in_scope), min_bins or 0)
        if required_bins > bins_in_scope:
            raise ValueError(
                f"required_bins={required_bins} exceeds bins_in_scope={bins_in_scope} for reducer {reducer.name!r}"
            )
        return required_bins

    def _resolve_consumed_inputs(self, group: pd.DataFrame, reducer: ReducerSpec, feature_key: str) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for ref in reducer.consumes:
            if ref.level == "binned":
                key = feature_key if ref.key == "target" else ref.key
                if key not in group.columns:
                    raise KeyError(f"Missing binned key {key!r}")
                resolved[ref.key] = group[key]
            elif ref.level == "bin_meta":
                if ref.key not in group.columns:
                    raise KeyError(f"Missing bin_meta key {ref.key!r}")
                resolved[ref.key] = group[ref.key]
            elif ref.level == "embryo_meta":
                meta_row = self.levels.embryo_meta.set_index(self.embryo_id_col).loc[group[self.embryo_id_col].iloc[0]]
                if ref.key not in meta_row.index:
                    raise KeyError(f"Missing embryo_meta key {ref.key!r}")
                resolved[ref.key] = pd.Series([meta_row[ref.key]] * len(group), index=group.index)
            else:
                raise ValueError(f"cross_bin reducers do not consume raw inputs in this implementation: {ref}")
        return resolved

    def _apply_reducer(self, group: pd.DataFrame, reducer: ReducerSpec, feature_key: str) -> dict[str, Any]:
        if reducer.func is None:
            raise ValueError(f"Reducer {reducer.name!r} has no callable implementation")
        resolved = self._resolve_consumed_inputs(group, reducer, feature_key)
        return reducer.func(group, resolved)

    def _cross_bin_output_name(self, feature_key: str, reducer_name: str, time_window: tuple[float, float], suffix: str | None = None) -> str:
        feature_id = feature_key.removeprefix("bin__")
        base = f"xbin__{feature_id}__{reducer_name}__{int(time_window[0])}_{int(time_window[1])}"
        if suffix:
            base += f"__{suffix}"
        return base

    def validate_reducer(self, reducer: str | ReducerSpec, feature_key: str, *, time_window: tuple[float, float]) -> None:
        reducer = self._resolve_reducer(reducer)
        if feature_key not in self.levels.binned.columns:
            raise KeyError(f"Unknown binned feature {feature_key!r}")
        bins = self._select_bins(time_window)
        self._required_bins(bins_in_scope=len(bins), reducer=reducer, bin_fract=1.0, min_bins=None)
        missing_inputs = []
        for ref in reducer.consumes:
            if ref.level == "binned" and ref.key != "target" and ref.key not in self.levels.binned.columns:
                missing_inputs.append(f"binned:{ref.key}")
            if ref.level == "bin_meta" and ref.key not in self.levels.bin_meta.columns:
                missing_inputs.append(f"bin_meta:{ref.key}")
            if ref.level == "embryo_meta" and ref.key not in self.levels.embryo_meta.columns:
                missing_inputs.append(f"embryo_meta:{ref.key}")
        if missing_inputs:
            raise KeyError(f"Reducer {reducer.name!r} is missing required inputs: {missing_inputs}")

    def summarize_cross_bin_by_group(
        self,
        *,
        features: str,
        reducer: str | ReducerSpec = "mean_equal_bin",
        time_window: tuple[float, float],
        group_key: str,
        bin_fract: float = 1.0,
        min_bins: int | None = None,
    ) -> pd.DataFrame:
        """Compute group-level summary from a cross-bin reduction result.

        Returns one row per group with the mean of the embryo-level reduced value.
        """
        meta_df, _ = self.cross_bin_reduce(
            features=features,
            reducer=reducer,
            time_window=time_window,
            bin_fract=bin_fract,
            min_bins=min_bins,
            overwrite=True,
        )
        if meta_df.empty:
            raise ValueError("No embryos retained; cannot compute group summary")
        value_cols = [c for c in meta_df.columns if c.startswith("xbin__")]
        if len(value_cols) != 1:
            raise ValueError(f"Expected exactly one xbin output column, found {value_cols}")
        value_col = value_cols[0]
        if group_key not in self.levels.embryo_meta.columns:
            raise KeyError(f"Unknown embryo_meta group key {group_key!r}")
        group_df = self.levels.embryo_meta[[self.embryo_id_col, group_key]].copy()
        merged = meta_df.merge(group_df, on=self.embryo_id_col, how="left")
        return (
            merged.groupby(group_key, dropna=False)[value_col]
            .mean()
            .reset_index()
            .rename(columns={value_col: "group_mean"})
        )

    def build_centered_reducer_from_group(
        self,
        *,
        features: str,
        time_window: tuple[float, float],
        group_key: str,
        reference_group: str,
        base_reducer: str | ReducerSpec = "mean_equal_bin",
        reducer_name: str | None = None,
        bin_fract: float = 1.0,
        min_bins: int | None = None,
    ) -> ReducerSpec:
        """Auto-create a centered reducer using one reference group baseline.

        Typical use: center all embryos by WT mean in the selected window.
        """
        summary = self.summarize_cross_bin_by_group(
            features=features,
            reducer=base_reducer,
            time_window=time_window,
            group_key=group_key,
            bin_fract=bin_fract,
            min_bins=min_bins,
        )
        idx = summary[group_key].astype(str) == str(reference_group)
        if not idx.any():
            raise KeyError(f"Reference group {reference_group!r} not found in retained cohort")
        baseline_value = float(summary.loc[idx, "group_mean"].iloc[0])
        name = reducer_name or f"centered__{get_reducer(base_reducer).name}__{group_key}_{reference_group}"
        return make_centered_reducer(
            name=name,
            base_reducer=base_reducer,
            baseline_value=baseline_value,
            register=True,
        )

    def build_group_centered_reducer(
        self,
        *,
        features: str,
        time_window: tuple[float, float],
        group_key: str,
        base_reducer: str | ReducerSpec = "mean_equal_bin",
        reducer_name: str | None = None,
        bin_fract: float = 1.0,
        min_bins: int | None = None,
    ) -> ReducerSpec:
        """Auto-create a group-centered reducer using per-group cohort means."""
        summary = self.summarize_cross_bin_by_group(
            features=features,
            reducer=base_reducer,
            time_window=time_window,
            group_key=group_key,
            bin_fract=bin_fract,
            min_bins=min_bins,
        )
        baseline_by_group = {
            str(row[group_key]): float(row["group_mean"])
            for _, row in summary.iterrows()
        }
        name = reducer_name or f"group_centered__{get_reducer(base_reducer).name}__{group_key}"
        return make_group_centered_reducer(
            name=name,
            group_key=group_key,
            baseline_by_group=baseline_by_group,
            base_reducer=base_reducer,
            register=True,
        )

    def build_group_difference_reducer(
        self,
        *,
        features: str,
        time_window: tuple[float, float],
        group_key: str,
        reference_group: str,
        base_reducer: str | ReducerSpec = "mean_equal_bin",
        reducer_name: str | None = None,
        bin_fract: float = 1.0,
        min_bins: int | None = None,
    ) -> ReducerSpec:
        """Auto-create reducer returning value - mean(reference_group) for all groups."""
        summary = self.summarize_cross_bin_by_group(
            features=features,
            reducer=base_reducer,
            time_window=time_window,
            group_key=group_key,
            bin_fract=bin_fract,
            min_bins=min_bins,
        )
        mean_by_group = {
            str(row[group_key]): float(row["group_mean"])
            for _, row in summary.iterrows()
        }
        name = reducer_name or f"group_diff__{get_reducer(base_reducer).name}__vs_{group_key}_{reference_group}"
        return make_group_difference_reducer(
            name=name,
            group_key=group_key,
            reference_group=str(reference_group),
            mean_by_group=mean_by_group,
            base_reducer=base_reducer,
            register=True,
        )

    def cross_bin_reduce(
        self,
        *,
        features: str,
        reducer: str | ReducerSpec,
        time_window: tuple[float, float],
        bin_fract: float = 1.0,
        min_bins: int | None = None,
        verbose: bool = False,
        label_col: str | None = None,
        overwrite: bool = False,
    ) -> tuple[pd.DataFrame, SupportReport]:
        reducer = self._resolve_reducer(reducer)
        if features not in self.levels.binned.columns:
            raise KeyError(f"Unknown binned feature {features!r}")

        bins = self._select_bins(time_window)
        required_bins = self._required_bins(bins_in_scope=len(bins), reducer=reducer, bin_fract=bin_fract, min_bins=min_bins)
        selected_bin_ids = tuple(bins["bin_id"].tolist())
        selected_bin_centers = tuple(bins["bin_center_time"].astype(float).tolist())

        working = self.levels.binned.merge(bins[["bin_id"]], on="bin_id", how="inner")
        working = working.sort_values([self.embryo_id_col, "bin_center_time"])
        present = working[working[features].notna()].groupby(self.embryo_id_col)["bin_id"].nunique()
        kept = present[present >= required_bins].index
        kept_df = working[working[self.embryo_id_col].isin(kept)].copy()

        output_name = self._cross_bin_output_name(features, reducer.name, time_window)

        rows: list[dict[str, Any]] = []
        dropped = tuple(sorted(set(working[self.embryo_id_col].astype(str).unique()) - set(map(str, kept))))
        drop_reasons = {eid: f"bins_present<{required_bins}" for eid in dropped}

        for embryo_id, group in kept_df.groupby(self.embryo_id_col, sort=True):
            result = self._apply_reducer(group, reducer, features)
            row = {self.embryo_id_col: embryo_id, output_name: result["value"]}
            rows.append(row)

        meta_df = pd.DataFrame(rows)
        if not meta_df.empty:
            self._upsert_level_features(
                level="cross_bin",
                values_df=meta_df,
                feature_cols=[output_name],
                overwrite=overwrite,
            )

        class_drop_warning = None
        if label_col and label_col in self.levels.embryo_meta.columns and not meta_df.empty:
            embryo_labels = self.levels.embryo_meta.set_index(self.embryo_id_col)[label_col]
            kept_labels = embryo_labels.loc[embryo_labels.index.intersection(pd.Index(meta_df[self.embryo_id_col]))]
            drop_rates = {}
            for label in embryo_labels.dropna().astype(str).unique():
                total = int((embryo_labels.astype(str) == label).sum())
                kept_n = int((kept_labels.astype(str) == label).sum())
                drop_rates[label] = 1.0 - (kept_n / total if total else 0.0)
            if drop_rates and (max(drop_rates.values()) - min(drop_rates.values()) > 0.15):
                class_drop_warning = "Class-imbalanced exclusions detected"

        report = SupportReport(
            time_window=time_window,
            selected_bin_ids=selected_bin_ids,
            selected_bin_centers=selected_bin_centers,
            bins_in_scope=len(bins),
            required_bins=required_bins,
            bin_fract=bin_fract,
            min_bins=min_bins,
            math_min_bins=reducer.math_min_bins,
            kept_embryos=tuple(meta_df[self.embryo_id_col].astype(str).tolist()) if not meta_df.empty else tuple(),
            dropped_embryos=dropped,
            drop_reasons=drop_reasons,
            confounding_warning=class_drop_warning,
            reducer_name=reducer.name,
            consumed_inputs=tuple(f"{ref.level}:{ref.key}" for ref in reducer.consumes),
            provenance={
                "feature": features,
                "output_name": output_name,
                "reducer_version": reducer.version,
            },
        )

        if verbose:
            print(report.as_dict())
        return meta_df, report

    def cross_bin_reduce_batch(
        self,
        *,
        features: list[str],
        reducer: str | ReducerSpec,
        time_window: tuple[float, float],
        bin_fract: float = 1.0,
        min_bins: int | None = None,
        verbose: bool = False,
        label_col: str | None = None,
        overwrite: bool = False,
    ) -> tuple[pd.DataFrame, SupportReport]:
        reducer = self._resolve_reducer(reducer)
        bins = self._select_bins(time_window)
        required_bins = self._required_bins(bins_in_scope=len(bins), reducer=reducer, bin_fract=bin_fract, min_bins=min_bins)
        selected_bin_ids = tuple(bins["bin_id"].tolist())
        selected_bin_centers = tuple(bins["bin_center_time"].astype(float).tolist())

        working = self.levels.binned.merge(bins[["bin_id"]], on="bin_id", how="inner")
        working = working.sort_values([self.embryo_id_col, "bin_center_time"])
        valid_mask = working[features].notna().all(axis=1)
        present = working[valid_mask].groupby(self.embryo_id_col)["bin_id"].nunique()
        kept = present[present >= required_bins].index
        kept_df = working[working[self.embryo_id_col].isin(kept)].copy()

        rows: list[dict[str, Any]] = []
        output_names = {f: self._cross_bin_output_name(f, reducer.name, time_window) for f in features}

        dropped = tuple(sorted(set(working[self.embryo_id_col].astype(str).unique()) - set(map(str, kept))))
        drop_reasons = {eid: f"bins_present<{required_bins}" for eid in dropped}
        for embryo_id, group in kept_df.groupby(self.embryo_id_col, sort=True):
            row = {self.embryo_id_col: embryo_id}
            for feature in features:
                result = self._apply_reducer(group, reducer, feature)
                row[output_names[feature]] = result["value"]
            rows.append(row)

        meta_df = pd.DataFrame(rows)
        if not meta_df.empty:
            self._upsert_level_features(
                level="cross_bin",
                values_df=meta_df,
                feature_cols=list(output_names.values()),
                overwrite=overwrite,
            )

        report = SupportReport(
            time_window=time_window,
            selected_bin_ids=selected_bin_ids,
            selected_bin_centers=selected_bin_centers,
            bins_in_scope=len(bins),
            required_bins=required_bins,
            bin_fract=bin_fract,
            min_bins=min_bins,
            math_min_bins=reducer.math_min_bins,
            kept_embryos=tuple(meta_df[self.embryo_id_col].astype(str).tolist()) if not meta_df.empty else tuple(),
            dropped_embryos=dropped,
            drop_reasons=drop_reasons,
            reducer_name=reducer.name,
            consumed_inputs=tuple(f"{ref.level}:{ref.key}" for ref in reducer.consumes),
            provenance={
                "features": tuple(features),
                "output_names": output_names,
                "reducer_version": reducer.version,
            },
        )
        if verbose:
            print(report.as_dict())
        return meta_df, report
