from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .specs import FeatureSpec, LevelName


_RESERVED_COLUMNS = {
    "embryo_id",
    "bin_id",
    "bin_start",
    "bin_end",
    "bin_center_time",
    "n_frames",
    "bin_width_seconds",
}


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
        if embryo_meta_cols:
            nunique = raw.groupby(embryo_id_col)[embryo_meta_cols].nunique(dropna=False)
            inconsistent = nunique[(nunique > 1).any(axis=1)]
            if not inconsistent.empty:
                offending = {}
                for eid, row in inconsistent.iterrows():
                    bad_cols = [c for c, n in row.items() if n > 1]
                    offending[str(eid)] = bad_cols
                raise ValueError(
                    f"embryo_meta columns vary within embryo_id; each must be constant per embryo. "
                    f"Offending embryo_id -> columns: {offending}"
                )
        embryo_meta = raw[[embryo_id_col] + embryo_meta_cols].drop_duplicates(subset=[embryo_id_col]).copy()

        cross_bin = embryo_meta[[embryo_id_col]].copy()
        levels = LevelCollection(binned=binned, raw=raw, bin_meta=bin_meta, embryo_meta=embryo_meta, cross_bin=cross_bin)
        return cls(levels=levels, embryo_id_col=embryo_id_col, time_col=time_col, bin_width=bin_width, _feature_specs=feature_specs)

    def _find_key_level(self, key: str) -> str | None:
        for level in ["raw", "binned", "bin_meta", "embryo_meta", "cross_bin"]:
            if key in self.levels._level_columns(level):
                return level
        return None

    def _normalize_level(self, level: str | LevelName) -> str:
        level_name = str(level)
        valid = {"binned", "bin_meta", "embryo_meta", "cross_bin"}
        if level_name not in valid:
            raise ValueError(
                f"Cannot write to level {level!r}; expected one of {sorted(valid)} (raw is read-only)"
            )
        return level_name

    def _level_grain_columns(self, level: str) -> list[str]:
        if level in {"raw", "binned", "bin_meta"}:
            return [self.embryo_id_col, "bin_id"]
        if level in {"embryo_meta", "cross_bin"}:
            return [self.embryo_id_col]
        raise ValueError(f"Unknown level {level!r}")

    def _coerce_feature_values(self, level: str, values: pd.Series | pd.DataFrame, key: str) -> pd.DataFrame:
        grain_cols = self._level_grain_columns(level)

        if isinstance(values, pd.Series):
            series = values.rename(key)
            if len(grain_cols) == 1:
                if series.index.name != grain_cols[0]:
                    raise KeyError(f"Series index must be named {grain_cols[0]!r} for level {level!r}")
            else:
                if not isinstance(series.index, pd.MultiIndex) or list(series.index.names) != grain_cols:
                    raise KeyError(f"Series index must be a MultiIndex named {grain_cols} for level {level!r}")
            return series.reset_index()[grain_cols + [key]].copy()

        if isinstance(values, pd.DataFrame):
            incoming = values.copy()
            if all(c in incoming.columns for c in grain_cols):
                pass
            elif len(grain_cols) == 1 and incoming.index.name == grain_cols[0]:
                incoming = incoming.reset_index()
            elif isinstance(incoming.index, pd.MultiIndex) and list(incoming.index.names) == grain_cols:
                incoming = incoming.reset_index()
            else:
                raise KeyError(f"DataFrame must include grain columns {grain_cols} as columns or index names")

            if key not in incoming.columns:
                value_cols = [c for c in incoming.columns if c not in grain_cols]
                if len(value_cols) != 1:
                    raise ValueError(
                        f"DataFrame values must contain exactly one feature column when key {key!r} is absent"
                    )
                incoming = incoming.rename(columns={value_cols[0]: key})
            return incoming[grain_cols + [key]].copy()

        raise TypeError("values must be a pandas Series or DataFrame")

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

        current = getattr(self.levels, level)
        if not current.empty:
            current_keys = set(map(tuple, current[grain_cols].itertuples(index=False, name=None)))
            incoming_keys = set(map(tuple, values_df[grain_cols].itertuples(index=False, name=None)))
            extra = incoming_keys - current_keys
            if extra:
                raise ValueError(
                    f"Incoming values reference unknown grain rows for level {level!r}: {sorted(extra)}"
                )

        for key in feature_cols:
            self._validate_write_contract(level=level, key=key, overwrite=overwrite)

        incoming = values_df[grain_cols + feature_cols].copy()
        if current.empty:
            setattr(self.levels, level, incoming)
            return

        base = current.copy()
        to_drop = [c for c in feature_cols if c in base.columns]
        if to_drop:
            base = base.drop(columns=to_drop)

        merged = base.merge(incoming, on=grain_cols, how="left", validate="one_to_one")
        setattr(self.levels, level, merged)

    def add_feature(
        self,
        *,
        level: str | LevelName,
        values: pd.Series | pd.DataFrame,
        key: str,
        overwrite: bool = False,
    ) -> None:
        """Add one calculated feature to a target level with strict grain validation.

        Parameters
        ----------
        level
            Target level name (e.g., "binned", "cross_bin").
        values
            Feature values as either:
              - pd.Series indexed by the target level grain, or
              - single-column pd.DataFrame containing grain columns (or grain index).
        key
            Output feature column name.
        overwrite
            Whether replacing an existing key in the same level is allowed.
        """
        level_name = self._normalize_level(level)
        incoming = self._coerce_feature_values(level_name, values, key)

        self._upsert_level_features(
            level=level_name,
            values_df=incoming,
            feature_cols=[key],
            overwrite=overwrite,
        )
