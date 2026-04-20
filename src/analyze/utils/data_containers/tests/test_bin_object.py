from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pandas as pd
import pytest

from analyze.utils.data_containers import BinObject, InputRef, ReducerSpec


def _toy_raw() -> pd.DataFrame:
    rows = []
    for embryo_id in ["e1", "e2", "e3"]:
        for t, value in [(30.0, 1.0), (31.0, 2.0), (33.0, 3.0), (35.0, 4.0)]:
            rows.append(
                {
                    "embryo_id": embryo_id,
                    "predicted_stage_hpf": t,
                    "z_mu_b_0": value + (0.1 if embryo_id == "e2" else 0.0),
                    "z_mu_b_1": value * 2,
                    "genotype": "wt" if embryo_id != "e3" else "mut",
                }
            )
    raw = pd.DataFrame(rows)
    raw = raw[~((raw["embryo_id"] == "e3") & (raw["predicted_stage_hpf"] >= 33.0))].copy()
    return raw


def test_from_raw_materializes_mean_vae_bins() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    assert "bin__vae__0__mean" in bo.levels.binned.columns
    assert "bin__vae__1__mean" in bo.levels.binned.columns
    first_bin = bo.levels.binned[(bo.levels.binned["embryo_id"] == "e1") & (bo.levels.binned["bin_id"] == 30.0)]
    assert pytest.approx(first_bin["bin__vae__0__mean"].iloc[0]) == 1.5


def test_cross_bin_reduce_returns_embryo_level_output_and_report() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    meta_df, report = bo.cross_bin_reduce(
        features="bin__vae__0__mean",
        reducer="max",
        time_window=(30.0, 35.0),
        bin_fract=0.5,
    )
    assert "xbin__vae__0__mean__max__30_35" in meta_df.columns
    assert set(meta_df["embryo_id"]) == {"e1", "e2"}
    assert "e3" in report.dropped_embryos
    assert report.bins_in_scope > 0
    assert report.required_bins >= 1


def test_batch_reduce_uses_shared_cohort() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    meta_df, report = bo.cross_bin_reduce_batch(
        features=["bin__vae__0__mean", "bin__vae__1__mean"],
        reducer="mean_equal_bin",
        time_window=(30.0, 35.0),
        bin_fract=0.5,
    )
    assert {"embryo_id", "xbin__vae__0__mean__mean_equal_bin__30_35", "xbin__vae__1__mean__mean_equal_bin__30_35"}.issubset(meta_df.columns)
    assert report.reducer_name == "mean_equal_bin"


def test_validate_reducer_fails_on_missing_feature() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    missing = ReducerSpec(
        name="needs_missing_bin_meta",
        consumes=(InputRef("bin_meta", "missing_column"),),
        output_schema=("value",),
        math_min_bins=1,
        func=lambda group, resolved: {"value": 0.0},
    )
    with pytest.raises(KeyError):
        bo.validate_reducer(missing, "bin__vae__0__mean", time_window=(30.0, 35.0))


def test_level_inspect_runs() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    text = bo.levels.inspect()
    assert "binned" in text


def test_build_centered_reducer_from_group() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    reducer = bo.build_centered_reducer_from_group(
        features="bin__vae__0__mean",
        time_window=(30.0, 35.0),
        group_key="genotype",
        reference_group="wt",
        base_reducer="mean_equal_bin",
    )
    meta_df, report = bo.cross_bin_reduce(
        features="bin__vae__0__mean",
        reducer=reducer.name,
        time_window=(30.0, 35.0),
        bin_fract=0.5,
        overwrite=True,
    )
    out_col = [c for c in meta_df.columns if c.startswith("xbin__")][0]
    wt_rows = meta_df.merge(bo.levels.embryo_meta[["embryo_id", "genotype"]], on="embryo_id", how="left")
    wt_mean = wt_rows.loc[wt_rows["genotype"] == "wt", out_col].mean()
    assert abs(float(wt_mean)) < 1e-8
    assert report.reducer_name == reducer.name


def test_build_group_difference_reducer_vs_wt() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    reducer = bo.build_group_difference_reducer(
        features="bin__vae__0__mean",
        time_window=(30.0, 35.0),
        group_key="genotype",
        reference_group="wt",
        base_reducer="mean_equal_bin",
    )
    meta_df, _ = bo.cross_bin_reduce(
        features="bin__vae__0__mean",
        reducer=reducer.name,
        time_window=(30.0, 35.0),
        bin_fract=0.5,
        overwrite=True,
    )
    out_col = [c for c in meta_df.columns if c.startswith("xbin__")][0]
    wt_rows = meta_df.merge(bo.levels.embryo_meta[["embryo_id", "genotype"]], on="embryo_id", how="left")
    wt_mean = wt_rows.loc[wt_rows["genotype"] == "wt", out_col].mean()
    assert abs(float(wt_mean)) < 1e-8


def test_overwrite_contract_blocks_duplicate_key_by_default() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    bo.cross_bin_reduce(
        features="bin__vae__0__mean",
        reducer="max",
        time_window=(30.0, 35.0),
        bin_fract=0.5,
    )
    with pytest.raises(ValueError):
        bo.cross_bin_reduce(
            features="bin__vae__0__mean",
            reducer="max",
            time_window=(30.0, 35.0),
            bin_fract=0.5,
        )


def test_overwrite_contract_allows_in_place_replace_when_true() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    first_df, _ = bo.cross_bin_reduce(
        features="bin__vae__0__mean",
        reducer="max",
        time_window=(30.0, 35.0),
        bin_fract=0.5,
    )
    key = [c for c in first_df.columns if c.startswith("xbin__")][0]

    second_df, _ = bo.cross_bin_reduce(
        features="bin__vae__0__mean",
        reducer="max",
        time_window=(30.0, 35.0),
        bin_fract=0.5,
        overwrite=True,
    )
    assert key in bo.levels.cross_bin.columns
    assert key in first_df.columns
    assert set(second_df["embryo_id"]) == {"e1", "e2"}


def test_overwrite_contract_disallows_cross_level_takeover() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    collision_key = "xbin__vae__0__mean__max__30_35"
    bo.levels.binned[collision_key] = 0.0
    with pytest.raises(ValueError):
        bo.cross_bin_reduce(
            features="bin__vae__0__mean",
            reducer="max",
            time_window=(30.0, 35.0),
            bin_fract=0.5,
            overwrite=True,
        )


def test_add_feature_accepts_series_for_cross_bin() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    embryo_ids = bo.levels.embryo_meta["embryo_id"].copy()
    values = pd.Series([0.1, 0.2, 0.3], index=embryo_ids, name="tmp")
    values.index.name = "embryo_id"

    bo.add_feature(level="cross_bin", values=values, key="xbin__toy__score", overwrite=False)
    assert "xbin__toy__score" in bo.levels.cross_bin.columns
    assert set(bo.levels.cross_bin["embryo_id"]) == set(embryo_ids)


def test_add_feature_accepts_dataframe_for_binned() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    subset = bo.levels.binned[["embryo_id", "bin_id"]].copy()
    subset["tmp"] = 1.0

    bo.add_feature(level="binned", values=subset, key="bin__toy__constant", overwrite=False)
    assert "bin__toy__constant" in bo.levels.binned.columns
    assert bo.levels.binned["bin__toy__constant"].notna().all()


def test_add_feature_enforces_overwrite_contract() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    values = bo.levels.embryo_meta[["embryo_id"]].copy()
    values["score"] = 1.0
    bo.add_feature(level="cross_bin", values=values, key="xbin__toy__score", overwrite=False)

    with pytest.raises(ValueError):
        bo.add_feature(level="cross_bin", values=values, key="xbin__toy__score", overwrite=False)

    values2 = bo.levels.embryo_meta[["embryo_id"]].copy()
    values2["score"] = 2.0
    bo.add_feature(level="cross_bin", values=values2, key="xbin__toy__score", overwrite=True)
    assert set(bo.levels.cross_bin["xbin__toy__score"].dropna().unique()) == {2.0}


def test_add_feature_rejects_wrong_grain() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    bad = pd.DataFrame({"embryo_id": ["e1", "e2"], "value": [1.0, 2.0]})
    with pytest.raises(KeyError):
        bo.add_feature(level="binned", values=bad, key="bin__bad", overwrite=False)
