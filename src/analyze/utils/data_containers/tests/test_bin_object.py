from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pandas as pd
import pytest

from analyze.utils.data_containers import BinObject


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


def test_add_feature_accepts_dataframe_for_binned() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    subset = bo.levels.binned[["embryo_id", "bin_id"]].copy()
    subset["bin__toy__constant"] = 1.0

    bo.add_feature(level="binned", values=subset, key="bin__toy__constant", overwrite=False)
    assert "bin__toy__constant" in bo.levels.binned.columns
    assert bo.levels.binned["bin__toy__constant"].notna().all()


def test_add_feature_accepts_series_for_cross_bin() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    values = pd.Series([0.1, 0.2, 0.3], index=pd.Index(["e1", "e2", "e3"], name="embryo_id"), name="xbin__toy__score")

    bo.add_feature(level="cross_bin", values=values, key="xbin__toy__score", overwrite=False)
    assert "xbin__toy__score" in bo.levels.cross_bin.columns
    assert set(bo.levels.cross_bin["embryo_id"]) == {"e1", "e2", "e3"}


def test_add_feature_enforces_overwrite_contract() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    values = pd.Series([0.1, 0.2, 0.3], index=pd.Index(["e1", "e2", "e3"], name="embryo_id"), name="xbin__toy__score")

    bo.add_feature(level="cross_bin", values=values, key="xbin__toy__score", overwrite=False)

    with pytest.raises(ValueError):
        bo.add_feature(level="cross_bin", values=values, key="xbin__toy__score", overwrite=False)

    updated = pd.Series([1.1, 1.2, 1.3], index=pd.Index(["e1", "e2", "e3"], name="embryo_id"), name="xbin__toy__score")
    bo.add_feature(level="cross_bin", values=updated, key="xbin__toy__score", overwrite=True)
    assert set(bo.levels.cross_bin["xbin__toy__score"].dropna().unique()) == {1.1, 1.2, 1.3}


def test_add_feature_rejects_wrong_grain() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    bad = pd.DataFrame({"embryo_id": ["e1", "e2"], "value": [1.0, 2.0]})
    with pytest.raises(KeyError):
        bo.add_feature(level="binned", values=bad, key="bin__bad", overwrite=False)


def test_add_feature_allows_subset_and_fills_missing_with_nan() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    partial = pd.Series([0.1, 0.2], index=pd.Index(["e1", "e2"], name="embryo_id"))
    bo.add_feature(level="cross_bin", values=partial, key="xbin__partial", overwrite=False)
    col = bo.levels.cross_bin.set_index("embryo_id")["xbin__partial"]
    assert col.loc["e1"] == 0.1
    assert col.loc["e2"] == 0.2
    assert pd.isna(col.loc["e3"])


def test_add_feature_rejects_unknown_grain_rows() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    bogus = pd.Series([0.1], index=pd.Index(["e_nonexistent"], name="embryo_id"))
    with pytest.raises(ValueError, match="unknown grain"):
        bo.add_feature(level="cross_bin", values=bogus, key="xbin__bogus")


def test_add_feature_rejects_raw_level() -> None:
    bo = BinObject.from_raw(_toy_raw(), bin_width=2.0)
    series = pd.Series([0.1, 0.2, 0.3], index=pd.Index(["e1", "e2", "e3"], name="embryo_id"))
    with pytest.raises(ValueError, match="raw is read-only"):
        bo.add_feature(level="raw", values=series, key="bad")


def test_from_raw_rejects_varying_embryo_meta() -> None:
    raw = _toy_raw()
    mask = (raw["embryo_id"] == "e1") & (raw["predicted_stage_hpf"] == 31.0)
    raw.loc[mask, "genotype"] = "mut"
    with pytest.raises(ValueError, match="embryo_meta columns vary"):
        BinObject.from_raw(raw, bin_width=2.0)
