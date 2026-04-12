import json

import numpy as np
import pandas as pd

from analyze.classification import ClassificationResults


def _make_alias_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"positive_class": "Het", "time_bin_start": 24.0, "time_bin_end": 28.0, "auroc_observed": 0.61},
            {"positive_class": "Het", "time_bin_start": 28.0, "time_bin_end": 32.0, "auroc_observed": 0.64},
            {"positive_class": "Het", "time_bin_start": 32.0, "time_bin_end": 36.0, "auroc_observed": 0.66},
        ]
    )


def test_add_coerces_aliases_and_types():
    res = ClassificationResults()
    res.add("curvature", _make_alias_df(), metadata={"bin_width": 4.0})

    df = res.comparisons
    assert set(["metric", "tag", "positive", "time_bin_center", "auroc_obs"]).issubset(df.columns)
    assert set(df["metric"].unique()) == {"curvature"}
    assert set(df["tag"].unique()) == {"default"}
    assert set(df["positive"].unique()) == {"Het"}
    assert pd.api.types.is_float_dtype(df["time_bin_center"])
    assert pd.api.types.is_float_dtype(df["auroc_obs"])
    assert np.isclose(df["time_bin_center"].iloc[0], 26.0)


def test_add_overwrite_replaces_rows_for_metric_tag():
    res = ClassificationResults()
    res.add("m1", pd.DataFrame([{"positive": "A", "time_bin_center": 10.0, "auroc_obs": 0.6}]))
    res.add("m1", pd.DataFrame([{"positive": "A", "time_bin_center": 10.0, "auroc_obs": 0.9}]), overwrite=True)
    df = res.comparisons
    assert len(df) == 1
    assert float(df["auroc_obs"].iloc[0]) == 0.9


def test_load_injects_default_tag_for_v1_tables(tmp_path):
    out = tmp_path / "saved"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [{"metric": "m1", "positive": "A", "time_bin_center": 10.0, "auroc_obs": 0.6}],
    )
    df.to_parquet(out / "comparisons.parquet", index=False)
    (out / "metadata.json").write_text(json.dumps({"schema_version": 1, "run_metadata": {"m1": {"x": 1}}}) + "\n")

    loaded = ClassificationResults.load(out)
    assert "tag" in loaded.comparisons.columns
    assert set(loaded.comparisons["tag"].unique()) == {"default"}
    assert loaded.run_metadata["m1"]["default"]["x"] == 1


def test_save_load_roundtrip_writes_debug_metadata(tmp_path):
    res = ClassificationResults()
    res.add("m1", pd.DataFrame([{"positive": "A", "time_bin_center": 10.0, "auroc_obs": 0.6}]), metadata={"x": 1})

    out_dir = res.save(tmp_path / "bundle")
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["schema_version"] == 2
    assert "created_at" in meta
    assert "python_version" in meta
    assert "git_commit" in meta
    assert "comparisons_columns" in meta
    assert meta["run_metadata"]["m1"]["default"]["x"] == 1

    loaded = ClassificationResults.load(out_dir)
    assert set(loaded.metrics) == {"m1"}
    assert set(loaded.tags) == {"default"}


def test_add_sorts_deterministically_by_time():
    res = ClassificationResults()
    df = pd.DataFrame(
        [
            {"positive": "A", "time_bin_center": 14.0, "auroc_obs": 0.7},
            {"positive": "A", "time_bin_center": 10.0, "auroc_obs": 0.6},
            {"positive": "A", "time_bin_center": 12.0, "auroc_obs": 0.65},
        ]
    )
    res.add("m1", df)
    times = res.comparisons["time_bin_center"].to_numpy(dtype=float)
    assert np.all(np.diff(times) >= 0)


def test_subset_filters_time_and_metric():
    res = ClassificationResults()
    res.add("m1", pd.DataFrame([{"positive": "A", "time_bin_center": 10.0, "auroc_obs": 0.6}]))
    res.add("m2", pd.DataFrame([{"positive": "A", "time_bin_center": 30.0, "auroc_obs": 0.9}]))

    sub = res.subset(metric="m2", time_range=(24.0, 40.0))
    assert set(sub.metrics) == {"m2"}
    assert len(sub.comparisons) == 1
    assert float(sub.comparisons["time_bin_center"].iloc[0]) == 30.0

