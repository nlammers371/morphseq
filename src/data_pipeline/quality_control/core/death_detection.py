from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from data_pipeline.quality_control.config import get_qc_defaults
from ._shared import align_to_universe, assert_no_duplicate_columns


def validate_death_persistence(
    embryo_data: pd.DataFrame,
    inflection_time: float,
    threshold: float = 0.80,
    dead_fraction_threshold: float = 0.90,
) -> Tuple[bool, Dict[str, Any]]:
    post_inflection = embryo_data[embryo_data["time_int"] > inflection_time]
    if len(post_inflection) == 0:
        return False, {"post_count": 0, "dead_count": 0, "dead_fraction": 0.0}

    dead_count = int((post_inflection["fraction_alive"].astype(float) <= dead_fraction_threshold).sum())
    total_count = int(len(post_inflection))
    dead_fraction = dead_count / total_count
    return dead_fraction >= threshold, {
        "post_count": total_count,
        "dead_count": dead_count,
        "dead_fraction": dead_fraction,
        "threshold": threshold,
    }


def find_inflection_candidates(
    embryo_data: pd.DataFrame,
    min_decline_rate: float = 0.05,
) -> List[Tuple[float, float]]:
    if len(embryo_data) < 3:
        return []
    data = embryo_data.sort_values("time_int").copy()
    if "fraction_alive" not in data.columns or "time_int" not in data.columns:
        return []
    times = data["time_int"].to_numpy()
    fractions = data["fraction_alive"].to_numpy(dtype=float)
    if len(times) < 3 or np.all(np.isnan(fractions)):
        return []
    if len(fractions) >= 5:
        window = min(5, len(fractions))
        if window % 2 == 0:
            window -= 1
        fractions_smooth = savgol_filter(fractions, window_length=max(window, 3), polyorder=min(2, max(window - 1, 1)))
    else:
        fractions_smooth = fractions
    dt = np.diff(times)
    rates = np.diff(fractions_smooth) / dt
    candidates = []
    for index, is_declining in enumerate(rates < -min_decline_rate):
        if is_declining:
            candidates.append((float(times[index]), float(rates[index])))
    return candidates


def detect_persistent_death_inflection(
    embryo_data: pd.DataFrame,
    persistence_threshold: float | None = None,
    min_decline_rate: float | None = None,
    dead_fraction_threshold: float | None = None,
) -> Optional[Dict[str, Any]]:
    defaults = get_qc_defaults("death_detection")
    if persistence_threshold is None:
        persistence_threshold = float(defaults["persistence_threshold"])
    if min_decline_rate is None:
        min_decline_rate = float(defaults["decline_rate_threshold"])
    if dead_fraction_threshold is None:
        dead_fraction_threshold = float(defaults["dead_fraction_threshold"])

    current_data = embryo_data.copy()
    candidates_tested: list[dict[str, Any]] = []
    while len(current_data) >= 3:
        candidates = find_inflection_candidates(current_data, min_decline_rate)
        if not candidates:
            break
        earliest_time, earliest_rate = candidates[0]
        is_persistent, stats = validate_death_persistence(
            embryo_data,
            earliest_time,
            persistence_threshold,
            dead_fraction_threshold,
        )
        candidates_tested.append(
            {
                "time": earliest_time,
                "rate": earliest_rate,
                "is_persistent": is_persistent,
                "stats": stats,
            }
        )
        if is_persistent:
            return {
                "inflection_time": earliest_time,
                "persistence_stats": stats,
                "candidates_tested": candidates_tested,
            }
        current_data = current_data[current_data["time_int"] > earliest_time]
    return None


def compute_death_detection_flags(
    fraction_alive_df: pd.DataFrame,
    snip_universe_df: pd.DataFrame,
    *,
    persistence_threshold: float | None = None,
    lead_time_hr: float | None = None,
    decline_rate_threshold: float | None = None,
    dead_fraction_threshold: float | None = None,
) -> pd.DataFrame:
    required = {"snip_id", "embryo_id", "time_int", "fraction_alive"}
    missing = sorted(required - set(fraction_alive_df.columns))
    if missing:
        raise ValueError(f"death_detection: missing required columns {missing}")
    assert_no_duplicate_columns(fraction_alive_df, "death_detection input")

    defaults = get_qc_defaults("death_detection")
    if persistence_threshold is None:
        persistence_threshold = float(defaults["persistence_threshold"])
    if lead_time_hr is None:
        lead_time_hr = float(defaults["lead_time_hr"])
    if decline_rate_threshold is None:
        decline_rate_threshold = float(defaults["decline_rate_threshold"])
    if dead_fraction_threshold is None:
        dead_fraction_threshold = float(defaults["dead_fraction_threshold"])

    merged = fraction_alive_df[["snip_id", "embryo_id", "time_int", "fraction_alive"]].copy()
    if merged.isna().any(axis=None):
        raise ValueError("death_detection: fraction_alive table contains null rows")

    out = pd.DataFrame({"snip_id": merged["snip_id"].astype(str)})
    out["dead_flag"] = False
    out["death_inflection_time_int"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out["death_predicted_stage_hpf"] = pd.Series([pd.NA] * len(out), dtype="Float64")

    for embryo_id in merged["embryo_id"].astype(str).unique():
        embryo_mask = merged["embryo_id"].astype(str) == embryo_id
        embryo_data = merged.loc[embryo_mask].sort_values("time_int").copy()
        if len(embryo_data) < 3:
            continue
        result = detect_persistent_death_inflection(
            embryo_data,
            persistence_threshold=persistence_threshold,
            min_decline_rate=decline_rate_threshold,
            dead_fraction_threshold=dead_fraction_threshold,
        )
        if result is None:
            continue
        inflection_time = result["inflection_time"]
        embryo_indices = merged.index[embryo_mask]
        embryo_rows = merged.loc[embryo_indices]
        flag_mask = embryo_rows["time_int"] >= inflection_time - float(lead_time_hr)
        flagged_indices = embryo_indices[flag_mask.to_numpy()]
        out.loc[flagged_indices, "dead_flag"] = True
        out.loc[flagged_indices, "death_inflection_time_int"] = int(inflection_time)

    if "predicted_stage_hpf" in fraction_alive_df.columns:
        stage_lookup = fraction_alive_df[["snip_id", "predicted_stage_hpf"]].drop_duplicates(subset=["snip_id"]).set_index("snip_id")["predicted_stage_hpf"]
        out["death_predicted_stage_hpf"] = out["snip_id"].map(stage_lookup).astype("Float64")
        out.loc[~out["dead_flag"], "death_predicted_stage_hpf"] = pd.NA

    out["dead_flag"] = out["dead_flag"].astype(bool)
    aligned = align_to_universe(snip_universe_df, out, "death_detection")
    return aligned[["snip_id", "dead_flag", "death_inflection_time_int", "death_predicted_stage_hpf"]]


def compute_dead_flag2_persistence(df: pd.DataFrame, dead_lead_time: float = None) -> pd.DataFrame:
    """Backward-compatible wrapper that preserves the legacy dead_flag2 column."""
    defaults = get_qc_defaults("death_detection")
    if dead_lead_time is None:
        dead_lead_time = float(defaults["lead_time_hr"])
    result = compute_death_detection_flags(
        fraction_alive_df=df[["snip_id", "embryo_id", "time_int", "fraction_alive"]].copy(),
        snip_universe_df=df[["snip_id"]].copy(),
        lead_time_hr=dead_lead_time,
    )
    legacy = result.rename(columns={"dead_flag": "dead_flag2", "death_inflection_time_int": "dead_inflection_time_int"})
    return legacy
