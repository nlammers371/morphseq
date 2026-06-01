"""
Developmental stage (HPF) prediction from embryo features.

Predicts hours post-fertilization using Kimmel et al. (1995) temperature formula.
Extracted from build03A_process_images.py segment_wells_sam2_csv (lines 984-989).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def predict_stage_hpf(
    start_age_hpf: float,
    elapsed_time_s: float,
    temperature_c: float,
) -> float:
    """Predict developmental stage using Kimmel et al. (1995) formula."""
    time_hours = elapsed_time_s / 3600.0
    developmental_rate = 0.055 * temperature_c - 0.57
    predicted_hpf = start_age_hpf + (time_hours * developmental_rate)
    return float(predicted_hpf)


def infer_stage_from_area(
    area_um2: float,
    reference_curve: Optional[dict] = None,
) -> tuple:
    """Infer developmental stage from surface area using reference curve."""
    return np.nan, np.nan


def compute_stage_predictions_batch(
    tracking_df: pd.DataFrame,
    start_age_col: str = 'start_age_hpf',
    time_col: str = 'experiment_time_s',
    temp_col: str = 'temperature',
) -> pd.DataFrame:
    """Compute stage predictions for batch of snips."""
    results = []
    for _, row in tracking_df.iterrows():
        snip_id = row['snip_id']
        if start_age_col not in row.index or pd.isna(row[start_age_col]):
            raise ValueError(f"stage_predictions: missing required column '{start_age_col}' for snip_id={snip_id}")
        start_age = float(row[start_age_col])
        elapsed_time = row.get(time_col, np.nan)
        if pd.isna(elapsed_time):
            for alt_time_col in ('time_s',):
                if alt_time_col in row.index and pd.notna(row[alt_time_col]):
                    elapsed_time = row[alt_time_col]
                    break
        if pd.isna(elapsed_time):
            raise ValueError(f"stage_predictions: missing required time column '{time_col}' for snip_id={snip_id}")
        temperature = row.get(temp_col, np.nan)
        if pd.isna(temperature):
            for alt_temp_col in ('temperature_c', 'temperature'):
                if alt_temp_col in row.index and pd.notna(row[alt_temp_col]):
                    temperature = row[alt_temp_col]
                    break
        if pd.isna(temperature):
            raise ValueError(f"stage_predictions: missing required temperature column '{temp_col}' for snip_id={snip_id}")
        predicted_hpf = predict_stage_hpf(start_age, float(elapsed_time), float(temperature))
        results.append({
            'snip_id': snip_id,
            'predicted_stage_hpf': predicted_hpf,
            'stage_confidence': 1.0,
        })
    return pd.DataFrame(results)
