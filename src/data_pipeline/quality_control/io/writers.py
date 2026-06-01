from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.io.savers import save_csv
from data_pipeline.quality_control.io.paths import qc_sentinel_path
from data_pipeline.quality_control.validators import (
    validate_auxiliary_mask_qc_flags,
    validate_death_detection_flags,
    validate_focus_qc_flags,
    validate_motion_qc_flags,
    validate_qc_flags,
    validate_segmentation_qc_flags,
    validate_surface_area_qc_flags,
    validate_viability_qc_flags,
)


def write_qc_stage_contract(df: pd.DataFrame, output_csv: Path, *, validator) -> None:
    validator(df)
    save_csv(df, output_csv)
    qc_sentinel_path(output_csv).write_text("ok\n")


def write_qc_contract(df: pd.DataFrame, output_csv: Path) -> None:
    validate_qc_flags(df)
    save_csv(df, output_csv)
    qc_sentinel_path(output_csv).write_text("ok\n")
