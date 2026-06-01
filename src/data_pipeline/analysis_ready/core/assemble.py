from __future__ import annotations

import pandas as pd

from data_pipeline.analysis_ready.validators import (
    assert_1to1_join,
    assert_no_column_collisions,
    assert_unique_snip_id,
)
from data_pipeline.schemas.analysis_ready import ANALYSIS_READY_QC_COLUMNS


def assemble_analysis_ready(
    features_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    *,
    key: str = "snip_id",
    embedding_calculated: bool = False,
) -> pd.DataFrame:
    """Join feature and QC tables into the analysis-ready contract."""
    assert_unique_snip_id(features_df, key=key)
    assert_unique_snip_id(qc_df, key=key)

    qc_keep = [key] + [column for column in ANALYSIS_READY_QC_COLUMNS if column in qc_df.columns]
    qc_projection = qc_df.loc[:, qc_keep].copy()

    assert_no_column_collisions(features_df, qc_projection, key=key)
    assert_1to1_join(features_df, qc_projection, key=key)

    assembled = features_df.merge(qc_projection, on=key, how="inner", validate="one_to_one")
    assembled["embedding_calculated"] = bool(embedding_calculated)
    return assembled
