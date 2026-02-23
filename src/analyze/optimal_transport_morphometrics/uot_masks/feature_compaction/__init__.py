"""Feature compaction and storage utilities for UOT outputs."""

from .storage import (
    FEATURE_SCHEMA_VERSION,
    build_pair_id,
    build_pair_metrics_record,
    apply_contract_dtypes,
    upsert_pair_metrics,
    upsert_ot_pair_metrics_parquet,
    compute_barycentric_projection,
    save_pair_artifacts,
)
from .features import (
    FEATURE_VECTOR_SCHEMA_VERSION,
    dct_radial_band_energy_fractions,
    extract_pair_feature_record,
    upsert_ot_feature_matrix_parquet,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "build_pair_id",
    "build_pair_metrics_record",
    "apply_contract_dtypes",
    "upsert_pair_metrics",
    "upsert_ot_pair_metrics_parquet",
    "compute_barycentric_projection",
    "save_pair_artifacts",
    "FEATURE_VECTOR_SCHEMA_VERSION",
    "dct_radial_band_energy_fractions",
    "extract_pair_feature_record",
    "upsert_ot_feature_matrix_parquet",
]
