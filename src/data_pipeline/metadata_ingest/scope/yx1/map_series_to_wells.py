from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_pipeline.metadata_ingest.mapping.series_well_mapper_yx1 import (
    DEFAULT_REF_XY_PATH,
    map_nd2_to_wells_by_xy,
)


REQUIRED_COLUMNS_SERIES_WELL_MAPPING = [
    "experiment_id",
    "series_number",  # 1-based, matches ND2 P + 1
    "well_index",     # canonical (A01..)
    "mapping_method",
]


def _find_nd2_file(raw_yx1_experiment_dir: Path) -> Path:
    nd2_files = sorted(raw_yx1_experiment_dir.glob("*.nd2"))
    if not nd2_files:
        raise FileNotFoundError(f"No ND2 files found in {raw_yx1_experiment_dir}")
    if len(nd2_files) > 1:
        raise RuntimeError(f"Multiple ND2 files found in {raw_yx1_experiment_dir}: {nd2_files}")
    return nd2_files[0]


def map_series_to_wells_yx1(
    *,
    raw_yx1_experiment_dir: Path,
    experiment_id: str,
    output_mapping_csv: Path,
    output_provenance_json: Path,
    ref_xy_csv: Path | None = None,
    max_distance_um: float = 4500.0,
) -> pd.DataFrame:
    """
    Physical series-to-well mapping for YX1 using ND2 stage positions + reference XY coordinates.

    This mapping is scope-native and does NOT require plate metadata.
    """
    nd2_path = _find_nd2_file(raw_yx1_experiment_dir)
    mapping, diagnostics = map_nd2_to_wells_by_xy(
        nd2_path=nd2_path,
        ref_xy_csv=ref_xy_csv,
        max_distance_um=float(max_distance_um),
    )

    # mapping is P index (0-based) -> well_index (A01..). Convert to series_number (1-based).
    rows: list[dict[str, object]] = []
    for p_idx, well_index in sorted(mapping.items(), key=lambda kv: int(kv[0])):
        rows.append(
            {
                "experiment_id": str(experiment_id),
                "series_number": int(p_idx) + 1,
                "well_index": str(well_index),
                "mapping_method": "yx1_xy_reference",
            }
        )

    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_COLUMNS_SERIES_WELL_MAPPING if c not in df.columns]
    if missing:
        raise ValueError(f"series_well_mapping missing required columns: {missing}")
    if df.empty:
        raise ValueError("series_well_mapping is empty (no ND2 positions mapped)")

    output_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_mapping_csv, index=False)

    provenance = {
        "experiment_id": str(experiment_id),
        "nd2_path": str(nd2_path),
        "ref_xy_csv": str(ref_xy_csv or DEFAULT_REF_XY_PATH),
        "max_distance_um": float(max_distance_um),
        "n_series_mapped": int(len(df)),
        "diagnostics": diagnostics,
    }
    output_provenance_json.parent.mkdir(parents=True, exist_ok=True)
    output_provenance_json.write_text(json.dumps(provenance, indent=2))

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-yx1-experiment-dir", type=Path, required=True)
    ap.add_argument("--experiment-id", type=str, required=True)
    ap.add_argument("--output-mapping-csv", type=Path, required=True)
    ap.add_argument("--output-provenance-json", type=Path, required=True)
    ap.add_argument("--ref-xy-csv", type=Path, default=None)
    ap.add_argument("--max-distance-um", type=float, default=4500.0)
    args = ap.parse_args()

    map_series_to_wells_yx1(
        raw_yx1_experiment_dir=args.raw_yx1_experiment_dir,
        experiment_id=args.experiment_id,
        output_mapping_csv=args.output_mapping_csv,
        output_provenance_json=args.output_provenance_json,
        ref_xy_csv=args.ref_xy_csv,
        max_distance_um=args.max_distance_um,
    )


if __name__ == "__main__":
    main()

