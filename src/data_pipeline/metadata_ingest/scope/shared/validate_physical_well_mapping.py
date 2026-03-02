"""Validate physical (plate-free) series->well mapping.

This validator exists to prevent silently generating downstream artifacts under the
wrong well IDs. It checks that `series_well_mapping.csv` can fully map the scope
rows to `well_index` values, and that those `well_index` values are canonical
(A01-style) unless an explicit override is enabled.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from data_pipeline.metadata_ingest.time_helpers import ensure_frame_time_alias


CANONICAL_WELL_RE = re.compile(r"^[A-H](0[1-9]|1[0-2])$")
OVERRIDE_WELL_RE = re.compile(r"^S\d{2}$")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_series_lookup(mapping_df: pd.DataFrame) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for _, row in mapping_df.iterrows():
        series_number = row.get("series_number")
        well_index = row.get("well_index")
        if pd.isna(series_number) or pd.isna(well_index):
            continue
        try:
            lookup[int(series_number)] = str(well_index)
        except Exception:
            continue
    return lookup


def _resolve_mapped_well(
    scope_well_index: str,
    series_lookup: dict[int, str],
    known_wells: set[str],
    *,
    prefer_zero_based: bool,
) -> str:
    # If scope is already mapped to plate-like well names (e.g. A04), keep it.
    if scope_well_index in known_wells:
        return scope_well_index

    # Try convention where scope well index is 0-based series index.
    try:
        scope_idx_int = int(scope_well_index)
    except (TypeError, ValueError):
        return scope_well_index

    if prefer_zero_based:
        if (scope_idx_int + 1) in series_lookup:
            return series_lookup[scope_idx_int + 1]
        return scope_well_index

    # Prefer 1-based mapping, but allow 0-based scopes as a fallback.
    if scope_idx_int in series_lookup:
        return series_lookup[scope_idx_int]
    if (scope_idx_int + 1) in series_lookup:
        return series_lookup[scope_idx_int + 1]
    return scope_well_index


def validate_physical_well_mapping(
    *,
    scope_metadata_csv: Path,
    mapping_csv: Path,
    allow_unmapped_wells: bool,
) -> dict:
    scope_df = ensure_frame_time_alias(pd.read_csv(scope_metadata_csv), stage_name="validate_physical_well_mapping.scope")
    mapping_df = pd.read_csv(mapping_csv)
    if mapping_df.empty:
        raise ValueError(f"Empty mapping_csv: {mapping_csv}")

    for required in ["series_number", "well_index"]:
        if required not in mapping_df.columns:
            raise ValueError(f"mapping_csv missing required column: {required}")

    # Basic uniqueness checks.
    dup_series = mapping_df["series_number"].duplicated(keep=False)
    if dup_series.any():
        preview = mapping_df.loc[dup_series, ["series_number", "well_index"]].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate series_number entries in mapping_csv (preview): {preview}")

    dup_well = mapping_df["well_index"].duplicated(keep=False)
    if dup_well.any():
        preview = mapping_df.loc[dup_well, ["series_number", "well_index"]].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate well_index entries in mapping_csv (preview): {preview}")

    series_lookup = _build_series_lookup(mapping_df)
    known_wells = set(str(w) for w in mapping_df["well_index"].dropna().unique())

    raw_scope_wells = sorted(set(scope_df["well_index"].astype(str).unique().tolist()))
    raw_ints = []
    for w in raw_scope_wells:
        try:
            raw_ints.append(int(w))
        except Exception:
            continue
    prefer_zero_based = (0 in raw_ints)

    resolved = {
        w: _resolve_mapped_well(str(w), series_lookup, known_wells, prefer_zero_based=prefer_zero_based)
        for w in raw_scope_wells
    }

    unmapped = [raw for raw, mapped in resolved.items() if str(mapped) == str(raw) and raw not in known_wells]
    if unmapped:
        # If we can't map any raw wells, Phase 3 should not run unless the user explicitly opted in.
        if not allow_unmapped_wells:
            raise ValueError(
                "Physical well mapping appears incomplete (raw scope wells not mapped). "
                f"Unmapped preview: {unmapped[:10]}. "
                "Fix mapping inputs (YX1 XY reference / Keyence layout) or set scope_ingest.allow_unmapped_wells=true."
            )

    # Ensure scope wells map to unique well_index values (otherwise Phase 3 would write collisions).
    mapped_vals = [str(v) for v in resolved.values()]
    if len(set(mapped_vals)) != len(mapped_vals):
        # If we're in override mode, duplicates are still bad: they indicate we cannot uniquely identify wells.
        dup = pd.Series(mapped_vals).value_counts()
        dup = dup[dup > 1]
        preview = dup.head(10).to_dict()
        raise ValueError(f"Resolved scope wells map to duplicate well_index values (preview): {preview}")

    bad_well_indices: list[str] = []
    for mapped in resolved.values():
        mapped = str(mapped)
        if CANONICAL_WELL_RE.match(mapped):
            continue
        if allow_unmapped_wells and OVERRIDE_WELL_RE.match(mapped):
            continue
        bad_well_indices.append(mapped)

    if bad_well_indices:
        preview = sorted(set(bad_well_indices))[:10]
        raise ValueError(
            "Mapped well_index values are not canonical A01-style. "
            f"Bad well_index preview: {preview}. "
            "If you truly want to proceed with non-canonical IDs, set scope_ingest.allow_unmapped_wells=true and use S00-style IDs."
        )

    diagnostics = {
        "n_scope_wells": int(len(raw_scope_wells)),
        "n_mapping_rows": int(len(mapping_df)),
        "allow_unmapped_wells": bool(allow_unmapped_wells),
        "raw_scope_wells_preview": raw_scope_wells[:12],
        "resolved_mapping_preview": [{"raw": k, "mapped": v} for k, v in list(resolved.items())[:12]],
        "unmapped_raw_wells_preview": unmapped[:12],
    }
    return diagnostics


def validate_physical_well_mapping_file(
    *,
    scope_metadata_csv: Path,
    mapping_csv: Path,
    output_flag: Path,
    diagnostics_json: Path | None = None,
    allow_unmapped_wells: bool = False,
) -> dict:
    diagnostics = validate_physical_well_mapping(
        scope_metadata_csv=scope_metadata_csv,
        mapping_csv=mapping_csv,
        allow_unmapped_wells=bool(allow_unmapped_wells),
    )

    output_flag = Path(output_flag)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n")

    if diagnostics_json is not None:
        diagnostics_json = Path(diagnostics_json)
        diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_json.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    return diagnostics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope-metadata-csv", type=Path, required=True)
    p.add_argument("--mapping-csv", type=Path, required=True)
    p.add_argument("--output-flag", type=Path, required=True)
    p.add_argument("--diagnostics-json", type=Path, required=False, default=None)
    p.add_argument("--allow-unmapped-wells", default="false")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    validate_physical_well_mapping_file(
        scope_metadata_csv=args.scope_metadata_csv,
        mapping_csv=args.mapping_csv,
        output_flag=args.output_flag,
        diagnostics_json=args.diagnostics_json,
        allow_unmapped_wells=_parse_bool(args.allow_unmapped_wells),
    )


if __name__ == "__main__":
    main()
