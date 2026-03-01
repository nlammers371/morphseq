"""Validate YX1 XY-reference grid file used for physical (plate-free) well mapping.

This validator exists to catch "looks plausible but wrong" reference grids early.
It enforces simple geometric invariants:

- Wells in the same *row* (A..H) should have nearly-constant y_um
- Wells in the same *column* (01..12) should have nearly-constant x_um
- Adjacent grid spacing should be roughly regular (via CV tolerance)

Output is a sentinel flag plus a small diagnostics JSON for debugging.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


WELL_RE = re.compile(r"^([A-H])([0-1][0-9])$")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def validate_xy_reference_grid_df(
    df: pd.DataFrame,
    *,
    row_y_tol_um: float,
    col_x_tol_um: float,
    dx_cv_tol: float,
    dy_cv_tol: float,
) -> dict:
    required = {"well", "x_um", "y_um"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"XY reference grid missing required columns: {missing}")

    keep = df.copy()
    keep["well"] = keep["well"].astype(str).str.strip()
    keep = keep[keep["well"] != ""].copy()
    if keep.empty:
        raise ValueError("XY reference grid contains no non-empty wells.")

    parsed = keep["well"].map(lambda w: WELL_RE.match(w))
    if parsed.isna().any():
        bad = keep.loc[parsed.isna(), "well"].head(10).tolist()
        raise ValueError(f"XY reference grid has invalid well labels (preview): {bad}")

    keep["row"] = keep["well"].map(lambda w: WELL_RE.match(w).group(1))
    keep["col"] = keep["well"].map(lambda w: int(WELL_RE.match(w).group(2)))
    keep["x_um"] = pd.to_numeric(keep["x_um"], errors="coerce")
    keep["y_um"] = pd.to_numeric(keep["y_um"], errors="coerce")

    if keep["x_um"].isna().any() or keep["y_um"].isna().any():
        bad = keep.loc[keep["x_um"].isna() | keep["y_um"].isna(), ["well", "x_um", "y_um"]].head(10).to_dict(orient="records")
        raise ValueError(f"XY reference grid has non-numeric x_um/y_um (preview): {bad}")

    # Row consistency (y spread).
    row_spreads = (
        keep.groupby("row", as_index=False)
        .agg(y_min=("y_um", "min"), y_max=("y_um", "max"), n=("y_um", "size"))
    )
    row_spreads["y_spread"] = (row_spreads["y_max"] - row_spreads["y_min"]).astype(float)
    bad_rows = row_spreads[row_spreads["y_spread"] > float(row_y_tol_um)]
    if not bad_rows.empty:
        preview = bad_rows[["row", "n", "y_spread"]].head(10).to_dict(orient="records")
        raise ValueError(f"XY reference grid row y spread exceeds tolerance {row_y_tol_um} um (preview): {preview}")

    # Column consistency (x spread).
    col_spreads = (
        keep.groupby("col", as_index=False)
        .agg(x_min=("x_um", "min"), x_max=("x_um", "max"), n=("x_um", "size"))
    )
    col_spreads["x_spread"] = (col_spreads["x_max"] - col_spreads["x_min"]).astype(float)
    bad_cols = col_spreads[col_spreads["x_spread"] > float(col_x_tol_um)]
    if not bad_cols.empty:
        preview = bad_cols[["col", "n", "x_spread"]].head(10).to_dict(orient="records")
        raise ValueError(f"XY reference grid column x spread exceeds tolerance {col_x_tol_um} um (preview): {preview}")

    # Regular spacing: estimate dx per row, dy per col.
    dx_vals: list[float] = []
    for r, sub in keep.sort_values(["row", "col"]).groupby("row"):
        sub = sub.drop_duplicates(subset=["col"]).sort_values("col")
        if len(sub) >= 3:
            dx = np.diff(sub["x_um"].to_numpy(dtype=float))
            dx_vals.extend([float(v) for v in dx if np.isfinite(v)])

    dy_vals: list[float] = []
    for c, sub in keep.sort_values(["col", "row"]).groupby("col"):
        # Row ordering A..H
        sub = sub.drop_duplicates(subset=["row"]).copy()
        sub["row_ord"] = sub["row"].map(lambda x: ord(x) - ord("A"))
        sub = sub.sort_values("row_ord")
        if len(sub) >= 3:
            dy = np.diff(sub["y_um"].to_numpy(dtype=float))
            dy_vals.extend([float(v) for v in dy if np.isfinite(v)])

    def _cv(values: list[float]) -> float | None:
        if len(values) < 3:
            return None
        arr = np.asarray(values, dtype=float)
        m = float(np.mean(arr))
        s = float(np.std(arr))
        if m == 0.0:
            return None
        return abs(s / m)

    dx_cv = _cv(dx_vals)
    dy_cv = _cv(dy_vals)

    if dx_cv is not None and dx_cv > float(dx_cv_tol):
        raise ValueError(f"XY reference grid dx CV {dx_cv:.3f} exceeds tolerance {dx_cv_tol}.")
    if dy_cv is not None and dy_cv > float(dy_cv_tol):
        raise ValueError(f"XY reference grid dy CV {dy_cv:.3f} exceeds tolerance {dy_cv_tol}.")

    diagnostics = {
        "n_rows": int(len(keep)),
        "rows_present": sorted(keep["row"].unique().tolist()),
        "cols_present": sorted([int(c) for c in keep["col"].unique().tolist()]),
        "row_y_tol_um": float(row_y_tol_um),
        "col_x_tol_um": float(col_x_tol_um),
        "dx_cv_tol": float(dx_cv_tol),
        "dy_cv_tol": float(dy_cv_tol),
        "row_y_spreads": row_spreads[["row", "n", "y_spread"]].to_dict(orient="records"),
        "col_x_spreads": col_spreads[["col", "n", "x_spread"]].to_dict(orient="records"),
        "dx_cv": None if dx_cv is None else float(dx_cv),
        "dy_cv": None if dy_cv is None else float(dy_cv),
        "dx_median_um": None if not dx_vals else float(np.median(np.asarray(dx_vals, dtype=float))),
        "dy_median_um": None if not dy_vals else float(np.median(np.asarray(dy_vals, dtype=float))),
    }
    return diagnostics


def validate_xy_reference_grid_file(
    *,
    input_csv: Path,
    output_flag: Path,
    diagnostics_json: Path | None = None,
    row_y_tol_um: float = 1200.0,
    col_x_tol_um: float = 1200.0,
    dx_cv_tol: float = 0.15,
    dy_cv_tol: float = 0.15,
) -> dict:
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)
    diagnostics = validate_xy_reference_grid_df(
        df,
        row_y_tol_um=float(row_y_tol_um),
        col_x_tol_um=float(col_x_tol_um),
        dx_cv_tol=float(dx_cv_tol),
        dy_cv_tol=float(dy_cv_tol),
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
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-flag", type=Path, required=True)
    p.add_argument("--diagnostics-json", type=Path, required=False, default=None)
    p.add_argument("--row-y-tol-um", type=float, default=1200.0)
    p.add_argument("--col-x-tol-um", type=float, default=1200.0)
    p.add_argument("--dx-cv-tol", type=float, default=0.15)
    p.add_argument("--dy-cv-tol", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    validate_xy_reference_grid_file(
        input_csv=args.input_csv,
        output_flag=args.output_flag,
        diagnostics_json=args.diagnostics_json,
        row_y_tol_um=args.row_y_tol_um,
        col_x_tol_um=args.col_x_tol_um,
        dx_cv_tol=args.dx_cv_tol,
        dy_cv_tol=args.dy_cv_tol,
    )


if __name__ == "__main__":
    main()

