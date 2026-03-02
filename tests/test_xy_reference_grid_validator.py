from __future__ import annotations

import pandas as pd

from data_pipeline.metadata_ingest.scope.yx1.validate_xy_reference_grid import (
    validate_xy_reference_grid_df,
)


def _make_grid(dx: float = 9000.0, dy: float = 9000.0) -> pd.DataFrame:
    rows = []
    for r_i, r in enumerate("ABCDEFGH"):
        for c in range(1, 13):
            rows.append(
                {
                    "well": f"{r}{c:02d}",
                    "x_um": (12 - c) * dx,  # decreasing left->right is fine; spacing is what we validate
                    "y_um": -r_i * dy,
                }
            )
    return pd.DataFrame(rows)


def test_validate_xy_reference_grid_passes_on_regular_grid() -> None:
    df = _make_grid()
    diag = validate_xy_reference_grid_df(
        df,
        row_y_tol_um=1.0,
        col_x_tol_um=1.0,
        dx_cv_tol=1e-6,
        dy_cv_tol=1e-6,
    )
    assert diag["n_rows"] == 96


def test_validate_xy_reference_grid_fails_on_row_y_inconsistency() -> None:
    df = _make_grid()
    df.loc[df["well"] == "A01", "y_um"] = 12345.0
    try:
        validate_xy_reference_grid_df(
            df,
            row_y_tol_um=1.0,
            col_x_tol_um=1.0,
            dx_cv_tol=1e-6,
            dy_cv_tol=1e-6,
        )
    except ValueError as e:
        assert "row y spread exceeds tolerance" in str(e)
    else:
        raise AssertionError("Expected validator to raise ValueError")
