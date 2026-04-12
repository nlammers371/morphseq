from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.metadata_ingest.plate.plate_processing import process_plate_layout


def _make_plate_sheet(*, values: list[list[object]], row_labels: list[object]) -> pd.DataFrame:
    """
    Build a DataFrame shaped like the Numbers-exported MorphSeq plate sheets:
    - header row with columns: ['Unnamed: 0', 1..12]
    - 8 data rows A..H, but row label cells can be missing (NaN).
    """
    assert len(values) == 8
    assert all(len(r) == 12 for r in values)
    assert len(row_labels) == 8
    cols = ["Unnamed: 0"] + list(range(1, 13))
    rows = []
    for lab, row in zip(row_labels, values):
        rows.append([lab] + row)
    return pd.DataFrame(rows, columns=cols)


def test_process_plate_layout_excel_handles_missing_row_label_and_defaults_treatment(tmp_path: Path) -> None:
    # Create a synthetic workbook with the minimum required sheets plus series_number_map.
    xlsx_path = tmp_path / "plate.xlsx"

    row_labels = ["A", "B", "C", "D", "E", "F", "G", np.nan]  # missing "H" label cell

    # genotype: fill with distinct values for a couple wells we assert on.
    geno_vals = [["wt"] * 12 for _ in range(8)]
    geno_vals[7][0] = "hom"  # H01
    geno_vals[7][1] = "het"  # H02

    # chem_perturbation: blank everywhere -> should become treatment="none"
    # Use whitespace (not NaN) so the row is preserved when written to Excel, but still normalizes to "none".
    treat_vals = [[""] * 12 for _ in range(8)]
    treat_vals[7][0] = " "  # keep last row from being entirely empty in Excel

    # start_age_hpf: drop A01 by leaving it empty (NaN), keep all others.
    age_vals = [[13.0] * 12 for _ in range(8)]
    age_vals[0][0] = np.nan  # A01 missing

    # temperature: numeric
    temp_vals = [[30.0] * 12 for _ in range(8)]

    # medium: string
    medium_vals = [["MC10"] * 12 for _ in range(8)]

    # series_number_map: numeric grid (not strictly required by schema, but required by mapping helpers)
    series_vals = [[float(i + 1 + r * 12) for i in range(12)] for r in range(8)]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        _make_plate_sheet(values=medium_vals, row_labels=row_labels).to_excel(w, sheet_name="medium", index=False)
        _make_plate_sheet(values=geno_vals, row_labels=row_labels).to_excel(w, sheet_name="genotype", index=False)
        _make_plate_sheet(values=treat_vals, row_labels=row_labels).to_excel(w, sheet_name="chem_perturbation", index=False)
        _make_plate_sheet(values=age_vals, row_labels=["A", "B", "C", "D", "E", "F", "G", "H"]).to_excel(
            w, sheet_name="start_age_hpf", index=False
        )
        _make_plate_sheet(values=temp_vals, row_labels=["A", "B", "C", "D", "E", "F", "G", "H"]).to_excel(
            w, sheet_name="temperature", index=False
        )
        _make_plate_sheet(values=series_vals, row_labels=["A", "B", "C", "D", "E", "F", "G", "H"]).to_excel(
            w, sheet_name="series_number_map", index=False
        )

    out_csv = tmp_path / "plate_metadata.csv"
    df = process_plate_layout(xlsx_path, experiment_id="EXP", output_csv=out_csv)

    # We dropped A01 due to missing start_age_hpf.
    assert len(df) == 95
    assert "A01" not in set(df["well_index"].astype(str))

    # Missing chem_perturbation should be normalized into explicit "none" so validation passes.
    assert set(df["treatment"].astype(str).str.strip().unique()) == {"none"}

    # Ensure H row is still present even if the row label cell was missing in the genotype sheet.
    h = df[df["well_index"].astype(str).str.startswith("H")].set_index("well_index")
    assert h.loc["H01", "genotype"] == "hom"
    assert h.loc["H02", "genotype"] == "het"

    # series_number_map should be parsed as a per-well column (float is fine).
    assert "series_number_map" in df.columns
    assert df["series_number_map"].notna().all()
