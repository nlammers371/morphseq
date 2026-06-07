#!/usr/bin/env python
"""Backfill start_age_hpf in the Excel source for genotyped-but-unstaged wells.

3 single-stage cep290 plates have a column of wells (column 6 or 8) that carry a
genotype annotation but were left blank in the start_age_hpf sheet. The whole plate
is one stage, so fill those wells with the plate stage value -- writing it directly
into the start_age_hpf sheet so the source metadata is correct for any future rerun.

Only wells that (a) appear in the `genotype` sheet and (b) are blank in start_age get
filled. Ungenotyped blanks are left blank. Makes a one-time .bak_excel_fill backup.

Excel grid: row 1 = plate-col headers (B..M = cols 1..12), col A = row labels (A..H).
  well <R><CC> -> cell(row = ord(R)-65+2, col = CC+1).
"""
import shutil
from pathlib import Path

import openpyxl
import pandas as pd

REPO = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
PLATE_META = REPO / "metadata/plate_metadata"

# experiment -> single plate stage (verified: each is single-stage)
TARGETS = {
    "20260324_cep290_30hpf_plate02": 30.0,
    "20260415_cep290_30to48hpf_plate02_t01": 30.0,
    "20260416_cep290_30to48hpf_plate02_t02": 48.0,
}

WELLS = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]


def grid_map(xlf, sheet):
    if sheet not in xlf.sheet_names:
        return {}
    df = xlf.parse(sheet, header=0)
    block = df.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
    arr = block.to_numpy(dtype=str).ravel()
    return {w: v for w, v in zip(WELLS, arr) if v not in ("", "nan")}


def well_to_cell(well: str):
    row = ord(well[0]) - ord("A") + 2
    col = int(well[1:]) + 1
    return row, col


def main(apply: bool):
    for exp, stage in TARGETS.items():
        xl = PLATE_META / f"{exp}_well_metadata.xlsx"
        with pd.ExcelFile(xl) as xlf:
            geno = grid_map(xlf, "genotype")
            age = grid_map(xlf, "start_age_hpf")
        fill = sorted(set(geno) - set(age))  # genotyped, unstaged
        print(f"{exp}  stage={stage}  filling {len(fill)} wells: {fill}")
        if not fill:
            continue
        if not apply:
            continue
        bak = xl.with_suffix(xl.suffix + ".bak_excel_fill")
        if not bak.exists():
            shutil.copy2(xl, bak)
        wb = openpyxl.load_workbook(xl)
        ws = wb["start_age_hpf"]
        # stage value: write as int when whole (matches existing 30/48 ints)
        val = int(stage) if float(stage).is_integer() else stage
        for w in fill:
            r, c = well_to_cell(w)
            ws.cell(row=r, column=c, value=val)
        wb.save(xl)
        wb.close()
        # verify
        with pd.ExcelFile(xl) as xlf:
            age2 = grid_map(xlf, "start_age_hpf")
        ok = all(w in age2 for w in fill)
        print(f"    written & verified: {ok}  (start_age wells now {len(age2)})")


if __name__ == "__main__":
    import sys

    main(apply="--apply" in sys.argv)
