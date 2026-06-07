#!/usr/bin/env python
"""Manually backfill start_age_hpf + predicted_stage_hpf into existing build03/build06 CSVs.

Context: the plate-metadata Excel sheet had been renamed start_age_hpf -> start_stage_hpf,
so Build01 compiled an empty `start_age_hpf`, which made `predicted_stage_hpf` NaN at Build03.
The Excel sheet has since been renamed back to `start_age_hpf` (values intact). Rather than
re-running the full pipeline right now, we patch the already-produced CSVs in place:

  1. read the per-well `start_age_hpf` grid (8x12, row-major A01..H12) exactly as
     src/build/export_utils.py does,
  2. join start_age_hpf onto each row by `well` (handles multi-stage plates),
  3. recompute predicted_stage_hpf with the Build03 Kimmel formula:
        start_age_hpf + (Time Rel (s)/3600) * (0.055*temperature - 0.57)
     (build03A_process_images.py:_ensure_predicted_stage_hpf)

Idempotent: re-running overwrites start_age_hpf / predicted_stage_hpf from the Excel source.
A one-time .bak_prestage copy is made the first time a CSV is touched.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
PLATE_META = REPO / "metadata/plate_metadata"
B06 = REPO / "morphseq_playground/metadata/build06_output"
B03 = REPO / "morphseq_playground/metadata/build03_output"

EXPERIMENTS = """
20260319_cilia_crispant_18hpf
20260319_cilia_crispant_24hpf
20260319_cilia_crispant_30hpf
20260320_cilia_crispant_48hpf
20260324_cep290_18hpf_24hpf_plate02
20260324_cep290_18hpf_plate01
20260324_cep290_24hpf_plate01
20260324_cep290_24hpf_plate02
20260324_cep290_30hpf_plate01
20260324_cep290_30hpf_plate02
20260331_b9d2_18hpf_plate01
20260331_b9d2_18hpf_plate02
20260414_b9d2_14hpf_plate01
20260414_b9d2_30hpf_plate01
20260414_b9d2_30hpf_plate02
20260415_b9d2_30to48hpf_plate01_t02
20260415_cep290_18hpf_plate03
20260416_cep290_30to48hpf_plate01_t02
20260415_cep290_30to48hpf_plate02_t01
20260416_cep290_30to48hpf_plate02_t02
20260414_sci_b9d2_48hpf_plate01
20260415_sci_cep290_48hpf_plate01
""".split()

WELLS = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]


def well_start_age_map(exp: str) -> dict[str, float]:
    """well -> start_age_hpf (float), mirroring export_utils.py parsing of the 8x12 grid."""
    xl = PLATE_META / f"{exp}_well_metadata.xlsx"
    with pd.ExcelFile(xl) as xlf:
        if "start_age_hpf" not in xlf.sheet_names:
            raise ValueError(f"{exp}: no start_age_hpf sheet")
        df = xlf.parse("start_age_hpf", header=0)
        block = df.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
        arr = block.to_numpy(dtype=str).ravel()
    out = {}
    for w, v in zip(WELLS, arr):
        if v in ("", "nan"):
            continue
        try:
            out[w] = float(v)
        except ValueError:
            raise ValueError(f"{exp}: non-numeric start_age_hpf {v!r} at well {w}")
    return out


def patch_csv(path: Path, age_map: dict[str, float], dry: bool) -> str:
    if not path.exists():
        return f"  SKIP (missing): {path.name}"
    df = pd.read_csv(path, low_memory=False)
    if "well" not in df.columns:
        return f"  SKIP (no `well` col): {path.name}"

    sa = df["well"].map(age_map)
    n_unmapped = int(sa.isna().sum())
    df["start_age_hpf"] = sa

    temp_col = "temperature" if "temperature" in df.columns else "temperature_c"
    have_formula = {"Time Rel (s)", temp_col}.issubset(df.columns)
    if have_formula:
        df["predicted_stage_hpf"] = (
            df["start_age_hpf"].astype(float)
            + (pd.to_numeric(df["Time Rel (s)"], errors="coerce") / 3600.0)
            * (0.055 * pd.to_numeric(df[temp_col], errors="coerce") - 0.57)
        )
        nn = int(df["predicted_stage_hpf"].notna().sum())
        rng = (
            f"[{df['predicted_stage_hpf'].min():.2f}, {df['predicted_stage_hpf'].max():.2f}]"
            if nn
            else "n/a"
        )
        msg = (
            f"  OK: {path.name}  pred_hpf {nn}/{len(df)} {rng}"
            + (f"  (unmapped wells: {n_unmapped})" if n_unmapped else "")
        )
    else:
        msg = f"  WARN (no Time Rel (s)/temp): {path.name}  start_age set only"

    if not dry:
        bak = path.with_suffix(path.suffix + ".bak_prestage")
        if not bak.exists():
            shutil.copy2(path, bak)
        df.to_csv(path, index=False)
    return msg


def main():
    dry = "--apply" not in sys.argv
    print(f"{'DRY RUN' if dry else 'APPLYING'} — {len(EXPERIMENTS)} experiments\n")
    for exp in EXPERIMENTS:
        try:
            age_map = well_start_age_map(exp)
        except Exception as e:  # noqa: BLE001
            print(f"{exp}\n  ERROR reading Excel: {e}")
            continue
        print(f"{exp}  ({len(age_map)} wells, stages={sorted(set(age_map.values()))})")
        print(patch_csv(B06 / f"df03_final_output_with_latents_{exp}.csv", age_map, dry))
        print(patch_csv(B03 / f"expr_embryo_metadata_{exp}.csv", age_map, dry))
    if dry:
        print("\nNo files written. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
