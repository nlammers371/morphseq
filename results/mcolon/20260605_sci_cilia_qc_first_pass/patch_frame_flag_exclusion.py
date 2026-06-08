"""
Patch existing qc_staged CSVs: lift use_embryo_flag=False caused solely by frame_flag.

frame_flag is now informational-only (like focus_flag). This script updates existing
CSVs so they match the new embryo_flags.py logic without a full rebuild.

Scope: named cilia snapshot plates from 2026 only. Does NOT touch time-lapse plates.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3] / "morphseq_playground/metadata/build04_output"

SNAPSHOT_PLATES = [
    "qc_staged_20260319_cilia_crispant_18hpf.csv",
    "qc_staged_20260319_cilia_crispant_24hpf.csv",
    "qc_staged_20260319_cilia_crispant_30hpf.csv",
    "qc_staged_20260320_cilia_crispant_48hpf.csv",
    "qc_staged_20260324_cep290_18hpf_24hpf_plate02.csv",
    "qc_staged_20260324_cep290_18hpf_plate01.csv",
    "qc_staged_20260324_cep290_24hpf_plate01.csv",
    "qc_staged_20260324_cep290_24hpf_plate02.csv",
    "qc_staged_20260324_cep290_30hpf_plate01.csv",
    "qc_staged_20260324_cep290_30hpf_plate02.csv",
    "qc_staged_20260331_b9d2_18hpf_plate01.csv",
    "qc_staged_20260331_b9d2_18hpf_plate02.csv",
    "qc_staged_20260414_b9d2_14hpf_plate01.csv",
    "qc_staged_20260414_b9d2_30hpf_plate01.csv",
    "qc_staged_20260414_b9d2_30hpf_plate02.csv",
    "qc_staged_20260414_sci_b9d2_48hpf_plate01.csv",
    "qc_staged_20260415_b9d2_30to48hpf_plate01_t02.csv",
    "qc_staged_20260415_b9d2_30to48hpf_plate02_t02.csv",
    "qc_staged_20260415_cep290_18hpf_plate03.csv",
    "qc_staged_20260415_cep290_30to48hpf_plate02_t01.csv",
    "qc_staged_20260415_sci_cep290_48hpf_plate01.csv",
    "qc_staged_20260416_cep290_30to48hpf_plate01_t02.csv",
    "qc_staged_20260416_cep290_30to48hpf_plate02_t02.csv",
]

OTHER_EXCLUSION_FLAGS = ["dead_flag", "dead_flag2", "sa_outlier_flag", "sam2_qc_flag"]

total_recovered = 0

for fname in SNAPSHOT_PLATES:
    p = BASE / fname
    if not p.exists():
        print(f"SKIP (not found): {fname}")
        continue

    df = pd.read_csv(p, low_memory=False)

    if "frame_flag" not in df.columns:
        print(f"SKIP (no frame_flag col): {fname}")
        continue

    frame_true = df["frame_flag"].fillna(False).astype(bool)
    other_false = ~df[[c for c in OTHER_EXCLUSION_FLAGS if c in df.columns]].fillna(False).astype(bool).any(axis=1)
    frame_only_mask = frame_true & other_false

    n = int(frame_only_mask.sum())
    if n == 0:
        print(f"  {fname}: 0 recoverable wells (nothing to patch)")
        continue

    wells = df.loc[frame_only_mask, "well"].tolist() if "well" in df.columns else ["(no well col)"]

    bak = p.with_suffix(".csv.bak_pre_frame_flag_patch")
    if not bak.exists():
        shutil.copy2(p, bak)

    df.loc[frame_only_mask, "use_embryo_flag"] = True
    df.to_csv(p, index=False)

    print(f"  {fname}: recovered {n} wells -> {wells}")
    total_recovered += n

print(f"\nTotal wells recovered across all plates: {total_recovered}")
