"""
10_merge_chunks.py
==================
After the qsub array job completes, merge all chunk CSVs into one
rel_entropy_summaries.csv.

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/10_merge_chunks.py
"""

from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "10_scan_output"
CSV_OUT = OUT_DIR / "rel_entropy_summaries.csv"

chunks = sorted(OUT_DIR.glob("chunk_*.csv"))
print(f"Found {len(chunks)} chunk files")

dfs = [pd.read_csv(c) for c in chunks]
df = pd.concat(dfs, ignore_index=True).sort_values(["t", "p"]).reset_index(drop=True)
df = df.drop_duplicates(subset=["t", "p"], keep="last")

df.to_csv(CSV_OUT, index=False)
n_mask = int(df["has_mask"].sum())
print(f"Saved → {CSV_OUT}  ({len(df)} rows, {n_mask} with mask)")
print(df["rel_entropy_mean"].describe())
