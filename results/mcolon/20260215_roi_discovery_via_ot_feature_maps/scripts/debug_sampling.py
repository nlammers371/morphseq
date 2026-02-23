#!/usr/bin/env python3
"""Test the _sample_by_stage function"""

import sys
from pathlib import Path

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))

import pandas as pd
import numpy as np

data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
df = pd.read_csv(data_csv, low_memory=False)

# Test the sampling function
stage_lo, stage_hi = 47.0, 49.0
target_stage = 48.0
genotype = "cep290_wildtype"
n_samples = 3
seed = 42

print(f"Testing _sample_by_stage with:")
print(f"  Genotype: {genotype}")
print(f"  Stage window: {stage_lo}-{stage_hi} hpf")
print(f"  Target stage: {target_stage} hpf")
print(f"  Samples: {n_samples}\n")

# Filter to stage window
subset = df[
    (df["genotype"] == genotype) &
    (df["predicted_stage_hpf"] >= stage_lo) &
    (df["predicted_stage_hpf"] <= stage_hi)
].copy()

print(f"Filtered subset: {len(subset)} frames")

# Group by embryo and select closest to target
def select_frame(group_df: pd.DataFrame) -> pd.Series:
    group_df = group_df.copy()
    group_df["stage_diff"] = (group_df["predicted_stage_hpf"] - target_stage).abs()
    return group_df.sort_values("stage_diff").iloc[0]

per_embryo = subset.groupby("embryo_id").apply(select_frame).reset_index(drop=True)

print(f"Per-embryo selection: {len(per_embryo)} embryos")
print(f"\nFirst 5 embryos:")
for i in range(min(5, len(per_embryo))):
    row = per_embryo.iloc[i]
    print(f"  {row['embryo_id']} frame {row['frame_index']}: stage {row['predicted_stage_hpf']:.1f} hpf")

# Sample randomly
rng = np.random.default_rng(seed)
idx = rng.choice(len(per_embryo), size=n_samples, replace=False)
sampled = per_embryo.iloc[idx].reset_index(drop=True)

print(f"\nRandomly sampled {n_samples}:")
for i in range(len(sampled)):
    row = sampled.iloc[i]
    print(f"  {row['embryo_id']} frame {row['frame_index']}: stage {row['predicted_stage_hpf']:.1f} hpf")
