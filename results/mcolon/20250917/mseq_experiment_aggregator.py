#!/usr/bin/env python
"""Concatenate selected Build06 CSVs into a single file.

Edit the CSV_BASENAMES list below and run the script. It will grab each file
from BUILD06_DIR, tag rows with their source basename, and write the combined
table to OUTPUT_PATH.
"""

from pathlib import Path

import pandas as pd


# Directory containing the per-experiment Build06 CSVs.
BUILD06_DIR = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output"
)

# Experiment identifiers (typically YYYYMMDD or YYYYMMDD_label) to combine.
EXPERIMENTS = [
"20250415",
"20250416",
"20250425",
"20250501",
"20250509",
"20250512",
"20250515_part2",
"20250519",
"20250520",
"20250626",
"20250703",
"20250711",
"20250725",
"20250728"]

# File pattern used for each experiment’s Build06 CSV.
CSV_PATTERN = "df03_final_output_with_latents_{exp}.csv"

# Where to write the combined CSV.
OUTPUT_PATH = Path(__file__).parent / Path("data/mseq_TZ_experiments.csv")


def main() -> None:
    if not EXPERIMENTS:
        raise SystemExit("Populate EXPERIMENTS with the runs you want to merge.")

    frames = []
    for exp in EXPERIMENTS:
        expected = CSV_PATTERN.format(exp=exp)
        csv_path = (BUILD06_DIR / expected).resolve()
        if not csv_path.exists():
            matches = sorted(BUILD06_DIR.glob(f"*{exp}*.csv"))
            if len(matches) == 1:
                csv_path = matches[0]
            elif not matches:
                print(f"Warning: No Build06 CSV found for {exp}, skipping...")
                continue
            else:
                raise RuntimeError(
                    f"Multiple Build06 CSVs match {exp}: {[m.name for m in matches]}"
                )

        df = pd.read_csv(csv_path)
        df["source_experiment"] = exp
        df["source_csv"] = csv_path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, copy=False)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(combined)} rows from {len(frames)} CSVs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
