from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.classification.viz.misclassification import plot_margin_trends
from common import RESULTS_ROOT

DEFAULT_PAIRWISE_ROOT = (
    RESULTS_ROOT
    / "results"
    / "positioning"
    / "pairwise"
    / "combined_pairwise_5class_bin2_perm500_feature4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot signed-margin trajectories directly from predictions.parquet.")
    parser.add_argument("--pairwise-root", type=Path, default=DEFAULT_PAIRWISE_ROOT)
    parser.add_argument("--feature-set", default="vae")
    parser.add_argument("--results-subdir", default="plain_signed_margin_from_predictions")
    parser.add_argument("--max-embryos", type=int, default=1000000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = args.pairwise_root / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing predictions.parquet at {pred_path}. Re-run the all-pairs job with save_predictions=True."
        )

    predictions = pd.read_parquet(pred_path)

    figures_dir = RESULTS_ROOT / "figures" / "classification" / args.results_subdir
    results_dir = RESULTS_ROOT / "results" / "classification" / args.results_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison_ids = (
        predictions[predictions["feature_set"].astype(str) == args.feature_set]["comparison_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    if not comparison_ids:
        raise ValueError(f"No rows found for feature_set={args.feature_set!r} in {pred_path}")

    for comparison_id in sorted(comparison_ids):
        slug = comparison_id.replace("__vs__", "_vs_")
        pred_csv = results_dir / f"embryo_predictions_{slug}.csv"
        fig_path = figures_dir / f"embryo_trajectories_signed_margin_{slug}.png"

        predictions[
            (predictions["comparison_id"].astype(str) == comparison_id)
            & (predictions["feature_set"].astype(str) == args.feature_set)
        ].to_csv(pred_csv, index=False)

        plot_margin_trends(
            predictions,
            comparison_id=comparison_id,
            feature_id=args.feature_set,
            max_embryos=int(args.max_embryos),
            output_path=fig_path,
        )
        print(fig_path)


if __name__ == "__main__":
    main()
