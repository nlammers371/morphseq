from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_qc_reason_death_review_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from attrition_qc.config import BUILD04_DIR, EXPERIMENT_IDS, FIGURES_BASE, RESULTS_BASE  # noqa: E402
from attrition_qc.io import ensure_output_dirs, save_manifest  # noqa: E402
from attrition_qc.review import (  # noqa: E402
    load_review_dataframe,
    plot_dead_flag_review,
    plot_frame_reason_breakdown,
    plot_granular_exclusion_flags,
    plot_granular_exclusion_flags_alive,
    plot_sam2_reason_breakdown,
    select_dead_flag_review_embryos,
    summarize_dead_flag_agreement,
    summarize_frame_reasons,
    summarize_granular_exclusion_flags,
    summarize_alive_granular_exclusion_flags,
    summarize_sam2_reasons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Granular SAM2 reason review and dead-flag persistence review.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_BASE / "embryo_attrition_qc_audit" / "reason_and_death_review",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_BASE / "embryo_attrition_qc_audit" / "reason_and_death_review",
    )
    parser.add_argument("--build04-dir", type=Path, default=BUILD04_DIR)
    parser.add_argument("--experiment-ids", nargs="+", default=EXPERIMENT_IDS)
    parser.add_argument("--bin-width", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.results_dir, args.figures_dir)

    print("Loading review dataframe ...")
    df = load_review_dataframe(build_dir=args.build04_dir, experiment_ids=list(args.experiment_ids))
    print(f"  {len(df)} frame rows")

    print("Summarizing SAM2 QC reasons ...")
    sam2_summary = summarize_sam2_reasons(df, bin_width=float(args.bin_width))
    sam2_summary.to_csv(args.results_dir / "sam2_qc_reason_summary.csv", index=False)
    plot_sam2_reason_breakdown(
        sam2_summary,
        output_path=args.figures_dir / "sam2_qc_reason_fraction_of_present_over_time.png",
        denominator="present",
    )
    plot_sam2_reason_breakdown(
        sam2_summary,
        output_path=args.figures_dir / "sam2_qc_reason_fraction_of_excluded_over_time.png",
        denominator="excluded",
    )

    print("Summarizing reconstructed frame QC reasons ...")
    frame_summary = summarize_frame_reasons(df, bin_width=float(args.bin_width))
    frame_summary.to_csv(args.results_dir / "frame_qc_reason_summary.csv", index=False)
    plot_frame_reason_breakdown(
        frame_summary,
        output_path=args.figures_dir / "frame_qc_reason_fraction_of_present_over_time.png",
        denominator="present",
    )
    plot_frame_reason_breakdown(
        frame_summary,
        output_path=args.figures_dir / "frame_qc_reason_fraction_of_excluded_over_time.png",
        denominator="excluded",
    )

    print("Summarizing combined granular exclusion reasons ...")
    granular_summary = summarize_granular_exclusion_flags(df, bin_width=float(args.bin_width))
    granular_summary.to_csv(args.results_dir / "granular_exclusion_flag_summary.csv", index=False)
    plot_granular_exclusion_flags(
        granular_summary,
        output_path=args.figures_dir / "granular_exclusion_flag_fraction_of_present_over_time.png",
        denominator="present",
    )
    plot_granular_exclusion_flags(
        granular_summary,
        output_path=args.figures_dir / "granular_exclusion_flag_fraction_of_excluded_over_time.png",
        denominator="excluded",
    )

    print("Summarizing granular exclusion reasons for alive-only bins ...")
    alive_summary = summarize_alive_granular_exclusion_flags(df, bin_width=float(args.bin_width))
    alive_summary.to_csv(args.results_dir / "granular_exclusion_flag_summary_alive.csv", index=False)
    plot_granular_exclusion_flags_alive(
        alive_summary,
        output_path=args.figures_dir / "granular_exclusion_flag_fraction_of_alive_present_over_time.png",
        denominator="present",
    )
    plot_granular_exclusion_flags_alive(
        alive_summary,
        output_path=args.figures_dir / "granular_exclusion_flag_fraction_of_alive_excluded_over_time.png",
        denominator="excluded",
    )

    print("Selecting dead-flag review embryos ...")
    selected = select_dead_flag_review_embryos(df, bin_width=float(args.bin_width), max_per_genotype=3)
    selected.to_csv(args.results_dir / "dead_flag_review_embryos.csv", index=False)
    dead_agreement = summarize_dead_flag_agreement(df)
    dead_agreement.to_csv(args.results_dir / "dead_flag_agreement_summary.csv", index=False)
    plot_dead_flag_review(df, selected, output_path=args.figures_dir / "dead_flag_persistence_review.png")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(args.results_dir),
        "figures_dir": str(args.figures_dir),
        "build04_dir": str(args.build04_dir),
        "experiment_ids": list(args.experiment_ids),
        "bin_width": float(args.bin_width),
        "outputs": {
            "sam2_qc_reason_summary.csv": "Granular SAM2 subreason summary by genotype and time",
            "frame_qc_reason_summary.csv": "Best-effort reconstructed frame QC subreason summary by genotype and time",
            "granular_exclusion_flag_summary.csv": "Combined granular exclusion reason summary with death, no-yolk, SAM2, and reconstructed frame reasons",
            "granular_exclusion_flag_summary_alive.csv": "Same summary filtered to alive embryo-bins only",
            "dead_flag_review_embryos.csv": "Embryos selected from dead_flag spike bins for persistence review",
            "dead_flag_agreement_summary.csv": "Embryo-level agreement between raw dead_flag and persistence-based dead_flag2",
            "sam2_qc_reason_fraction_of_present_over_time.png": "SAM2 subreason fractions over present embryo-bins",
            "sam2_qc_reason_fraction_of_excluded_over_time.png": "SAM2 subreason fractions over excluded embryo-bins",
            "frame_qc_reason_fraction_of_present_over_time.png": "Reconstructed frame subreason fractions over present embryo-bins",
            "frame_qc_reason_fraction_of_excluded_over_time.png": "Reconstructed frame subreason fractions over excluded embryo-bins",
            "granular_exclusion_flag_fraction_of_present_over_time.png": "Combined granular exclusionary reason fractions over present embryo-bins",
            "granular_exclusion_flag_fraction_of_excluded_over_time.png": "Combined granular exclusionary reason fractions over excluded embryo-bins",
            "granular_exclusion_flag_fraction_of_alive_present_over_time.png": "Combined granular exclusionary reason fractions over present alive embryo-bins",
            "granular_exclusion_flag_fraction_of_alive_excluded_over_time.png": "Combined granular exclusionary reason fractions over excluded alive embryo-bins",
            "dead_flag_persistence_review.png": "Per-embryo fraction_alive review using current dead_flag2 persistence method",
        },
    }
    save_manifest(args.results_dir / "reason_and_death_review_manifest.json", manifest)

    print("Done.")
    print(f"  Results: {args.results_dir}")
    print(f"  Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
