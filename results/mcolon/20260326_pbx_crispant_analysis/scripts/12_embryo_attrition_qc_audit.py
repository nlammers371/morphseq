from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_attrition_qc_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from attrition_qc.config import (  # noqa: E402
    BUILD04_DIR,
    CANONICAL_QC_FLAGS,
    DEFAULT_BIN_WIDTH,
    DEFAULT_GENOTYPES,
    DEFAULT_RAW_TO_ANALYSIS_GENOTYPE,
    DEFAULT_RESULTS_SUBDIR,
    DEFAULT_TIME_COL,
    EXPERIMENT_IDS,
    FIGURES_BASE,
    INFO_QC_FLAGS,
    RESULTS_BASE,
)
from attrition_qc.data import build_embryo_bin_status, load_build04_dataframe  # noqa: E402
from attrition_qc.io import ensure_output_dirs, save_manifest  # noqa: E402
from attrition_qc.plots import (  # noqa: E402
    build_color_palette,
    plot_alive_only_qc_rates,
    plot_alive_only_use_embryo_rate,
    plot_alive_only_use_embryo_rate_by_experiment,
    plot_embryo_presence_over_time,
    plot_exclusionary_flag_fractions,
    plot_excluded_reasons_over_time,
    plot_flag_focus_over_time,
    plot_included_by_experiment,
    plot_included_vs_excluded_over_time,
)
from attrition_qc.summaries import (  # noqa: E402
    summarize_alive_only_qc,
    summarize_attrition,
    summarize_exclusionary_flag_rates,
    summarize_overall_attrition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embryo attrition and QC audit for PBX crispant analysis.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_BASE / DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_BASE / DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--build04-dir", type=Path, default=BUILD04_DIR)
    parser.add_argument("--experiment-ids", nargs="+", default=EXPERIMENT_IDS)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.results_dir, args.figures_dir)

    print("Loading build04 QC data ...")
    raw_df = load_build04_dataframe(
        build_dir=args.build04_dir,
        experiment_ids=list(args.experiment_ids),
        genotype_map=DEFAULT_RAW_TO_ANALYSIS_GENOTYPE,
    )
    print(f"  {len(raw_df)} build04 rows")

    print("Aggregating embryo-bin status ...")
    status_df = build_embryo_bin_status(
        raw_df,
        bin_width=float(args.bin_width),
        time_col=args.time_col,
    )
    print(f"  {len(status_df)} embryo-bin rows")

    print("Computing summary tables ...")
    overall_df = summarize_overall_attrition(status_df)
    by_genotype_time_df = summarize_attrition(status_df, group_cols=["genotype", "time_bin_start", "time_bin_center"])
    by_experiment_genotype_time_df = summarize_attrition(
        status_df,
        group_cols=["experiment_date", "genotype", "time_bin_start", "time_bin_center"],
    )
    flag_fraction_df = summarize_exclusionary_flag_rates(
        status_df,
        group_cols=["genotype", "time_bin_start", "time_bin_center"],
    )
    flag_fraction_by_experiment_df = summarize_exclusionary_flag_rates(
        status_df,
        group_cols=["experiment_date", "genotype", "time_bin_start", "time_bin_center"],
    )
    alive_only_df = summarize_alive_only_qc(status_df, group_cols=["genotype", "time_bin_start", "time_bin_center"])
    alive_only_by_experiment_df = summarize_alive_only_qc(
        status_df,
        group_cols=["experiment_date", "genotype", "time_bin_start", "time_bin_center"],
    )

    print("Writing CSV outputs ...")
    status_df.to_csv(args.results_dir / "embryo_bin_status.csv", index=False)
    overall_df.to_csv(args.results_dir / "overall_attrition_summary.csv", index=False)
    by_genotype_time_df.to_csv(args.results_dir / "attrition_summary_by_genotype_time.csv", index=False)
    by_experiment_genotype_time_df.to_csv(args.results_dir / "attrition_summary_by_experiment_genotype_time.csv", index=False)
    flag_fraction_df.to_csv(args.results_dir / "exclusionary_flag_fractions_by_genotype_time.csv", index=False)
    flag_fraction_by_experiment_df.to_csv(args.results_dir / "exclusionary_flag_fractions_by_experiment_genotype_time.csv", index=False)
    alive_only_df.to_csv(args.results_dir / "alive_only_qc_summary.csv", index=False)
    alive_only_by_experiment_df.to_csv(args.results_dir / "alive_only_qc_summary_by_experiment.csv", index=False)

    print("Rendering figures ...")
    color_palette = build_color_palette(DEFAULT_GENOTYPES)
    plot_embryo_presence_over_time(by_genotype_time_df, figures_dir=args.figures_dir, color_palette=color_palette)
    plot_included_vs_excluded_over_time(by_genotype_time_df, figures_dir=args.figures_dir, color_palette=color_palette)
    plot_excluded_reasons_over_time(by_genotype_time_df, figures_dir=args.figures_dir, detailed=False)
    plot_excluded_reasons_over_time(by_genotype_time_df, figures_dir=args.figures_dir, detailed=True)
    plot_exclusionary_flag_fractions(flag_fraction_df, figures_dir=args.figures_dir, denominator="present")
    plot_exclusionary_flag_fractions(flag_fraction_df, figures_dir=args.figures_dir, denominator="excluded")
    plot_flag_focus_over_time(flag_fraction_df, figures_dir=args.figures_dir, denominator="present")
    plot_flag_focus_over_time(flag_fraction_df, figures_dir=args.figures_dir, denominator="excluded")
    plot_alive_only_use_embryo_rate(alive_only_df, figures_dir=args.figures_dir, color_palette=color_palette)
    plot_alive_only_qc_rates(alive_only_df, figures_dir=args.figures_dir, color_palette=color_palette)
    plot_included_by_experiment(by_experiment_genotype_time_df, figures_dir=args.figures_dir, color_palette=color_palette)
    plot_alive_only_use_embryo_rate_by_experiment(
        alive_only_by_experiment_df,
        figures_dir=args.figures_dir,
        color_palette=color_palette,
    )

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(args.results_dir),
        "figures_dir": str(args.figures_dir),
        "build04_dir": str(args.build04_dir),
        "experiment_ids": list(args.experiment_ids),
        "bin_width": float(args.bin_width),
        "time_col": args.time_col,
        "analysis_unit": "experiment_date x genotype x embryo_id x time_bin",
        "canonical_qc_flags": CANONICAL_QC_FLAGS,
        "info_qc_flags": INFO_QC_FLAGS,
        "raw_to_analysis_genotype": DEFAULT_RAW_TO_ANALYSIS_GENOTYPE,
        "analysis_genotypes": DEFAULT_GENOTYPES,
        "status_logic": {
            "included": "any(use_embryo_flag) within embryo-bin",
            "dead_like": "any(dead_flag or dead_flag2) within embryo-bin",
            "canonical_qc_like": "any(sa_outlier_flag or sam2_qc_flag or frame_flag or no_yolk_flag) within embryo-bin",
            "excluded_non_death_qc": "excluded and not dead_like and canonical_qc_like",
            "excluded_death_involved": "excluded and dead_like",
        },
        "outputs": {
            "embryo_bin_status.csv": "Canonical embryo-bin status table",
            "overall_attrition_summary.csv": "Overall unique embryo summary by genotype",
            "attrition_summary_by_genotype_time.csv": "Genotype x time counts and exclusion splits",
            "attrition_summary_by_experiment_genotype_time.csv": "Experiment x genotype x time counts and exclusion splits",
            "exclusionary_flag_fractions_by_genotype_time.csv": "Per-genotype time series of non-exclusive exclusionary flag fractions over present and excluded embryo-bins",
            "exclusionary_flag_fractions_by_experiment_genotype_time.csv": "Experiment-stratified version of exclusionary flag fractions",
            "alive_only_qc_summary.csv": "Alive-only QC/pass-rate summary by genotype x time",
            "alive_only_qc_summary_by_experiment.csv": "Alive-only QC/pass-rate summary by experiment x genotype x time",
        },
    }
    save_manifest(args.results_dir / "attrition_manifest.json", manifest)

    print("Done.")
    print(f"  Results: {args.results_dir}")
    print(f"  Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
