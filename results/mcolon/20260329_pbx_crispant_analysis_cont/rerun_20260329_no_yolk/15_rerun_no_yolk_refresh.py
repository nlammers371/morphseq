from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import (
    EXPERIMENT_IDS,
    RERUN_ID,
    resolve_bin_width_roots,
    resolve_embedding_roots,
    write_manifest,
)


def _run(cmd: list[str]) -> None:
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PBX no-yolk refresh into a dated rerun namespace.")
    parser.add_argument("--bin-width", type=float, default=2.0, help="Time bin width for the refreshed classification branch.")
    parser.add_argument("--rerun-id", default=RERUN_ID, help="Relative rerun subdir under the PBX analysis results/figures roots.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Workers for classification reruns.")
    parser.add_argument("--n-permutations", type=int, default=500, help="Permutation count for 02_classification_only.py.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    results_root, figures_root = resolve_bin_width_roots(
        bin_width=args.bin_width,
        results_subdir=args.rerun_id,
        figures_subdir=args.rerun_id,
    )
    embedding_results_dir, embedding_figures_dir = resolve_embedding_roots(
        results_subdir=args.rerun_id,
        figures_subdir=args.rerun_id,
    )

    shared_subdir_args = ["--results-subdir", args.rerun_id, "--figures-subdir", args.rerun_id]
    classification_args = [
        "--bin-width",
        f"{args.bin_width:.1f}",
        "--n-jobs",
        str(args.n_jobs),
    ]

    commands: list[list[str]] = [
        [
            python,
            str(script_dir / "02_classification_only.py"),
            *classification_args,
            "--n-permutations",
            str(args.n_permutations),
            *shared_subdir_args,
            "--mode",
            "all",
        ],
        [
            python,
            str(script_dir / "01_all_crispants_vs_wik_ab.py"),
            *classification_args,
            "--n-permutations",
            str(args.n_permutations),
            *shared_subdir_args,
        ],
        [
            python,
            str(script_dir / "data_viz" / "01_plot_classification_heatmaps.py"),
            "--bin-width",
            f"{args.bin_width:.1f}",
            *shared_subdir_args,
        ],
        [
            python,
            str(script_dir / "data_viz" / "03_plot_all_vs_inj_ctrl_heatmaps.py"),
            "--bin-width",
            f"{args.bin_width:.1f}",
            *shared_subdir_args,
        ],
        [
            python,
            str(script_dir / "data_viz" / "04_plot_control_comparison_heatmaps.py"),
            "--bin-width",
            f"{args.bin_width:.1f}",
            *shared_subdir_args,
        ],
    ]

    pairwise_specs = [
        ("inj_ctrl", "wik_ab"),
        ("inj_ctrl", "pbx1b_crispant"),
        ("inj_ctrl", "pbx4_crispant"),
        ("inj_ctrl", "pbx1b_pbx4_crispant"),
        ("pbx4_crispant", "pbx1b_pbx4_crispant"),
    ]
    for group1, group2 in pairwise_specs:
        commands.append(
            [
                python,
                str(script_dir / "04_plot_pairwise_signed_margin.py"),
                "--group1",
                group1,
                "--group2",
                group2,
                "--bin-width",
                f"{args.bin_width:.1f}",
                "--n-jobs",
                str(args.n_jobs),
                "--results-dir",
                str(embedding_results_dir),
                "--figures-dir",
                str(embedding_figures_dir),
            ]
        )

    commands.extend(
        [
            [
                python,
                str(script_dir / "05_plot_vs_inj_ctrl_mean_signed_margin.py"),
                "--results-dir",
                str(embedding_results_dir),
                "--figures-dir",
                str(embedding_figures_dir),
            ],
            [
                python,
                str(script_dir / "06_rank_pbx4_morphology_drivers.py"),
                "--results-dir",
                str(embedding_results_dir),
                "--figures-dir",
                str(embedding_figures_dir),
                "--predictions",
                str(embedding_results_dir / "embryo_predictions_inj_ctrl_vs_pbx4_crispant.csv"),
                "--bin-width",
                f"{args.bin_width:.1f}",
            ],
            [
                python,
                str(script_dir / "07_rank_pairwise_continuum_drivers.py"),
                "--results-dir",
                str(embedding_results_dir),
                "--figures-dir",
                str(embedding_figures_dir),
                "--predictions",
                str(embedding_results_dir / "embryo_predictions_pbx4_crispant_vs_pbx1b_pbx4_crispant.csv"),
                "--bin-width",
                f"{args.bin_width:.1f}",
            ],
            [
                python,
                str(script_dir / "08_all_pairs_overlap_analysis.py"),
                "--results-dir",
                str(embedding_results_dir),
                "--figures-dir",
                str(embedding_figures_dir),
                "--bin-width",
                f"{args.bin_width:.1f}",
            ],
        ]
    )

    for cmd in commands:
        _run(cmd)

    manifest_path = results_root.parent / "rerun_manifest.json"
    summary_path = results_root.parent / "rerun_summary.txt"
    write_manifest(
        manifest_path,
        {
            "analysis": "pbx_no_yolk_refresh",
            "source_experiments": EXPERIMENT_IDS,
            "bin_width_hpf": args.bin_width,
            "n_jobs": args.n_jobs,
            "n_permutations_02_classification_only": args.n_permutations,
            "policy_change": "no_yolk_flag removed from build04 use_embryo_flag exclusion logic; downstream build06 refreshed before rerun",
            "commands": [list(map(str, cmd)) for cmd in commands],
        },
    )

    summary_lines = [
        f"rerun_id: {args.rerun_id}",
        f"source_experiments: {','.join(EXPERIMENT_IDS)}",
        f"bin_width_hpf: {args.bin_width:.1f}",
        "",
        "output_roots:",
        f"- results_bin_root: {results_root}",
        f"- figures_bin_root: {figures_root}",
        f"- embedding_results_root: {embedding_results_dir}",
        f"- embedding_figures_root: {embedding_figures_dir}",
        "",
        "notable refreshed figures:",
        f"- {figures_root / f'20260304_20260306_three_features_stacked.png'}",
        f"- {figures_root / 'classification' / '20260304_20260306_all_crispants_vs_wik_ab_heatmaps_v2.png'}",
        f"- {figures_root / 'classification' / '20260304_20260306_all_genotypes_vs_inj_ctrl_heatmaps_v2.png'}",
        f"- {figures_root / 'classification' / '20260304_20260306_wik_ab_vs_inj_ctrl_heatmaps_v2.png'}",
        f"- {embedding_figures_dir / 'pbx4_wildtype_like_morphology_trajectories.png'}",
        f"- {embedding_figures_dir / 'pairwise_auroc_heatmap_pbx_controls_embedding_all_pairs.png'}",
        "",
        f"manifest: {manifest_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
