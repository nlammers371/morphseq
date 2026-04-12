from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import RERUN_ID, resolve_embedding_roots, write_manifest

PAIR = ("pbx4_crispant", "pbx1b_pbx4_crispant")
SCOPES = {
    "20251207_pbx": ["20251207_pbx"],
    "20260304": ["20260304"],
    "20260306": ["20260306"],
    "combined": ["20251207_pbx", "20260304", "20260306"],
}


def _run(cmd: list[str]) -> None:
    print("$", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pbx4 vs pbx1b+4 pairwise classification per experiment and combined.")
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-embryos", type=int, default=30)
    parser.add_argument("--results-subdir", default=f"{RERUN_ID}_pbx4_vs_double")
    parser.add_argument("--figures-subdir", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    figures_subdir = args.figures_subdir or args.results_subdir
    results_dir, figures_dir = resolve_embedding_roots(
        results_subdir=args.results_subdir,
        figures_subdir=figures_subdir,
    )

    commands = []
    for scope, experiment_ids in SCOPES.items():
        cmd = [
            python,
            str(script_dir / "04_plot_pairwise_signed_margin.py"),
            "--group1",
            PAIR[0],
            "--group2",
            PAIR[1],
            "--bin-width",
            f"{args.bin_width:.1f}",
            "--n-jobs",
            str(args.n_jobs),
            "--n-permutations",
            str(args.n_permutations),
            "--n-splits",
            str(args.n_splits),
            "--max-embryos",
            str(args.max_embryos),
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
            "--experiment-ids",
            *experiment_ids,
        ]
        commands.append((scope, experiment_ids, cmd))

    for _, _, cmd in commands:
        _run(cmd)

    write_manifest(
        results_dir / "pairwise_pbx4_vs_double_manifest.json",
        {
            "analysis": "pairwise_pbx4_vs_double_by_experiment",
            "pair": list(PAIR),
            "scopes": {scope: ids for scope, ids, _ in commands},
            "bin_width_hpf": args.bin_width,
            "n_jobs": args.n_jobs,
            "n_permutations": args.n_permutations,
            "n_splits": args.n_splits,
            "max_embryos": args.max_embryos,
            "commands": [list(map(str, cmd)) for _, _, cmd in commands],
        },
    )
    print(results_dir)
    print(figures_dir)


if __name__ == "__main__":
    main()
