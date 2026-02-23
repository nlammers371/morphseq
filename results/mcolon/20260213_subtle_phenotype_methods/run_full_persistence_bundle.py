#!/usr/bin/env python
"""Run embryo-first persistence discovery + validation + enrichment end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PERSISTENCE_ROOT = HERE / "output" / "embryo_first_persistence"


def _latest_run(root: Path) -> Path:
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories found in {root}")
    return runs[-1]


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None, help="Use existing run dir for validation/enrichment")
    parser.add_argument("--persistence-root", type=Path, default=DEFAULT_PERSISTENCE_ROOT)

    parser.add_argument("--scope-mode", default="within_experiment")
    parser.add_argument("--feature-mode", default="auto")
    parser.add_argument("--dataset-ids", default="")
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.skip_discovery:
        cmd = [
            py,
            str(HERE / "run_embryo_first_persistence.py"),
            "--scope-mode",
            args.scope_mode,
            "--feature-mode",
            args.feature_mode,
            "--n-bootstrap",
            str(args.n_bootstrap),
            "--n-jobs",
            str(args.n_jobs),
        ]
        if args.dataset_ids:
            cmd.extend(["--dataset-ids", args.dataset_ids])
        _run(cmd)

    run_dir = args.run_dir.resolve() if args.run_dir else _latest_run(args.persistence_root)

    _run(
        [
            py,
            str(HERE / "validate_persistence_clusters.py"),
            "--run-dir",
            str(run_dir),
            "--n-permutations",
            str(args.n_permutations),
            "--n-jobs",
            str(args.n_jobs),
        ]
    )

    _run(
        [
            py,
            str(HERE / "cluster_enrichment_analysis.py"),
            "--run-dir",
            str(run_dir),
        ]
    )

    print(f"Bundle complete. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
