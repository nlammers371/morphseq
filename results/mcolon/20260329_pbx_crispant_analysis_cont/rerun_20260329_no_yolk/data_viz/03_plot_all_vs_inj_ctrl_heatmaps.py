#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import EXPERIMENT_LABEL, resolve_bin_width_roots


def _load_save_mode_heatmaps():
    helper_path = Path(__file__).with_name("01_plot_classification_heatmaps.py")
    spec = importlib.util.spec_from_file_location("pbx_rerun_heatmaps", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load heatmap helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.save_mode_heatmaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PBX rerun all-genotypes-vs-inj_ctrl AUROC heatmaps.")
    parser.add_argument("--bin-width", type=float, default=2.0, help="Time bin width in hpf.")
    parser.add_argument("--results-subdir", default=None, help="Relative results subdir under the PBX analysis root.")
    parser.add_argument("--figures-subdir", default=None, help="Relative figures subdir under the PBX analysis root.")
    args = parser.parse_args()

    results_dir, figures_dir = resolve_bin_width_roots(
        bin_width=args.bin_width,
        results_subdir=args.results_subdir,
        figures_subdir=args.figures_subdir,
    )
    out_path = _load_save_mode_heatmaps()(
        classification_dir=results_dir / "classification",
        output_path=figures_dir / "classification" / f"{EXPERIMENT_LABEL}_all_genotypes_vs_inj_ctrl_heatmaps_v2.png",
        mode_stem="all_genotypes_vs_inj_ctrl",
        title="All Genotypes vs inj_ctrl",
    )
    print(out_path)


if __name__ == "__main__":
    main()
