from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260407_subset_umap_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator")
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.trajectory_condensation import init_embedding, schema
from analyze.trajectory_condensation.viz import plotting

from common import GENOTYPE_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare aligned UMAP inits under row-subset conditions.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _render_bundle(
    *,
    out_dir: Path,
    title_prefix: str,
    x0: np.ndarray,
    data: schema.CondensationData,
) -> None:
    labels = np.asarray(data.labels, dtype=object)
    color_map = {g: GENOTYPE_COLORS.get(g, "#555555") for g in np.unique(labels)}

    fig, _ = plotting.plot_trajectories(
        x0,
        data.mask,
        data.time_values,
        labels=labels,
        color_map=color_map,
        title=f"{title_prefix} trajectories",
    )
    fig.savefig(out_dir / "plot_trajectories_init.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_stacked_3d(
        x0,
        data.mask,
        data.time_values,
        labels=labels,
        color_map=color_map,
        title=f"{title_prefix} stacked 3D",
    )
    fig.savefig(out_dir / "plot_stacked_3d_init.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        out_dir / "x0_init.npz",
        x0=x0,
        mask=data.mask,
        time_values=data.time_values,
        embryo_ids=data.embryo_ids,
        labels=labels,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        (
            "all_5class",
            ["inj_ctrl", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant", "wik_ab"],
            "All 5 classes",
        ),
        (
            "all_except_wikab",
            ["inj_ctrl", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant"],
            "All except wik_ab",
        ),
        (
            "all_except_injctrl",
            ["pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant", "wik_ab"],
            "All except inj_ctrl",
        ),
    ]

    full_data = schema.from_pairwise_margin_csv(args.input)

    for slug, genotypes, title in configs:
        run_dir = args.output_dir / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        data = schema.subset_pairwise(full_data, genotypes)
        x0 = init_embedding.aligned_umap_init(
            data.features,
            data.mask,
            n_neighbors=15,
            min_dist=0.1,
            alignment_regularisation=1e-2,
            alignment_window_size=3,
            random_state=int(args.seed),
        )
        _render_bundle(out_dir=run_dir, title_prefix=title, x0=x0, data=data)
        print(run_dir)


if __name__ == "__main__":
    main()
