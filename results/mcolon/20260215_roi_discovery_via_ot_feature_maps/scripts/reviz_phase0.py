#!/usr/bin/env python3
"""
Re-render cached Phase 0 contour figures from an existing run directory.

This script does not recompute OT. It reads the stored feature dataset,
reuses the cached reference mask and QC outlier flags, and writes one or
more contour-only variants for presentation or publication use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
ROI_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_NAME = "phase0_run_004"

sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))
sys.path.insert(0, str(ROI_DIR))

from viz import plot_cost_density_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


_PHASE0_LOADER_PATH = ROI_DIR / "io" / "phase0.py"

PRESETS = {
    "publication": {
        "contour_linewidth": 0.45,
        "outline_linewidth": 1.8,
        "contour_level_count": 12,
        "diff_level_count": 15,
        "dpi": 150,
    },
    "presentation": {
        "contour_linewidth": 1.05,
        "outline_linewidth": 2.5,
        "contour_level_count": 9,
        "diff_level_count": 11,
        "dpi": 300,
    },
}


def _load_phase0_loader():
    spec = importlib.util.spec_from_file_location("phase0_loader", _PHASE0_LOADER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Phase0Loader from {_PHASE0_LOADER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Phase0Loader


def _resolve_run_dir(run_name: str | None, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    chosen = run_name or DEFAULT_RUN_NAME
    return (SCRIPTS_DIR / "output" / chosen).resolve()


def _load_outlier_flag(loader) -> np.ndarray:
    if "qc/outlier_flag" in loader._z:
        return loader._z["qc/outlier_flag"][:].astype(bool)
    logger.warning("No cached qc/outlier_flag in run; proceeding with no exclusions")
    return np.zeros(loader.X_zarr.shape[0], dtype=bool)


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "feature_dataset" / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as fh:
        return json.load(fh)


def _build_palette_specs() -> dict[str, dict[str, object]]:
    nwdb_mean = LinearSegmentedColormap.from_list(
        "nwdb_mean",
        ["#f7f4ea", "#cfe9e6", "#79c9c4", "#2FB7B0", "#0d6f6b"],
    )
    nwdb_diff = LinearSegmentedColormap.from_list(
        "nwdb_diff",
        ["#1d7874", "#bfe3de", "#f7f4ea", "#f3c6d7", "#E76FA2", "#8d2b57"],
    )
    return {
        "warm": {
            "mean_cmap": "hot",
            "diff_cmap": "RdBu_r",
        },
        "nwdb": {
            "mean_cmap": nwdb_mean,
            "diff_cmap": nwdb_diff,
        },
        "cividis": {
            "mean_cmap": "cividis",
            "diff_cmap": "coolwarm",
        },
    }


def render_reviz(
    run_dir: Path,
    output_dir: Path | None = None,
    sigma: float = 2.0,
    preset: str = "both",
    background: str = "filled",
    palette: str = "warm",
    show_colorbars: bool = True,
) -> list[Path]:
    """Render publication and/or presentation contour figures from a cached run."""
    Phase0Loader = _load_phase0_loader()
    loader = Phase0Loader(run_dir)
    manifest = _load_manifest(run_dir)

    if output_dir is None:
        output_dir = run_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    X = loader.get_X()
    y = loader.get_y()
    mask_ref = loader.get_mask_ref()
    outlier_flag = _load_outlier_flag(loader)

    logger.info("Run directory: %s", run_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Reference mask id: %s", manifest.get("reference_mask_id", "unknown"))
    logger.info("Feature set: %s", manifest.get("feature_set", "unknown"))
    logger.info("Loaded X=%s, mask_ref=%s, outliers=%d", X.shape, mask_ref.shape, int(outlier_flag.sum()))

    preset_names = ["publication", "presentation"] if preset == "both" else [preset]
    palette_specs = _build_palette_specs()
    palette_names = list(palette_specs) if palette == "bundle" else [palette]
    written: list[Path] = []

    for palette_name in palette_names:
        if palette_name not in palette_specs:
            raise KeyError(f"Unknown palette {palette_name!r}. Available: {list(palette_specs)}")
        cmaps = palette_specs[palette_name]
        for preset_name in preset_names:
            style = PRESETS[preset_name]
            figs = plot_cost_density_suite(
                X,
                y,
                mask_ref,
                outlier_flag,
                sigma_grid=(sigma,),
                contour_linewidth=style["contour_linewidth"],
                outline_linewidth=style["outline_linewidth"],
                contour_level_count=style["contour_level_count"],
                diff_level_count=style["diff_level_count"],
                contour_background=background,
                mean_cmap=cmaps["mean_cmap"],
                diff_cmap=cmaps["diff_cmap"],
                show_contour_colorbars=show_colorbars,
                save_dir=None,
            )

            fig_key = f"cost_contour_sigma{sigma}"
            if fig_key not in figs:
                for fig in figs.values():
                    plt.close(fig)
                raise KeyError(f"Expected contour figure key '{fig_key}' not found; got {list(figs)}")

            bg_suffix = "" if background == "filled" else f"_{background}"
            out_path = output_dir / f"fig_A_cost_contour_sigma{sigma:.0f}{bg_suffix}_{palette_name}_{preset_name}.png"
            figs[fig_key].savefig(out_path, dpi=style["dpi"], bbox_inches="tight")
            written.append(out_path)
            logger.info("Saved %s", out_path)

            for fig in figs.values():
                plt.close(fig)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-render cached Phase 0 contour figures")
    parser.add_argument("run_name", nargs="?", default=DEFAULT_RUN_NAME,
                        help=f"Run directory name under scripts/output (default: {DEFAULT_RUN_NAME})")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Explicit run directory. Overrides positional run_name.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Destination for re-rendered figures (default: <run_dir>/viz)")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma for contour rendering")
    parser.add_argument("--preset", choices=("publication", "presentation", "both"), default="both",
                        help="Which style preset(s) to render")
    parser.add_argument("--background", choices=("filled", "raw"), default="filled",
                        help="Use filled contours or overlay contour lines on the raw heatmap")
    parser.add_argument("--palette", choices=("warm", "nwdb", "cividis", "bundle"), default="warm",
                        help="Continuous color palette or a 3-palette export bundle")
    parser.add_argument("--no-colorbars", action="store_true",
                        help="Disable per-panel colorbars on contour exports")
    args = parser.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_name, args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    render_reviz(
        run_dir=run_dir,
        output_dir=args.output_dir,
        sigma=float(args.sigma),
        preset=args.preset,
        background=args.background,
        palette=args.palette,
        show_colorbars=not args.no_colorbars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
