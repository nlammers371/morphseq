#!/usr/bin/env python
"""Generate toy single-cell-like 2D cluster panels.

The colors are approximated from the supplied embryo stack figure:
coral, gold, green, teal, and blue. The script writes one CSV containing
all simulated points plus one image per random configuration.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUT_DIR = Path(
    "/Users/nick/Projects/data/morphseq/results/20260528/sac_figs/"
    "simulated_single_cell_clusters"
)

PALETTE = [
    "#e75f3f",  # coral/red
    "#f1bf64",  # gold
    "#8bc57c",  # green
    "#5b9fa6",  # teal
    "#3d78bf",  # blue
]

CLUSTER_LABELS = ["coral", "gold", "green", "teal", "blue"]
PALETTE_BY_LABEL = dict(zip(CLUSTER_LABELS, PALETTE, strict=True))


def allocate_cells(n_cells: int, n_clusters: int) -> np.ndarray:
    """Split cells as evenly as possible across clusters."""
    base = n_cells // n_clusters
    counts = np.full(n_clusters, base, dtype=int)
    counts[: n_cells - base * n_clusters] += 1
    return counts


def sample_cluster_centers(
    rng: np.random.Generator,
    n_clusters: int,
    min_distance: float = 0.45,
    bounds: tuple[float, float] = (-1.05, 1.05),
) -> np.ndarray:
    """Sample well-separated cluster centers with simple rejection sampling."""
    centers: list[np.ndarray] = []
    attempts = 0
    while len(centers) < n_clusters and attempts < 10_000:
        attempts += 1
        candidate = rng.uniform(bounds[0], bounds[1], size=2)
        if not centers:
            centers.append(candidate)
            continue
        distances = np.linalg.norm(np.vstack(centers) - candidate, axis=1)
        if np.min(distances) >= min_distance:
            centers.append(candidate)

    if len(centers) < n_clusters:
        raise RuntimeError("Could not place cluster centers with the requested spacing.")
    return np.vstack(centers)


def sample_cluster_points(
    rng: np.random.Generator,
    center: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Sample a compact 2D Gaussian cluster with random orientation."""
    theta = rng.uniform(0, np.pi)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    scales = rng.uniform(0.16, 0.36, size=2)
    covariance = rotation @ np.diag(scales**2) @ rotation.T
    return rng.multivariate_normal(center, covariance, size=n_points)


def simulate_configuration(
    rng: np.random.Generator,
    config_id: int,
    n_cells: int,
    n_clusters: int,
    *,
    center_min_distance: float,
    center_bound: float,
) -> list[dict[str, object]]:
    """Create one simulated single-cell configuration."""
    counts = allocate_cells(n_cells, n_clusters)
    centers = sample_cluster_centers(
        rng,
        n_clusters,
        min_distance=center_min_distance,
        bounds=(-center_bound, center_bound),
    )

    rows: list[dict[str, object]] = []
    cell_id = 0
    for cluster_id, n_points in enumerate(counts):
        points = sample_cluster_points(rng, centers[cluster_id], int(n_points))
        for x, y in points:
            rows.append(
                {
                    "config": config_id,
                    "cell_id": cell_id,
                    "cluster_id": cluster_id,
                    "cluster_label": CLUSTER_LABELS[cluster_id % len(CLUSTER_LABELS)],
                    "x": float(x),
                    "y": float(y),
                    "color": PALETTE[cluster_id % len(PALETTE)],
                }
            )
            cell_id += 1
    return rows


def darken_hex(hex_color: str, amount: float = 0.52) -> str:
    """Return a darker version of a hex color for subtle marker outlines."""
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)])
    rgb = np.clip(rgb * amount, 0, 255).astype(int)
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def set_tight_equal_limits(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    *,
    pad_fraction: float = 0.08,
    min_span: float = 2.2,
) -> None:
    """Crop around a configuration while keeping equal x/y scaling."""
    x = np.array([row["x"] for row in rows], dtype=float)
    y = np.array([row["y"] for row in rows], dtype=float)
    x_mid = float((x.min() + x.max()) / 2)
    y_mid = float((y.min() + y.max()) / 2)
    span = max(float(x.max() - x.min()), float(y.max() - y.min()), min_span)
    half_width = span * (1 + pad_fraction) / 2
    ax.set_xlim(x_mid - half_width, x_mid + half_width)
    ax.set_ylim(y_mid - half_width, y_mid + half_width)


def plot_configuration(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    dpi: int = 300,
    marker_size_scale: float = 4.5,
    pad_fraction: float = 0.08,
    min_span: float = 2.2,
    uniform_color: str | None = None,
) -> None:
    """Plot one configuration as circular colored markers."""
    fig, ax = plt.subplots(figsize=(4.0, 4.0), facecolor="none")
    ax.set_facecolor("none")

    for cluster_id, color in enumerate(PALETTE):
        plot_color = uniform_color or color
        cluster_rows = [row for row in rows if row["cluster_id"] == cluster_id]
        x = [row["x"] for row in cluster_rows]
        y = [row["y"] for row in cluster_rows]
        ax.scatter(
            x,
            y,
            s=78 * marker_size_scale,
            marker="o",
            c=plot_color,
            edgecolors=darken_hex(plot_color),
            linewidths=1.0,
            alpha=0.88,
        )

    ax.set_aspect("equal", adjustable="box")
    set_tight_equal_limits(ax, rows, pad_fraction=pad_fraction, min_span=min_span)
    ax.axis("off")
    fig.savefig(output_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_overview(
    configs: list[list[dict[str, object]]],
    output_path: Path,
    *,
    dpi: int = 300,
    marker_size_scale: float = 4.5,
    pad_fraction: float = 0.08,
    min_span: float = 2.2,
    montage_hspace: float = 0.22,
    uniform_color: str | None = None,
) -> None:
    """Plot configurations as a vertical single-column montage."""
    fig, axes = plt.subplots(len(configs), 1, figsize=(3.6, 3.1 * len(configs)), facecolor="none")
    if len(configs) == 1:
        axes = np.array([axes])

    for ax, rows in zip(axes.ravel(), configs, strict=True):
        ax.set_facecolor("none")
        for cluster_id, color in enumerate(PALETTE):
            plot_color = uniform_color or color
            cluster_rows = [row for row in rows if row["cluster_id"] == cluster_id]
            ax.scatter(
                [row["x"] for row in cluster_rows],
                [row["y"] for row in cluster_rows],
                s=28 * marker_size_scale,
                marker="o",
                c=plot_color,
                edgecolors=darken_hex(plot_color),
                linewidths=0.55,
                alpha=0.88,
            )
        ax.set_aspect("equal", adjustable="box")
        set_tight_equal_limits(ax, rows, pad_fraction=pad_fraction, min_span=min_span)
        ax.axis("off")

    fig.subplots_adjust(hspace=montage_hspace)
    fig.savefig(output_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write simulated point coordinates and cluster metadata."""
    fieldnames = ["config", "cell_id", "cluster_id", "cluster_label", "x", "y", "color"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-configs", type=int, default=10)
    parser.add_argument("--n-montage-configs", type=int, default=6)
    parser.add_argument("--n-cells", type=int, default=100)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--marker-size-scale", type=float, default=4.5)
    parser.add_argument("--center-bound", type=float, default=1.05)
    parser.add_argument("--center-min-distance", type=float, default=0.45)
    parser.add_argument("--panel-pad-fraction", type=float, default=0.08)
    parser.add_argument("--min-panel-span", type=float, default=2.2)
    parser.add_argument("--montage-hspace", type=float, default=0.22)
    parser.add_argument(
        "--uniform-only",
        action="store_true",
        help="Only write uniform-color alternate figures, leaving standard outputs untouched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_clusters > len(PALETTE):
        raise ValueError(f"This simple palette supports up to {len(PALETTE)} clusters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    configs: list[list[dict[str, object]]] = []
    all_rows: list[dict[str, object]] = []
    for config_id in range(1, args.n_configs + 1):
        rows = simulate_configuration(
            rng,
            config_id,
            args.n_cells,
            args.n_clusters,
            center_min_distance=args.center_min_distance,
            center_bound=args.center_bound,
        )
        configs.append(rows)
        all_rows.extend(rows)

        if not args.uniform_only:
            for suffix in ["png", "pdf"]:
                plot_configuration(
                    rows,
                    args.output_dir / f"simulated_single_cell_config_{config_id:02d}.{suffix}",
                    dpi=args.dpi,
                    marker_size_scale=args.marker_size_scale,
                    pad_fraction=args.panel_pad_fraction,
                    min_span=args.min_panel_span,
                )

    if not args.uniform_only:
        write_csv(all_rows, args.output_dir / "simulated_single_cell_points.csv")
    montage_configs = configs[: min(args.n_montage_configs, len(configs))]
    if not args.uniform_only:
        for suffix in ["png", "pdf"]:
            plot_overview(
                montage_configs,
                args.output_dir / f"simulated_single_cell_all_configs.{suffix}",
                dpi=args.dpi,
                marker_size_scale=args.marker_size_scale,
                pad_fraction=args.panel_pad_fraction,
                min_span=args.min_panel_span,
                montage_hspace=args.montage_hspace,
            )
            plot_overview(
                montage_configs,
                args.output_dir / f"simulated_single_cell_six_panel_montage.{suffix}",
                dpi=args.dpi,
                marker_size_scale=args.marker_size_scale,
                pad_fraction=args.panel_pad_fraction,
                min_span=args.min_panel_span,
                montage_hspace=args.montage_hspace,
            )

    for color_label, uniform_color in PALETTE_BY_LABEL.items():
        for suffix in ["png", "pdf"]:
            plot_overview(
                montage_configs,
                args.output_dir / f"simulated_single_cell_six_panel_montage_uniform_{color_label}.{suffix}",
                dpi=args.dpi,
                marker_size_scale=args.marker_size_scale,
                pad_fraction=args.panel_pad_fraction,
                min_span=args.min_panel_span,
                montage_hspace=args.montage_hspace,
                uniform_color=uniform_color,
            )
        for config_id, rows in enumerate(configs, start=1):
            for suffix in ["png", "pdf"]:
                plot_configuration(
                    rows,
                    args.output_dir / f"simulated_single_cell_config_{config_id:02d}_uniform_{color_label}.{suffix}",
                    dpi=args.dpi,
                    marker_size_scale=args.marker_size_scale,
                    pad_fraction=args.panel_pad_fraction,
                    min_span=args.min_panel_span,
                    uniform_color=uniform_color,
                )

    print(f"Wrote {args.n_configs} configurations to {args.output_dir}")


if __name__ == "__main__":
    main()
