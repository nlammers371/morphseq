"""
ranked_stack_figure.py
======================
Reusable figure function: one column per embryo, ranked by a chosen metric.

Each column (top → bottom):
  [A] Focus-stacked JPEG cropped to embryo with mask contour
  [B] 15 Z slices (3 rows × 5 cols) cropped to embryo;
      red border on Z slice i if NCC(i, i+1) < ncc_thresh
  [C] Ranked metric bars (green = good, red = bad) with value labels

Public API
----------
    make_ranked_figure(
        examples       : list of dicts with keys:
                           well, t, p, label, color
                           (plus any metric columns, passed through to bars)
        metrics        : list of (key, label, good_high: bool)
                           where key matches a column in the example dicts
        nd2_path       : Path to ND2 file
        masks_dir      : Path to directory of mask PNGs
        images_dir     : Path to directory of focus-stacked JPEGs
        ncc_grids_dir  : Path to directory of .npz grid files
        out_path       : Path to save the figure PNG
        ncc_thresh     : float, threshold for red Z-slice border (default 0.90)
        col_width      : float, inches per column (default 3.6)
        fig_height     : float, total figure height in inches (default 14.0)
        dpi            : int (default 180)
    ) -> matplotlib.figure.Figure
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nd2
from PIL import Image


NCC_THRESH_DEFAULT = 0.90
CROP_MARGIN        = 0.20


# ── image helpers ─────────────────────────────────────────────────────────────

def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)


def _crop_box(mask: np.ndarray, margin: float = CROP_MARGIN) -> tuple[int,int,int,int]:
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + margin)
    H, W = mask.shape
    y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half))
    x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half))
    sh, sw = y1 - y0, x1 - x0
    side = max(sh, sw)
    cy2, cx2 = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, cy2 - side // 2); y1 = min(H, y0 + side)
    x0 = max(0, cx2 - side // 2); x1 = min(W, x0 + side)
    return y0, y1, x0, x1


def _crop(img: np.ndarray, box: tuple) -> np.ndarray:
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def _add_contour(ax, mask_crop: np.ndarray, color: str, lw: float = 1.5) -> None:
    if mask_crop.any():
        ax.contour(mask_crop.astype(float), levels=[0.5],
                   colors=[color], linewidths=lw)


def _mask_path(masks_dir: Path, date: str, well: str, t: int) -> Path:
    return masks_dir / f"{date}_{well}_ch00_t{t:04d}_masks_emnum_1.png"


def _jpeg_path(images_dir: Path, date: str, well: str, t: int) -> Path:
    return images_dir / f"{date}_{well}" / f"{date}_{well}_ch00_t{t:04d}.jpg"


def _npz_path(grids_dir: Path, t: int, p: int) -> Path:
    return grids_dir / f"t{t:03d}_p{p:03d}.npz"


# ── ncc z-pair values from grid ───────────────────────────────────────────────

def _pair_ncc_from_grid(grids_dir: Path, t: int, p: int) -> dict[int, float]:
    """
    Returns {z0: mean_masked_tile_ncc} for each Z-pair from the .npz grid.
    Uses the whole-frame mean (no mask) since this is for border colouring only.
    Returns empty dict if grid missing/corrupt.
    """
    path = _npz_path(grids_dir, t, p)
    if not path.exists():
        return {}
    try:
        import numpy as np
        data = np.load(str(path))
        ncc_grid = data["ncc_grid"]   # (Z-1, Ny, Nx)
        return {z: float(np.nanmean(ncc_grid[z])) for z in range(ncc_grid.shape[0])}
    except Exception:
        return {}


# ── main figure function ──────────────────────────────────────────────────────

def make_ranked_figure(
    examples: list[dict],
    metrics: list[tuple[str, str, bool]],
    nd2_path: Path,
    masks_dir: Path,
    images_dir: Path,
    ncc_grids_dir: Path,
    out_path: Path,
    series_well_map_csv: Path | None = None,
    date: str = "20250912",
    ncc_thresh: float = NCC_THRESH_DEFAULT,
    col_width: float = 3.6,
    fig_height: float = 14.0,
    dpi: int = 180,
) -> plt.Figure:
    """
    Build and save the ranked Z-stack quality figure.

    examples : list of dicts, one per column. Required keys: well, t, p, label, color.
               Any additional keys matching a metric key in `metrics` will be used
               for the bar panel.
    metrics  : list of (key, display_label, good_high).
               key must be present in the example dicts.
    """
    N = len(examples)
    if N == 0:
        raise ValueError("examples list is empty")

    # global metric ranges across all examples for consistent bar scaling
    metric_ranges: dict[str, tuple[float, float]] = {}
    for key, _, _ in metrics:
        vals = [float(ex[key]) for ex in examples
                if key in ex and ex[key] is not None and not np.isnan(float(ex[key]))]
        if vals:
            metric_ranges[key] = (min(vals), max(vals))

    # pre-load masks, focused images, crop boxes, ncc pairs (small, keep in memory)
    # ND2 stacks are loaded one at a time inside the column loop to avoid OOM
    print(f"Pre-loading masks and focused images for {N} embryos...")
    masks_arr:    dict = {}
    focused_imgs: dict = {}
    crop_boxes:   dict = {}
    ncc_pairs:    dict = {}

    for ex in examples:
        key = (ex["well"], ex["t"])
        mp = _mask_path(masks_dir, date, ex["well"], ex["t"])
        masks_arr[key] = np.array(Image.open(mp)).astype(bool) if mp.exists() else None

        jp = _jpeg_path(images_dir, date, ex["well"], ex["t"])
        focused_imgs[key] = (np.array(Image.open(jp).convert("L")).astype(np.float32)
                             if jp.exists() else None)

        if masks_arr[key] is not None:
            crop_boxes[key] = _crop_box(masks_arr[key])

        ncc_pairs[key] = _pair_ncc_from_grid(ncc_grids_dir, ex["t"], ex["p"])

    # ── figure layout ─────────────────────────────────────────────────────────
    FIG_W = N * col_width
    fig = plt.figure(figsize=(FIG_W, fig_height), facecolor="#1a1a1a")

    col_gs = gridspec.GridSpec(
        1, N, figure=fig,
        left=0.01, right=0.99, top=0.94, bottom=0.02,
        wspace=0.06,
    )

    cmap_rg = matplotlib.colormaps["RdYlGn"]

    # build well -> p lookup from series_well_map if provided
    # (p from embryo_ncc_summaries was derived at T=0; use series_well_map for correct dask indexing)
    well_to_p: dict[str, int] = {}
    if series_well_map_csv is not None and Path(series_well_map_csv).exists():
        import pandas as _pd
        _sm = _pd.read_csv(series_well_map_csv)
        well_to_p = {row["well_index"]: int(row["series_number"]) - 1
                     for _, row in _sm.iterrows()}
        print(f"Loaded well->p lookup: {len(well_to_p)} wells from {series_well_map_csv}")

    # open ND2 once for the dask handle; load each stack individually then discard
    print(f"Rendering {N} columns (loading ND2 stacks one at a time)...")
    nd2_file = nd2.ND2File(str(nd2_path))
    dask_arr = nd2_file.to_dask()

    for col_idx, ex in enumerate(examples):
        key     = (ex["well"], ex["t"])
        # use series_well_map p if available, else fall back to ex["p"]
        p_nd2   = well_to_p.get(ex["well"], ex["p"])
        print(f"  [{col_idx+1}/{N}] {ex['label']:8s}  well={ex['well']}  t={ex['t']}  p_nd2={p_nd2}")
        stack   = dask_arr[ex["t"], p_nd2, :, :, :].compute().astype(np.float32)
        mask    = masks_arr[key]
        focused = focused_imgs[key]
        box     = crop_boxes.get(key)
        ncc_v   = ncc_pairs[key]
        color   = ex["color"]

        inner = gridspec.GridSpecFromSubplotSpec(
            5, 1,
            subplot_spec=col_gs[col_idx],
            hspace=0.08,
            height_ratios=[2.5, 1, 1, 1, 3.2],
        )

        # [A] focus-stacked image
        ax_f = fig.add_subplot(inner[0])
        ax_f.set_facecolor("#111")
        if focused is not None and box is not None:
            ax_f.imshow(_crop(_norm01(focused), box), cmap="gray",
                        vmin=0, vmax=1, interpolation="lanczos")
            _add_contour(ax_f, _crop(mask, box), color, lw=2)
        ax_f.set_xticks([]); ax_f.set_yticks([])
        for sp in ax_f.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5)
        ax_f.set_title(f"{ex['label']}\n{ex['well']}  t={ex['t']}",
                       fontsize=9, color=color, fontweight="bold", pad=4)

        # [B] Z slices 3×5
        z_gs = gridspec.GridSpecFromSubplotSpec(
            3, 5,
            subplot_spec=gridspec.GridSpecFromSubplotSpec(
                3, 1, subplot_spec=inner[1:4], hspace=0.0
            )[:],
            hspace=0.04, wspace=0.04,
        )
        z_axes = [fig.add_subplot(z_gs[r, c]) for r in range(3) for c in range(5)]

        for z in range(min(15, stack.shape[0])):
            ax_z = z_axes[z]
            ax_z.set_facecolor("#111")
            if box is not None and mask is not None:
                ax_z.imshow(_crop(_norm01(stack[z]), box), cmap="gray",
                            vmin=0, vmax=1, interpolation="nearest")
                _add_contour(ax_z, _crop(mask, box), color, lw=0.8)
            ax_z.set_xticks([]); ax_z.set_yticks([])

            ncc = ncc_v.get(z)
            if ncc is not None and ncc < ncc_thresh:
                bc, blw = "#FF3333", 2.0
            else:
                bc, blw = "#333333", 0.5
            for sp in ax_z.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(blw)

            if ncc is not None:
                tc = "#FF3333" if ncc < ncc_thresh else "#555555"
                ax_z.set_xlabel(f"{ncc:.2f}", fontsize=4.5, color=tc, labelpad=1)
            ax_z.set_title(f"Z{z}", fontsize=4, color="#666", pad=1)

        # [C] metric bars
        ax_m = fig.add_subplot(inner[4])
        ax_m.set_facecolor("#111")
        ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1)
        ax_m.set_xticks([]); ax_m.set_yticks([])
        for sp in ax_m.spines.values():
            sp.set_edgecolor("#333"); sp.set_linewidth(0.5)

        n_m = len(metrics)
        y_positions = np.linspace(0.88, 0.10, n_m)
        bar_h = 0.10

        for m_i, (mkey, mlabel, good_high) in enumerate(metrics):
            y = y_positions[m_i]
            raw = ex.get(mkey)
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                val = float(raw)
                lo, hi = metric_ranges.get(mkey, (0.0, 1.0))
                span  = hi - lo if hi > lo else 1.0
                frac  = np.clip((val - lo) / span, 0, 1)
                bar_f = frac if good_high else (1 - frac)

                ax_m.barh(y, 1.0, height=bar_h, left=0,
                          color="#2a2a2a", zorder=1)
                ax_m.barh(y, bar_f, height=bar_h, left=0,
                          color=cmap_rg(bar_f), zorder=2)
                ax_m.text(0.02, y + bar_h * 0.6, mlabel,
                          fontsize=7, color="white", va="bottom", ha="left",
                          fontweight="bold")
                val_str = (f"{val:.3f}" if abs(val) < 10
                           else f"{val:.1f}" if abs(val) < 1000
                           else f"{int(val)}")
                ax_m.text(0.98, y, val_str,
                          fontsize=9, color="white", va="center", ha="right",
                          fontweight="bold")
            else:
                ax_m.text(0.5, y, f"{mlabel}: n/a",
                          fontsize=6, color="#555", va="center", ha="center")

    nd2_file.close()

    fig.text(
        0.5, 0.97,
        "Z-stack quality survey  |  "
        "Top: focus-stacked output  |  "
        f"Middle: 15 Z slices (red border = mean NCC < {ncc_thresh:.2f})  |  "
        "Bottom: ranked metrics (green = good, red = bad)",
        ha="center", va="top", fontsize=9, color="white",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    return fig
