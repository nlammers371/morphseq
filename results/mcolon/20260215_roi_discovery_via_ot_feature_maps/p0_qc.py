"""
Phase 0 Step 2: QC + Outlier Filtering.

Provides IQR-based outlier flagging on total_cost_C and the three
required QC deliverables (gate to proceed):
  QC-1: histogram/violin of total_cost_C (before/after filtering)
  QC-2: montage of top-N highest-cost samples with their cost maps
  QC-3: summary table of dropped samples

Gate: post-filter mean maps must not be dominated by alignment failures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from outlier_detection import OutlierDetectionConfig, detect_outliers

logger = logging.getLogger(__name__)


def compute_iqr_outliers(
    total_cost_C: np.ndarray,
    multiplier: float = 2.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Flag outliers using IQR method on total_cost_C.

    Returns
    -------
    outlier_flag : (N,) bool
    stats : dict with q1, q3, iqr, lower, upper, n_outliers
    """
    q1 = float(np.percentile(total_cost_C, 25))
    q3 = float(np.percentile(total_cost_C, 75))
    iqr = q3 - q1

    result = detect_outliers(
        total_cost_C,
        OutlierDetectionConfig(method="iqr", iqr_multiplier=multiplier),
    )
    outlier_flag = result.outlier_flag

    stats = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": float(result.lower_bound),
        "upper_bound": float(result.upper_bound),
        "n_total": int(result.n_total),
        "n_outliers": int(result.n_outliers),
        "n_retained": int(result.n_total - result.n_outliers),
        "multiplier": multiplier,
        "method": "iqr",
    }
    logger.info(
        "IQR QC: %d/%d flagged (bounds=[%.4f, %.4f])",
        stats["n_outliers"],
        stats["n_total"],
        stats["lower_bound"],
        stats["upper_bound"],
    )
    return outlier_flag, stats


def plot_qc1_cost_histogram(
    total_cost_C: np.ndarray,
    outlier_flag: np.ndarray,
    stats: Dict,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    QC-1: Histogram/violin of total_cost_C showing before/after filtering.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full distribution with outlier bounds
    ax = axes[0]
    ax.hist(total_cost_C, bins=40, alpha=0.7, color="steelblue", edgecolor="k",
            label="All samples")
    ax.axvline(stats["lower_bound"], color="red", linestyle="--", linewidth=1.5,
               label=f"IQR bounds (×{stats['multiplier']:.1f})")
    ax.axvline(stats["upper_bound"], color="red", linestyle="--", linewidth=1.5)
    ax.axvline(stats["q1"], color="orange", linestyle=":", alpha=0.7, label="Q1/Q3")
    ax.axvline(stats["q3"], color="orange", linestyle=":", alpha=0.7)

    # Mark outliers
    outlier_costs = total_cost_C[outlier_flag]
    if len(outlier_costs) > 0:
        ax.hist(outlier_costs, bins=40, alpha=0.5, color="red", edgecolor="k",
                label=f"Outliers ({stats['n_outliers']})")
    ax.set_xlabel("Total OT Cost (C)")
    ax.set_ylabel("Count")
    ax.set_title(f"QC-1: Total Cost Distribution\n"
                 f"N={stats['n_total']}, outliers={stats['n_outliers']}")
    ax.legend(fontsize=8)

    # Right: retained only
    ax = axes[1]
    retained = total_cost_C[~outlier_flag]
    ax.hist(retained, bins=40, alpha=0.7, color="seagreen", edgecolor="k")
    ax.set_xlabel("Total OT Cost (C)")
    ax.set_ylabel("Count")
    ax.set_title(f"QC-1: After Filtering (N={stats['n_retained']})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved QC-1: {save_path}")
    return fig


def plot_qc2_worst_samples(
    X: np.ndarray,
    total_cost_C: np.ndarray,
    mask_ref: np.ndarray,
    sample_ids: List[str],
    metadata_df: Optional[pd.DataFrame] = None,
    target_masks_canonical: Optional[np.ndarray] = None,
    top_n: int = 8,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    QC-2: Montage of high/low cost samples with their cost density maps.

    X : (N, 512, 512, C) — channel 0 is cost_density
    """
    n_total = len(total_cost_C)
    n_high = min(4, n_total)
    n_low = min(4, max(0, n_total - n_high))
    high_idx = np.argsort(total_cost_C)[::-1][:n_high]
    low_idx = np.argsort(total_cost_C)[:n_low]
    order = np.concatenate([high_idx, low_idx])

    if target_masks_canonical is not None:
        if target_masks_canonical.shape[:3] != X.shape[:3]:
            raise ValueError(
                "target_masks_canonical must have shape (N, H, W) matching X[:, :, :, 0]. "
                f"Got target_masks_canonical={target_masks_canonical.shape}, X={X.shape}"
            )

    ncols = min(4, top_n)
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    ref_mask_bool = mask_ref.astype(bool)

    # Shared color scale across all panels
    all_costs = [X[order[rank], :, :, 0][ref_mask_bool] for rank in range(min(top_n, len(order)))]
    vmax_shared = float(np.nanmax(np.concatenate(all_costs))) if all_costs else 1.0
    vmin_shared = 0.0

    for idx, rank in enumerate(range(top_n)):
        if rank >= len(order):
            break
        i = order[rank]
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        cost_map = X[i, :, :, 0]
        display = np.where(ref_mask_bool, cost_map, np.nan)

        # Optional target overlay for visual diagnosis of why cost is high/low.
        if target_masks_canonical is not None:
            target_mask = target_masks_canonical[i].astype(bool)
            target_overlay = np.where(target_mask, 1.0, np.nan)
            ax.imshow(
                target_overlay,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                alpha=0.35,
                interpolation="nearest",
                origin="upper",
            )
            # Show mismatch regions explicitly:
            # target-only pixels (blue), reference-only pixels (purple).
            target_only = np.where(target_mask & (~ref_mask_bool), 1.0, np.nan)
            ref_only = np.where(ref_mask_bool & (~target_mask), 1.0, np.nan)
            ax.imshow(target_only, cmap="Blues", vmin=0.0, vmax=1.0, alpha=0.45, interpolation="nearest", origin="upper")
            ax.imshow(ref_only, cmap="Purples", vmin=0.0, vmax=1.0, alpha=0.35, interpolation="nearest", origin="upper")

        im = ax.imshow(display, cmap="hot", alpha=0.90, interpolation="nearest",
                       vmin=vmin_shared, vmax=vmax_shared, origin="upper")

        # Draw explicit contours so target/reference boundaries are always visible.
        ax.contour(mask_ref.astype(float), levels=[0.5], colors=["white"], linewidths=0.5, alpha=0.8, origin="upper")
        if target_masks_canonical is not None:
            ax.contour(target_mask.astype(float), levels=[0.5], colors=["#9ecae1"], linewidths=0.9, alpha=0.95, origin="upper")

        if metadata_df is not None and "genotype" in metadata_df.columns:
            genotype = str(metadata_df.iloc[i]["genotype"])
        else:
            genotype = "NA"

        bucket = "HIGH" if rank < n_high else "LOW"
        bucket_rank = (rank + 1) if rank < n_high else (rank - n_high + 1)
        ax.set_title(
            f"{bucket} #{bucket_rank}: {sample_ids[i]}\n{genotype}\nC={total_cost_C[i]:.4f}",
            fontsize=8,
        )
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off unused axes
    for idx in range(top_n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        "QC-2: Top-4 Highest + Top-4 Lowest Cost Samples\n"
        "(target underlay gray; target contour blue; ref contour white; "
        "target-only blue fill; ref-only purple fill)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved QC-2: {save_path}")
    return fig


def build_qc3_dropped_table(
    metadata_df: pd.DataFrame,
    outlier_flag: np.ndarray,
    total_cost_C: np.ndarray,
) -> pd.DataFrame:
    """
    QC-3: Summary table of dropped samples.

    Returns DataFrame with columns: sample_id, embryo_id, snip_id, total_cost_C, reason.
    """
    dropped = metadata_df[outlier_flag].copy()
    dropped["total_cost_C"] = total_cost_C[outlier_flag]
    dropped["reason"] = "IQR_outlier"

    # Sort by cost descending
    dropped = dropped.sort_values("total_cost_C", ascending=False).reset_index(drop=True)
    logger.info(f"QC-3: {len(dropped)} dropped samples")
    return dropped


def plot_qc_mean_maps(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    outlier_flag: np.ndarray,
    label_names: Dict[int, str] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Post-filter mean cost density maps by class (gate visual check).

    Shows WT mean, mutant mean, and difference. This is the gate check:
    if these are dominated by alignment failures, do NOT proceed.
    """
    if label_names is None:
        label_names = {0: "cep290_wildtype", 1: "cep290_homozygous"}

    valid = ~outlier_flag
    X_valid = X[valid]
    y_valid = y[valid]
    cost_ch = X_valid[:, :, :, 0]  # channel 0 = cost_density

    # Use an explicit image-style coordinate system: x increases to the right,
    # y increases downward, row 0 at the top.
    h, w = mask_ref.shape
    extent = [0, w, h, 0]  # [left, right, bottom, top] for upper origin
    origin = "upper"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Use a shared color scale for both class mean cost panels.
    mean_maps = {}
    for label_int in label_names:
        mask_class = y_valid == label_int
        if mask_class.sum() == 0:
            continue
        mean_map = np.mean(cost_ch[mask_class], axis=0)
        mean_map = np.where(mask_ref.astype(bool), mean_map, np.nan)
        mean_maps[label_int] = mean_map

    if mean_maps:
        vmax_cost = max(
            float(np.nanmax(m)) for m in mean_maps.values() if np.any(np.isfinite(m))
        )
        if not np.isfinite(vmax_cost) or vmax_cost <= 0:
            vmax_cost = 1.0
    else:
        vmax_cost = 1.0

    for label_int, label_str in label_names.items():
        mask_class = y_valid == label_int
        if mask_class.sum() == 0:
            continue
        mean_map = mean_maps[label_int]

        ax_idx = label_int  # 0=WT, 1=mutant
        im = axes[ax_idx].imshow(
            mean_map,
            cmap="hot",
            vmin=0.0,
            vmax=vmax_cost,
            aspect="equal",
            extent=extent,
            origin=origin,
            interpolation="nearest",
        )
        axes[ax_idx].set_title(f"Mean Cost: {label_str} (n={mask_class.sum()})")
        axes[ax_idx].set_xlabel("x (px)")
        axes[ax_idx].set_ylabel("y (px)")
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)

    # Difference
    wt_mask = y_valid == 0
    mut_mask = y_valid == 1
    if wt_mask.sum() > 0 and mut_mask.sum() > 0:
        diff = np.mean(cost_ch[mut_mask], axis=0) - np.mean(cost_ch[wt_mask], axis=0)
        diff = np.where(mask_ref.astype(bool), diff, np.nan)
        vabs = np.nanmax(np.abs(diff)) if np.any(np.isfinite(diff)) else 1.0
        im = axes[2].imshow(
            diff,
            cmap="RdBu_r",
            vmin=-vabs,
            vmax=vabs,
            aspect="equal",
            extent=extent,
            origin=origin,
            interpolation="nearest",
        )
        axes[2].set_title(f"Difference ({label_names[1]} - {label_names[0]})")
        axes[2].set_xlabel("x (px)")
        axes[2].set_ylabel("y (px)")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Δ cost")

    fig.suptitle("QC Gate: Post-Filter Mean Cost Maps", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved QC mean maps: {save_path}")
    return fig


def _classify_outlier_cause(
    overlap_iou_src_tgt: float,
    tgt_retained_ratio: float,
    yolk_offset_px: float,
    target_ref_area_ratio: float,
) -> str:
    """Heuristic label to separate likely alignment failures from mask defects."""
    alignment_flags = (
        (np.isfinite(overlap_iou_src_tgt) and overlap_iou_src_tgt < 0.55)
        or (np.isfinite(tgt_retained_ratio) and tgt_retained_ratio < 0.985)
        or (np.isfinite(yolk_offset_px) and yolk_offset_px > 20.0)
    )
    mask_flags = (
        np.isfinite(target_ref_area_ratio)
        and (target_ref_area_ratio < 0.65 or target_ref_area_ratio > 1.50)
    )
    if alignment_flags and mask_flags:
        return "mixed_alignment_and_mask"
    if alignment_flags:
        return "likely_alignment"
    if mask_flags:
        return "likely_mask_quality"
    return "unclear"


def plot_qc4_outlier_diagnostics(
    X: np.ndarray,
    total_cost_C: np.ndarray,
    mask_ref: np.ndarray,
    sample_ids: List[str],
    metadata_df: pd.DataFrame,
    outlier_flag: np.ndarray,
    target_masks_canonical: np.ndarray,
    alignment_debug_df: Optional[pd.DataFrame],
    out_dir: str | Path,
) -> pd.DataFrame:
    """
    QC-4: Per-outlier diagnostic plots to inspect alignment-vs-mask failure modes.

    Saves one plot per outlier and returns a summary DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outlier_idx = np.where(outlier_flag)[0]
    if len(outlier_idx) == 0:
        logger.info("QC-4: no outliers to plot")
        return pd.DataFrame(
            columns=[
                "sample_id",
                "embryo_id",
                "genotype",
                "total_cost_C",
                "overlap_iou_src_tgt",
                "tgt_retained_ratio",
                "yolk_offset_px",
                "target_ref_area_ratio",
                "diagnostic_label",
            ]
        )

    ref_mask_bool = mask_ref.astype(bool)
    ref_area = float(ref_mask_bool.sum())
    alignment_by_sample = {}
    if alignment_debug_df is not None and not alignment_debug_df.empty and "sample_id" in alignment_debug_df.columns:
        alignment_by_sample = {
            str(row["sample_id"]): row for _, row in alignment_debug_df.iterrows()
        }

    summary_rows = []
    outlier_idx_sorted = outlier_idx[np.argsort(total_cost_C[outlier_idx])[::-1]]
    for rank, i in enumerate(outlier_idx_sorted, start=1):
        sample_id = str(sample_ids[i])
        meta_row = metadata_df.iloc[i]
        align_row = alignment_by_sample.get(sample_id)

        target_mask = target_masks_canonical[i].astype(bool)
        target_area = float(target_mask.sum())
        area_ratio = target_area / max(ref_area, 1.0)

        overlap_iou = np.nan
        tgt_retained_ratio = np.nan
        yolk_offset_px = np.nan
        if align_row is not None:
            overlap_iou = float(align_row.get("overlap_iou_src_tgt", np.nan))
            tgt_retained_ratio = float(align_row.get("tgt_retained_ratio", np.nan))
            src_y = float(align_row.get("src_yolk_y_final", np.nan))
            src_x = float(align_row.get("src_yolk_x_final", np.nan))
            tgt_y = float(align_row.get("tgt_yolk_y_final", np.nan))
            tgt_x = float(align_row.get("tgt_yolk_x_final", np.nan))
            if np.all(np.isfinite([src_y, src_x, tgt_y, tgt_x])):
                yolk_offset_px = float(np.hypot(tgt_y - src_y, tgt_x - src_x))

        label = _classify_outlier_cause(
            overlap_iou_src_tgt=overlap_iou,
            tgt_retained_ratio=tgt_retained_ratio,
            yolk_offset_px=yolk_offset_px,
            target_ref_area_ratio=area_ratio,
        )

        summary_rows.append(
            {
                "sample_id": sample_id,
                "embryo_id": str(meta_row.get("embryo_id", "NA")),
                "genotype": str(meta_row.get("genotype", "NA")),
                "total_cost_C": float(total_cost_C[i]),
                "overlap_iou_src_tgt": overlap_iou,
                "tgt_retained_ratio": tgt_retained_ratio,
                "yolk_offset_px": yolk_offset_px,
                "target_ref_area_ratio": float(area_ratio),
                "diagnostic_label": label,
            }
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: ref/target overlay and mismatch
        ax = axes[0]
        ax.imshow(np.where(ref_mask_bool, 1.0, np.nan), cmap="gray", alpha=0.35, interpolation="nearest", origin="upper")
        ax.imshow(np.where(target_mask, 1.0, np.nan), cmap="Blues", alpha=0.35, interpolation="nearest", origin="upper")
        target_only = np.where(target_mask & (~ref_mask_bool), 1.0, np.nan)
        ref_only = np.where(ref_mask_bool & (~target_mask), 1.0, np.nan)
        ax.imshow(target_only, cmap="Blues", alpha=0.6, interpolation="nearest", origin="upper")
        ax.imshow(ref_only, cmap="Purples", alpha=0.45, interpolation="nearest", origin="upper")
        ax.contour(ref_mask_bool.astype(float), levels=[0.5], colors=["white"], linewidths=0.8, alpha=0.9, origin="upper")
        ax.contour(target_mask.astype(float), levels=[0.5], colors=["#9ecae1"], linewidths=0.9, alpha=0.95, origin="upper")
        ax.set_title("Ref vs Target Mask Overlay")
        ax.axis("off")

        # Panel 2: cost map on reference support
        ax = axes[1]
        cost_map = X[i, :, :, 0]
        cost_display = np.where(ref_mask_bool, cost_map, np.nan)
        im = ax.imshow(cost_display, cmap="hot", interpolation="nearest", origin="upper")
        ax.contour(ref_mask_bool.astype(float), levels=[0.5], colors=["white"], linewidths=0.8, alpha=0.9, origin="upper")
        ax.contour(target_mask.astype(float), levels=[0.5], colors=["#9ecae1"], linewidths=0.9, alpha=0.95, origin="upper")
        ax.set_title("Cost Density on Canonical Grid")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cost")

        # Panel 3: diagnostics text
        ax = axes[2]
        ax.axis("off")
        ax.text(
            0.03,
            0.97,
            "\n".join(
                [
                    f"sample_id: {sample_id}",
                    f"embryo_id: {meta_row.get('embryo_id', 'NA')}",
                    f"genotype: {meta_row.get('genotype', 'NA')}",
                    f"total_cost_C: {float(total_cost_C[i]):.4f}",
                    f"diagnostic_label: {label}",
                    "---",
                    f"overlap_iou_src_tgt: {overlap_iou:.4f}" if np.isfinite(overlap_iou) else "overlap_iou_src_tgt: n/a",
                    f"tgt_retained_ratio: {tgt_retained_ratio:.4f}" if np.isfinite(tgt_retained_ratio) else "tgt_retained_ratio: n/a",
                    f"yolk_offset_px: {yolk_offset_px:.2f}" if np.isfinite(yolk_offset_px) else "yolk_offset_px: n/a",
                    f"target_ref_area_ratio: {area_ratio:.4f}",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )

        fig.suptitle(f"QC-4 Outlier #{rank}: {sample_id}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = out_dir / f"qc4_outlier_{rank:02d}_{sample_id}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved QC-4: {save_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values("total_cost_C", ascending=False).reset_index(drop=True)
    summary_path = out_dir / "qc4_outlier_diagnostics.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved QC-4 summary: {summary_path}")
    return summary_df


def run_qc_suite(
    X: np.ndarray,
    y: np.ndarray,
    total_cost_C: np.ndarray,
    mask_ref: np.ndarray,
    metadata_df: pd.DataFrame,
    sample_ids: List[str],
    out_dir: str | Path,
    iqr_multiplier: float = 2.0,
    target_masks_canonical: Optional[np.ndarray] = None,
    alignment_debug_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Run the full QC suite (steps 2.1 + 2.2 from Phase 0 spec).

    Returns outlier_flag and stats dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 IQR filter
    outlier_flag, stats = compute_iqr_outliers(total_cost_C, multiplier=iqr_multiplier)

    # 2.2 QC deliverables
    plot_qc1_cost_histogram(total_cost_C, outlier_flag, stats,
                            save_path=out_dir / "qc1_cost_histogram.png")
    plot_qc2_worst_samples(X, total_cost_C, mask_ref, sample_ids,
                           metadata_df=metadata_df,
                           target_masks_canonical=target_masks_canonical,
                           save_path=out_dir / "qc2_worst_samples.png")

    dropped_df = build_qc3_dropped_table(metadata_df, outlier_flag, total_cost_C)
    dropped_df.to_csv(out_dir / "qc3_dropped_samples.csv", index=False)

    # Gate check: mean maps
    plot_qc_mean_maps(X, y, mask_ref, outlier_flag,
                      save_path=out_dir / "qc_gate_mean_maps.png")

    if target_masks_canonical is not None:
        plot_qc4_outlier_diagnostics(
            X=X,
            total_cost_C=total_cost_C,
            mask_ref=mask_ref,
            sample_ids=sample_ids,
            metadata_df=metadata_df,
            outlier_flag=outlier_flag,
            target_masks_canonical=target_masks_canonical,
            alignment_debug_df=alignment_debug_df,
            out_dir=out_dir,
        )

    plt.close("all")
    logger.info(f"QC suite complete: {out_dir}")
    return outlier_flag, stats


__all__ = [
    "compute_iqr_outliers",
    "plot_qc1_cost_histogram",
    "plot_qc2_worst_samples",
    "build_qc3_dropped_table",
    "plot_qc_mean_maps",
    "plot_qc4_outlier_diagnostics",
    "run_qc_suite",
]
