"""
Motion Artifact Detection: Z-Stack Metric Exploration
======================================================
Goal: Cast a wide net of per-slice, adjacent-pair, and whole-stack metrics
      computed inside each embryo mask, to find which signals cleanly separate
      "Bad" (motion artifact) from "Great" (clean) stacks.

Data:
  - ND2: (T=113, P=95, Z=15, Y=2189, X=2189), Z-step=50µm, pixel=3.23µm
  - Masks: binary PNGs, one per embryo per timepoint (emnum_1, emnum_2)
  - Labeled examples from docs/refactors/motion_blur_filtering_zstack/frame_nd2_lookup.csv

Output (saved to results/mcolon/20260421_motion_artifact_detection/):
  - slice_metrics.csv       per (label, well, time_int, embryo, z)
  - pair_metrics.csv        per (label, well, time_int, embryo, z, z+1)
  - stack_metrics.csv       per (label, well, time_int, embryo)
  - winner_z_maps/          .npy files: for each example, which Z slice "won" per pixel
  - figures/                diagnostic plots
"""

from __future__ import annotations
import sys
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import nd2
import scipy.ndimage as ndi
from scipy.stats import entropy as scipy_entropy
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))

ND2_PATH = BASE / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
MASKS_DIR = BASE / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
IMAGES_DIR = BASE / "morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images"
LOOKUP_CSV = BASE / "docs/refactors/motion_blur_filtering_zstack/frame_nd2_lookup.csv"

OUT_DIR = BASE / "results/mcolon/20260421_motion_artifact_detection"
FIGURES_DIR = OUT_DIR / "figures"
WINNER_Z_DIR = OUT_DIR / "winner_z_maps"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
WINNER_Z_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants from ND2 metadata
Z_STEP_UM = 50.0
PIXEL_UM = 3.2308

# ---------------------------------------------------------------------------
# Load lookup table
# ---------------------------------------------------------------------------
lookup_df = pd.read_csv(LOOKUP_CSV)
print(f"Loaded {len(lookup_df)} labeled examples:")
print(lookup_df[["category", "well", "time_int"]].to_string())

# ---------------------------------------------------------------------------
# Helpers: load data
# ---------------------------------------------------------------------------

def load_z_stack(nd2_path: Path, series_num: int, time_int: int) -> np.ndarray:
    """Load all 15 Z slices for one (well, timepoint). Returns float32 (Z, Y, X)."""
    with nd2.ND2File(str(nd2_path)) as f:
        arr = f.to_dask()  # (T, P, Z, Y, X)
        stack = arr[time_int, series_num - 1, :, :, :].compute()
    return stack.astype(np.float32)


def load_masks(masks_dir: Path, date: str, well: str, time_int: int) -> dict[int, np.ndarray]:
    """
    Load all embryo masks for a given (well, timepoint).
    Returns {embryo_num: binary_mask (Y, X)} — mask pixels = 1.
    """
    masks = {}
    emnum = 1
    while True:
        fname = f"{date}_{well}_ch00_t{time_int:04d}_masks_emnum_{emnum}.png"
        p = masks_dir / fname
        if not p.exists():
            break
        arr = np.array(Image.open(p)).astype(bool)
        if arr.any():
            masks[emnum] = arr
        emnum += 1
    return masks


def load_focused_image(images_dir: Path, date: str, well: str, time_int: int) -> np.ndarray:
    """Load the LoG-focus-stacked JPEG (the pipeline output)."""
    p = images_dir / f"{date}_{well}" / f"{date}_{well}_ch00_t{time_int:04d}.jpg"
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("L")).astype(np.float32)


# ---------------------------------------------------------------------------
# Metric families
# ---------------------------------------------------------------------------

def _f64(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64)


def variance_of_laplacian(img: np.ndarray) -> float:
    lap = cv2.Laplacian(_f64(img), cv2.CV_64F, ksize=3)
    return float(lap.var())


def tenengrad(img: np.ndarray) -> float:
    gx = cv2.Sobel(_f64(img), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(_f64(img), cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def brenner_gradient(img: np.ndarray) -> float:
    diff = img[:, 2:].astype(np.float64) - img[:, :-2].astype(np.float64)
    return float(np.sum(diff**2))


def modified_laplacian(img: np.ndarray) -> float:
    """Sum of absolute modified Laplacian values."""
    lx = 2 * img - np.roll(img, 1, axis=1) - np.roll(img, -1, axis=1)
    ly = 2 * img - np.roll(img, 1, axis=0) - np.roll(img, -1, axis=0)
    return float(np.sum(np.abs(lx) + np.abs(ly)))


def gray_variance(img: np.ndarray) -> float:
    return float(img.var())


def normalized_variance(img: np.ndarray) -> float:
    mu = img.mean()
    if mu < 1e-6:
        return 0.0
    return float(img.var() / mu)


def image_entropy(img: np.ndarray) -> float:
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 65535))
    hist = hist[hist > 0].astype(np.float64)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist + 1e-12)))


def hf_energy_ratio(img: np.ndarray) -> float:
    """Ratio of high-freq to total energy via FFT."""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    low_mask = (y - cy)**2 + (x - cx)**2 < r**2
    total = mag.sum() + 1e-9
    low = mag[low_mask].sum()
    return float((total - low) / total)


def log_sharp_fraction(img: np.ndarray, mask: np.ndarray, threshold_pct: float = 75.0) -> float:
    """Fraction of mask pixels whose |LoG| exceeds the threshold percentile."""
    lap = cv2.Laplacian(_f64(img), cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap)
    if not mask.any():
        return 0.0
    vals = lap_abs[mask]
    thresh = np.percentile(vals, threshold_pct)
    return float((vals >= thresh).mean())


def log_response_mean(img: np.ndarray, mask: np.ndarray) -> float:
    lap = cv2.Laplacian(_f64(img), cv2.CV_64F, ksize=3)
    if not mask.any():
        return 0.0
    return float(np.abs(lap)[mask].mean())


def edge_density(img: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of mask pixels that are Canny edges."""
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(img_u8, 50, 150)
    if not mask.any():
        return 0.0
    return float(edges[mask].mean() / 255.0)


def gradient_anisotropy(img: np.ndarray, mask: np.ndarray) -> float:
    """
    Ratio of horizontal to vertical gradient energy inside mask.
    Close to 1 = isotropic. Far from 1 = directional blur.
    """
    gx = cv2.Sobel(_f64(img), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(_f64(img), cv2.CV_64F, 0, 1, ksize=3)
    if not mask.any():
        return 1.0
    ex = float((gx[mask]**2).mean())
    ey = float((gy[mask]**2).mean())
    denom = ey + 1e-9
    return float(ex / denom)


def compute_per_slice_metrics(slice_img: np.ndarray, mask: np.ndarray) -> dict:
    """All per-slice metrics, computed on pixels inside mask."""
    if not mask.any():
        return {k: np.nan for k in [
            "lap_var", "tenengrad", "brenner", "mod_lap", "gray_var",
            "norm_var", "entropy", "hf_ratio", "log_sharp_frac",
            "log_mean", "edge_density", "grad_anisotropy",
            "mask_area", "mask_cx", "mask_cy",
        ]}
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = slice_img[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop_masked = crop.copy()
    crop_masked[~crop_mask] = 0.0

    cy = float(ys.mean())
    cx = float(xs.mean())
    area = int(mask.sum())

    return {
        "lap_var":          variance_of_laplacian(crop_masked),
        "tenengrad":        tenengrad(crop_masked),
        "brenner":          brenner_gradient(crop_masked),
        "mod_lap":          modified_laplacian(crop_masked),
        "gray_var":         gray_variance(slice_img[mask]),
        "norm_var":         normalized_variance(slice_img[mask]),
        "entropy":          image_entropy(slice_img[mask]),
        "hf_ratio":         hf_energy_ratio(crop_masked),
        "log_sharp_frac":   log_sharp_fraction(crop, crop_mask),
        "log_mean":         log_response_mean(crop, crop_mask),
        "edge_density":     edge_density(crop, crop_mask),
        "grad_anisotropy":  gradient_anisotropy(crop, crop_mask),
        "mask_area":        area,
        "mask_cx":          cx,
        "mask_cy":          cy,
    }


def compute_background_metrics_per_z(stack: np.ndarray, all_masks: dict[int, np.ndarray]) -> list[dict]:
    """
    For each Z slice, compute focus metrics on the background region —
    everything outside all embryo masks. This is the per-Z reference level
    against which each embryo's metrics are normalized.

    Returns a list of dicts, one per Z slice, with keys:
      bg_log_mean, bg_tenengrad, bg_lap_var, bg_entropy
    """
    # Union of all embryo masks = pixels to exclude
    combined_mask = np.zeros(stack.shape[1:], dtype=bool)
    for m in all_masks.values():
        combined_mask |= m
    background = ~combined_mask

    # Dilate slightly so we don't include pixels right at embryo edges
    background = ndi.binary_erosion(background, iterations=10)

    bg_rows = []
    for z in range(stack.shape[0]):
        sl = stack[z]
        if not background.any():
            bg_rows.append({k: np.nan for k in
                            ["bg_log_mean", "bg_tenengrad", "bg_lap_var", "bg_entropy"]})
            continue

        # Sample background pixels (cap at 50k for speed)
        bg_ys, bg_xs = np.where(background)
        if len(bg_ys) > 50000:
            idx = np.random.choice(len(bg_ys), 50000, replace=False)
            bg_ys, bg_xs = bg_ys[idx], bg_xs[idx]

        # Compute metrics on a tight bounding box of the background sample
        # For LoG-based metrics, use the full slice (background is spread across frame)
        lap = cv2.Laplacian(_f64(sl), cv2.CV_64F, ksize=3)
        lap_abs = np.abs(lap)

        gx = cv2.Sobel(_f64(sl), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(_f64(sl), cv2.CV_64F, 0, 1, ksize=3)
        grad2 = gx**2 + gy**2

        bg_px = sl[bg_ys, bg_xs]
        bg_lap = lap_abs[bg_ys, bg_xs]
        bg_grad = grad2[bg_ys, bg_xs]

        hist, _ = np.histogram(bg_px, bins=256, range=(0, 65535))
        hist = hist[hist > 0].astype(np.float64)
        hist /= hist.sum()
        ent = float(-np.sum(hist * np.log2(hist + 1e-12)))

        bg_rows.append({
            "bg_log_mean":   float(bg_lap.mean()),
            "bg_tenengrad":  float(bg_grad.mean()),
            "bg_lap_var":    float(bg_lap.var()),
            "bg_entropy":    ent,
        })

    return bg_rows


def compute_pair_metrics(
    s0: np.ndarray, s1: np.ndarray,
    m0: np.ndarray, m1: np.ndarray
) -> dict:
    """Metrics between adjacent Z slices, computed inside the union of masks."""
    union_mask = m0 | m1
    if not union_mask.any():
        return {k: np.nan for k in [
            "ncc", "ssim_score", "phase_shift_dy", "phase_shift_dx",
            "phase_shift_mag", "post_align_residual",
            "mask_iou", "centroid_shift", "area_ratio",
        ]}

    ys, xs = np.where(union_mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    c0 = s0[y0:y1, x0:x1].copy()
    c1 = s1[y0:y1, x0:x1].copy()
    um = union_mask[y0:y1, x0:x1]

    # NCC
    a = c0[um].astype(np.float64)
    b = c1[um].astype(np.float64)
    a -= a.mean(); b -= b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    ncc = float(np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else 0.0

    # SSIM (manual, lightweight)
    mu_a = a.mean(); mu_b = b.mean()
    var_a = a.var(); var_b = b.var()
    cov = float(np.mean((a - mu_a) * (b - mu_b)))
    c1_c = (0.01 * 255)**2; c2_c = (0.03 * 255)**2
    ssim_score = float(
        (2*mu_a*mu_b + c1_c) * (2*cov + c2_c) /
        ((mu_a**2 + mu_b**2 + c1_c) * (var_a + var_b + c2_c))
    )

    # Phase correlation shift
    f0 = np.fft.fft2(c0); f1 = np.fft.fft2(c1)
    cross = f0 * np.conj(f1)
    denom = np.abs(cross) + 1e-9
    r = np.fft.ifft2(cross / denom).real
    peak = np.unravel_index(np.argmax(r), r.shape)
    dy = float(peak[0] if peak[0] < r.shape[0]//2 else peak[0] - r.shape[0])
    dx = float(peak[1] if peak[1] < r.shape[1]//2 else peak[1] - r.shape[1])
    shift_mag = float(np.sqrt(dy**2 + dx**2))

    # Post-alignment residual (shift c1, measure NCC improvement)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    c1_aligned = cv2.warpAffine(c1, M, (c1.shape[1], c1.shape[0]))
    b_al = c1_aligned[um].astype(np.float64)
    b_al -= b_al.mean()
    nb_al = np.linalg.norm(b_al)
    ncc_aligned = float(np.dot(a, b_al) / (na * nb_al)) if (na > 0 and nb_al > 0) else 0.0
    post_align_residual = float(ncc_aligned - ncc)  # positive = alignment helped

    # Mask IoU
    inter = float((m0 & m1).sum())
    union_ = float((m0 | m1).sum())
    mask_iou = inter / union_ if union_ > 0 else 1.0

    # Centroid shift (pixels)
    cy0 = float(np.where(m0)[0].mean()) if m0.any() else 0.0
    cx0 = float(np.where(m0)[1].mean()) if m0.any() else 0.0
    cy1 = float(np.where(m1)[0].mean()) if m1.any() else 0.0
    cx1 = float(np.where(m1)[1].mean()) if m1.any() else 0.0
    centroid_shift = float(np.sqrt((cy1-cy0)**2 + (cx1-cx0)**2))

    area_ratio = float(m1.sum() / (m0.sum() + 1e-6))

    return {
        "ncc":                  ncc,
        "ssim_score":           ssim_score,
        "phase_shift_dy":       dy,
        "phase_shift_dx":       dx,
        "phase_shift_mag":      shift_mag,
        "post_align_residual":  post_align_residual,
        "mask_iou":             mask_iou,
        "centroid_shift":       centroid_shift,
        "area_ratio":           area_ratio,
    }


def compute_winner_z_map(stack: np.ndarray) -> np.ndarray:
    """
    Replicate the LoG focus-stacking winner selection.
    For each pixel, which Z index has the highest |LoG| response?
    Returns array of shape (Y, X) with integer Z indices.
    """
    from src.build.export_utils import LoG_focus_stacker
    import torch

    norm = stack.copy().astype(np.float32)
    lo, hi = norm.min(), norm.max()
    if hi > lo:
        norm = (norm - lo) / (hi - lo)

    tensor = torch.from_numpy(norm)
    _, log_responses = LoG_focus_stacker(tensor, filter_size=3, device="cpu")
    # log_responses: (Z, Y, X)
    winner_z = log_responses.argmax(dim=0).numpy().astype(np.uint8)
    return winner_z


def winner_z_diagnostics(winner_z: np.ndarray, mask: np.ndarray) -> dict:
    """Stack-level diagnostics derived from the winner Z map."""
    inside = winner_z[mask].astype(np.float64) if mask.any() else np.array([])
    outside = winner_z[~mask].astype(np.float64)

    # Entropy of winning-Z distribution
    def _entropy(arr):
        if len(arr) == 0:
            return np.nan
        counts = np.bincount(arr.astype(int), minlength=15)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    # Spatial discontinuity: how often does winner_z differ from its neighbor
    wy = np.abs(np.diff(winner_z, axis=0))
    wx = np.abs(np.diff(winner_z, axis=1))
    # pad to same shape
    disc_map = np.zeros_like(winner_z, dtype=np.float32)
    disc_map[:-1, :] += wy
    disc_map[:, :-1] += wx

    inside_disc = disc_map[mask].mean() if mask.any() else np.nan
    outside_disc = disc_map[~mask].mean() if (~mask).any() else np.nan

    return {
        "winner_z_entropy_inside":     _entropy(inside),
        "winner_z_entropy_outside":    _entropy(outside),
        "winner_z_mean_inside":        float(inside.mean()) if len(inside) else np.nan,
        "winner_z_std_inside":         float(inside.std()) if len(inside) else np.nan,
        "winner_z_disc_inside":        float(inside_disc),
        "winner_z_disc_outside":       float(outside_disc),
        "winner_z_disc_ratio":         float(inside_disc / (outside_disc + 1e-6)),
        "winner_z_mode_inside":        int(np.bincount(inside.astype(int)).argmax()) if len(inside) else -1,
    }


def compute_stack_summaries(slice_rows: list[dict], pair_rows: list[dict]) -> dict:
    """Aggregate per-slice and per-pair metrics into whole-stack summaries."""
    from scipy.signal import find_peaks

    sl = pd.DataFrame(slice_rows)
    pr = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame()

    out = {}
    for col in ["lap_var", "tenengrad", "brenner", "log_mean", "entropy", "log_sharp_frac", "edge_density"]:
        if col in sl.columns:
            vals = sl[col].dropna().values
            out[f"{col}_max"]    = float(vals.max()) if len(vals) else np.nan
            out[f"{col}_median"] = float(np.median(vals)) if len(vals) else np.nan
            out[f"{col}_std"]    = float(vals.std()) if len(vals) else np.nan
            out[f"z_at_{col}_max"] = int(sl[col].idxmax()) if len(vals) else -1
            # Peak analysis
            peaks, props = find_peaks(vals, prominence=0)
            out[f"{col}_n_peaks"] = len(peaks)
            out[f"{col}_peak_prominence"] = float(props["prominences"].max()) if len(peaks) else 0.0

    if not pr.empty:
        for col in ["ncc", "ssim_score", "phase_shift_mag", "centroid_shift", "mask_iou"]:
            if col in pr.columns:
                vals = pr[col].dropna().values
                out[f"{col}_min"]    = float(vals.min()) if len(vals) else np.nan
                out[f"{col}_median"] = float(np.median(vals)) if len(vals) else np.nan
                out[f"{col}_std"]    = float(vals.std()) if len(vals) else np.nan

        # Bad-pair fraction (NCC < 0.9)
        if "ncc" in pr.columns:
            ncc_vals = pr["ncc"].dropna().values
            out["bad_pair_frac_ncc"]  = float((ncc_vals < 0.90).mean()) if len(ncc_vals) else np.nan
            out["bad_pair_frac_ncc95"] = float((ncc_vals < 0.95).mean()) if len(ncc_vals) else np.nan
        if "ssim_score" in pr.columns:
            ssim_vals = pr["ssim_score"].dropna().values
            out["bad_pair_frac_ssim"] = float((ssim_vals < 0.90).mean()) if len(ssim_vals) else np.nan

        # Longest run of bad NCC pairs
        if "ncc" in pr.columns:
            bad = (pr["ncc"].fillna(1.0).values < 0.90).astype(int)
            max_run = 0
            cur_run = 0
            for b in bad:
                cur_run = cur_run + 1 if b else 0
                max_run = max(max_run, cur_run)
            out["longest_bad_ncc_run"] = max_run

        # Max phase shift and centroid shift
        if "phase_shift_mag" in pr.columns:
            out["max_phase_shift_px"] = float(pr["phase_shift_mag"].max())
            out["max_phase_shift_um"] = float(pr["phase_shift_mag"].max() * PIXEL_UM)
        if "centroid_shift" in pr.columns:
            out["max_centroid_shift_px"] = float(pr["centroid_shift"].max())
            out["max_centroid_shift_um"] = float(pr["centroid_shift"].max() * PIXEL_UM)

    return out


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

all_slice_rows = []
all_pair_rows = []
all_stack_rows = []

print("\n" + "="*60)
print("Processing labeled examples...")
print("="*60)

with nd2.ND2File(str(ND2_PATH)) as nd2_file:
    dask_arr = nd2_file.to_dask()  # (T, P, Z, Y, X)

    for _, row in lookup_df.iterrows():
        label    = row["category"]
        well     = row["well"]
        time_int = int(row["time_int"])
        series   = int(row["nd2_series_num"])
        date     = str(row["date"])

        print(f"\n[{label}] {well} t={time_int}")

        # Load Z stack (15, 2189, 2189)
        stack = dask_arr[time_int, series - 1, :, :, :].compute().astype(np.float32)
        print(f"  Stack shape: {stack.shape}")

        # Load masks
        masks = load_masks(MASKS_DIR, date, well, time_int)
        if not masks:
            print(f"  WARNING: no masks found, skipping")
            continue
        print(f"  Found {len(masks)} embryo mask(s): emnum {list(masks.keys())}")

        # Compute winner-Z map (expensive but informative)
        print(f"  Computing winner-Z map...")
        try:
            winner_z = compute_winner_z_map(stack)
            wz_path = WINNER_Z_DIR / f"{date}_{well}_t{time_int:04d}_winner_z.npy"
            np.save(wz_path, winner_z)
        except Exception as e:
            print(f"  WARNING: winner_z failed: {e}")
            winner_z = None

        for emnum, mask in masks.items():
            key = dict(label=label, well=well, time_int=time_int, embryo=emnum,
                       date=date, series=series)
            print(f"  Embryo {emnum}: mask area = {mask.sum()} px")

            # --- Per-slice metrics ---
            slice_metrics_list = []
            for z in range(stack.shape[0]):
                m = compute_per_slice_metrics(stack[z], mask)
                m.update(key)
                m["z"] = z
                m["z_um"] = z * Z_STEP_UM
                slice_metrics_list.append(m)
                all_slice_rows.append(m)

            # --- Per-pair metrics ---
            pair_metrics_list = []
            for z in range(stack.shape[0] - 1):
                pm = compute_pair_metrics(
                    stack[z], stack[z+1], mask, mask
                )
                pm.update(key)
                pm["z0"] = z
                pm["z1"] = z + 1
                pair_metrics_list.append(pm)
                all_pair_rows.append(pm)

            # --- Stack summaries ---
            stack_row = compute_stack_summaries(slice_metrics_list, pair_metrics_list)
            stack_row.update(key)

            # Winner-Z diagnostics
            if winner_z is not None:
                wz_diag = winner_z_diagnostics(winner_z, mask)
                stack_row.update(wz_diag)

            all_stack_rows.append(stack_row)

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
slice_df = pd.DataFrame(all_slice_rows)
pair_df  = pd.DataFrame(all_pair_rows)
stack_df = pd.DataFrame(all_stack_rows)

slice_df.to_csv(OUT_DIR / "slice_metrics.csv", index=False)
pair_df.to_csv(OUT_DIR / "pair_metrics.csv", index=False)
stack_df.to_csv(OUT_DIR / "stack_metrics.csv", index=False)

print(f"\nSaved CSVs to {OUT_DIR}")
print(f"  slice_metrics:  {len(slice_df)} rows")
print(f"  pair_metrics:   {len(pair_df)} rows")
print(f"  stack_metrics:  {len(stack_df)} rows")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "Bad Images":   "#B2182B",
    "Okay Images":  "#F7B267",
    "Great Images": "#2166AC",
}

def label_color(lbl):
    return LABEL_COLORS.get(lbl, "gray")


# --- Figure 1: Focus curves per example (per-slice lap_var vs Z) ---
focus_metrics = ["lap_var", "tenengrad", "log_mean", "log_sharp_frac", "entropy"]
n_metrics = len(focus_metrics)
examples = slice_df[["label", "well", "time_int", "embryo"]].drop_duplicates()

fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=False)
for ax, metric in zip(axes, focus_metrics):
    for _, ex in examples.iterrows():
        sub = slice_df[
            (slice_df["well"] == ex["well"]) &
            (slice_df["time_int"] == ex["time_int"]) &
            (slice_df["embryo"] == ex["embryo"])
        ].sort_values("z")
        lbl = ex["label"]
        ax.plot(sub["z"].values, sub[metric].values,
                color=label_color(lbl), alpha=0.8,
                label=f"{lbl[:3]} {ex['well']} t{ex['time_int']}",
                linewidth=1.5, marker="o", markersize=4)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_xlabel("Z index")
    ax.grid(True, alpha=0.3)
    ax.axvline(7, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="homeIndex")

# Legend only on first axis
handles, labels_ = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels_, handles))
axes[0].legend(by_label.values(), by_label.keys(), fontsize=7, ncol=3, loc="upper right")
plt.suptitle("Per-slice focus metrics across Z (Bad=red, Okay=amber, Great=blue)", fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "focus_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved focus_curves.png")


# --- Figure 2: Adjacent-pair NCC and phase-shift magnitude ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
pair_metrics_to_plot = ["ncc", "phase_shift_mag"]
for ax, metric in zip(axes, pair_metrics_to_plot):
    examples_p = pair_df[["label", "well", "time_int", "embryo"]].drop_duplicates()
    for _, ex in examples_p.iterrows():
        sub = pair_df[
            (pair_df["well"] == ex["well"]) &
            (pair_df["time_int"] == ex["time_int"]) &
            (pair_df["embryo"] == ex["embryo"])
        ].sort_values("z0")
        lbl = ex["label"]
        ax.plot(sub["z0"].values, sub[metric].values,
                color=label_color(lbl), alpha=0.8,
                label=f"{lbl[:3]} {ex['well']} t{ex['time_int']}",
                linewidth=1.5, marker="o", markersize=4)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_xlabel("Z pair (z0, z0+1)")
    ax.grid(True, alpha=0.3)

axes[0].axhline(0.90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="NCC=0.90")
axes[1].axhline(5.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="5px shift")
for ax in axes:
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=3, loc="best")

plt.suptitle("Adjacent Z-pair: NCC and phase-shift (Bad=red, Okay=amber, Great=blue)", fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "pair_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pair_metrics.png")


# --- Figure 3: Stack-level metric summary (dot plot, Bad vs Great) ---
stack_cols = [
    "lap_var_max", "lap_var_std", "tenengrad_max", "log_mean_max",
    "bad_pair_frac_ncc", "ncc_min", "phase_shift_mag_max",
    "max_centroid_shift_px", "winner_z_entropy_inside",
    "winner_z_disc_ratio", "longest_bad_ncc_run",
]
stack_cols = [c for c in stack_cols if c in stack_df.columns]

fig, axes = plt.subplots(len(stack_cols), 1, figsize=(8, 2.5 * len(stack_cols)))
for ax, col in zip(axes, stack_cols):
    for _, row_ in stack_df.iterrows():
        val = row_.get(col, np.nan)
        if pd.isna(val):
            continue
        ax.scatter(val, 0,
                   color=label_color(row_["label"]),
                   s=100, alpha=0.8, zorder=3)
        ax.text(val, 0.05, f"{row_['well']}", fontsize=7,
                ha="center", va="bottom", rotation=45)
    ax.set_xlabel(col, fontsize=9)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(col, fontsize=8, pad=2)

plt.suptitle("Stack-level metrics (Bad=red, Okay=amber, Great=blue)\n(each dot = one embryo-stack)",
             fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "stack_metrics_dotplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved stack_metrics_dotplot.png")


# --- Figure 4: Winner-Z maps side by side for bad vs great ---
wz_examples = [
    ("Bad Images",   "B10", 92),
    ("Great Images", "C04", 28),
    ("Great Images", "G09", 31),
]
fig, axes = plt.subplots(1, len(wz_examples), figsize=(5 * len(wz_examples), 5))
for ax, (lbl, well, t) in zip(axes, wz_examples):
    wz_path = WINNER_Z_DIR / f"20250912_{well}_t{t:04d}_winner_z.npy"
    if not wz_path.exists():
        ax.set_title(f"missing: {well} t{t}")
        continue
    wz = np.load(wz_path)
    # Load mask
    m = load_masks(MASKS_DIR, "20250912", well, t)
    im = ax.imshow(wz, cmap="viridis", vmin=0, vmax=14, interpolation="nearest")
    # Overlay mask contour
    for emnum, mask in m.items():
        ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=1)
    plt.colorbar(im, ax=ax, label="winning Z index", fraction=0.046)
    ax.set_title(f"{lbl[:3]}: {well} t={t}", fontsize=9)
    ax.axis("off")

plt.suptitle("Winner-Z map: which Z slice 'won' the LoG stacker\nWhite contour = embryo mask",
             fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "winner_z_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved winner_z_maps.png")


# --- Figure 5: Focused JPEG + Z-slice strip for bad and great ---
strip_examples = [
    ("Bad Images",   "B10", 92,  79),
    ("Great Images", "C04", 28,  30),
]
for lbl, well, t, series in strip_examples:
    focused = load_focused_image(IMAGES_DIR, "20250912", well, t)
    masks_ex = load_masks(MASKS_DIR, "20250912", well, t)

    with nd2.ND2File(str(ND2_PATH)) as f:
        arr = f.to_dask()
        stack_ex = arr[t, series - 1, :, :, :].compute().astype(np.float32)

    n_z = stack_ex.shape[0]
    fig, axes = plt.subplots(2, n_z + 1, figsize=(3 * (n_z + 1), 7))

    # Focused image (top-left)
    if focused is not None:
        axes[0, 0].imshow(focused, cmap="gray")
        axes[0, 0].set_title("Focus-stacked\n(pipeline output)", fontsize=8)
    else:
        axes[0, 0].set_title("No JPEG found")
    axes[0, 0].axis("off")

    # Winner-Z map (bottom-left)
    wz_path = WINNER_Z_DIR / f"20250912_{well}_t{t:04d}_winner_z.npy"
    if wz_path.exists():
        wz = np.load(wz_path)
        im = axes[1, 0].imshow(wz, cmap="viridis", vmin=0, vmax=14)
        for emnum, mask in masks_ex.items():
            axes[1, 0].contour(mask.astype(float), levels=[0.5], colors="white", linewidths=1)
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        axes[1, 0].set_title("Winner-Z map", fontsize=8)
    axes[1, 0].axis("off")

    # Z slices
    norm_stack = stack_ex - stack_ex.min()
    norm_stack = norm_stack / (norm_stack.max() + 1e-6)

    for z in range(n_z):
        sl = norm_stack[z]
        axes[0, z + 1].imshow(sl, cmap="gray", vmin=0, vmax=1)
        # overlay mask contour
        for emnum, mask in masks_ex.items():
            axes[0, z + 1].contour(mask.astype(float), levels=[0.5],
                                    colors="red", linewidths=0.8)
        # Lap var annotation
        sub = slice_df[
            (slice_df["well"] == well) &
            (slice_df["time_int"] == t) &
            (slice_df["z"] == z)
        ]
        if not sub.empty:
            lv = sub["lap_var"].values[0]
            axes[0, z + 1].set_title(f"z={z}\nLV={lv:.0f}", fontsize=7)
        else:
            axes[0, z + 1].set_title(f"z={z}", fontsize=7)
        axes[0, z + 1].axis("off")

        # NCC row
        if z < n_z - 1:
            sub_p = pair_df[
                (pair_df["well"] == well) &
                (pair_df["time_int"] == t) &
                (pair_df["z0"] == z)
            ]
            if not sub_p.empty:
                ncc_val = sub_p["ncc"].values[0]
                ps_val  = sub_p["phase_shift_mag"].values[0]
                axes[1, z + 1].text(
                    0.5, 0.5,
                    f"NCC\n{ncc_val:.3f}\nΔpx={ps_val:.1f}",
                    ha="center", va="center",
                    fontsize=9,
                    color="red" if ncc_val < 0.90 else "green",
                    transform=axes[1, z + 1].transAxes
                )
        axes[1, z + 1].axis("off")

    plt.suptitle(f"{lbl}: {well} t={t}\nTop=Z slices (Lap var), Bottom=pair NCC / shift",
                 fontsize=10)
    plt.tight_layout()
    fname = FIGURES_DIR / f"zstrip_{lbl.replace(' ', '_')}_{well}_t{t}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname.name}")


# --- Final summary table ---
print("\n" + "="*60)
print("STACK-LEVEL METRIC SUMMARY")
print("="*60)
summary_cols = ["label", "well", "time_int", "embryo",
                "lap_var_max", "bad_pair_frac_ncc", "ncc_min",
                "max_phase_shift_px", "max_centroid_shift_px",
                "winner_z_entropy_inside", "winner_z_disc_ratio"]
summary_cols = [c for c in summary_cols if c in stack_df.columns]
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)
print(stack_df[summary_cols].sort_values("label").to_string(index=False))

print(f"\nAll outputs in: {OUT_DIR}")
