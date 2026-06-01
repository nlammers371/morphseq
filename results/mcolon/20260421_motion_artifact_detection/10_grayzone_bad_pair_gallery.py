"""
10_grayzone_bad_pair_gallery.py
===============================
Create visual galleries for gray-zone embryos (0.80 <= ncc_p05 < 0.85), grouped
by bad_pair_frac tranches so threshold behavior can be inspected qualitatively.

For each bad_pair_frac bin, sample up to 50 examples and render a 10x5 panel grid.
Each tile is labeled with well, timepoint, ncc_p05, and bad_pair_frac.

Outputs:
  results/mcolon/20260421_motion_artifact_detection/figures/threshold_bins/
    v3_bad_pair_frac_analysis/gray_zone_galleries/
      - gray_zone_badpair_*.png
      - gray_zone_badpair_*.csv

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260421_motion_artifact_detection/10_grayzone_bad_pair_gallery.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
BASE_DIR = Path(__file__).parent

SUMMARIES_CSV = BASE_DIR / "07_embryo_ncc_output/embryo_ncc_summaries.csv"
IMAGES_DIR = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images"

OUT_DIR = BASE_DIR / "figures/threshold_bins/v3_bad_pair_frac_analysis/gray_zone_galleries"

GRAY_NCC_MIN = 0.80
GRAY_NCC_MAX = 0.85

N_ROWS = 10
N_COLS = 5
MAX_PER_BIN = N_ROWS * N_COLS
RANDOM_SEED = 7

BAD_PAIR_BINS = [
    (0.00, 0.05),
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 0.20),
    (0.20, 0.25),
    (0.25, 0.30),
    (0.30, np.inf),
]


def image_path(well: str, t: int) -> Path:
    return IMAGES_DIR / f"20250912_{well}" / f"20250912_{well}_ch00_t{t:04d}.jpg"


def load_gray_zone() -> pd.DataFrame:
    df = pd.read_csv(SUMMARIES_CSV)
    required = {"well", "t", "ncc_p05", "bad_pair_frac"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {SUMMARIES_CSV}: {missing}")

    df = df.dropna(subset=["well", "t", "ncc_p05", "bad_pair_frac"]).copy()
    gray = df[(df["ncc_p05"] >= GRAY_NCC_MIN) & (df["ncc_p05"] < GRAY_NCC_MAX)].copy()
    gray["t"] = gray["t"].astype(int)
    gray["well"] = gray["well"].astype(str)
    return gray


def bin_label(lo: float, hi: float) -> str:
    if np.isinf(hi):
        return f"{lo:.2f}_plus"
    return f"{lo:.2f}_to_{hi:.2f}"


def in_bin(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    if np.isinf(hi):
        return df[df["bad_pair_frac"] >= lo].copy()
    return df[(df["bad_pair_frac"] >= lo) & (df["bad_pair_frac"] < hi)].copy()


def sample_examples(df_bin: pd.DataFrame, n: int = MAX_PER_BIN) -> pd.DataFrame:
    if len(df_bin) <= n:
        return df_bin.sort_values(["bad_pair_frac", "ncc_p05", "well", "t"]).reset_index(drop=True)
    return (
        df_bin.sample(n=n, random_state=RANDOM_SEED)
        .sort_values(["bad_pair_frac", "ncc_p05", "well", "t"])
        .reset_index(drop=True)
    )


def draw_gallery(df_sel: pd.DataFrame, lo: float, hi: float, out_png: Path) -> None:
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3.0, N_ROWS * 2.8), facecolor="black")
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.set_facecolor("black")
        ax.axis("off")

    for idx, (_, row) in enumerate(df_sel.iterrows()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        well = row["well"]
        t = int(row["t"])
        ncc = float(row["ncc_p05"])
        bpf = float(row["bad_pair_frac"])
        jp = image_path(well, t)

        if jp.exists():
            img = np.array(Image.open(jp).convert("L"))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.text(
                0.5,
                0.5,
                "JPEG missing",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                transform=ax.transAxes,
            )

        ax.set_title(
            f"{well} t={t}\nncc_p05={ncc:.3f}  bad_pair={bpf:.3f}",
            fontsize=7,
            color="white",
            pad=2,
        )

    hi_txt = "inf" if np.isinf(hi) else f"{hi:.2f}"
    fig.suptitle(
        f"Gray zone (0.80<=ncc_p05<0.85) | bad_pair_frac [{lo:.2f}, {hi_txt}) | n={len(df_sel)}",
        color="white",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gray = load_gray_zone()
    print(f"Loaded gray-zone rows: {len(gray)}")

    for lo, hi in BAD_PAIR_BINS:
        df_bin = in_bin(gray, lo, hi)
        if len(df_bin) == 0:
            print(f"Skipping bin [{lo:.2f}, {'inf' if np.isinf(hi) else f'{hi:.2f}'}) — 0 rows")
            continue

        df_sel = sample_examples(df_bin, MAX_PER_BIN)
        label = bin_label(lo, hi)
        out_png = OUT_DIR / f"gray_zone_badpair_{label}.png"
        out_csv = OUT_DIR / f"gray_zone_badpair_{label}.csv"

        df_sel[["well", "t", "ncc_p05", "bad_pair_frac", "p"] if "p" in df_sel.columns else ["well", "t", "ncc_p05", "bad_pair_frac"]].to_csv(out_csv, index=False)
        draw_gallery(df_sel, lo, hi, out_png)
        print(f"Saved {out_png.name} (n={len(df_sel)})")


if __name__ == "__main__":
    main()
