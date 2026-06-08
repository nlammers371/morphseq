#!/usr/bin/env python
"""Render static embryo-review contact sheets from training snips.

Inputs:
  portfolio/embryo_portfolio_manifest.csv
  morphseq_playground/training_data/bf_embryo_snips/<query_experiment>/<snip_id>.jpg

Outputs:
  portfolio/contact_sheets/<dataset>_contact_sheet.pdf
  portfolio/contact_sheets/<dataset>_pageNNN.png
  portfolio/contact_sheets/contact_sheet_manifest.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
DEFAULT_MANIFEST = RUN_DIR / "portfolio" / "embryo_portfolio_manifest.csv"
DEFAULT_SNIP_ROOT = PROJECT_ROOT / "morphseq_playground" / "training_data" / "bf_embryo_snips"
DEFAULT_OUT = RUN_DIR / "portfolio" / "contact_sheets"

COLORS = {
    "wildtype": "#2166AC",
    "heterozygous": "#F7B267",
    "homozygous": "#B2182B",
    "CE": "#1b9e77",
    "HTA": "#d95f02",
    "High_to_Low": "#E76FA2",
    "Low_to_High": "#2FB7B0",
    "Not Penetrant": "#BBBBBB",
    "missing": "#777777",
    "included": "#1b9e77",
    "excluded": "#B2182B",
}


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

FONT_TITLE = font(22, True)
FONT_HEADER = font(12, True)
FONT_SMALL = font(10, False)
FONT_TINY = font(8, False)


def coerce_text(value, empty="NA") -> str:
    if pd.isna(value):
        return empty
    s = str(value)
    return empty if s == "" or s.lower() == "nan" else s


def resolve_snip(row: pd.Series, root: Path) -> Path | None:
    exp = coerce_text(row.get("query_experiment"), "")
    snip = coerce_text(row.get("snip_id"), "")
    if not exp or not snip:
        return None
    base = root / exp
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        p = base / f"{snip}{ext}"
        if p.exists():
            return p
    hits = list(base.glob(f"{snip}.*")) if base.exists() else []
    return hits[0] if hits else None


def fill_rect(draw, xy, color):
    draw.rectangle(xy, fill=color, outline="#222222", width=1)


def text(draw, xy, s, fill="#111111", fnt=FONT_SMALL):
    draw.text(xy, s, fill=fill, font=fnt)


def card(row: pd.Series, image_path: Path | None, card_w: int, card_h: int, image_h: int) -> Image.Image:
    bg = "#fff9ef" if bool(row.get("excluded")) else "#ffffff"
    im = Image.new("RGB", (card_w, card_h), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, card_w - 1, card_h - 1], outline="#b9b0a3", width=1)

    well = coerce_text(row.get("well"), "")
    embryo_id = coerce_text(row.get("embryo_id"), "")
    true_g = coerce_text(row.get("true_genotype"))
    pred_g = coerce_text(row.get("predicted_genotype"))
    pred_p = coerce_text(row.get("predicted_phenotype"))
    dataset = coerce_text(row.get("dataset"), "")
    homo_col = f"{dataset}_homo_only_predicted_phenotype"
    homo_p = coerce_text(row.get(homo_col), "") if homo_col in row.index else ""
    qc = "excluded" if bool(row.get("excluded")) else "included"
    qc_reason = coerce_text(row.get("qc_reason"), "")
    seq = coerce_text(row.get("sequenced"), "")

    y = 6
    text(d, (8, y), f"{well}  {embryo_id[-18:]}", fnt=FONT_HEADER)
    y += 16
    text(d, (8, y), f"seq={seq} | true={true_g}", fnt=FONT_SMALL)
    y += 14

    x = 8
    for label, value, color_key in [
        ("predG", pred_g, pred_g),
        ("predP", pred_p, pred_p),
        ("homo", homo_p, homo_p),
        ("QC", qc, qc),
    ]:
        if label == "homo" and not value:
            continue
        color = COLORS.get(color_key, COLORS["missing"])
        fill_rect(d, [x, y, x + 7, y + 11], color)
        text(d, (x + 10, y - 1), f"{label}:{value}", fnt=FONT_TINY)
        x += min(112, 18 + 6 * len(f"{label}:{value}"))
    y += 15
    if qc == "excluded" and qc_reason:
        text(d, (8, y), qc_reason[:54], fill="#B2182B", fnt=FONT_TINY)
    y += 6

    img_top = 64
    img_box = [8, img_top, card_w - 8, img_top + image_h]
    if image_path is None:
        d.rectangle(img_box, fill="#eeeeee", outline="#bbbbbb")
        text(d, (img_box[0] + 18, img_box[1] + image_h // 2 - 8), "missing training snip", fill="#777777", fnt=FONT_HEADER)
    else:
        try:
            src = Image.open(image_path).convert("RGB")
            src = ImageOps.contain(src, (img_box[2] - img_box[0], image_h), method=Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (img_box[2] - img_box[0], image_h), "#111111")
            ox = (canvas.width - src.width) // 2
            oy = (canvas.height - src.height) // 2
            canvas.paste(src, (ox, oy))
            im.paste(canvas, (img_box[0], img_box[1]))
        except Exception as e:
            d.rectangle(img_box, fill="#eeeeee", outline="#bbbbbb")
            text(d, (img_box[0] + 8, img_box[1] + 20), f"image read error: {type(e).__name__}", fill="#777777", fnt=FONT_SMALL)

    text(d, (8, card_h - 18), coerce_text(row.get("snip_id"), "")[:64], fill="#555555", fnt=FONT_TINY)
    return im


def page_title(draw, dataset, page_idx, total_pages, sub, page_w):
    title = f"{dataset} embryo portfolio | page {page_idx + 1}/{total_pages} | n={len(sub)}"
    draw.text((24, 16), title, font=FONT_TITLE, fill="#111111")


def render_dataset(df: pd.DataFrame, dataset: str, out_dir: Path, snip_root: Path, *, cols: int, rows: int, card_w: int, card_h: int, image_h: int, png: bool, pdf: bool) -> list[Path]:
    sub = df[df["dataset"].astype(str).eq(dataset)].copy()
    sub = sub.sort_values(["review_stage_hpf", "query_experiment", "well", "embryo_id"], kind="stable")
    per_page = cols * rows
    pages = []
    page_w = cols * card_w + 48
    page_h = rows * card_h + 74
    total_pages = math.ceil(len(sub) / per_page) if len(sub) else 0
    for page_idx in range(total_pages):
        chunk = sub.iloc[page_idx * per_page:(page_idx + 1) * per_page]
        page = Image.new("RGB", (page_w, page_h), "#f5f1e9")
        d = ImageDraw.Draw(page)
        page_title(d, dataset, page_idx, total_pages, sub, page_w)
        for n, (_, row) in enumerate(chunk.iterrows()):
            r = n // cols
            c = n % cols
            x = 24 + c * card_w
            y = 54 + r * card_h
            p = resolve_snip(row, snip_root)
            page.paste(card(row, p, card_w - 8, card_h - 8, image_h), (x, y))
        if png:
            path = out_dir / f"{dataset}_page{page_idx + 1:03d}.png"
            page.save(path)
            pages.append(path)
        else:
            pages.append(page)
    if pdf and total_pages:
        pdf_path = out_dir / f"{dataset}_contact_sheet.pdf"
        pil_pages = []
        if png:
            pil_pages = [Image.open(p).convert("RGB") for p in pages]
        else:
            pil_pages = pages
        pil_pages[0].save(pdf_path, save_all=True, append_images=pil_pages[1:])
        if not png:
            pages = [pdf_path]
        else:
            pages.append(pdf_path)
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--snip-root", default=str(DEFAULT_SNIP_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--dataset", action="append", help="Dataset to render. Repeatable. Default: all.")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--card-width", type=int, default=320)
    ap.add_argument("--card-height", type=int, default=292)
    ap.add_argument("--image-height", type=int, default=205)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snip_root = Path(args.snip_root)
    df = pd.read_csv(args.manifest, low_memory=False)
    df["training_snip_path"] = df.apply(lambda r: str(resolve_snip(r, snip_root) or ""), axis=1)
    df["training_snip_found"] = df["training_snip_path"].astype(str).ne("")
    df.to_csv(out_dir / "contact_sheet_manifest.csv", index=False)

    datasets = args.dataset or sorted(df["dataset"].dropna().astype(str).unique())
    for dataset in datasets:
        paths = render_dataset(
            df, dataset, out_dir, snip_root,
            cols=args.cols, rows=args.rows, card_w=args.card_width, card_h=args.card_height,
            image_h=args.image_height, png=not args.no_png, pdf=not args.no_pdf,
        )
        print(f"{dataset}: wrote {len(paths)} files")
    print("\nSnip coverage by dataset:")
    print(df.groupby("dataset")["training_snip_found"].agg(["sum", "count"]).to_string())
    print(f"\nOutputs under: {out_dir}")


if __name__ == "__main__":
    main()
