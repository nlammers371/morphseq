#!/usr/bin/env python
"""Render sequenced-focused embryo portfolio PDFs.

Views:
  1. By plate: one PDF page per query_experiment, 4 columns, rows as needed.
  2. By timepoint: one PDF page per review_stage_hpf, 4 columns, rows as needed.

Images come from:
  morphseq_playground/training_data/bf_embryo_snips/<query_experiment>/<snip_id>.jpg
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
DEFAULT_OUT = RUN_DIR / "portfolio" / "sequenced_views"

COLORS = {
    "AB": "#999999",
    "ab": "#999999",
    "ab_wildtype": "#999999",
    "wildtype": "#2166AC",
    "heterozygous": "#F7B267",
    "homozygous": "#B2182B",
    "unknown": "#808080",
    "CE": "#1b9e77",
    "HTA": "#d95f02",
    "High_to_Low": "#E76FA2",
    "Low_to_High": "#2FB7B0",
    "Not Penetrant": "#BBBBBB",
    "included": "#1b9e77",
    "excluded": "#B2182B",
    "missing": "#777777",
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

FONT_TITLE = font(28, True)
FONT_SUBTITLE = font(16, False)
FONT_HEADER = font(13, True)
FONT_SMALL = font(11, False)
FONT_TINY = font(10, False)


def clean(value, empty="NA") -> str:
    if pd.isna(value):
        return empty
    s = str(value)
    return empty if s == "" or s.lower() == "nan" else s


def resolve_snip(row: pd.Series, root: Path) -> Path | None:
    exp = clean(row.get("query_experiment"), "")
    snip = clean(row.get("snip_id"), "")
    if not exp or not snip:
        return None
    base = root / exp
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        p = base / f"{snip}{ext}"
        if p.exists():
            return p
    hits = list(base.glob(f"{snip}.*")) if base.exists() else []
    return hits[0] if hits else None


def label_color(value: str) -> str:
    if str(value) in {"AB", "ab", "ab_wildtype"}:
        return COLORS["AB"]
    if value in COLORS:
        return COLORS[value]
    for suffix in ("wildtype", "heterozygous", "homozygous", "unknown"):
        if str(value).endswith(suffix):
            return COLORS[suffix]
    return COLORS["missing"]


def draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str) -> int:
    color = label_color(value)
    text = f"{label}: {value}"
    w = max(70, min(190, int(7.0 * len(text) + 22)))
    draw.rectangle([x, y, x + 9, y + 15], fill=color, outline="#222222")
    draw.text((x + 13, y - 1), text, fill="#111111", font=FONT_TINY)
    return x + w


def card(row: pd.Series, snip_root: Path, card_w: int, card_h: int, image_h: int) -> Image.Image:
    excluded = bool(row.get("excluded"))
    bg = "#fff4f4" if excluded else "#ffffff"
    im = Image.new("RGB", (card_w, card_h), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, card_w - 1, card_h - 1], outline="#beb6aa", width=1)

    embryo_id = clean(row.get("embryo_id"), "")
    true_g = clean(row.get("true_genotype"))
    pred_g = clean(row.get("predicted_genotype"))
    dataset = clean(row.get("dataset"), "")
    homo_col = f"{dataset}_homo_only_predicted_phenotype"
    homo_p = clean(row.get(homo_col), "NA") if homo_col in row.index else "NA"
    qc = "excluded" if excluded else "included"
    qc_reason = clean(row.get("qc_reason"), "")

    y = 6
    d.text((8, y), embryo_id, fill="#111111", font=FONT_HEADER)
    y += 18
    x = 8
    x = draw_badge(d, x, y, "true_geno", true_g)
    x = draw_badge(d, x, y, "pred_geno", pred_g)
    y += 18
    x = 8
    x = draw_badge(d, x, y, "pred_Homo_pheno", homo_p)
    x = draw_badge(d, x, y, "QC", qc)
    if excluded:
        d.text((x, y - 1), qc_reason[:34], fill="#B2182B", font=FONT_TINY)

    img_top = 52
    img_box = [8, img_top, card_w - 8, img_top + image_h]
    snip = resolve_snip(row, snip_root)
    if snip is None:
        d.rectangle(img_box, fill="#eeeeee", outline="#bbbbbb")
        d.text((img_box[0] + 18, img_box[1] + image_h // 2 - 8), "missing snip", fill="#777777", font=FONT_HEADER)
    else:
        try:
            src = Image.open(snip).convert("RGB")
            # Training snips are portrait-oriented; rotate into the wide card to avoid side gutters.
            if src.height > src.width:
                src = src.rotate(90, expand=True)
            src = ImageOps.contain(src, (img_box[2] - img_box[0], image_h), method=Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (img_box[2] - img_box[0], image_h), "#111111")
            canvas.paste(src, ((canvas.width - src.width) // 2, (canvas.height - src.height) // 2))
            im.paste(canvas, (img_box[0], img_box[1]))
        except Exception:
            d.rectangle(img_box, fill="#eeeeee", outline="#bbbbbb")
            d.text((img_box[0] + 12, img_box[1] + 20), "image read error", fill="#777777", font=FONT_HEADER)
    return im


def group_title(group_by: str, key, sub: pd.DataFrame) -> str:
    if group_by == "plate":
        stages = sorted(sub["review_stage_hpf"].dropna().astype(float).unique().tolist())
        stage_label = ", ".join(f"{int(s)} hpf" for s in stages) if stages else "stage NA"
        return f"{key} | {stage_label} | n={len(sub)} sequenced embryos"
    if group_by == "timepoint" and isinstance(key, tuple):
        dataset, genotype_group, stage = key
        return f"{dataset} time ranges | {genotype_group} | {int(float(stage))} hpf | n={len(sub)} sequenced embryos"
    return f"{int(float(key))} hpf | n={len(sub)} sequenced embryos"


def genotype_group_label(row: pd.Series) -> str:
    stratum = clean(row.get("stratum"), "")
    true_genotype = clean(row.get("true_genotype"), "")
    if stratum and stratum != "NA":
        return stratum
    if true_genotype and true_genotype != "NA":
        return true_genotype
    return "unknown"


def group_sort_key(key) -> tuple:
    if isinstance(key, tuple):
        dataset, genotype_group, stage = key
        order = {
            "AB": 0,
            "wildtype_sibling": 1,
            "wildtype": 1,
            "heterozygous": 2,
            "homozygous": 3,
            "unknown": 99,
        }
        return (str(dataset), order.get(str(genotype_group), 20), str(genotype_group), float(stage))
    return (float(key),)


def render_group_page(sub: pd.DataFrame, title: str, snip_root: Path, *, cols: int, card_w: int, card_h: int, image_h: int) -> Image.Image:
    rows = max(1, math.ceil(len(sub) / cols))
    margin_x = 28
    header_h = 58
    gap = 8
    page_w = margin_x * 2 + cols * card_w + (cols - 1) * gap
    page_h = header_h + rows * card_h + (rows - 1) * gap + 26
    page = Image.new("RGB", (page_w, page_h), "#f5f1e9")
    d = ImageDraw.Draw(page)
    d.text((margin_x, 16), title, fill="#111111", font=FONT_TITLE)

    for i, (_, row) in enumerate(sub.iterrows()):
        r = i // cols
        c = i % cols
        x = margin_x + c * (card_w + gap)
        y = header_h + r * (card_h + gap)
        page.paste(card(row, snip_root, card_w, card_h, image_h), (x, y))
    return page


def write_pdf(pages: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not pages:
        return
    pages[0].save(path, save_all=True, append_images=pages[1:])


def render_view(df: pd.DataFrame, group_by: str, out_dir: Path, snip_root: Path, *, cols: int, card_w: int, card_h: int, image_h: int, write_png: bool) -> Path:
    if group_by == "plate":
        keys = sorted(df["query_experiment"].dropna().unique(), key=lambda k: (df.loc[df["query_experiment"].eq(k), "review_stage_hpf"].min(), str(k)))
        group_iter = [(k, df[df["query_experiment"].eq(k)]) for k in keys]
    elif group_by == "timepoint":
        work = df.copy()
        work["genotype_group"] = work.apply(genotype_group_label, axis=1)
        keys = sorted(
            work[["dataset", "genotype_group", "review_stage_hpf"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None),
            key=group_sort_key,
        )
        group_iter = [
            (
                k,
                work[
                    work["dataset"].eq(k[0])
                    & work["genotype_group"].eq(k[1])
                    & work["review_stage_hpf"].astype(float).eq(float(k[2]))
                ],
            )
            for k in keys
        ]
    else:
        raise ValueError(group_by)

    pages = []
    png_dir = out_dir / f"{group_by}_pages"
    if write_png:
        png_dir.mkdir(parents=True, exist_ok=True)
    for idx, (key, sub) in enumerate(group_iter, start=1):
        sub = sub.sort_values(["dataset", "query_experiment", "well", "embryo_id"], kind="stable")
        page = render_group_page(sub, group_title(group_by, key, sub), snip_root, cols=cols, card_w=card_w, card_h=card_h, image_h=image_h)
        pages.append(page)
        if write_png:
            safe = "__".join(str(x) for x in key) if isinstance(key, tuple) else str(key)
            safe = safe.replace("/", "_").replace(" ", "_")
            page.save(png_dir / f"{idx:02d}_{safe}.png")
    pdf_path = out_dir / f"sequenced_by_{group_by}.pdf"
    write_pdf(pages, pdf_path)
    return pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--snip-root", default=str(DEFAULT_SNIP_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--card-width", type=int, default=410)
    ap.add_argument("--card-height", type=int, default=255)
    ap.add_argument("--image-height", type=int, default=195)
    ap.add_argument("--include-unsequenced", action="store_true")
    ap.add_argument("--png-pages", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snip_root = Path(args.snip_root)
    df = pd.read_csv(args.manifest, low_memory=False)
    if not args.include_unsequenced:
        df = df[df["sequenced"].fillna(0).astype(float) > 0].copy()
    df["training_snip_path"] = df.apply(lambda r: str(resolve_snip(r, snip_root) or ""), axis=1)
    df["training_snip_found"] = df["training_snip_path"].ne("")
    df.to_csv(out_dir / "sequenced_portfolio_manifest.csv", index=False)

    plate_pdf = render_view(df, "plate", out_dir, snip_root, cols=args.cols, card_w=args.card_width, card_h=args.card_height, image_h=args.image_height, write_png=args.png_pages)
    time_pdf = render_view(df, "timepoint", out_dir, snip_root, cols=args.cols, card_w=args.card_width, card_h=args.card_height, image_h=args.image_height, write_png=args.png_pages)

    print(f"wrote {plate_pdf}")
    print(f"wrote {time_pdf}")
    print("\nSequenced snip coverage:")
    print(df.groupby("dataset")["training_snip_found"].agg(["sum", "count"]).to_string())
    print("\nSequenced rows by plate:")
    print(df.groupby(["review_stage_hpf", "query_experiment"]).size().to_string())


if __name__ == "__main__":
    main()
