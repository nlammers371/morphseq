#!/usr/bin/env python
"""Build embryo-review portfolio pages for cilia QC sequenced/model review."""
from __future__ import annotations

import argparse
import html
import math
import re
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import build_reference_and_transfer as T  # noqa: E402

TR = RUN_DIR / "transfer_results"
OUT = RUN_DIR / "portfolio"
B6 = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"
B4 = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build04_output"

QC_FLAGS = [
    "use_embryo_flag", "dead_flag", "dead_flag2", "sa_outlier_flag", "sam2_qc_flag",
    "frame_flag", "no_yolk_flag", "focus_flag", "bubble_flag", "well_qc_flag",
]
IMAGE_COLS = [
    "embryo_id", "snip_id", "image_id", "image_path", "exported_mask_path",
    "query_experiment", "experiment_id", "genotype", "well", "well_id", "frame_index", "is_seed_frame",
    "start_age_hpf", "predicted_stage_hpf", "bbox_x_min", "bbox_y_min", "bbox_x_max",
    "bbox_y_max", "Width (px)", "Height (px)", "width_px", "height_px", "mask_confidence",
    "area_px", "area_um2", "surface_area_um", "total_length_um", "baseline_deviation_um",
    "baseline_deviation_normalized", *QC_FLAGS,
]
PALETTES = {
    "zyg": {"wildtype": "#2166AC", "heterozygous": "#F7B267", "homozygous": "#B2182B", "unknown": "#808080", "NA": "#999999"},
    "phenotype": {"CE": "#1b9e77", "HTA": "#d95f02", "wildtype": "#2166AC", "High_to_Low": "#E76FA2", "Low_to_High": "#2FB7B0", "Not Penetrant": "#BBBBBB", "NA": "#999999"},
}
OLD_ROOT_PATTERN = re.compile(r"^/net/trapnell/vol1/home/mdcolon/proj/morphseq[^/]*/")


def _infer_stage_from_experiment(exp) -> float:
    exp = str(exp)
    if "30to48" in exp:
        return 48.0 if "_t02" in exp else 30.0
    m = re.search(r"_(14|18|24|30|48)hpf", exp)
    return float(m.group(1)) if m else np.nan


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _query_experiments() -> list[str]:
    exps = []
    for cfg in T.DATASETS.values():
        exps.extend(cfg["queries"])
    return sorted(dict.fromkeys(exps))


def _load_image_qc_metadata(experiments: list[str]) -> pd.DataFrame:
    parts = []
    for exp in experiments:
        path = B4 / f"qc_staged_{exp}.csv"
        if not path.exists():
            path = B6 / f"df03_final_output_with_latents_{exp}.csv"
        if not path.exists():
            continue
        cols = pd.read_csv(path, nrows=0).columns
        usecols = [c for c in IMAGE_COLS if c in cols]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        if "query_experiment" not in df.columns:
            df["query_experiment"] = exp
        df["dataset_meta"] = T._route_gene(exp)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    meta = pd.concat(parts, ignore_index=True)
    if "is_seed_frame" in meta.columns:
        meta["_seed_sort"] = (~meta["is_seed_frame"].fillna(False).astype(bool)).astype(int)
        sort_cols = ["embryo_id", "_seed_sort"] + (["frame_index"] if "frame_index" in meta.columns else [])
    else:
        sort_cols = ["embryo_id"] + (["frame_index"] if "frame_index" in meta.columns else [])
    meta = meta.sort_values(sort_cols, kind="stable").drop_duplicates("embryo_id", keep="first")
    return meta.drop(columns=["_seed_sort"], errors="ignore")


def _load_prediction_tables() -> pd.DataFrame:
    geno = _read_csv(TR / "genotype_transfer_predictions.csv")
    pheno = _read_csv(TR / "phenotype_transfer_predictions.csv")
    if geno.empty:
        raise FileNotFoundError(TR / "genotype_transfer_predictions.csv")
    keep = ["query_embryo_id", "dataset", "query_experiment", "true_genotype", "true_zygosity", "sequenced", "stratum", "predicted_label", "top_probability", "argmax_margin", "consistency_score", "status"]
    g = geno[[c for c in keep if c in geno.columns]].copy().rename(columns={
        "query_embryo_id": "embryo_id", "predicted_label": "predicted_genotype",
        "top_probability": "predicted_genotype_prob", "argmax_margin": "predicted_genotype_margin",
        "consistency_score": "predicted_genotype_consistency", "status": "predicted_genotype_status",
    })
    if not pheno.empty:
        pkeep = ["query_embryo_id", "predicted_label", "top_probability", "argmax_margin", "consistency_score", "status"]
        p = pheno[[c for c in pkeep if c in pheno.columns]].copy().rename(columns={
            "query_embryo_id": "embryo_id", "predicted_label": "predicted_phenotype",
            "top_probability": "predicted_phenotype_prob", "argmax_margin": "predicted_phenotype_margin",
            "consistency_score": "predicted_phenotype_consistency", "status": "predicted_phenotype_status",
        })
        g = g.merge(p, on="embryo_id", how="left")
    for gene, path in {"cep290": TR / "cep290_homo_low_to_high_predictions.csv", "b9d2": TR / "b9d2_homo_ce_hta_predictions.csv"}.items():
        h = _read_csv(path)
        if h.empty:
            continue
        cols = ["query_embryo_id", "predicted_label", "top_probability", "argmax_margin"]
        h = h[[c for c in cols if c in h.columns]].copy().rename(columns={
            "query_embryo_id": "embryo_id",
            "predicted_label": f"{gene}_homo_only_predicted_phenotype",
            "top_probability": f"{gene}_homo_only_predicted_phenotype_prob",
            "argmax_margin": f"{gene}_homo_only_predicted_phenotype_margin",
        })
        g = g.merge(h, on="embryo_id", how="left")
    return g


def _normalize_path(value, image_root: str | None) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "None"}:
        return ""
    path = str(value)
    if image_root:
        marker = "/morphseq_playground/"
        if marker in path:
            return str(Path(image_root) / path.split(marker, 1)[1])
    m = OLD_ROOT_PATTERN.match(path)
    if m:
        return str(PROJECT_ROOT / path[m.end():])
    return path


def _file_url(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).as_uri() if Path(path).is_absolute() else quote(path)
    except Exception:
        return quote(path)


def _qc_reason(row: pd.Series) -> str:
    reasons = []
    for flag in QC_FLAGS:
        if flag == "use_embryo_flag" or flag not in row.index:
            continue
        val = row.get(flag)
        if pd.notna(val) and bool(val):
            reasons.append(flag)
    if not reasons and "use_embryo_flag" in row.index and pd.notna(row["use_embryo_flag"]):
        return "included" if bool(row["use_embryo_flag"]) else "excluded_unknown_flag"
    return "+".join(reasons) if reasons else "unknown"


def build_manifest(image_root: str | None = None) -> pd.DataFrame:
    pred = _load_prediction_tables()
    meta = _load_image_qc_metadata(_query_experiments())
    manifest = pred.merge(meta, on=["embryo_id", "query_experiment"], how="outer", suffixes=("", "_meta"))
    if "dataset_meta" in manifest.columns:
        manifest["dataset"] = manifest["dataset"].fillna(manifest["dataset_meta"])
    if "true_genotype" in manifest.columns and "genotype" in manifest.columns:
        manifest["true_genotype"] = manifest["true_genotype"].fillna(manifest["genotype"].astype(str).map(lambda v: T.GENOTYPE_RENAME.get(v, v)))
    if "true_zygosity" in manifest.columns and "true_genotype" in manifest.columns:
        manifest["true_zygosity"] = manifest["true_zygosity"].fillna(manifest["true_genotype"].astype(str).map(T.to_zygosity))
    manifest["resolved_image_path"] = manifest["image_path"].map(lambda v: _normalize_path(v, image_root)) if "image_path" in manifest.columns else ""
    manifest["resolved_mask_path"] = manifest["exported_mask_path"].map(lambda v: _normalize_path(v, image_root)) if "exported_mask_path" in manifest.columns else ""
    manifest["excluded"] = ~manifest["use_embryo_flag"].fillna(False).astype(bool) if "use_embryo_flag" in manifest.columns else np.nan
    manifest["qc_reason"] = manifest.apply(_qc_reason, axis=1)
    manifest["review_decision"] = ""
    manifest["review_notes"] = ""
    for c in ["start_age_hpf", "stage", "predicted_stage_hpf"]:
        if c not in manifest.columns:
            manifest[c] = np.nan
    manifest["inferred_stage_hpf"] = manifest["query_experiment"].map(_infer_stage_from_experiment)
    manifest["review_stage_hpf"] = manifest["start_age_hpf"].fillna(manifest["stage"]).fillna(manifest["predicted_stage_hpf"]).fillna(manifest["inferred_stage_hpf"])
    sort_cols = [c for c in ["dataset", "review_stage_hpf", "query_experiment", "well", "embryo_id"] if c in manifest.columns]
    return manifest.sort_values(sort_cols, kind="stable")


def _badge(label: str, value, palette: dict | None = None) -> str:
    if pd.isna(value) or str(value) == "":
        value = "NA"
    color = palette.get(str(value), palette.get("NA", "#777")) if palette else "#777"
    return f'<span class="badge" style="--c:{color}"><b>{html.escape(label)}</b>{html.escape(str(value))}</span>'


def _fmt_prob(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _bbox(row: pd.Series) -> str:
    try:
        width = float(row.get("Width (px)", np.nan))
        height = float(row.get("Height (px)", np.nan))
        if math.isnan(width):
            width = float(row.get("width_px", np.nan))
        if math.isnan(height):
            height = float(row.get("height_px", np.nan))
        x0 = float(row.get("bbox_x_min", np.nan)) / width * 100
        x1 = float(row.get("bbox_x_max", np.nan)) / width * 100
        y0 = float(row.get("bbox_y_min", np.nan)) / height * 100
        y1 = float(row.get("bbox_y_max", np.nan)) / height * 100
        return f'<div class="bbox" style="left:{x0:.3f}%;top:{y0:.3f}%;width:{(x1-x0):.3f}%;height:{(y1-y0):.3f}%"></div>'
    except Exception:
        return ""


def _card(row: pd.Series) -> str:
    image_url = _file_url(str(row.get("resolved_image_path", "")))
    mask_url = _file_url(str(row.get("resolved_mask_path", "")))
    excluded = bool(row.get("excluded"))
    qc_color = "#B2182B" if excluded else "#1b9e77"
    dataset = str(row.get("dataset", ""))
    homo_col = f"{dataset}_homo_only_predicted_phenotype"
    homo_pheno = row.get(homo_col, "") if homo_col in row.index else ""
    image_block = "<div class=\"missing-image\">no image path</div>"
    if image_url:
        image_block = (
            f'<div class="image-wrap"><img loading="lazy" src="{image_url}" '
            f'alt="{html.escape(str(row.get("snip_id", "")))}">{_bbox(row)}</div>'
        )
    qc_rows = [f"{flag}={row.get(flag)}" for flag in QC_FLAGS if flag in row.index]
    extras = [
        f"geno p={_fmt_prob(row.get('predicted_genotype_prob'))}",
        f"pheno p={_fmt_prob(row.get('predicted_phenotype_prob'))}",
        f"mask={html.escape(mask_url) if mask_url else 'NA'}",
        html.escape('; '.join(qc_rows)),
    ]
    class_name = "card is-excluded" if excluded else "card"
    homo_badge = _badge("homo-only", homo_pheno, PALETTES["phenotype"]) if str(homo_pheno) not in {"", "nan"} else ""
    return (
        f'<article class="{class_name}">'
        '<div class="card-head">'
        f'<div class="ids">{html.escape(str(row.get("well", "")))} | {html.escape(str(row.get("embryo_id", "")))}<br><small>{html.escape(str(row.get("snip_id", "")))}</small></div>'
        '<div class="badges">'
        f'{_badge("seq", row.get("true_genotype"), PALETTES["zyg"])}'
        f'{_badge("pred geno", row.get("predicted_genotype"), PALETTES["zyg"])}'
        f'{_badge("pred pheno", row.get("predicted_phenotype"), PALETTES["phenotype"])}'
        f'{homo_badge}'
        f'<span class="badge" style="--c:{qc_color}"><b>QC</b>{"excluded" if excluded else "included"}: {html.escape(str(row.get("qc_reason", "")))}</span>'
        '</div></div>'
        f'{image_block}'
        f'<details><summary>details</summary><div class="details">{"<br>".join(extras)}</div></details>'
        '</article>'
    )


def _html_page(df: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for (stage, plate), sub in df.groupby(["review_stage_hpf", "query_experiment"], dropna=False, sort=True):
        cards = "\n".join(_card(row) for _, row in sub.iterrows())
        sections.append(f'<section><h2>{html.escape(str(stage))} hpf · {html.escape(str(plate))} · n={len(sub)}</h2><div class="grid">{cards}</div></section>')
    style = """
    :root { color-scheme: light; --bg:#f5f1e9; --ink:#181714; --muted:#666; --line:#d7d0c4; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--ink); }
    header { position:sticky; top:0; z-index:5; background:rgba(245,241,233,.96); border-bottom:1px solid var(--line); padding:14px 18px; }
    h1 { margin:0; font-size:20px; letter-spacing:.01em; } h2 { margin:28px 18px 12px; font-size:15px; color:#322; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(360px, 1fr)); gap:14px; padding:0 18px 28px; }
    .card { background:#fff; border:1px solid var(--line); box-shadow:0 1px 6px rgba(0,0,0,.08); }
    .card.is-excluded { outline:3px solid rgba(178,24,43,.35); } .card-head { padding:10px 10px 8px; border-bottom:1px solid var(--line); }
    .ids { font-size:12px; line-height:1.25; margin-bottom:8px; overflow-wrap:anywhere; } .ids small { color:var(--muted); }
    .badges { display:flex; flex-wrap:wrap; gap:5px; } .badge { display:inline-flex; gap:4px; align-items:center; border-left:6px solid var(--c); background:#f3f3f3; padding:3px 6px; font-size:11px; }
    .badge b { text-transform:uppercase; font-size:9px; color:#555; }
    .image-wrap { position:relative; width:100%; min-height:240px; background:#111; display:flex; align-items:center; justify-content:center; overflow:hidden; }
    .image-wrap img { width:100%; height:auto; display:block; } .bbox { position:absolute; border:2px solid #ffe600; box-shadow:0 0 0 1px #000, 0 0 12px rgba(255,230,0,.65); pointer-events:none; }
    .missing-image { min-height:240px; display:flex; align-items:center; justify-content:center; color:#777; background:#eee; font-size:13px; }
    details { padding:8px 10px 10px; font-size:11px; color:#555; } summary { cursor:pointer; } .details { margin-top:6px; overflow-wrap:anywhere; }
    """
    path.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{style}</style></head><body><header><h1>{html.escape(title)}</h1><div>{len(df)} embryos · grouped by stage and plate</div></header>{"".join(sections)}</body></html>\n')


def write_outputs(manifest: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "embryo_portfolio_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    links = []
    for dataset, sub in manifest.groupby("dataset", dropna=False):
        page = out_dir / f"{dataset}_embryo_portfolio.html"
        _html_page(sub, page, f"{dataset} embryo review portfolio")
        links.append(f'<li><a href="{page.name}">{html.escape(str(dataset))}</a></li>')
    (out_dir / "index.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Embryo portfolios</title></head><body><h1>Embryo portfolios</h1><ul>{''.join(links)}</ul><p>Manifest: <a href='embryo_portfolio_manifest.csv'>embryo_portfolio_manifest.csv</a></p></body></html>\n")
    print(f"wrote {manifest_path.relative_to(RUN_DIR)}")
    print(f"wrote {len(links)} HTML portfolio pages under {out_dir.relative_to(RUN_DIR)}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", default=None, help="Optional replacement root containing morphseq_playground image-data suffixes.")
    ap.add_argument("--out-dir", default=str(OUT), help="Output directory for manifest and HTML.")
    args = ap.parse_args()
    manifest = build_manifest(args.image_root)
    write_outputs(manifest, Path(args.out_dir))
    print("\nCounts by dataset/stage:")
    print(manifest.groupby(["dataset", "review_stage_hpf"]).size().to_string())
    print("\nQC included/excluded:")
    print(manifest.groupby(["dataset", "excluded"]).size().to_string())


if __name__ == "__main__":
    main()
