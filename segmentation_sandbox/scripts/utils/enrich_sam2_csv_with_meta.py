#!/usr/bin/env python3
"""
Enrich SAM2 per-experiment CSV with raw image metadata from per-experiment metadata JSON.

Use this when export_sam2_metadata_to_csv.py missed Height/Width/Time fields because it
looked only for a monolithic experiment_metadata.json.

Inputs:
- --sam2-csv: per-experiment CSV (e.g., sam2_metadata_{exp}.csv)
- --meta-json: per-experiment metadata JSON (raw_data_organized/{exp}/experiment_metadata_{exp}.json)
- --out: output CSV path (default: overwrite input CSV)

Behavior:
- For each row keyed by image_id, fill any missing fields from metadata JSON:
  ['Height (um)','Height (px)','Width (um)','Width (px)','BF Channel','Objective',
   'Time (s)','Time Rel (s)','height_um','height_px','width_um','width_px',
   'bf_channel','objective','raw_time_s','relative_time_s']
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any

FIELDS = [
    'Height (um)', 'Height (px)', 'Width (um)', 'Width (px)',
    'BF Channel', 'Objective', 'Time (s)', 'Time Rel (s)',
    'height_um', 'height_px', 'width_um', 'width_px',
    'bf_channel', 'objective', 'raw_time_s', 'relative_time_s',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich SAM2 CSV with per-experiment raw metadata")
    p.add_argument("--sam2-csv", required=True, help="Path to per-experiment SAM2 CSV")
    p.add_argument("--meta-json", required=True, help="Path to per-experiment metadata JSON")
    p.add_argument("--out", help="Output CSV (default: overwrite input CSV)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def build_meta_map(meta_path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(meta_path.read_text())
    result: Dict[str, Dict[str, Any]] = {}
    exps = data.get("experiments") or {}
    # Flatten videos -> image_ids -> raw fields
    for exp_id, exp_data in exps.items():
        vids = exp_data.get("videos", {})
        for vid, vid_data in vids.items():
            imgs = vid_data.get("image_ids", {})
            for image_id, image_data in imgs.items():
                # image_data may directly include the raw fields
                entry = {}
                for k in FIELDS:
                    if k in image_data:
                        entry[k] = image_data.get(k)
                # Some metadata may be nested under aliases; merge any present
                if entry:
                    result[image_id] = entry
    # Fallback shape: top-level 'image_ids'
    if not result and "image_ids" in data:
        imgs = data["image_ids"]
        if isinstance(imgs, dict):
            for image_id, image_data in imgs.items():
                entry = {}
                for k in FIELDS:
                    if k in image_data:
                        entry[k] = image_data.get(k)
                if entry:
                    result[image_id] = entry
    return result


def enrich_csv(csv_path: Path, meta_map: Dict[str, Dict[str, Any]], out_path: Path, verbose: bool = False) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    # Ensure all fields present in header
    new_fields = [k for k in FIELDS if k not in fieldnames]
    fieldnames = list(fieldnames) + new_fields

    updated = 0
    for r in rows:
        img = r.get("image_id", "")
        if not img:
            continue
        meta = meta_map.get(img)
        if not meta:
            continue
        for k, v in meta.items():
            if k not in r or r.get(k) in (None, ""):
                if v is not None and v != "":
                    r[k] = v
                    updated += 1
    if verbose:
        print(f"📈 Filled {updated} fields across {len(rows)} rows")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return updated


def main() -> int:
    args = parse_args()
    csv_path = Path(args.sam2_csv)
    meta_path = Path(args.meta_json)
    out_path = Path(args.out) if args.out else csv_path

    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return 2
    if not meta_path.exists():
        print(f"❌ Metadata JSON not found: {meta_path}")
        return 2
    if args.verbose:
        print(f"📄 CSV: {csv_path}")
        print(f"🧾 Meta: {meta_path}")
        print(f"🖊️  Out: {out_path}")

    meta_map = build_meta_map(meta_path)
    if args.verbose:
        print(f"🔎 Mapped metadata for {len(meta_map)} images")
    updated = enrich_csv(csv_path, meta_map, out_path, verbose=args.verbose)
    if args.verbose:
        print(f"✅ Done. Updated fields: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

