#!/usr/bin/env python3
"""
Quick helper to map representative JPEG frames back to their originating ND2
sequence indices. Reads the curated lists from bad_image_examples.md and looks
up (well, time_int, nd2_series_num) using the per-experiment metadata CSV.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYGROUND_ROOT = REPO_ROOT / "morphseq_playground"
METADATA_ROOT = PLAYGROUND_ROOT / "metadata" / "build03_output"
RAW_ROOT = PLAYGROUND_ROOT / "raw_image_data"
EXAMPLE_DOC = Path(__file__).with_name("bad_image_examples.md")

PATH_PATTERN = re.compile(
    r"(?P<date>\d{8})_(?P<well>[A-H]\d{2})_ch\d{2}_t(?P<time>\d{4})\.jpg$"
)


@dataclass
class FrameRecord:
    category: str
    image_path: Path
    date: str
    well: str
    time_int: int
    nd2_series_num: int | None
    nd2_path: Path | None


def iter_example_entries(markdown_path: Path) -> Iterable[tuple[str, Path]]:
    """Yield (category, image_path) pairs from the markdown list."""
    category = "unknown"
    for raw_line in markdown_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            category = line[2:].strip()
            continue
        if ".jpg" not in line:
            continue
        # bullet lines: "- path"
        if line.startswith("-"):
            _, path_str = line.split("-", 1)
            token = path_str.strip().strip("`'\"")
            img = Path(token)
            if not img.is_absolute():
                img = (REPO_ROOT / img).resolve()
            yield category, img


def load_metadata(date: str) -> dict[tuple[str, int], int]:
    csv_path = METADATA_ROOT / f"expr_embryo_metadata_{date}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV missing for {date}: {csv_path}")
    mapping: dict[tuple[str, int], int] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        missing = {"well", "time_int", "nd2_series_num"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")
        for row in reader:
            try:
                well = str(row["well"])
                time_int = int(row["time_int"])
                nd2_series = int(row["nd2_series_num"])
            except (KeyError, ValueError):
                continue
            mapping[(well, time_int)] = nd2_series
    return mapping


def find_nd2_file(date: str) -> Path | None:
    for scope in ("YX1", "Keyence"):
        root = RAW_ROOT / scope / date
        if root.exists():
            nd2_files = list(root.glob("*.nd2"))
            if len(nd2_files) == 1:
                return nd2_files[0]
            if len(nd2_files) > 1:
                # prefer the smallest lexicographically to keep deterministic
                return sorted(nd2_files)[0]
    return None


def main() -> None:
    records: list[FrameRecord] = []
    cache: dict[str, dict[tuple[str, int], int]] = {}
    nd2_cache: dict[str, Path | None] = {}

    for category, img_path in iter_example_entries(EXAMPLE_DOC):
        match = PATH_PATTERN.search(str(img_path))
        if not match:
            print(f"[WARN] Could not parse path: {img_path}")
            continue
        date = match.group("date")
        well = match.group("well")
        time_int = int(match.group("time"))

        if date not in cache:
            cache[date] = load_metadata(date)
        metadata = cache[date]

        nd2_series = None
        if (well, time_int) in metadata:
            nd2_series = metadata[(well, time_int)]
        else:
            print(f"[WARN] No metadata row for {well} t{time_int:04} (date {date})")

        if date not in nd2_cache:
            nd2_cache[date] = find_nd2_file(date)

        records.append(
            FrameRecord(
                category=category,
                image_path=img_path,
                date=date,
                well=well,
                time_int=time_int,
                nd2_series_num=nd2_series,
                nd2_path=nd2_cache[date],
            )
        )

    if not records:
        print("No frames parsed from markdown; nothing to do.")
        return

    out_path = EXAMPLE_DOC.with_name("frame_nd2_lookup.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["category", "image_path", "date", "well", "time_int", "nd2_series_num", "nd2_path"]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.category,
                    rec.image_path,
                    rec.date,
                    rec.well,
                    rec.time_int,
                    rec.nd2_series_num if rec.nd2_series_num is not None else "",
                    rec.nd2_path if rec.nd2_path is not None else "",
                ]
            )

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
