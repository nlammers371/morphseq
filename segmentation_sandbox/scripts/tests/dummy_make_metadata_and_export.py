#!/usr/bin/env python3
"""
Dummy script: generate a minimal enhanced metadata JSON and export to CSV

This does not depend on real data; it fabricates a tiny structure that matches
the expected schema (experiments → videos → image_ids dict with raw_image_data_info)
and then runs the CSV exporter to validate schema and column presence.
"""

import json
from pathlib import Path
from datetime import datetime
import argparse

from scripts.utils.export_sam2_metadata_to_csv import SAM2MetadataExporter


def make_dummy_metadata() -> dict:
    exp_id = "20990101"
    video_id = f"{exp_id}_A01"
    image_id = f"{video_id}_ch00_t0000"

    return {
        "segmentation_format": "rle",
        "experiments": {
            exp_id: {
                "videos": {
                    video_id: {
                        "video_id": video_id,
                        "well_id": "A01",
                        "processed_jpg_images_dir": "/tmp/nonexistent",  # not used by exporter
                        "image_ids": {
                            image_id: {
                                "frame_index": 0,
                                "raw_image_data_info": {
                                    "Height (um)": 7000.0,
                                    "Height (px)": 2000,
                                    "Width (um)": 7000.0,
                                    "Width (px)": 2000,
                                    "BF Channel": 0,
                                    "Objective": "4x",
                                    "Time (s)": 0.0,
                                    "Time Rel (s)": 0.0,
                                    "height_um": 7000.0,
                                    "height_px": 2000,
                                    "width_um": 7000.0,
                                    "width_px": 2000,
                                    "bf_channel": 0,
                                    "objective": "4x",
                                    "raw_time_s": 0.0,
                                    "relative_time_s": 0.0,
                                    "microscope": "DUMMY",
                                    "nd2_series_num": 1
                                },
                                "embryos": {
                                    f"{video_id}_e01": {
                                        "snip_id": f"{video_id}_e01_0000",
                                        "segmentation": {"counts": "", "size": [10, 10]},
                                        "segmentation_format": "rle",
                                        "bbox": [0, 0, 10, 10],
                                        "area": 100.0,
                                        "mask_confidence": 0.9
                                    }
                                }
                            }
                        },
                        "seed_frame_info": {"seed_frame": image_id},
                        "medium": "E3",
                        "genotype": "wt",
                        "chem_perturbation": "none",
                        "start_age_hpf": 24,
                        "embryos_per_well": 1,
                        "temperature": 28.5,
                        "well_qc_flag": 0
                    }
                }
            }
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    data = make_dummy_metadata()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(data, f, indent=2)

    exporter = SAM2MetadataExporter(args.out_json)
    exporter.load_and_validate_json()
    df = exporter.export_to_csv(args.out_csv)
    print(f"Wrote CSV with {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()

