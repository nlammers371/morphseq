#!/usr/bin/env python3
"""
Lightweight verification that SAM2 snip_id creation uses the canonical t#### suffix
and that parsing_utils can parse the resulting IDs.

This test reads a small CSV from segmentation_sandbox/tests/dummy_sam2/dummy_ids.csv
with columns: embryo_id, image_id, expected_snip_id
"""

import csv
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    dummy_csv = here.parent.parent / "tests" / "dummy_sam2" / "dummy_ids.csv"
    if not dummy_csv.exists():
        print(f"ERROR: Missing dummy IDs CSV: {dummy_csv}")
        return 1

    # Import locally to avoid heavy runtime side effects
    from .sam2_utils import create_snip_id
    from .parsing_utils import parse_entity_id, get_entity_type, is_snip_t_style, validate_snip_t_style

    failures = 0
    with dummy_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            embryo_id = row["embryo_id"].strip()
            image_id = row["image_id"].strip()
            expected = row["expected_snip_id"].strip()

            got = create_snip_id(embryo_id, image_id)
            if got != expected:
                print(f"❌ Mismatch: expected {expected}, got {got}")
                failures += 1
                continue

            # Basic parse checks
            etype = get_entity_type(got)
            if etype != "snip":
                print(f"❌ Wrong entity type for {got}: {etype}")
                failures += 1
                continue

            parsed = parse_entity_id(got, entity_type="snip")
            if parsed.get("embryo_id") != embryo_id:
                print(f"❌ Embryo mismatch for {got}: {parsed.get('embryo_id')} != {embryo_id}")
                failures += 1
                continue

            # Canonical style validators
            if not is_snip_t_style(got):
                print(f"❌ Not canonical t-style snip: {got}")
                failures += 1
                continue
            try:
                validate_snip_t_style(got)
            except Exception as e:
                print(f"❌ Validation failed for {got}: {e}")
                failures += 1
                continue

            print(f"✅ {got}")

    if failures:
        print(f"Finished with {failures} failures")
        return 2
    print("All dummy SAM2 snip_id checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
