Dummy SAM2 test set

This minimal test set verifies snip_id creation using SAM2 utilities with the canonical t#### suffix.

Contents:
- dummy_ids.csv: Small table of embryo_id and image_id pairs.

Usage:
- Run `python segmentation_sandbox/scripts/utils/test_sam2_snip_style.py` to validate that snip_ids are generated with `_t####` and that parsing_utils can parse them.

