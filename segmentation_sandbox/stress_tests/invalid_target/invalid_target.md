✅ Loaded configuration from: /workspace/morphseq/segmentation_sandbox/scripts/annotations/config.json
Creating new annotations from SAM2: /tmp/tmp2i8kxny9.json
Caught exception as expected:
ValueError - Invalid target format: 'abc'. Use 'all', frame number, or 'start:end' range

### Summary
- **Action**: Targeted phenotype addition with non-numeric target `abc`.
- **Result**: `ValueError` for invalid target format.
- **Cause**: Target must be 'all', numeric frame, or range.
- **Suggested Fix**: Validate user input before calling `add_phenotype` or extend parser for richer formats.
