# EmbryoMetadata Stress Test Report

This document aggregates the failure modes observed while intentionally
misusing the `EmbryoMetadata` class with dummy SAM2 data. Each test was executed
separately; detailed output and reproduction steps live alongside the scripts in
subdirectories of `stress_tests/`.

## Summary of Failures

1. **Missing SAM2 file** – initialization fails with `FileNotFoundError`.
2. **Malformed SAM2 JSON** – invalid JSON triggers a `ValueError` during parsing.
3. **Missing `experiments` key** – incorrect SAM2 schema raises a `ValueError`.
4. **Annotations directory absent** – constructor raises `FileNotFoundError` when annotations path directory is missing.
5. **Ambiguous `add_phenotype` parameters** – providing both `embryo_id` and `snip_ids` causes a `ValueError`.
6. **Invalid phenotype label** – unrecognized phenotype triggers a `ValueError` listing allowed options.
7. **Invalid target format** – non-numeric or malformed targets raise a `ValueError`.
8. **Existing output file** – pipeline script refuses to update when annotations JSON already exists.

## Observations
- The class performs early validation on SAM2 inputs, preventing bad data from
  propagating.
- Parameter guards in `add_phenotype` are effective at catching ambiguous or
  unsupported usage patterns.
- Error messages are generally descriptive, often suggesting corrective action.

## Suggested Improvements
- Automatically create annotation directories when missing.
- Provide clearer guidance or examples for target formats.
- Consider exposing hooks for custom phenotype lists to reduce invalid value
  errors.
- Allow the pipeline update script to merge new snips when output files already exist.
