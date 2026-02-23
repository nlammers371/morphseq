First run (create):
SAM2 input: /workspace/morphseq/segmentation_sandbox/stress_tests/existing_output_file/sam2_annotations.json
Output path: /workspace/morphseq/segmentation_sandbox/stress_tests/existing_output_file/data/embryo_metadata/sam2_biology.json
Loading SAM2 data...
Creating new metadata structure...

Result summary:
Operation: create
Embryos: 1
Total snips: 1

Saved metadata to: /workspace/morphseq/segmentation_sandbox/stress_tests/existing_output_file/data/embryo_metadata/sam2_biology.json

Second run (attempt update without --force):
SAM2 input: /workspace/morphseq/segmentation_sandbox/stress_tests/existing_output_file/sam2_annotations_v2.json
Output path: /workspace/morphseq/segmentation_sandbox/stress_tests/existing_output_file/data/embryo_metadata/sam2_biology.json
Output file exists: /workspace/morphseq/segmentation_sandbox/stress_tests/existing_output_file/data/embryo_metadata/sam2_biology.json
Use --force to overwrite or --dry-run to preview


### Summary
- **Action**: Ran 07_embryo_metadata_update twice, second time with new SAM2 frame and existing output file.
- **Result**: Script aborted with message "Output file exists" instead of merging new snips.
- **Cause**: Early output-file check prevents update mode when annotations JSON already exists.
- **Suggested Fix**: Provide an update mode that merges new snips instead of exiting.
