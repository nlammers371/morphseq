# Stress Tests for `EmbryoMetadata`

Each subfolder contains a Python script that performs a deliberately invalid
operation on the `EmbryoMetadata` class and captures the resulting error output
in a Markdown report of the same name.

| Test Folder | Purpose |
|-------------|---------|
| `missing_sam_file` | Initialize with a non-existent SAM2 file. |
| `invalid_sam_json` | Load SAM2 file containing malformed JSON. |
| `no_experiments_key` | Provide SAM2 JSON missing the required `experiments` key. |
| `invalid_annotations_dir` | Use an annotations path located in a missing directory. |
| `ambiguous_parameters` | Call `add_phenotype` with both `embryo_id` and `snip_ids`. |
| `invalid_phenotype` | Add a phenotype not in the allowed list. |
| `invalid_target` | Target phenotype addition using an unsupported format. |
| `existing_output_file` | Running pipeline script with preexisting output aborts instead of updating with new snips. |

These tests are intentionally destructive and are meant to highlight how the
system fails under misuse. No fixes are implemented.
