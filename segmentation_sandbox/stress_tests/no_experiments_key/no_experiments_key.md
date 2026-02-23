Caught exception as expected:
ValueError - Error reading SAM2 file /tmp/tmpyv21di96.json: Invalid SAM2 format: missing 'experiments' key in /tmp/tmpyv21di96.json

### Summary
- **Action**: SAM2 file missing the required `experiments` key.
- **Result**: `ValueError` reporting invalid SAM2 format.
- **Cause**: Data structure did not meet expected schema.
- **Suggested Fix**: Ensure SAM2 JSON contains the top-level `experiments` field before processing.
