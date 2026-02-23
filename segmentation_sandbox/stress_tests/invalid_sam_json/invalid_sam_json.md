Caught exception as expected:
ValueError - Invalid JSON in SAM2 file /tmp/tmpz6zfvrxx.json: Expecting property name enclosed in double quotes: line 1 column 3 (char 2)

### Summary
- **Action**: Loaded `EmbryoMetadata` with malformed JSON file.
- **Result**: `ValueError` due to JSON parsing failure.
- **Cause**: SAM2 file contained invalid JSON syntax.
- **Suggested Fix**: Validate SAM2 files before use and surface clearer error messages to users.
