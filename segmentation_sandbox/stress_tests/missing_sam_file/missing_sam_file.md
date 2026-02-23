Caught exception as expected:
FileNotFoundError - SAM2 file not found: nonexistent_sam2.json

### Summary
- **Action**: Initialized `EmbryoMetadata` with nonexistent SAM2 file.
- **Result**: `FileNotFoundError`.
- **Cause**: Provided SAM2 path does not exist.
- **Suggested Fix**: Verify SAM2 file path before initialization or handle missing file with user-friendly message.
