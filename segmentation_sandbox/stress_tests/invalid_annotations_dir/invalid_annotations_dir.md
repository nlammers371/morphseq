Caught exception as expected:
FileNotFoundError - Directory for annotations file does not exist: nonexistent_dir

### Summary
- **Action**: Provided annotations path in a directory that doesn't exist.
- **Result**: `FileNotFoundError` for missing annotations directory.
- **Cause**: Code expects directory to pre-exist.
- **Suggested Fix**: Automatically create directories or prompt user to do so.
