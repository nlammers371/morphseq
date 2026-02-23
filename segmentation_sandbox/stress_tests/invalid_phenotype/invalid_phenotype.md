✅ Loaded configuration from: /workspace/morphseq/segmentation_sandbox/scripts/annotations/config.json
Creating new annotations from SAM2: /tmp/tmpw5kxi9a2.json
❌ ERROR: Invalid phenotype 'NOT_REAL'
   Valid options: ['NORMAL', 'EDEMA', 'DEAD', 'CONVERGENCE_EXTENSION', 'BLUR', 'CORRUPT']
Caught exception as expected:
ValueError - Invalid phenotype 'NOT_REAL'. Valid options: ['NORMAL', 'EDEMA', 'DEAD', 'CONVERGENCE_EXTENSION', 'BLUR', 'CORRUPT']

### Summary
- **Action**: Used unrecognized phenotype label `NOT_REAL`.
- **Result**: `ValueError` listing allowed phenotypes.
- **Cause**: Strict validation against predefined phenotype list.
- **Suggested Fix**: Expand phenotype list or provide clearer error guidance for custom values.
