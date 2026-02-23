✅ Loaded configuration from: /workspace/morphseq/segmentation_sandbox/scripts/annotations/config.json
Creating new annotations from SAM2: /tmp/tmp54o46o9r.json
❌ ERROR: add_phenotype() called with both approaches:
   Embryo approach: embryo_id='exp1_e01', target='None'
   Snip approach: snip_ids=['exp1_e01_s0100']
   SOLUTION: Use either (embryo_id + target) OR snip_ids, not both
Caught exception as expected:
ValueError - Ambiguous parameters: cannot use both embryo and snip approaches

### Summary
- **Action**: Called `add_phenotype` with both `embryo_id` and `snip_ids`.
- **Result**: `ValueError` for ambiguous parameter usage.
- **Cause**: Method prevents simultaneous embryo and snip modes.
- **Suggested Fix**: Ensure callers use either `embryo_id` with `target` or `snip_ids` exclusively.
