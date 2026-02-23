# Restore Candidates (morphseq_CORRUPT_OLD -> morphseq)

Date: 2026-02-12
Scope: tracked, script-like files that exist in `morphseq_CORRUPT_OLD` but are missing in current `morphseq`.

## Summary

- `src/` missing entries (all selected extensions): 666
- High-signal code/notebook candidates in `src/` (excluding `_Archive` and docs artifacts): 183
- `docs/` missing: 9
- `results/` script/notebook/doc-style missing: 28

## Critical Non-Tracked Asset

The restore candidate lists above are git-tracked files only. They do not include
the local `morphseq_playground/` data directory, which pipeline scripts expect.

To restore the playground (including its internal symlink layout) from
`morphseq_CORRUPT_OLD`, run:

```bash
docs/recovery/restore_morphseq_playground.sh
```

Optional:
- `--copy` to copy contents instead of creating a top-level symlink
- `--force` to replace an existing target

## Prioritized Restore Tiers

1. Tier 1 (core pipeline/runtime code): 47 files
Path list: `docs/recovery/2026-02-12_restore_candidates_tier1.txt`
Includes:
- `src/data/*`
- `src/functions/*`
- `src/models/*`
- `src/run/*`
- `src/segmentation/ml_preprocessing/*`
- `src/tools/*`

2. Tier 2 (training/model stacks likely legacy but potentially reusable): 118 files
Path list: `docs/recovery/2026-02-12_restore_candidates_tier2.txt`
Includes:
- `src/diffusion/*`
- `src/flux/*`
- `src/lightning/*`
- `src/losses/*`
- `src/vae/*`

3. Tier 3 (analysis/support and exploratory code): 18 files
Path list: `docs/recovery/2026-02-12_restore_candidates_tier3.txt`
Includes:
- `src/crossmodal/*`
- `src/seq/*`

4. Lower priority (mostly archive/log/context docs)
- `docs/recovery/2026-02-12_restore_candidates_results_scripts_low_priority.txt`
- `docs/recovery/2026-02-12_restore_candidates_docs_low_priority.txt`

## Recommended Next Restore Action

Restore Tier 1 first into a dedicated recovery branch, run import/tests/smoke checks, then decide on Tier 2 modules selectively.
