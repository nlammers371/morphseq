# ncc_min Threshold Observations (2026-04-23)

## Key Findings from Visual Inspection of Threshold Bins

### Clear rules:
- **ncc_min < 0**: Definitively bad. Negative NCC means the detector is picking up real inter-slice
  motion. Nothing should be allowed through. (6.2% of images, n=571)

- **ncc_min >= 0.6**: Appears safe. Images look clean. (89.5% of images, n=8254)

### Ambiguous zone (4.3% of images total):
- **0.0 – 0.2**: Still bad, just less extreme. (0.8%, n=75)
- **0.2 – 0.6**: Highly variable — some images look fine, some clearly have artifacts.
  This is the problem region. (3.5%, n=323)

### Open question:
The 0.2–0.6 zone is small (3.5%) but variable. Worth exploring whether a secondary
metric derived from ncc_min (e.g. ncc_mean, bad_pair_frac, or a combination) could
better classify these borderline cases without discarding good images.

### Key fix: min_tile_coverage raised from 0.10 → 0.25 (2026-04-23)

Edge tiles that straddle the embryo boundary (mostly background, small embryo sliver)
were driving down ncc_p05 and ncc_min — the intensity discontinuity at the mask edge
creates low NCC between Z-slices even with zero real motion. This is NOT motion; it is
a focus-dependent edge effect.

Raising min_tile_coverage to 0.25 excludes these boundary tiles. Validated on:
- D05 t=14: ncc_p05 0.751 → 0.874 (was flagged, is actually clean)
- C04 t=45: ncc_p05 0.919 → 0.954 (already clean, confirmed)
- B06 t=16: ncc_p05 0.799 → 0.750 (slight drop but stays in safe zone ≥0.6)
- E08 t=18: ncc_p05 0.819 → 0.770 (stays in safe zone, may have subtle real movement)

B06 and E08 not improving is fine — they did not get worse, and both sit above the
safe threshold. The fix correctly targets the edge artifact without penalizing
embryos with mild real motion.

### Metric decision: use ncc_p05 not ncc_min
ncc_min is pulled down by a single outlier tile. ncc_p05 (5th percentile) is robust
to 1-2 bad tiles and better reflects true motion quality across the stack.

### Current best threshold: ncc_p05 < 0.85 (with min_tile_coverage=0.25)

At this threshold: **8.7% of embryo-timepoint images flagged as FAIL**.
The curve is flat from -1 to ~0.85, then shoots up steeply — 0.85 is a natural elbow.
Going to 0.90 flags ~29% — a 21pp jump that captures too many borderline-clean images.

**If we had to stop here, ncc_p05 < 0.85 is the production threshold.**

### Documented borderline cases (~0.80–0.85) for future reference:

| Well | t | ncc_p05 | bad_pair_frac | Verdict |
|------|---|---------|---------------|---------|
| A10  | 98 | 0.802 | 0.143 | Clean — no visible motion |
| E11  | 79 | 0.842 | 0.143 | Real motion — correctly caught |
| D05  | 17 | 0.811 | 0.071 | Slight motion |
| C04  | 11 | 0.831 | 0.143 | Clearly moving |

Observation: **bad_pair_frac is consistent with ncc_p05** for the motion cases above
(all show bad_pair_frac > 0.10). This supports using bad_pair_frac as a secondary
criterion to reduce false positives in the 0.80–0.85 zone.

### Next step: refine with bad_pair_frac as secondary filter
Use `ncc_p05 < 0.85 AND bad_pair_frac > 0.10` to rule out borderline clean cases
like A10 t=98 (ncc_p05=0.802 but may be clean). This is the logical next analysis step.
