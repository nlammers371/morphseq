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

### Current threshold recommendation:
- Hard cutoff at **ncc_min < 0** for definite failures
- Flag for review: **0.0 – 0.6** (borderline, ~4.3%)
- Accept: **>= 0.6** (89.5%)
