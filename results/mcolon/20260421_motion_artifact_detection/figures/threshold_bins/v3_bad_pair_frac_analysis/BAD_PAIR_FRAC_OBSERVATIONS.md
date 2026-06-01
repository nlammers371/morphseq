# bad_pair_frac Threshold Analysis (2026-04-23)

## Anchor cohort summary

- Good anchor (`ncc_p05 >= 0.90`): n=8049
  - mean bad_pair_frac = 0.0005
  - median bad_pair_frac = 0.0000
  - 95th percentile = 0.0000
  - 99th percentile = 0.0000

- Gray zone (`0.80 <= ncc_p05 < 0.85`): n=114
  - mean bad_pair_frac = 0.0877
  - median bad_pair_frac = 0.0714

## Suggested threshold candidates

- **Conservative candidate**: `bad_pair_frac > 0.000`
  - Anchored to 99th percentile of clearly good images.
  - Minimizes false positives on clean embryos.

- **Balanced candidate**: `bad_pair_frac > 0.000`
  - Anchored to 95th percentile of clearly good images.
  - More sensitive in borderline motion cases.

- **Proxy-separation best** (using `ncc_p05 < 0.80` as likely-motion and `>=0.90` as good):
  - `bad_pair_frac > 0.000`
  - good-anchor flag rate = 0.006
  - likely-motion flag rate = 0.994
  - gray-zone flag rate = 0.798

## User-provided examples

| Well | t | ncc_p05 | bad_pair_frac |
|------|---|---------|---------------|
| A10 | 98 | 0.802 | 0.143 |
| E11 | 79 | 0.842 | 0.143 |
| D05 | 17 | 0.811 | 0.071 |
| C04 | 11 | 0.831 | 0.143 |
| E11 | 230 | NA | NA |

## Practical next rule to test

Use a two-stage rule in borderline cases:

1. Primary fail: `ncc_p05 < 0.85`
2. Secondary confirmation in borderline zone: if `0.80 <= ncc_p05 < 0.85`, require `bad_pair_frac` above a chosen threshold from this analysis.

See CSV/plots in this folder for tranche-level tradeoffs.
