# Execution Pack (Phases 0 → 2.5)

This folder turns the high-level plans into a practical execution workflow:

1. `TASK_LIST.md` is the short, operator-friendly run order.
2. Each task has one execution file with:
   - scope and dependencies
   - concrete checks
   - minimal pseudo-logic
   - expected artifacts
   - pass/fail notes

## Phase 0 outlier detector

A dedicated module now exists at:

- `results/mcolon/20260215_roi_discovery_via_ot_feature_maps/outlier_detection.py`

Use this as the single source of truth for Phase 0 outlier detection behavior.
Default workflow remains IQR-based filtering on `total_cost_C`, with optional MAD and
z-score detectors for diagnostics.

## Core biological inspection rule for cep290

Across Phase 1, Phase 2.0, and Phase 2.5, include a **tail-localization check** whenever an ROI, weight map, or learned mask is produced:

- Compute `tail_fraction = signal_in_tail / signal_total` using a pre-declared tail band on canonical coordinates.
- Flag runs where `tail_fraction` is low relative to head/trunk concentration.
- Treat this as an inspection gate (not a hard blocker) unless the dataset/age bin is known to violate the prior.

## Suggested tail band default

- Canonical y-axis increases rostral→caudal.
- Default tail band: bottom 30% of valid embryo pixels (`y_norm >= 0.70`).
- Always write the exact tail-band definition to run metadata.
