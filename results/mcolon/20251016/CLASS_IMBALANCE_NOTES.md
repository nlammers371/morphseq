# Class imbalance handling notes (2025 results)

## Where this was evaluated
- `results/mcolon/20251014/class_imbalance_method_comparison.py` explicitly compares baseline vs class-imbalance strategies (including `class_weight='balanced'`).

## Current default behavior in the 2025 refactor
- `predictive_signal_test(..., use_class_weights: bool = True)` defaults to using class weights.
- When enabled, it sets `class_weight = 'balanced'` and passes that into `LogisticRegression`.
- `run_analysis.py` wires this through `use_class_weights=config.USE_CLASS_WEIGHTS`.
- `config.py` sets `USE_CLASS_WEIGHTS = True` by default.

## Source tutorial / routed implementation status
- In `src/analyze/difference_detection/classification_test_multiclass.py`, the routed API `run_classification_test()` delegates to `run_multiclass_classification_test()`.
- The underlying classifier factory (`_make_logistic_classifier`) applies `class_weight='balanced'` for both binary and multiclass logistic regression.
- With `verbose=True`, the implementation now logs the balancing policy and per-bin class counts (so balanced-weight assumptions are explicit during runs).

## Important historical caveat
- In the older 20251014 `robust_phenotype_emergence_classifaction.py`, `use_class_weights` existed in the signature/docstring, but the model instantiation shown there did not apply `class_weight`.
- The 20251016 `classification/predictive_test.py` includes an explicit "FIXED" comment indicating the parameter is now actually used.
