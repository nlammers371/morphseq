---
name: analyze-classification
description: morphseq classification — AUROC comparisons, genotype classification, difference detection, phenotype emergence, misclassification deep-dives
---

You are a morphseq classification expert. When the user asks about AUROC comparisons, genotype classification, difference detection, phenotype emergence, or misclassification deep-dives, use the `src/analyze/classification/` module.

## Canonical references (read first)

- **`src/analyze/classification/README.md`** — user-facing API, input contract, layer matrix, comparison-mode decision table, troubleshooting, full `run_classification` signature.
- **`src/analyze/classification/viz/README.md`** — plotting cookbook organized by "I want to see X → use Y".
- **`src/analyze/classification/DESIGN.md`** — approved design spec. Open when deciding whether to change the signature, or when confused about `positive` / `negative` / `comparisons` semantics.
- **`src/analyze/classification/emergence/ALGORITHM.md`** + **`DESIGN.md`** — emergence algorithm.

## When to use this module

- "AUROC over time" for any class vs class(es) comparison across developmental bins
- Phenotype emergence ordering (read from `result.scores`)
- Per-embryo misclassification inspection
- Interpretable classifier geometry: `directions/` (preferred) or contrast coordinates (legacy-but-supported)

## Guardrails

- **Canonical module is `analyze.classification`.** `analyze.difference_detection` is a deprecated re-export shim — still works, but prefer the canonical path in new code.
- **Primary entry point is `run_classification`**, not `run_classification_test`. The latter is legacy and emits `FutureWarning`.
- **Prefer `directions/`** over contrast coordinates for new interpretable-geometry work. Contrast coordinates remain supported because `trajectory_condensation` and some pre-April-2026 result scripts depend on them.
- `save_contrast_coordinates=True` and `save_classifier_directions=True` are **binary-path only** — they error in the multiclass fast path. Pass an explicit `comparisons=`, `positive=/negative=`.
- `save_contrast_coordinates=True` requires `n_permutations > 0` (needs the null for shrinkage).

## Setup

```python
import matplotlib
matplotlib.use("Agg")

from analyze.classification import run_classification, ClassificationAnalysis
```

## One-shot quickstart

```python
result = run_classification(
    df,
    class_col="genotype",
    id_col="embryo_id",
    time_col="predicted_stage_hpf",
    features={"emb": "z_mu_b", "shape": ["total_length_um"]},
    comparisons="all_pairs",
    bin_width=2.0,
    n_splits=5,
    n_permutations=500,
    n_jobs=-1,
    save_predictions=True,        # enables misclass / margin plots downstream
    save_dir="results/my_run/",
)

result.plot_aurocs(output_path="aurocs.png")
# Full plot menu: see viz/README.md
```

## Where the input df comes from

Feature-labeled embryo dataframes in this repo are produced by `src/data_pipeline/feature_extraction/` + VAE embedding export. VAE latents are `z_mu_b_*`; shape features include `total_length_um`, `yolk_area_um2`, etc. See README § "Input contract" for the full column requirements.
