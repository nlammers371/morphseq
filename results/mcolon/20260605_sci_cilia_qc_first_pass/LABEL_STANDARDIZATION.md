# Label Standardization — cilia QC first-pass label transfer

**Date:** 2026-06-06 · **Owner:** mdcolon
**Scope:** the label-transfer QC of the new QC-first-pass experiments
(`20260605_sci_cilia_qc_first_pass.txt`) against our old labeled reference data.

This file is the **progeny record** of how we standardize labels for this analysis so the
choices are not buried in a script. The actual rename is a **pre-processing step applied in
place** to the `genotype` column (no separate standardization module) — see
`build_reference_and_transfer.py`.

---

## Why this exists

The reference datasets and the new query experiments label the **same biology with different
strings**, in two different vocabularies:

- **Reference** (old, labeled reference): `genotype` uses long canonical forms (`cep290_homozygous`,
  `b9d2_wildtype`, …) and additionally carries a **phenotype** label (`cluster_categories`).
- **Query** (new QC experiments): `genotype` uses short forms (`cep290_homo`, `b9d2_wt`,
  `uncertain`, …) and has **no** phenotype ground truth.

So we standardize the **`genotype`** column to one canonical vocabulary, then run two checks:

1. **Genotype check (first):** does a new experiment's embryos map to the right
   **gene + zygosity** reference class (b9d2 → b9d2, cep290 → cep290, crispant → its gene)?
2. **Phenotype check (second):** within a matched dataset, do they fall into the expected
   phenotype region (cep290: Low_to_High / High_to_Low / Not Penetrant; b9d2: CE / HTA /
   wildtype)?

---

## Canonical genotype vocabulary

**Zygosity tokens** (canonical long forms — match `GENOTYPE_COLORS` in
`src/analyze/viz/styling/color_mapping_config.py`):

| Canonical | Meaning |
|---|---|
| `homozygous` | homozygous mutant |
| `heterozygous` | heterozygous |
| `wildtype` | wild type |
| `unknown` | genotype not resolved |

**The label keeps its gene prefix** (`{gene}_{zygosity}`, e.g. `cep290_homozygous`,
`b9d2_heterozygous`) so the class encodes **gene-of-origin AND zygosity in one string** — this
is what lets the genotype check verify "b9d2 maps to b9d2, cep290 to cep290."

### Genotype rename map (applied in place to `genotype`)

| Raw token(s) seen in data | Canonical |
|---|---|
| `cep290_homo`, `cep290_homozygous` | `cep290_homozygous` |
| `cep290_het`, `cep290_heterozygous` | `cep290_heterozygous` |
| `cep290_wt`, `cep290_wildtype` | `cep290_wildtype` |
| `cep290_unknown`, `cep290_uncertain`, `uncertain` (in cep290 exps) | `cep290_unknown` |
| `b9d2_homo`, `b9d2_homozygous` | `b9d2_homozygous` |
| `b9d2_het`, `b9d2_heterozygous` | `b9d2_heterozygous` |
| `b9d2_wt`, `b9d2_wildtype` | `b9d2_wildtype` |
| `b9d2_unknown`, `b9d2_uncertain` | `b9d2_unknown` |

### Crispant tokens (per-gene classes — no zygosity)

CRISPR-injected crispants are knockdowns, not zygosity classes. They are kept as **per-gene
classes**:

| Raw token(s) | Canonical |
|---|---|
| `foxj1a_crispant` | `foxj1a_crispant` |
| `ift88_crispant` | `ift88_crispant` |
| `sspo_crispant`, `scospondin_crispant` | `sspo_crispant` |
| `if88_ift74_crispant` | `ift88_ift74_crispant` *(double knockdown; fixes the `if88` typo)* |

### Controls

| Raw token | Canonical | Note |
|---|---|---|
| `ab` | `ab_wildtype` | AB wild-type background; behaves as the wildtype-like control |
| `inj-ctrl` | `injection_control` | injection control for crispant experiments |

> Controls are **kept distinct** from mutant/crispant classes so the transfer can show whether
> they land in a wildtype-like region rather than being forced into a mutant class.

---

## Phenotype label conventions (`cluster_categories`) — reference only

The query has no phenotype ground truth, so this only standardizes the **reference**.

**Update for cep290 Phase A:** see `cep290_LABEL_TRANSFER_README.md`. The standalone cep290
Phase-A audit keeps `Not Penetrant` as a real class and merges `Intermediate` into `Low_to_High`.
The older general cep290 plots may still show the two-directional-class view for continuity.

**cep290** — grounded in the most recent reference logic
(`results/mcolon/20260302_NWDB_talk_figures_analysis/02_run_reference_genotype_condensation.py:213`),
which folds **Intermediate into Low_to_High**:

| Raw | Canonical | Note |
|---|---|---|
| `Low_to_High` | `Low_to_High` | |
| `Intermediate` | `Low_to_High` | **merged per the updated condensation script** |
| `High_to_Low` | `High_to_Low` | |
| `Not Penetrant` | `Not Penetrant` | |

**b9d2** — kept as-is for this first pass (deliberately; this is part of what we want to see):

| Raw | Canonical |
|---|---|
| `CE` | `CE` |
| `HTA` | `HTA` |
| `wildtype` | `wildtype` |
| `BA_rescue` | `BA_rescue` |
| `unlabeled` | dropped from the reference (no usable label) |

---

## Known limitations / open items

- **No `strain` column** exists in any of the build06 CSVs (reference or query), so the
  "resolve an `unknown` by checking the original parent strain" rule **cannot run off these
  files** — it would require joining plate metadata. For this first pass, `unknown`/`uncertain`
  are mapped to canonical `{gene}_unknown` and **excluded from the genotype reference** (you
  can't transfer a class you don't trust). Revisit by joining strain from plate metadata.
- **b9d2 phenotype labels intentionally left un-harmonized** with cep290's trajectory labels —
  the two label spaces mean different things and we want to observe how they behave before
  forcing a merge.
- Per-gene crispant separability (foxj1a vs ift88 vs sspo) is the strict test; if it's too hard
  early, fall back to a pooled `cilia_crispant` class.
