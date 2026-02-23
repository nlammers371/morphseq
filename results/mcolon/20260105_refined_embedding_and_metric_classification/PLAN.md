# Cryptic Phenotype Detection Framework — Implementation Plan v3

**Output Directory:** `results/mcolon/20260105_refined_embedding_and_metric_classification`

> **Key Design Decision:** This framework composes existing `src/analyze/difference_detection` APIs rather than reimplementing. All classification and divergence logic reuses production code; only plotting and cryptic window detection are new.

---

## Scientific Purpose: Detecting Cryptic Phenotypes

### What is a Cryptic Phenotype?

A **cryptic phenotype** is a morphological difference that exists but is NOT detectable using standard morphometric measurements (like curvature or length). These subtle differences are only revealed when analyzing high-dimensional embedding space from the VAE model.

**Why this matters:**
- If embedding-based classification detects a difference BEFORE metric-based classification → the early difference is "cryptic"
- This indicates the phenotype involves subtle shape changes not captured by simple metrics
- Understanding cryptic phenotypes reveals the biological mechanism: what's changing before gross morphology is affected?

### The Core Hypothesis

**Embedding space captures shape information that precedes measurable morphometric changes.**

By comparing WHEN embeddings vs metrics detect phenotypes, we can:
1. Identify the earliest time point of phenotype emergence
2. Determine if early changes are "cryptic" (embedding-only) or "overt" (metric-detectable)
3. Understand the temporal sequence: cryptic → overt → severe

---

## Scientific Hypotheses by Phenotype

### CEP290 (Cep290 Mutants)

**Background:** Cep290 is a ciliopathy gene. Mutants show body axis curvature defects.

**Hypothesis:**
- Earliest morphological difference (~18 hpf) is ONLY detectable in embedding space
- Curvature metric differences appear later (>24 hpf)
- The ~18 hpf signal represents a **cryptic phenotype** - subtle shape changes before overt curvature

**Scientific Question:** What is the nature of the early (~18 hpf) phenotype if it's not curvature?

**24h Landmark:** Important developmental checkpoint where many morphological features emerge

### B9D2 HTA (Head-Trunk Angle Phenotype)

**Background:** B9d2 is a ciliopathy gene. HTA phenotype shows abnormal head-trunk angle, appearing after 60 hpf.

**Hypothesis:**
- HTA is a late-onset phenotype (>60 hpf)
- Non-penetrant heterozygotes should be similar to wildtype (no cryptic phenotype for HTA)
- Both curvature AND length metrics should detect HTA at similar times as embeddings

**Scientific Question:** Is HTA purely a late morphometric defect, or are there earlier cryptic signals?

### B9D2 CE (Convergent Extension Phenotype)

**Background:** CE phenotype shows shortened body axis due to defective convergent extension movements during gastrulation.

**Hypothesis:**
- Embedding-based detection: ~10 hpf (CE vs WT), ~15 hpf (CE vs non-pen hets)
- Metric-based detection: ~20 hpf (length and curvature differences)
- The 5-10 hour gap between embedding and metric detection = **cryptic phenotype window**

**Critical Insight - Non-Penetrant Hets Have a Cryptic Phenotype:**
- Non-penetrant hets are genetically heterozygous but show NO visible phenotype
- However, they differ from WT in embedding space
- This means hets carry a subtle, cryptic phenotype even without overt defects
- CE detection vs non-pen hets is LATER than vs WT because hets are already "partway" to CE

**Scientific Question:** What is the nature of the cryptic phenotype in non-penetrant hets?

### Non-Penetrant Hets vs WT (Control Validation)

**Purpose:** Validate that non-penetrant hets carry a cryptic phenotype distinct from WT.

**Hypothesis:**
- Embedding-based AUROC will show significant difference (cryptic phenotype)
- Metric-based AUROC may show minimal or no difference (phenotype is truly cryptic)

**Why This Matters:**
- If confirmed, non-penetrant hets are NOT equivalent to WT
- They represent an intermediate state between WT and penetrant phenotypes
- This has implications for using hets as controls in experiments

---

## What We're Committing To

This plan should answer the scientific questions (cryptic vs overt windows; WT vs non-pen hets vs phenotypes; consistent 3-panel summaries). The main risk is *over-consolidation* too early (wrappers on wrappers). The implementation choices below keep things simple now and make later migration into `src/` straightforward.

- **Reuse core inference**: do not reimplement AUROC/CV/permutation testing; call `src/analyze/difference_detection` (`compare_groups()` / `predictive_signal_test()`).
- **Thin helpers only**: keep local `utils/` functions minimal (data prep convenience, looping divergence, plotting, cryptic-window logic). If a wrapper becomes a one-liner, call the underlying API directly from the analysis scripts.
- **One clear data contract**: standardize the “prepared comparison dataframe” schema (`embryo_id`, time column, `group`, embedding features, metric columns) and keep everything else as plain DataFrame-in/DataFrame-out functions.
- **Explicit AUROC direction in every comparison**: **positive = phenotype**, **negative = reference** (WT or non-pen hets). Always pass `group1=phenotype`, `group2=reference`, and label plots accordingly (e.g. `AUROC (positive=CE, negative=WT)`).
- **Optional separability score**: if we want direction-free “detectability”, compute `AUROC_sep = max(AUROC, 1-AUROC)` for display/thresholding, but keep the sign convention explicit so this is never a silent assumption.

## Quick Reference: Existing APIs We Use

```python
# Core classification API (handles everything)
from src.analyze.difference_detection.comparison import compare_groups, add_group_column

# Public metric divergence API (loop this for multiple metrics)
from src.analyze.difference_detection.comparison import compute_metric_divergence

# Back-compat note: older code may import `_compute_divergence` (alias),
# but treat that name as internal/legacy.

# Optional: distribution shift tests
from src.analyze.difference_detection.distribution_test import permutation_test_energy
```

---


## Plot Inventory (18 total)

**Panel A Clarification:**
- **"Metrics only"**: Panel A shows metric-based AUROC (using curvature/length as features)
- **"With embeddings"**: Panel A shows metric AUROC + embedding AUROC overlaid

### CEP290 (2 plots)
| # | Name | Panel A | Panels B&C | Scientific Purpose |
|---|------|---------|------------|-------------------|
| 1 | `cep290_metric_auroc_only.png` | Curvature AUROC | Curvature divergence | Baseline: when do metrics detect? |
| 2 | `cep290_embedding_vs_metric.png` | Embedding + Curvature AUROC | Curvature divergence | Compare: embedding detects earlier? |

### B9D2 HTA (8 plots)
| # | Name | Panel A | Panels B&C | Scientific Purpose |
|---|------|---------|------------|-------------------|
| 3 | `hta_vs_wt_metrics_only.png` | Metric AUROC | Z-score curvature + length | HTA detection with metrics |
| 4 | `hta_vs_nonpen_hets_metrics_only.png` | Metric AUROC | Z-score curvature + length | HTA vs het control |
| 5 | `hta_overlay_metrics_only.png` | Both metric AUROCs | Both overlaid | Compare WT vs het as controls |
| 6 | `hta_vs_wt_with_embeddings.png` | Metric + Embedding AUROC | Z-score metrics | Does embedding detect HTA earlier? |
| 7 | `hta_vs_nonpen_hets_with_embeddings.png` | Metric + Embedding AUROC | Z-score metrics | Embedding vs metric timing |
| 8 | `hta_overlay_with_embeddings.png` | Both with embeddings | Both overlaid | Full comparison |
| 9 | `nonpen_hets_vs_wt_metrics_only.png` | Metric AUROC | Z-score metrics | Do hets differ from WT in metrics? |
| 10 | `nonpen_hets_vs_wt_with_embeddings.png` | Metric + Embedding AUROC | Z-score metrics | **Cryptic het phenotype?** |

### B9D2 CE (8 plots)
| # | Name | Panel A | Panels B&C | Scientific Purpose |
|---|------|---------|------------|-------------------|
| 11 | `ce_vs_wt_metrics_only.png` | Metric AUROC | Z-score length + curvature | CE detection with metrics |
| 12 | `ce_vs_nonpen_hets_metrics_only.png` | Metric AUROC | Z-score length + curvature | CE vs cryptic het baseline |
| 13 | `ce_overlay_metrics_only.png` | Both metric AUROCs | Both overlaid | Compare WT vs het as reference |
| 14 | `ce_vs_wt_with_embeddings.png` | Metric + Embedding AUROC | Z-score metrics | **Cryptic window: embedding earlier?** |
| 15 | `ce_vs_nonpen_hets_with_embeddings.png` | Metric + Embedding AUROC | Z-score metrics | **CE vs already-cryptic hets** |
| 16 | `ce_overlay_with_embeddings.png` | Both with embeddings | Both overlaid | Full comparison of both references |
| 17 | `ce_nonpen_hets_vs_wt_metrics_only.png` | Metric AUROC | Z-score metrics | Het vs WT metric difference |
| 18 | `ce_nonpen_hets_vs_wt_with_embeddings.png` | Metric + Embedding AUROC | Z-score metrics | **Confirm het cryptic phenotype** |


## Implementation Tasks

### Phase 1: Core Utils (Day 1)

#### Task 1.1: `utils/data_prep.py`

**Purpose:** Optional convenience wrapper for `add_group_column()` + phenotype file loading

> **CAUTION:** If this becomes a one-liner wrapper (add group column + trivial filter),
> prefer calling `add_group_column()` directly in the analysis scripts to avoid hiding
> assumptions (column names, subset logic).

```python
"""Data preparation utilities - optional convenience around comparison.add_group_column()."""
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.analyze.difference_detection.comparison import add_group_column


def load_ids_from_file(filepath: Path) -> List[str]:
    """Load embryo IDs from file (one per line)."""
    return [line.strip() for line in filepath.read_text().strip().split('\n') if line.strip()]


def prepare_comparison_data(
    df: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    group1_label: str,
    group2_label: str,
    subset_by: Optional[str] = None,
    subset_values: Optional[List] = None,
) -> pd.DataFrame:
    """
    Prepare data for group comparison.
    
    Wraps add_group_column() with optional filtering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw trajectory data with embryo_id column
    group1_ids, group2_ids : List[str]
        Embryo IDs for each group
    group1_label, group2_label : str
        Labels for the groups (e.g., 'CE', 'WT')
    subset_by : Optional[str]
        Column to filter by (e.g., 'pair')
    subset_values : Optional[List]
        Values to keep (e.g., ['b9d2_pair_7', 'b9d2_pair_8'])
    
    Returns
    -------
    pd.DataFrame
        Data with 'group' column added, filtered to specified embryos
    """
    # Optional filtering
    if subset_by and subset_values:
        df = df[df[subset_by].isin(subset_values)].copy()
    
    # Use existing API
    return add_group_column(
        df, 
        groups={group1_label: group1_ids, group2_label: group2_ids},
        column_name='group'
    )
```

**Test:** Load CEP290 data, add group column, verify shape.

---

#### Task 1.2: `utils/divergence.py`

**Purpose:** Multi-metric divergence by looping existing `compute_metric_divergence()`

> **CAUTION:** Avoid adding wrappers whose main purpose is renaming/reshaping.
> This should be a thin loop over the stable API (not private internals).

```python
"""Multi-metric divergence computation - wraps comparison.compute_metric_divergence()."""
from typing import List
import pandas as pd

from src.analyze.difference_detection.comparison import compute_metric_divergence


def compute_multi_metric_divergence(
    df: pd.DataFrame,
    group_col: str,
    group1_label: str,
    group2_label: str,
    metric_cols: List[str],
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
) -> pd.DataFrame:
    """
    Compute trajectory divergence for multiple metrics.
    
    Loops compute_metric_divergence() and concatenates results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with group column and metric columns
    group_col : str
        Column containing group labels
    group1_label, group2_label : str
        Labels for the two groups
    metric_cols : List[str]
        Metrics to compute divergence for (e.g., ['baseline_deviation_normalized', 'total_length_um'])
    
    Returns
    -------
    pd.DataFrame
        Long-format divergence with columns: time, group1_mean, group2_mean, abs_diff, metric
    """
    dfs = []
    for metric in metric_cols:
        div = compute_metric_divergence(
            df, group_col, group1_label, group2_label,
            metric, time_col, embryo_id_col
        )
        div['metric'] = metric
        dfs.append(div)
    
    return pd.concat(dfs, ignore_index=True)


def zscore_divergence(divergence_df: pd.DataFrame, value_col: str = 'abs_diff') -> pd.DataFrame:
    """
    Z-score normalize divergence within each metric for multi-metric comparison.
    
    Parameters
    ----------
    divergence_df : pd.DataFrame
        Output from compute_multi_metric_divergence()
    value_col : str
        Column to normalize (default: 'abs_diff')
    
    Returns
    -------
    pd.DataFrame
        Same DataFrame with '{value_col}_zscore' column added
    """
    def zscore(x):
        return (x - x.mean()) / x.std()
    
    divergence_df = divergence_df.copy()
    divergence_df[f'{value_col}_zscore'] = divergence_df.groupby('metric')[value_col].transform(zscore)
    return divergence_df
```

**Test:** Compute divergence for curvature + length, verify both metrics present.

---

#### Task 1.3: `utils/cryptic_window.py`

**Purpose:** Detect time window where embedding signal precedes metric divergence (GENUINELY NEW)

```python
"""Cryptic window detection - identifies embedding-before-metric signal gaps."""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def detect_cryptic_window(
    embedding_auroc: pd.DataFrame,
    metric_divergence: pd.DataFrame,
    auroc_threshold: float = 0.6,
    auroc_pval_threshold: float = 0.05,
    divergence_zscore_threshold: float = 1.0,
    time_col: str = 'time_bin',
) -> Dict[str, Any]:
    """
    Detect time window where embedding signal precedes metric divergence.
    
    A "cryptic window" exists when embeddings detect a phenotype significantly
    BEFORE standard metrics (curvature/length) show divergence.
    
    Parameters
    ----------
    embedding_auroc : pd.DataFrame
        Output from compare_groups()['classification'] using embedding features.
        Must have: time_col, 'auroc_observed', 'pval'
    metric_divergence : pd.DataFrame
        Output from compute_multi_metric_divergence() with zscore normalization.
        Must have: 'time', 'abs_diff_zscore', 'metric'
    auroc_threshold : float
        AUROC value to consider "significant signal" (default: 0.6)
    auroc_pval_threshold : float
        p-value threshold for AUROC significance (default: 0.05)
    divergence_zscore_threshold : float
        Z-score threshold for metric divergence (default: 1.0 = 1 SD above mean)
    
    Returns
    -------
    Dict with keys:
        - has_cryptic_window: bool
        - embedding_first_signal_hpf: float or None
        - metric_first_divergence_hpf: float or None  
        - cryptic_window_duration_hours: float or None
        - details: Dict with per-metric breakdown
    """
    # Find first significant embedding signal
    sig_embedding = embedding_auroc[
        (embedding_auroc['auroc_observed'] > auroc_threshold) &
        (embedding_auroc['pval'] < auroc_pval_threshold)
    ]
    emb_first = sig_embedding[time_col].min() if len(sig_embedding) > 0 else None
    
    # Find first significant metric divergence (any metric)
    sig_metric = metric_divergence[
        metric_divergence['abs_diff_zscore'] > divergence_zscore_threshold
    ]
    metric_first = sig_metric['time'].min() if len(sig_metric) > 0 else None
    
    # Per-metric breakdown
    metric_details = {}
    for metric in metric_divergence['metric'].unique():
        metric_data = metric_divergence[metric_divergence['metric'] == metric]
        sig = metric_data[metric_data['abs_diff_zscore'] > divergence_zscore_threshold]
        metric_details[metric] = {
            'first_signal_hpf': sig['time'].min() if len(sig) > 0 else None
        }
    
    # Determine cryptic window
    has_window = False
    duration = None
    if emb_first is not None and metric_first is not None:
        has_window = emb_first < metric_first
        duration = metric_first - emb_first if has_window else 0.0
    
    return {
        'has_cryptic_window': has_window,
        'embedding_first_signal_hpf': emb_first,
        'metric_first_divergence_hpf': metric_first,
        'cryptic_window_duration_hours': duration,
        'thresholds': {
            'auroc': auroc_threshold,
            'auroc_pval': auroc_pval_threshold,
            'divergence_zscore': divergence_zscore_threshold,
        },
        'per_metric_details': metric_details,
    }


def summarize_cryptic_windows(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Summarize cryptic window results across multiple comparisons.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Mapping of comparison name → detect_cryptic_window() output
    
    Returns
    -------
    pd.DataFrame
        Summary table with one row per comparison
    """
    rows = []
    for name, cw in results.items():
        rows.append({
            'comparison': name,
            'has_cryptic_window': cw['has_cryptic_window'],
            'embedding_first_hpf': cw['embedding_first_signal_hpf'],
            'metric_first_hpf': cw['metric_first_divergence_hpf'],
            'window_hours': cw['cryptic_window_duration_hours'],
        })
    return pd.DataFrame(rows)
```

**Test:** Create mock AUROC and divergence data, verify window detection logic.

---

### Phase 2: Plotting Module (Day 1-2)

#### Task 2.1: `utils/plotting.py`

**Purpose:** 3-panel comparison figures (GENUINELY NEW layout)

```python
"""Plotting utilities for cryptic phenotype comparison figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any


def create_comparison_figure(
    auroc_df: pd.DataFrame,
    divergence_df: pd.DataFrame,
    df_trajectories: pd.DataFrame,
    group1_label: str,  # POSITIVE class (phenotype)
    group2_label: str,  # NEGATIVE class (reference: WT or non-pen hets)
    metric_cols: List[str],
    embedding_auroc_df: Optional[pd.DataFrame] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    time_landmarks: Optional[Dict[float, str]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Create 3-panel comparison figure.

    Panel A: AUROC over time (metric-based, optionally with embedding overlay)
    Panel B: Metric divergence (Z-scored for multi-metric comparison)
    Panel C: Raw trajectories with group means

    IMPORTANT: group1_label is the POSITIVE class (phenotype), group2_label is NEGATIVE (reference).
    This matches compare_groups() convention where group1='phenotype', group2='reference'.

    Parameters
    ----------
    auroc_df : pd.DataFrame
        Metric-based AUROC from compare_groups()['classification']
        Must have: time_bin, auroc_observed, auroc_null_std, pval
    divergence_df : pd.DataFrame
        From compute_multi_metric_divergence() with zscore
        Must have: time, metric, abs_diff_zscore
    df_trajectories : pd.DataFrame
        Raw trajectory data for Panel C (with 'group' column added by add_group_column())
    group1_label, group2_label : str
        Labels for positive (phenotype) and negative (reference) groups
    embedding_auroc_df : Optional[pd.DataFrame]
        Embedding-based AUROC to overlay on Panel A
    metric_labels : Dict[str, str]
        Display labels for metrics (e.g., {'baseline_deviation_normalized': 'Curvature'})
    time_landmarks : Dict[float, str]
        Vertical lines to mark (e.g., {24.0: '24 hpf'})

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[1, 1, 1.2])

    metric_labels = metric_labels or {m: m for m in metric_cols}

    # Panel A: AUROC
    ax_auroc = axes[0]
    ax_auroc.plot(auroc_df['time_bin'], auroc_df['auroc_observed'],
                  'o-', label='Metric AUROC', color='#2ca02c', linewidth=2)
    ax_auroc.fill_between(auroc_df['time_bin'],
                          auroc_df['auroc_observed'] - auroc_df.get('auroc_null_std', 0),
                          auroc_df['auroc_observed'] + auroc_df.get('auroc_null_std', 0),
                          alpha=0.2, color='#2ca02c')

    if embedding_auroc_df is not None:
        ax_auroc.plot(embedding_auroc_df['time_bin'], embedding_auroc_df['auroc_observed'],
                      's--', label='Embedding AUROC', color='#1f77b4', linewidth=2)
        ax_auroc.fill_between(embedding_auroc_df['time_bin'],
                              embedding_auroc_df['auroc_observed'] - embedding_auroc_df.get('auroc_null_std', 0),
                              embedding_auroc_df['auroc_observed'] + embedding_auroc_df.get('auroc_null_std', 0),
                              alpha=0.2, color='#1f77b4')

    ax_auroc.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    ax_auroc.set_ylabel('AUROC')
    ax_auroc.set_title(f'A. Classification (positive={group1_label}, negative={group2_label})')
    ax_auroc.legend(loc='upper left')
    ax_auroc.set_ylim(0.3, 1.0)
    
    # Panel B: Divergence
    ax_div = axes[1]
    colors = plt.cm.Set1(np.linspace(0, 1, len(metric_cols)))
    for i, metric in enumerate(metric_cols):
        metric_data = divergence_df[divergence_df['metric'] == metric]
        label = metric_labels.get(metric, metric)
        ax_div.plot(metric_data['time'], metric_data['abs_diff_zscore'],
                    '-', label=label, color=colors[i], linewidth=2)
    
    ax_div.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax_div.set_ylabel('Divergence (Z-score)')
    ax_div.set_title('B. Metric Divergence Over Time')
    ax_div.legend(loc='upper left')
    
    # Panel C: Trajectories (use first metric)
    ax_traj = axes[2]
    primary_metric = metric_cols[0]
    
    for group_label, color in [(group1_label, '#d62728'), (group2_label, '#1f77b4')]:
        group_data = df_trajectories[df_trajectories['group'] == group_label]
        
        # Individual trajectories (faint)
        for embryo_id in group_data['embryo_id'].unique()[:20]:  # Limit for clarity
            emb_data = group_data[group_data['embryo_id'] == embryo_id]
            ax_traj.plot(emb_data['predicted_stage_hpf'], emb_data[primary_metric],
                        alpha=0.1, color=color, linewidth=0.5)
        
        # Group mean
        mean_traj = group_data.groupby('predicted_stage_hpf')[primary_metric].mean()
        ax_traj.plot(mean_traj.index, mean_traj.values, 
                    linewidth=3, color=color, label=group_label)
    
    ax_traj.set_xlabel('Time (hpf)')
    ax_traj.set_ylabel(metric_labels.get(primary_metric, primary_metric))
    ax_traj.set_title(f'C. Individual Trajectories: {metric_labels.get(primary_metric, primary_metric)}')
    ax_traj.legend(loc='upper left')
    
    # Add time landmarks
    if time_landmarks:
        for t, label in time_landmarks.items():
            for ax in axes:
                ax.axvline(t, color='red', linestyle='--', alpha=0.5)
            axes[0].text(t, axes[0].get_ylim()[1], label, ha='center', va='bottom', fontsize=9)
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
```

---

### Phase 3: Analysis Scripts (Day 2-3)

#### Task 3.1: `cep290_analysis.py`

**Purpose:** CEP290 cryptic phenotype analysis using existing APIs

```python
"""CEP290 cryptic phenotype analysis.

Tests hypothesis: Embedding detects phenotype at ~18 hpf, before curvature at ~24 hpf.
"""
from pathlib import Path
import pandas as pd

# Existing APIs (NO reimplementation)
from src.analyze.difference_detection.comparison import compare_groups

# Local thin wrappers
from utils.data_prep import prepare_comparison_data, load_ids_from_file
from utils.divergence import compute_multi_metric_divergence, zscore_divergence
from utils.cryptic_window import detect_cryptic_window
from utils.plotting import create_comparison_figure


def main():
    # =========================================================================
    # Data Loading
    # =========================================================================
    DATA_PATH = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv")
    OUTPUT_DIR = Path("results/mcolon/20260105_refined_embedding_and_metric_classification/output/cep290")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Define groups
    penetrant_categories = ['Low_to_High', 'High_to_Low', 'Intermediate']
    penetrant_ids = df[df['cluster_categories'].isin(penetrant_categories)]['embryo_id'].unique().tolist()
    control_ids = df[df['cluster_categories'] == 'Not Penetrant']['embryo_id'].unique().tolist()
    
    print(f"Penetrant embryos: {len(penetrant_ids)}")
    print(f"Control (Not Penetrant) embryos: {len(control_ids)}")
    
    # Prepare data with group column
    df_prep = prepare_comparison_data(
        df, 
        group1_ids=penetrant_ids, 
        group2_ids=control_ids,
        group1_label='Penetrant',
        group2_label='Control'
    )
    
    # =========================================================================
    # Classification: Use existing API directly
    # =========================================================================
    
    # Metric-based classification (curvature)
    metric_results = compare_groups(
        df_prep,
        group_col='group',
        group1='Penetrant',
        group2='Control',
        features=['baseline_deviation_normalized'],  # Curvature as features
        morphology_metric=None,  # Don't compute internal divergence
        bin_width=4.0,
        n_permutations=100,
    )
    metric_auroc = metric_results['classification']
    
    # Embedding-based classification
    embedding_results = compare_groups(
        df_prep,
        group_col='group', 
        group1='Penetrant',
        group2='Control',
        features='z_mu_b',  # Auto-detects embedding columns
        morphology_metric=None,
        bin_width=4.0,
        n_permutations=100,
    )
    embedding_auroc = embedding_results['classification']
    
    # =========================================================================
    # Divergence: Use multi-metric wrapper
    # =========================================================================
    divergence = compute_multi_metric_divergence(
        df_prep,
        group_col='group',
        group1_label='Penetrant',
        group2_label='Control',
        metric_cols=['baseline_deviation_normalized', 'total_length_um'],
    )
    divergence = zscore_divergence(divergence)
    
    # =========================================================================
    # Cryptic Window Detection
    # =========================================================================
    cryptic = detect_cryptic_window(
        embedding_auroc,
        divergence,
        auroc_threshold=0.6,
        divergence_zscore_threshold=1.0,
    )
    
    print(f"\nCryptic Window Analysis:")
    print(f"  Has cryptic window: {cryptic['has_cryptic_window']}")
    print(f"  Embedding first signal: {cryptic['embedding_first_signal_hpf']} hpf")
    print(f"  Metric first divergence: {cryptic['metric_first_divergence_hpf']} hpf")
    print(f"  Window duration: {cryptic['cryptic_window_duration_hours']} hours")
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    # Plot 1: Metric AUROC only
    fig1 = create_comparison_figure(
        auroc_df=metric_auroc,
        divergence_df=divergence,
        df_trajectories=df_prep,
        group1_label='Penetrant',
        group2_label='Control',
        metric_cols=['baseline_deviation_normalized', 'total_length_um'],
        metric_labels={
            'baseline_deviation_normalized': 'Curvature',
            'total_length_um': 'Body Length',
        },
        time_landmarks={24.0: '24 hpf'},
        title='CEP290: Metric-Based Classification',
        save_path=OUTPUT_DIR / 'cep290_metric_auroc_only.png',
    )
    
    # Plot 2: Embedding vs Metric overlay
    fig2 = create_comparison_figure(
        auroc_df=metric_auroc,
        divergence_df=divergence,
        df_trajectories=df_prep,
        group1_label='Penetrant',
        group2_label='Control',
        metric_cols=['baseline_deviation_normalized', 'total_length_um'],
        embedding_auroc_df=embedding_auroc,  # Add overlay
        metric_labels={
            'baseline_deviation_normalized': 'Curvature',
            'total_length_um': 'Body Length',
        },
        time_landmarks={24.0: '24 hpf'},
        title='CEP290: Embedding vs Metric Classification',
        save_path=OUTPUT_DIR / 'cep290_embedding_vs_metric.png',
    )
    
    # Save results
    metric_auroc.to_csv(OUTPUT_DIR / 'metric_auroc.csv', index=False)
    embedding_auroc.to_csv(OUTPUT_DIR / 'embedding_auroc.csv', index=False)
    divergence.to_csv(OUTPUT_DIR / 'divergence.csv', index=False)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
```

---

### Phase 4: B9D2 Analysis Scripts (Day 3-4)

Similar pattern to CEP290, with:
- `b9d2_hta_analysis.py` — HTA phenotype comparisons
- `b9d2_ce_analysis.py` — CE phenotype comparisons + non-penetrant het validation

---

## Directory Structure

```
results/mcolon/20260105_refined_embedding_and_metric_classification/
├── PLAN.md                       # This file
├── _Archive/
│   ├── PLAN_v0.md               # Original plan
│   └── PLAN_v1.md               # Plan with audit concerns
├── utils/
│   ├── __init__.py
│   ├── data_prep.py              # ⚠️ Wraps comparison.add_group_column()
│   ├── divergence.py             # ⚠️ Loops comparison.compute_metric_divergence()
│   ├── cryptic_window.py         # NEW: Detect embedding-before-metric signal
│   └── plotting.py               # NEW: 3-panel figure creation
├── cep290_analysis.py            # CEP290 analysis
├── b9d2_hta_analysis.py          # B9D2 HTA analysis
├── b9d2_ce_analysis.py           # B9D2 CE analysis
├── run_all.py                    # Master runner
└── output/
    ├── cep290/
    ├── b9d2_hta/
    └── b9d2_ce/
```

---


## Data Sources

### CEP290
- **Data:** `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
- **Penetrant:** `cluster_categories` in ['Low_to_High', 'High_to_Low', 'Intermediate']
- **Control:** `cluster_categories` == 'Not Penetrant'

### B9D2
- **Data:** `results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv` (combined 20251121 + 20251125 experiments)
- **HTA IDs:** `results/mcolon/20251219_b9d2_phenotype_extraction/phenotype_lists/b9d2-HTA-embryos.txt`
- **CE IDs:** `results/mcolon/20251219_b9d2_phenotype_extraction/phenotype_lists/b9d2-CE-phenotype.txt`
- **WT:** genotype == 'b9d2_wildtype'
- **Non-penetrant hets:** genotype contains 'b9d2_pair' AND NOT in any phenotype list

---

## Validation Checklist

### CEP290
- [ ] Embedding AUROC significant at ~18 hpf
- [ ] Curvature AUROC significant later (>24 hpf)
- [ ] 24h landmark clearly marked
- [ ] Gap between embedding and metric detection demonstrates cryptic phenotype

### B9D2 HTA
- [ ] HTA vs WT: Metric difference after 60 hpf
- [ ] HTA vs non-pen hets: Similar pattern (validates het as control)
- [ ] Non-pen hets vs WT: Minimal difference (no cryptic phenotype for HTA pathway)

### B9D2 CE
- [ ] CE vs WT: Embedding detects at ~10 hpf, metrics at ~20 hpf
- [ ] CE vs non-pen hets: Embedding detects at ~15 hpf (later due to het cryptic phenotype)
- [ ] Non-pen hets vs WT: Embedding shows cryptic difference, metrics show minimal

### Framework Quality
- [ ] All 18 plots generated
- [ ] Z-score normalization allows multi-metric comparison
- [ ] AUROC overlay clearly shows embedding vs metric timing
- [ ] utils/ is modular and ready for migration to src/

### Audit Compliance (NEW)
- [ ] No reimplementation of `predictive_signal_test()` or `compare_groups()` — use existing APIs
- [ ] `data_prep.py` uses `add_group_column()` from `comparison.py`
- [ ] `divergence.py` loops `_compute_divergence()` instead of reimplementing
- [ ] AUROC direction is explicit (document which group is positive class)
- [ ] `cryptic_window.py` contains genuinely new logic (not in src/)
- [ ] Migration path to src/ is documented

## Key Implementation Principles

### 1. NO Reimplementation of Classification

```python
# ❌ DON'T DO THIS (reimplementing)
def compute_auroc_over_time(df, ...):
    # 200 lines of logistic regression, CV, permutation tests...
    
# ✅ DO THIS (compose existing)
from src.analyze.difference_detection.comparison import compare_groups
results = compare_groups(df, group_col='group', group1='CE', group2='WT', ...)
auroc_df = results['classification']
```

### 2. AUROC Direction Handling

```python
# `compare_groups()` enforces AUROC direction explicitly:
# group1 is the positive/"phenotype" class and group2 is the negative/"reference" class.

# group1='Penetrant' means AUROC > 0.5 = classifier predicts Penetrant
results = compare_groups(df, group_col='group', group1='Penetrant', group2='Control', ...)

# Optional: direction-free separability (useful for detection timing)
auroc_df['auroc_sep'] = auroc_df['auroc_observed'].apply(lambda a: max(a, 1 - a))
```

### 3. Data Format Contract

**Input to `compare_groups()`:**
```python
df_prep = pd.DataFrame({
    'embryo_id': ['e1', 'e1', 'e2', ...],
    'predicted_stage_hpf': [8.2, 8.5, 12.1, ...],
    'group': ['CE', 'CE', 'WT', ...],           # Added by add_group_column()
    'z_mu_b_0': [0.1, 0.2, -0.1, ...],          # Embeddings
    'z_mu_b_1': [...],
    'baseline_deviation_normalized': [...],     # Metrics
    'total_length_um': [...],
})
```

**Output from `compare_groups()`:**
```python
    results = {
        'classification': pd.DataFrame({  # Per time-bin AUROC
            'time_bin': [8, 12, 16, ...],
            'auroc_observed': [0.52, 0.61, 0.73, ...],
        'auroc_null_mean': [0.50, 0.50, 0.50, ...],
        'auroc_null_std': [0.05, 0.04, 0.06, ...],
        'pval': [0.4, 0.02, 0.001, ...],
        'n_positive': [15, 18, 20, ...],
        'n_negative': [12, 14, 16, ...],
        'positive_class': ['CE', 'CE', 'CE', ...],
        'negative_class': ['WT', 'WT', 'WT', ...],
        }),
    'divergence': pd.DataFrame({  # Morphology divergence (if morphology_metric set)
        'time': [8.0, 8.5, 9.0, ...],
        'group1_mean': [...],
        'group2_mean': [...],
        'abs_diff': [...],
    }),
}
```

---

## Validation Checklist

### Implementation Quality
- [ ] `utils/data_prep.py` uses `add_group_column()` — no reimplementation
- [ ] `utils/divergence.py` loops `compute_metric_divergence()` — no reimplementation
- [ ] `utils/cryptic_window.py` is genuinely new logic
- [ ] `utils/plotting.py` is genuinely new layout
- [ ] Analysis scripts call `compare_groups()` directly

### Scientific Validation
- [ ] CEP290: Embedding AUROC significant before metric divergence
- [ ] B9D2 CE: ~5-10 hour cryptic window detected
- [ ] Non-penetrant hets vs WT: Embedding signal present, metric signal minimal

### Output Quality
- [ ] All 18 plots generated
- [ ] Cryptic window summary table created
- [ ] CSVs saved for each comparison

## Expected Outcomes

### If Hypotheses Are Confirmed:

1. **CEP290:** ~6 hour cryptic window (18-24 hpf) where phenotype exists but isn't curvature
2. **CE:** ~10 hour cryptic window (10-20 hpf) where phenotype exists but isn't length/curvature
3. **Non-pen hets:** Carry a cryptic phenotype, explaining why CE detection is later vs hets than vs WT

### Scientific Implications:

- Cryptic phenotypes represent early developmental perturbations before gross morphology changes
- VAE embeddings capture shape information beyond simple metrics
- Non-penetrant hets may be an intermediate state, not true controls
- Earliest detection time helps understand mechanism (what's changing first?)


---

## Migration Path to `src/`

When ready to productionize:

| Local Module | Migration Target | Notes |
|--------------|------------------|-------|
| `utils/divergence.py` | Extend `comparison.compute_metric_divergence()` | Add `metric_cols: List[str]` parameter |
| `utils/cryptic_window.py` | New function in `comparison.py` | `detect_cryptic_window()` |
| `utils/plotting.py` | New module `difference_detection/plotting.py` | 3-panel layout |
| `utils/data_prep.py` | Keep as-is (trivial wrapper) | Or inline in scripts |
| Analysis scripts | Stay in `results/` | Phenotype-specific |
