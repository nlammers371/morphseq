import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251013"
data_dir = os.path.join(results_dir, "data")
plot_dir = os.path.join(results_dir, "plots")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

morphseq_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"
build06_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output"

print("="*80)
print("NaN INVESTIGATION SCRIPT")
print("="*80)

# Experiments
WT_experiments = ["20230615","20230531", "20230525", "20250912"]
b9d2_experiments = ["20250519","20250520"]
cep290_experiments = ["20250305", "20250416", "20250512", "20250515_part2", "20250519"]
tmem67_experiments = ["20250711"]
experiments = WT_experiments + b9d2_experiments + cep290_experiments + tmem67_experiments

# Load all experiments and track NaN statistics
dfs = []
nan_stats = []

print("\n1. LOADING DATA AND CHECKING FOR NaNs PER EXPERIMENT")
print("-"*80)

for exp in experiments:
    try:
        file_path = f"{build06_dir}/df03_final_output_with_latents_{exp}.csv"
        df = pd.read_csv(file_path)
        df['source_experiment'] = exp

        # Get latent columns
        z_cols = [c for c in df.columns if "z_mu_b" in c]

        # Count NaNs
        total_rows = len(df)
        total_nans = df[z_cols].isna().sum().sum()
        rows_with_nans = df[z_cols].isna().any(axis=1).sum()
        pct_rows_with_nans = (rows_with_nans / total_rows) * 100

        nan_stats.append({
            'experiment': exp,
            'total_rows': total_rows,
            'total_nans': total_nans,
            'rows_with_nans': rows_with_nans,
            'pct_rows_with_nans': pct_rows_with_nans,
            'num_latent_cols': len(z_cols),
            'genotypes': df['genotype'].unique().tolist() if 'genotype' in df.columns else []
        })

        print(f"{exp}: {total_rows:,} rows, {rows_with_nans:,} ({pct_rows_with_nans:.1f}%) with NaNs")

        dfs.append(df)

    except Exception as e:
        print(f"Missing/Error {exp}: {e}")

# Create NaN stats dataframe
nan_stats_df = pd.DataFrame(nan_stats)
print(f"\nTotal experiments loaded: {len(dfs)}")

# Combine all data
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined data: {len(combined_df):,} rows")

# Filter to CEP290 genotypes
print("\n2. FILTERING TO CEP290 GENOTYPES")
print("-"*80)
cep290_genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
df_cep290 = combined_df[combined_df['genotype'].isin(cep290_genotypes)].copy()
print(f"CEP290 filtered data: {len(df_cep290):,} rows")
print(f"Genotype distribution:\n{df_cep290['genotype'].value_counts()}")

# Deep dive into NaN patterns
print("\n3. ANALYZING NaN PATTERNS IN CEP290 DATA")
print("-"*80)

z_cols = [c for c in df_cep290.columns if "z_mu_b" in c]
print(f"Number of latent columns: {len(z_cols)}")

# Check if NaNs are consistent across all latent columns
nan_mask = df_cep290[z_cols].isna()
rows_with_any_nan = nan_mask.any(axis=1)
rows_with_all_nan = nan_mask.all(axis=1)
rows_with_partial_nan = rows_with_any_nan & ~rows_with_all_nan

print(f"\nRows with ANY NaN in latents: {rows_with_any_nan.sum():,} ({rows_with_any_nan.sum()/len(df_cep290)*100:.1f}%)")
print(f"Rows with ALL NaNs in latents: {rows_with_all_nan.sum():,} ({rows_with_all_nan.sum()/len(df_cep290)*100:.1f}%)")
print(f"Rows with PARTIAL NaNs: {rows_with_partial_nan.sum():,} ({rows_with_partial_nan.sum()/len(df_cep290)*100:.1f}%)")

# Check NaN patterns by experiment
print("\n4. NaN PATTERNS BY EXPERIMENT (CEP290 only)")
print("-"*80)
for exp in df_cep290['source_experiment'].unique():
    exp_df = df_cep290[df_cep290['source_experiment'] == exp]
    nan_rows = exp_df[z_cols].isna().any(axis=1).sum()
    pct = (nan_rows / len(exp_df)) * 100
    print(f"{exp}: {nan_rows:,}/{len(exp_df):,} rows with NaNs ({pct:.1f}%)")

# Check NaN patterns by genotype
print("\n5. NaN PATTERNS BY GENOTYPE")
print("-"*80)
for genotype in cep290_genotypes:
    geno_df = df_cep290[df_cep290['genotype'] == genotype]
    nan_rows = geno_df[z_cols].isna().any(axis=1).sum()
    pct = (nan_rows / len(geno_df)) * 100
    print(f"{genotype}: {nan_rows:,}/{len(geno_df):,} rows with NaNs ({pct:.1f}%)")

# Check NaN patterns by embryo
print("\n6. NaN PATTERNS BY EMBRYO (Top 20 embryos with most NaNs)")
print("-"*80)
embryo_nan_counts = []
for embryo_id in df_cep290['embryo_id'].unique():
    emb_df = df_cep290[df_cep290['embryo_id'] == embryo_id]
    nan_rows = emb_df[z_cols].isna().any(axis=1).sum()
    total_rows = len(emb_df)
    pct = (nan_rows / total_rows) * 100 if total_rows > 0 else 0

    embryo_nan_counts.append({
        'embryo_id': embryo_id,
        'experiment': emb_df['source_experiment'].iloc[0],
        'genotype': emb_df['genotype'].iloc[0],
        'total_rows': total_rows,
        'nan_rows': nan_rows,
        'pct_nan': pct
    })

embryo_nan_df = pd.DataFrame(embryo_nan_counts).sort_values('nan_rows', ascending=False)
print(embryo_nan_df.head(20).to_string(index=False))

# Check NaN patterns by time
print("\n7. NaN PATTERNS BY DEVELOPMENTAL TIME")
print("-"*80)
if 'predicted_stage_hpf' in df_cep290.columns:
    df_cep290_copy = df_cep290.copy()
    df_cep290_copy['time_bin'] = (np.floor(df_cep290_copy['predicted_stage_hpf'] / 2) * 2).astype(int)

    time_nan_stats = []
    for time_bin in sorted(df_cep290_copy['time_bin'].unique()):
        time_df = df_cep290_copy[df_cep290_copy['time_bin'] == time_bin]
        nan_rows = time_df[z_cols].isna().any(axis=1).sum()
        pct = (nan_rows / len(time_df)) * 100
        time_nan_stats.append({
            'time_bin': time_bin,
            'total_rows': len(time_df),
            'nan_rows': nan_rows,
            'pct_nan': pct
        })

    time_nan_df = pd.DataFrame(time_nan_stats)
    print(time_nan_df.to_string(index=False))

# Check which specific latent dimensions have the most NaNs
print("\n8. NaN COUNTS BY LATENT DIMENSION (Top 10)")
print("-"*80)
latent_nan_counts = df_cep290[z_cols].isna().sum().sort_values(ascending=False)
print(latent_nan_counts.head(10))

# Check if there are any other columns with NaNs that might be relevant
print("\n9. OTHER COLUMNS WITH NaNs")
print("-"*80)
other_cols = [c for c in df_cep290.columns if c not in z_cols]
other_nan_counts = df_cep290[other_cols].isna().sum()
other_nan_counts = other_nan_counts[other_nan_counts > 0].sort_values(ascending=False)
if len(other_nan_counts) > 0:
    print(other_nan_counts.head(20))
else:
    print("No NaNs in non-latent columns")

# Create visualizations
print("\n10. GENERATING VISUALIZATIONS")
print("-"*80)

# Plot 1: NaN percentage by experiment
fig, ax = plt.subplots(figsize=(12, 6))
nan_stats_df_sorted = nan_stats_df.sort_values('pct_rows_with_nans', ascending=False)
ax.bar(range(len(nan_stats_df_sorted)), nan_stats_df_sorted['pct_rows_with_nans'])
ax.set_xticks(range(len(nan_stats_df_sorted)))
ax.set_xticklabels(nan_stats_df_sorted['experiment'], rotation=45, ha='right')
ax.set_ylabel('% Rows with NaNs')
ax.set_title('Percentage of Rows with NaN Values by Experiment')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'nan_by_experiment.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(plot_dir, 'nan_by_experiment.png')}")
plt.close()

# Plot 2: NaN percentage by genotype (CEP290 only)
fig, ax = plt.subplots(figsize=(10, 6))
genotype_nan_stats = []
for genotype in cep290_genotypes:
    geno_df = df_cep290[df_cep290['genotype'] == genotype]
    nan_rows = geno_df[z_cols].isna().any(axis=1).sum()
    pct = (nan_rows / len(geno_df)) * 100
    genotype_nan_stats.append({'genotype': genotype, 'pct_nan': pct})

geno_nan_df = pd.DataFrame(genotype_nan_stats)
ax.bar(geno_nan_df['genotype'], geno_nan_df['pct_nan'])
ax.set_ylabel('% Rows with NaNs')
ax.set_title('Percentage of Rows with NaN Values by CEP290 Genotype')
ax.set_xticklabels(geno_nan_df['genotype'], rotation=45, ha='right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'nan_by_genotype_cep290.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(plot_dir, 'nan_by_genotype_cep290.png')}")
plt.close()

# Plot 3: NaN percentage by time bin
if 'time_bin' in df_cep290_copy.columns:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_nan_df['time_bin'], time_nan_df['pct_nan'], marker='o', linewidth=2)
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('% Rows with NaNs')
    ax.set_title('Percentage of Rows with NaN Values Over Developmental Time')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'nan_by_time.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(plot_dir, 'nan_by_time.png')}")
    plt.close()

# Plot 4: Heatmap of NaN patterns across embryos
fig, ax = plt.subplots(figsize=(10, 12))
top_embryos = embryo_nan_df.head(30)['embryo_id'].values
embryo_subset = df_cep290[df_cep290['embryo_id'].isin(top_embryos)]

# Create a matrix showing which embryos have NaNs in which latent dimensions
nan_matrix = []
embryo_labels = []
for emb_id in top_embryos:
    emb_data = embryo_subset[embryo_subset['embryo_id'] == emb_id]
    nan_pcts = emb_data[z_cols].isna().mean().values
    nan_matrix.append(nan_pcts)
    embryo_labels.append(f"{emb_id[:15]}...")  # Truncate long IDs

nan_matrix = np.array(nan_matrix)

im = ax.imshow(nan_matrix, aspect='auto', cmap='YlOrRd')
ax.set_yticks(range(len(embryo_labels)))
ax.set_yticklabels(embryo_labels, fontsize=8)
ax.set_xlabel('Latent Dimension Index')
ax.set_ylabel('Embryo ID')
ax.set_title('NaN Pattern Heatmap: Top 30 Embryos with Most NaNs')
plt.colorbar(im, ax=ax, label='Fraction of Rows with NaN')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'nan_heatmap_embryos.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(plot_dir, 'nan_heatmap_embryos.png')}")
plt.close()

# Save detailed reports
print("\n11. SAVING DETAILED REPORTS")
print("-"*80)
nan_stats_df.to_csv(os.path.join(data_dir, 'nan_stats_by_experiment.csv'), index=False)
print(f"Saved: {os.path.join(data_dir, 'nan_stats_by_experiment.csv')}")

embryo_nan_df.to_csv(os.path.join(data_dir, 'nan_stats_by_embryo.csv'), index=False)
print(f"Saved: {os.path.join(data_dir, 'nan_stats_by_embryo.csv')}")

if 'time_bin' in df_cep290_copy.columns:
    time_nan_df.to_csv(os.path.join(data_dir, 'nan_stats_by_time.csv'), index=False)
    print(f"Saved: {os.path.join(data_dir, 'nan_stats_by_time.csv')}")

# Sample rows with NaNs for manual inspection
print("\n12. SAMPLING ROWS WITH NaNs FOR INSPECTION")
print("-"*80)
rows_with_nans = df_cep290[df_cep290[z_cols].isna().any(axis=1)]
sample_size = min(100, len(rows_with_nans))
sample_nans = rows_with_nans.sample(n=sample_size, random_state=42)

# Save relevant columns
inspection_cols = ['embryo_id', 'source_experiment', 'genotype', 'predicted_stage_hpf'] + z_cols[:10]
sample_nans[inspection_cols].to_csv(os.path.join(data_dir, 'sample_rows_with_nans.csv'), index=False)
print(f"Saved {sample_size} sample rows: {os.path.join(data_dir, 'sample_rows_with_nans.csv')}")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print(f"\nResults saved to:")
print(f"  Data: {data_dir}")
print(f"  Plots: {plot_dir}")
