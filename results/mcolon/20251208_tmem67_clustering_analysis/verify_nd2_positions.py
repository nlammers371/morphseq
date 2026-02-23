"""
Verify ND2 Well Position Ordering for 20250711

This script checks if the ND2 file's well positions follow the expected
plate layout by plotting XY coordinates with position labels and verifying
grid consistency.

Author: Generated via Claude Code
Date: 2025-12-08
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nd2


# ============================================================================
# Helper Functions
# ============================================================================

def p_to_well_name(P: int, n_cols: int = 12) -> str:
    """
    Convert P index to well name (e.g., 0 → 'A01', 12 → 'B01').

    Note: P indices in ND2 are 0-based, but appear to be 1-indexed in practice,
    so we add 1 before converting to row/col.

    Parameters
    ----------
    P : int
        Position index (0-based from ND2 file)
    n_cols : int
        Number of columns per row (default 12)

    Returns
    -------
    well_name : str
        Well name like 'A01', 'B02', etc.
    """
    # Add 1 to convert from 0-based to 1-based indexing
    P_1based = P + 1
    row = (P_1based - 1) // n_cols
    col = (P_1based - 1) % n_cols
    row_letter = chr(ord('A') + row)
    return f"{row_letter}{col+1:02d}"


def well_name_to_p(well_name: str, n_cols: int = 12) -> int:
    """
    Convert well name to P index (e.g., 'A01' → 0, 'B02' → 13).

    Parameters
    ----------
    well_name : str
        Well name like 'A01'
    n_cols : int
        Number of columns per row (default 12)

    Returns
    -------
    P : int
        Position index (0-based)
    """
    row_letter = well_name[0]
    col_num = int(well_name[1:])
    row = ord(row_letter) - ord('A')
    col = col_num - 1
    return row * n_cols + col


# ============================================================================
# ND2 Position Extraction
# ============================================================================

def find_nd2_file(exp_id: str, base_path: Path) -> Path:
    """Locate the ND2 file for a given experiment."""
    nd2_dir = base_path / "raw_image_data" / "YX1" / exp_id
    nd2_files = list(nd2_dir.glob("*.nd2"))

    if len(nd2_files) == 0:
        raise FileNotFoundError(f"No ND2 file found in {nd2_dir}")
    elif len(nd2_files) > 1:
        print(f"⚠️ Multiple ND2 files found in {nd2_dir}, using first one")

    nd2_path = nd2_files[0]
    print(f"Found ND2 file: {nd2_path}")
    return nd2_path


def get_stage_positions_fast(f):
    """
    Return one stage position per P (first T,Z,C) — fast version.

    The P dimension corresponds to different positions in the experiment.
    We extract stage position for each P index.

    Parameters
    ----------
    f : nd2.ND2File
        Opened ND2 file

    Returns
    -------
    df : DataFrame
        DataFrame with columns [P, x_um, y_um, z_um]
    """
    sizes = f.sizes
    P = sizes.get("P", sizes.get("W", 1))
    T = sizes.get("T", 1)
    Z = sizes.get("Z", 1)
    C = sizes.get("C", 1)
    records = []

    for w in range(P):
        idx = ((0 * P) + w) * (Z * C)  # T=0, Z=0, C=0
        try:
            md = f.frame_metadata(idx)
        except Exception as e:
            print(f"⚠️ Skipping P={w}: {e}")
            continue

        ch = getattr(md, "channels", [None])[0]
        if ch is None:
            continue

        pos = getattr(ch, "position", None)
        if not pos or not hasattr(pos, "stagePositionUm"):
            continue

        stage = pos.stagePositionUm
        records.append({
            "P": w,  # Position index in file (1-based will be assigned later via Excel mapping)
            "x_um": getattr(stage, "x", np.nan),
            "y_um": getattr(stage, "y", np.nan),
            "z_um": getattr(stage, "z", np.nan)
        })

    df = pd.DataFrame(records)
    df = df.sort_values("P").reset_index(drop=True)
    print(f"  Extracted {len(df)} positions from ND2 file")

    return df


def load_series_number_map(exp_id: str) -> dict:
    """
    Load explicit series-to-well mapping from Excel metadata.

    Mimics build01B_compile_yx1_images_torch.py lines 490-530 exactly.
    Iterates column-major: all rows in column 1, then all rows in column 2, etc.

    Parameters
    ----------
    exp_id : str
        Experiment ID

    Returns
    -------
    mapping : dict
        Dictionary mapping series number (1-based) to well name (e.g., "A01")
    """
    # Check both possible metadata locations
    candidate_paths = [
        Path(f"/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/plate_metadata/{exp_id}_well_metadata.xlsx"),
        Path(f"/net/trapnell/vol1/home/mdcolon/proj/morphseq/metadata/plate_metadata/{exp_id}_well_metadata.xlsx"),
    ]
    
    metadata_path = None
    for p in candidate_paths:
        if p.exists():
            metadata_path = p
            break

    if metadata_path is None:
        print(f"  ⚠️ No plate metadata found for {exp_id}")
        return {}

    try:
        # Read series_number_map sheet (follows build01B exactly)
        sm_raw = pd.read_excel(metadata_path, sheet_name='series_number_map', header=None)

        # Detect and handle header row (lines 493-500 of build01B)
        data_rows = sm_raw
        try:
            header_like = list(sm_raw.iloc[0, 1:13].astype(object))
            if header_like == list(range(1, 13)):
                data_rows = sm_raw.iloc[1:9, :]  # rows A..H in 1..8
            else:
                data_rows = sm_raw.iloc[:8, :]
        except Exception:
            data_rows = sm_raw.iloc[:8, :]

        series_map = data_rows.iloc[:, 1:13]  # 8x12 numeric grid

        # Build mapping (lines 504-530 of build01B)
        mapping = {}
        col_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        row_letter_list = ["A", "B", "C", "D", "E", "F", "G", "H"]

        for c in range(len(col_id_list)):  # Columns 1-12 (column-major order)
            for r in range(len(row_letter_list)):  # Rows A-H
                val = series_map.iloc[r, c]
                if pd.notna(val):
                    try:
                        series_idx_1b = int(val)
                        well_name = row_letter_list[r] + f"{col_id_list[c]:02d}"
                        mapping[series_idx_1b] = well_name
                    except Exception:
                        continue

        print(f"  Loaded explicit series mapping: {len(mapping)} wells")
        return mapping

    except Exception as e:
        print(f"  ⚠️ Could not load series mapping: {e}")
        return {}


def load_nd2_positions(exp_id: str, base_path: Path) -> pd.DataFrame:
    """
    Load stage positions from ND2 file and map to well names using Excel metadata.

    Parameters
    ----------
    exp_id : str
        Experiment ID (e.g., '20250711')
    base_path : Path
        Base directory for morphseq_playground

    Returns
    -------
    df : DataFrame
        DataFrame with columns [P, series_num, x_um, y_um, z_um, well_name]
    """
    nd2_path = find_nd2_file(exp_id, base_path)

    with nd2.ND2File(str(nd2_path)) as f:
        df = get_stage_positions_fast(f)

    # Load explicit series mapping from Excel
    series_map = load_series_number_map(exp_id)

    # Map P indices to series numbers
    # The P index (0-93) maps directly to series numbers (1-94) in the acquisition order
    # We need the inverse mapping: series_num -> well_name
    if series_map:
        # Create inverse mapping for faster lookup
        # series_map: series_num -> well_name
        # We want: P -> well_name via P -> series_num -> well_name

        # First, we need to figure out which series numbers are actually in the file
        # The acquired series numbers are 1, 2, 3, ..., up to max acquired series
        # If some wells are deselected, the mapping will have gaps

        # For now, assume P index maps to series numbers 1, 2, 3, ... in order
        df['series_num'] = df['P'] + 1  # Convert 0-based P to 1-based series number

        # Map series number to well name
        df['well_name'] = df['series_num'].apply(
            lambda s: series_map.get(int(s), f"S{int(s)}") if pd.notna(s) else "Unknown"
        )
    else:
        # Fall back to default row-major mapping
        df['series_num'] = df['P'] + 1
        df['well_name'] = df['P'].apply(p_to_well_name)

    print(f"\nLoaded {len(df)} stage positions from ND2 file")
    print(f"  P range: {df['P'].min()} - {df['P'].max()}")
    if 'series_num' in df.columns:
        print(f"  Series number range: {df['series_num'].min():.0f} - {df['series_num'].max():.0f}")
    print(f"  X range: {df['x_um'].min():.1f} - {df['x_um'].max():.1f} µm")
    print(f"  Y range: {df['y_um'].min():.1f} - {df['y_um'].max():.1f} µm")

    return df


# ============================================================================
# Clustering Results
# ============================================================================

def load_clustering_results(analysis_dir: Path) -> dict:
    """
    Load k=5 clustering results.

    Parameters
    ----------
    analysis_dir : Path
        Analysis directory containing output folder

    Returns
    -------
    results : dict
        Dictionary with 'cluster_chars' DataFrame and 'mutant_clusters' list
    """
    tables_dir = analysis_dir / "output" / "20250711" / "tables"

    # Load cluster characteristics
    cluster_chars = pd.read_csv(tables_dir / "cluster_characteristics_k5.csv")

    # Get mutant cluster IDs
    mutant_clusters = cluster_chars[cluster_chars['is_putative_mutant']]['cluster_id'].tolist()

    print(f"\nLoaded clustering results:")
    print(f"  Total clusters: {len(cluster_chars)}")
    print(f"  Mutant clusters: {mutant_clusters}")
    print(f"  WT-like clusters: {[c for c in cluster_chars['cluster_id'] if c not in mutant_clusters]}")

    return {
        'cluster_chars': cluster_chars,
        'mutant_clusters': mutant_clusters
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_xy_with_labels(df: pd.DataFrame,
                        exp_name: str = "experiment",
                        output_path: Path = None,
                        use_well_names: bool = True,
                        figsize: tuple = (14, 10)):
    """
    Plot XY stage positions with P or well name labels.

    Parameters
    ----------
    df : DataFrame
        DataFrame with columns [P, x_um, y_um, well_name]
    exp_name : str
        Experiment name for title
    output_path : Path
        Path to save figure (optional)
    use_well_names : bool
        If True, show "A01" instead of "0"
    figsize : tuple
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(df['x_um'], df['y_um'], s=100, alpha=0.5, c='steelblue', edgecolors='black')

    # Annotate with labels
    for _, row in df.iterrows():
        label = row['well_name'] if use_well_names else str(row['P'])
        ax.annotate(label, (row['x_um'], row['y_um']),
                   fontsize=7, ha='center', va='center', fontweight='bold')

    # Invert axes (microscope convention)
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_xlabel('X stage position (µm)', fontsize=12)
    ax.set_ylabel('Y stage position (µm)', fontsize=12)

    label_type = "Well Names" if use_well_names else "P indices"
    ax.set_title(f'ND2 Well Positions - {exp_name}\n({label_type})',
                fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.show()


# ============================================================================
# Grid Verification
# ============================================================================

def verify_grid_consistency(df: pd.DataFrame,
                            exp_name: str = "experiment",
                            n_rows: int = 8,
                            n_cols: int = 12,
                            tolerance_um: float = 50.0) -> dict:
    """
    Verify well assignments using clustering (mimics _qc_well_assignments from build01B).

    Uses KMeans to cluster stage positions by X and Y coordinates, then checks if
    the clustered labels match the expected well names.

    Parameters
    ----------
    df : DataFrame
        DataFrame with columns [well_name, x_um, y_um]
    exp_name : str
        Experiment name for reporting
    n_rows : int
        Expected number of rows (default 8 for A-H)
    n_cols : int
        Expected number of columns (default 12)
    tolerance_um : float
        (Unused - clustering-based verification doesn't use tolerance)

    Returns
    -------
    results : dict
        Dictionary with clustering verification results
    """
    from sklearn.cluster import KMeans

    print(f"\n{'='*80}")
    print(f"GRID VERIFICATION - {exp_name} (Clustering-based)")
    print(f"{'='*80}\n")

    results = {'rows': {}, 'cols': {}, 'summary': {}}

    # Extract row and column information from well names
    df = df[df['well_name'].str.match(r"^[A-H]\d{2}$", na=False)].copy()

    if len(df) == 0:
        print("⚠️ No valid well names found")
        results['summary'] = {
            'rows_passing': 0,
            'rows_total': 0,
            'cols_passing': 0,
            'cols_total': 0,
            'all_pass': False
        }
        return results

    row_letter_vec = np.asarray([id[0] for id in df['well_name']])
    col_num_vec = np.asarray([int(id[1:]) for id in df['well_name']])
    row_index = np.unique(row_letter_vec)
    col_index = np.unique(col_num_vec)

    # Prepare stage position array
    stage_xyz_array = df[['x_um', 'y_um']].values
    stage_xyz_array = np.column_stack([stage_xyz_array[:, 0], stage_xyz_array[:, 1], np.zeros(len(stage_xyz_array))])

    # Row verification (Y-axis clustering)
    print("ROW VERIFICATION (KMeans clustering on Y):")
    print("-" * 60)

    try:
        row_clusters = KMeans(n_init="auto", n_clusters=len(row_index)).fit(stage_xyz_array[:, 1].reshape(-1, 1))
        row_si = np.argsort(np.argsort(row_clusters.cluster_centers_.ravel()))
        row_ind_pd = row_si[row_clusters.labels_]
        row_letter_pd = row_index[row_ind_pd]

        # Check if predicted row letters match actual
        row_match = np.all(row_letter_pd == row_letter_vec)

        # Print per-row statistics
        for i, row_letter in enumerate(row_index):
            mask = row_letter_pd == row_letter
            if mask.sum() > 0:
                y_vals = stage_xyz_array[mask, 1]
                y_mean = y_vals.mean()
                y_std = y_vals.std()
                results['rows'][row_letter] = {
                    'mean': y_mean,
                    'std': y_std,
                    'n_wells': mask.sum(),
                    'matched': row_match
                }
                status = "✓" if row_match else "✗"
                print(f"{status} Row {row_letter}: Y = {y_mean:7.1f} ± {y_std:5.1f} µm ({mask.sum()} wells)")

        row_pass_count = 1 if row_match else 0
        print(f"\nRow verification: {'PASS' if row_match else 'FAIL'}")

    except Exception as e:
        print(f"✗ Row clustering failed: {e}")
        row_pass_count = 0
        row_match = False

    # Column verification (X-axis clustering)
    print("\nCOLUMN VERIFICATION (KMeans clustering on X):")
    print("-" * 60)

    try:
        col_clusters = KMeans(n_init="auto", n_clusters=len(col_index)).fit(stage_xyz_array[:, 0].reshape(-1, 1))
        col_si = np.argsort(np.argsort(col_clusters.cluster_centers_.ravel()))
        col_ind_pd = col_si[col_clusters.labels_]
        col_num_pd = col_index[len(col_index) - col_ind_pd - 1]  # Reverse index (matches build01B)

        # Check if predicted columns match actual
        col_match = np.all(col_num_pd == col_num_vec)

        # Print per-column statistics
        for i, col_num in enumerate(col_index):
            mask = col_num_pd == col_num
            if mask.sum() > 0:
                x_vals = stage_xyz_array[mask, 0]
                x_mean = x_vals.mean()
                x_std = x_vals.std()
                results['cols'][col_num] = {
                    'mean': x_mean,
                    'std': x_std,
                    'n_wells': mask.sum(),
                    'matched': col_match
                }
                status = "✓" if col_match else "✗"
                print(f"{status} Column {col_num:02d}: X = {x_mean:8.1f} ± {x_std:5.1f} µm ({mask.sum()} wells)")

        col_pass_count = 1 if col_match else 0
        print(f"\nColumn verification: {'PASS' if col_match else 'FAIL'}")

    except Exception as e:
        print(f"✗ Column clustering failed: {e}")
        col_pass_count = 0
        col_match = False

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_pass = row_match and col_match

    if all_pass:
        print("\n✅ GRID VERIFIED: Well names match clustered stage positions")
        print("   Row labels match Y-axis clustering")
        print("   Column labels match X-axis clustering")
    else:
        print("\n⚠️ GRID VERIFICATION FAILED:")
        if not row_match:
            print("   ✗ Row labels do NOT match Y-axis clustering")
        else:
            print("   ✓ Row labels match Y-axis clustering")
        if not col_match:
            print("   ✗ Column labels do NOT match X-axis clustering")
        else:
            print("   ✓ Column labels match X-axis clustering")
        print("\n   Possible causes:")
        print("   1. Incorrect well-to-series mapping in Excel metadata")
        print("   2. Row/column swap in ND2 position ordering")
        print("   3. Deselected wells causing offset in series numbering")

    results['summary'] = {
        'row_clusters_match': bool(row_match),
        'col_clusters_match': bool(col_match),
        'all_pass': all_pass
    }

    return results


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Main analysis pipeline."""
    import sys

    # Get experiment ID from command line or default to 20251225
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "20251121"

    print("="*80)
    print(f"ND2 POSITION VERIFICATION - {exp_id}")
    print("="*80)

    # Paths
    analysis_dir = Path(__file__).resolve().parent
    base_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    output_dir = analysis_dir / "output" / exp_id / "figures" / "nd2_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ND2 positions
    print("\n[1/4] Loading ND2 stage positions...")
    df_positions = load_nd2_positions(exp_id, base_path)

    # Load clustering results (optional - for future use)
    # Only load if this is the 20250711 experiment
    if exp_id == "20250711":
        print("\n[2/4] Loading clustering results...")
        clustering_results = load_clustering_results(analysis_dir)
    else:
        print("\n[2/4] Skipping clustering results (only available for 20250711)...")
        clustering_results = None

    # Plot with P labels
    print("\n[3/4] Plotting XY positions with P labels...")
    plot_xy_with_labels(
        df_positions,
        exp_name=exp_id,
        output_path=output_dir / "xy_positions_with_P_labels.png",
        use_well_names=False
    )

    # Plot with well names
    print("\n[4/4] Plotting XY positions with well names...")
    plot_xy_with_labels(
        df_positions,
        exp_name=exp_id,
        output_path=output_dir / "xy_positions_with_well_names.png",
        use_well_names=True
    )

    # Verify grid consistency
    print("\n[5/5] Verifying grid consistency...")
    verification_results = verify_grid_consistency(
        df_positions,
        exp_name=exp_id,
        tolerance_um=50.0
    )

    # Save verification report
    report_path = analysis_dir / "output" / exp_id / "tables" / "grid_verification_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"ND2 Grid Verification Report (Clustering-based) - {exp_id}\n")
        f.write("="*80 + "\n\n")

        f.write("METHOD: KMeans clustering of stage positions by X and Y coordinates\n")
        f.write("Verification checks if well labels match clustered positions.\n\n")

        f.write("ROW VERIFICATION:\n")
        f.write("-"*60 + "\n")
        for row_letter, stats in verification_results['rows'].items():
            f.write(f"  Row {row_letter}: Y = {stats['mean']:8.1f} ± {stats['std']:5.1f} µm ({stats['n_wells']} wells)\n")

        f.write("\nCOLUMN VERIFICATION:\n")
        f.write("-"*60 + "\n")
        for col_num, stats in verification_results['cols'].items():
            f.write(f"  Column {col_num:02d}: X = {stats['mean']:8.1f} ± {stats['std']:5.1f} µm ({stats['n_wells']} wells)\n")

        f.write("\nSUMMARY:\n")
        f.write("="*80 + "\n")
        summary = verification_results['summary']
        f.write(f"Row clustering match: {'YES' if summary['row_clusters_match'] else 'NO'}\n")
        f.write(f"Column clustering match: {'YES' if summary['col_clusters_match'] else 'NO'}\n")
        f.write(f"Overall: {'PASS ✓' if summary['all_pass'] else 'FAIL ✗'}\n")

    print(f"\nSaved verification report: {report_path}")

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'xy_positions_with_P_labels.png'}")
    print(f"  - {output_dir / 'xy_positions_with_well_names.png'}")
    print(f"  - {report_path}")


if __name__ == '__main__':
    main()
