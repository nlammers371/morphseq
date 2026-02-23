"""
Verify ND2 Well Position Ordering for All YX1 Experiments

This script checks all YX1 experiments to verify that ND2 file positions
match the expected plate layout using KMeans clustering verification.

The verification uses the Excel series_number_map to assign well names,
then checks if those well names are consistent with the actual XY positions
using KMeans clustering.

Adapted from verify_nd2_positions.py in tmem67 analysis.

Usage:
    python verify_all_yx1_experiments.py

Author: Generated via Claude Code
Date: 2025-12-10
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nd2
from sklearn.cluster import KMeans
import sys


# ============================================================================
# Helper Functions
# ============================================================================

def p_to_well_name(P: int, n_cols: int = 12) -> str:
    """
    Convert P index to well name (e.g., 0 → 'A01', 12 → 'B01').
    Fallback when no Excel mapping available.
    """
    P_1based = P + 1
    row = (P_1based - 1) // n_cols
    col = (P_1based - 1) % n_cols
    row_letter = chr(ord('A') + row)
    return f"{row_letter}{col+1:02d}"


def find_nd2_file(exp_id: str, base_path: Path) -> Path:
    """Locate the ND2 file for a given experiment."""
    nd2_dir = base_path / "raw_image_data" / "YX1" / exp_id
    nd2_files = list(nd2_dir.glob("*.nd2"))
    
    if len(nd2_files) == 0:
        raise FileNotFoundError(f"No ND2 file found in {nd2_dir}")
    
    return nd2_files[0]


def get_stage_positions_fast(f):
    """
    Return one stage position per P (first T,Z,C) — fast version.
    """
    sizes = f.sizes
    P = sizes.get("P", sizes.get("W", 1))
    Z = sizes.get("Z", 1)
    C = sizes.get("C", 1)
    records = []
    
    for w in range(P):
        idx = ((0 * P) + w) * (Z * C)
        try:
            md = f.frame_metadata(idx)
        except Exception:
            continue
        
        ch = getattr(md, "channels", [None])[0]
        if ch is None:
            continue
        
        pos = getattr(ch, "position", None)
        if not pos or not hasattr(pos, "stagePositionUm"):
            continue
        
        stage = pos.stagePositionUm
        records.append({
            "P": w,
            "x_um": getattr(stage, "x", np.nan),
            "y_um": getattr(stage, "y", np.nan),
            "z_um": getattr(stage, "z", np.nan)
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values("P").reset_index(drop=True)
    
    return df


def load_series_number_map(exp_id: str, metadata_dir: Path) -> dict:
    """
    Load explicit series-to-well mapping from Excel metadata.
    """
    metadata_path = metadata_dir / f"{exp_id}_well_metadata.xlsx"
    
    if not metadata_path.exists():
        return {}
    
    try:
        sm_raw = pd.read_excel(metadata_path, sheet_name='series_number_map', header=None)
        
        data_rows = sm_raw
        try:
            header_like = list(sm_raw.iloc[0, 1:13].astype(object))
            if header_like == list(range(1, 13)):
                data_rows = sm_raw.iloc[1:9, :]
            else:
                data_rows = sm_raw.iloc[:8, :]
        except Exception:
            data_rows = sm_raw.iloc[:8, :]
        
        series_map = data_rows.iloc[:, 1:13]
        
        mapping = {}
        col_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        row_letter_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
        
        for c in range(len(col_id_list)):
            for r in range(len(row_letter_list)):
                val = series_map.iloc[r, c]
                if pd.notna(val):
                    try:
                        series_idx_1b = int(val)
                        well_name = row_letter_list[r] + f"{col_id_list[c]:02d}"
                        mapping[series_idx_1b] = well_name
                    except Exception:
                        continue
        
        return mapping
    
    except Exception:
        return {}


def load_nd2_positions(exp_id: str, base_path: Path, metadata_dir: Path) -> pd.DataFrame:
    """
    Load stage positions from ND2 file and map to well names.
    """
    nd2_path = find_nd2_file(exp_id, base_path)
    
    with nd2.ND2File(str(nd2_path)) as f:
        df = get_stage_positions_fast(f)
    
    series_map = load_series_number_map(exp_id, metadata_dir)
    
    if series_map:
        df['series_num'] = df['P'] + 1
        df['well_name'] = df['series_num'].apply(
            lambda s: series_map.get(int(s), f"S{int(s)}") if pd.notna(s) else "Unknown"
        )
        df['has_excel_map'] = True
    else:
        df['series_num'] = df['P'] + 1
        df['well_name'] = df['P'].apply(p_to_well_name)
        df['has_excel_map'] = False
    
    return df


def verify_grid_consistency(df: pd.DataFrame, exp_name: str = "experiment") -> dict:
    """
    Verify well assignments using KMeans clustering.
    """
    results = {'rows': {}, 'cols': {}, 'summary': {}}
    
    df = df[df['well_name'].str.match(r"^[A-H]\d{2}$", na=False)].copy()
    
    if len(df) == 0:
        results['summary'] = {
            'row_clusters_match': False,
            'col_clusters_match': False,
            'all_pass': False,
            'error': 'No valid well names'
        }
        return results
    
    row_letter_vec = np.asarray([id[0] for id in df['well_name']])
    col_num_vec = np.asarray([int(id[1:]) for id in df['well_name']])
    row_index = np.unique(row_letter_vec)
    col_index = np.unique(col_num_vec)
    
    stage_xyz_array = df[['x_um', 'y_um']].values
    stage_xyz_array = np.column_stack([
        stage_xyz_array[:, 0], 
        stage_xyz_array[:, 1], 
        np.zeros(len(stage_xyz_array))
    ])
    
    # Row verification (Y-axis clustering)
    row_match = False
    try:
        row_clusters = KMeans(
            n_init="auto", 
            n_clusters=len(row_index)
        ).fit(stage_xyz_array[:, 1].reshape(-1, 1))
        row_si = np.argsort(np.argsort(row_clusters.cluster_centers_.ravel()))
        row_ind_pd = row_si[row_clusters.labels_]
        row_letter_pd = row_index[row_ind_pd]
        
        row_match = np.all(row_letter_pd == row_letter_vec)
    except Exception:
        pass
    
    # Column verification (X-axis clustering)
    col_match = False
    try:
        col_clusters = KMeans(
            n_init="auto", 
            n_clusters=len(col_index)
        ).fit(stage_xyz_array[:, 0].reshape(-1, 1))
        col_si = np.argsort(np.argsort(col_clusters.cluster_centers_.ravel()))
        col_ind_pd = col_si[col_clusters.labels_]
        col_num_pd = col_index[len(col_index) - col_ind_pd - 1]
        
        col_match = np.all(col_num_pd == col_num_vec)
    except Exception:
        pass
    
    all_pass = row_match and col_match
    
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
    """Verify all YX1 experiments."""
    print("="*80)
    print("ND2 POSITION VERIFICATION - ALL YX1 EXPERIMENTS")
    print("="*80)
    
    base_path = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq")
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/metadata/plate_metadata")
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yx1_dir = base_path / "raw_image_data" / "YX1"
    exp_ids = sorted([d.name for d in yx1_dir.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(exp_ids)} YX1 experiments")
    print("="*80)
    
    all_results = []
    passed = []
    failed = []
    errors = []
    
    for i, exp_id in enumerate(exp_ids, 1):
        print(f"\n[{i}/{len(exp_ids)}] Verifying {exp_id}...", end=" ")
        
        try:
            df_positions = load_nd2_positions(exp_id, base_path, metadata_dir)
            has_excel = df_positions['has_excel_map'].iloc[0] if len(df_positions) > 0 else False
            
            verification_results = verify_grid_consistency(df_positions, exp_name=exp_id)
            
            summary = verification_results['summary']
            row_pass = summary['row_clusters_match']
            col_pass = summary['col_clusters_match']
            all_pass = summary['all_pass']
            
            result = {
                'experiment_id': exp_id,
                'n_positions': len(df_positions),
                'has_excel_map': has_excel,
                'row_match': row_pass,
                'col_match': col_pass,
                'all_pass': all_pass,
                'status': 'PASS' if all_pass else 'FAIL'
            }
            all_results.append(result)
            
            if all_pass:
                passed.append(exp_id)
                print(f"✅ PASS ({len(df_positions)} wells)")
            else:
                failed.append(exp_id)
                print(f"⚠️  FAIL - Row: {row_pass}, Col: {col_pass}")
        
        except Exception as e:
            errors.append(exp_id)
            result = {
                'experiment_id': exp_id,
                'n_positions': 0,
                'has_excel_map': False,
                'row_match': False,
                'col_match': False,
                'all_pass': False,
                'status': f'ERROR: {str(e)[:50]}'
            }
            all_results.append(result)
            print(f"❌ ERROR: {e}")
    
    df_results = pd.DataFrame(all_results)
    results_path = output_dir / "verification_summary.csv"
    df_results.to_csv(results_path, index=False)
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total experiments:  {len(exp_ids)}")
    print(f"Passed:             {len(passed)} ✅")
    print(f"Failed:             {len(failed)} ⚠️")
    print(f"Errors:             {len(errors)} ❌")
    
    if failed:
        print("\n⚠️  FAILED experiments (need reprocessing):")
        for exp_id in failed:
            print(f"  - {exp_id}")
    
    if errors:
        print("\n❌ ERROR experiments:")
        for exp_id in errors:
            print(f"  - {exp_id}")
    
    print(f"\nResults saved to: {results_path}")
    print("="*80)


if __name__ == '__main__':
    main()
