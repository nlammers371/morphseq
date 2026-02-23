"""
YX1 series-to-well mapping.

Maps YX1 ND2 series numbers to plate well positions using:
1. XY-based reference matching (DEFAULT) - nearest-neighbor matching against reference coordinates
2. Explicit mapping from plate metadata series_number_map
3. Implicit positional mapping (fallback)
"""

from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Default path to YX1 reference coordinates
DEFAULT_REF_XY_PATH = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/YX1_nd2_ref_plate_xy_coordinates.csv")


def _parse_series_number_map(plate_df: pd.DataFrame) -> dict | None:
    """
    Extract series_number_map from plate metadata if present.

    Args:
        plate_df: Plate metadata DataFrame

    Returns:
        Dictionary mapping series numbers to well names, or None if not found
    """
    # Check if there's a series_number_map column or similar
    if 'series_number_map' in plate_df.columns:
        # Build mapping from series to well
        mapping = {}
        for _, row in plate_df.iterrows():
            series = row.get('series_number_map')
            well = row.get('well')
            if pd.notna(series) and pd.notna(well):
                try:
                    series_num = int(series)
                    mapping[series_num] = well
                except (ValueError, TypeError):
                    continue

        if mapping:
            log.info(f"Found explicit series_number_map with {len(mapping)} entries")
            return mapping

    log.info("No explicit series_number_map found in plate metadata")
    return None


def _build_implicit_mapping(scope_df: pd.DataFrame, plate_df: pd.DataFrame) -> dict:
    """
    Build implicit series-to-well mapping based on positional order.

    For YX1, we assume series numbers map to wells in row-major order
    (A01, A02, ..., A12, B01, B02, ..., H12) unless explicit mapping exists.

    Args:
        scope_df: Scope metadata DataFrame
        plate_df: Plate metadata DataFrame

    Returns:
        Dictionary mapping series numbers (1-based) to well names
    """
    # Get unique wells from scope metadata (these are the ND2 series)
    scope_wells = scope_df['well_index'].unique()
    n_series = len(scope_wells)

    # Get wells from plate metadata (these are the experiment wells)
    plate_wells = sorted(plate_df['well'].unique())

    log.info(f"Building implicit mapping: {n_series} series → {len(plate_wells)} plate wells")

    # Build positional mapping
    mapping = {}

    # YX1 series are typically 1-based indices
    for series_idx, well_index in enumerate(scope_wells):
        # Convert well_index to 1-based series number
        try:
            series_num = int(well_index) + 1  # Convert from 0-based to 1-based
        except ValueError:
            series_num = series_idx + 1

        # Map to plate well by position
        if series_idx < len(plate_wells):
            mapping[series_num] = plate_wells[series_idx]
        else:
            log.warning(f"Series {series_num} has no corresponding plate well")

    return mapping


# ============================================================================
# XY-Based Reference Mapping Functions
# ============================================================================

def _load_reference_xy_coordinates(ref_csv_path: Path = None) -> pd.DataFrame | None:
    """
    Load reference plate XY coordinates from CSV.

    Args:
        ref_csv_path: Path to reference CSV file. Uses DEFAULT_REF_XY_PATH if None.

    Returns:
        DataFrame with columns [well, x_um, y_um], or None if file not found
    """
    if ref_csv_path is None:
        ref_csv_path = DEFAULT_REF_XY_PATH

    if not ref_csv_path.exists():
        log.warning(f"Reference XY coordinates file not found: {ref_csv_path}")
        return None

    df = pd.read_csv(ref_csv_path)
    log.info(f"Loaded reference XY coordinates: {len(df)} wells from {ref_csv_path}")
    return df


def extract_nd2_stage_positions(nd2_path: Path) -> pd.DataFrame:
    """
    Extract stage positions from ND2 file.

    Args:
        nd2_path: Path to ND2 file

    Returns:
        DataFrame with columns [P, x_um, y_um]
    """
    import nd2

    log.info(f"Extracting stage positions from: {nd2_path}")

    with nd2.ND2File(str(nd2_path)) as f:
        sizes = f.sizes
        P = sizes.get("P", sizes.get("W", 1))
        Z = sizes.get("Z", 1)
        C = sizes.get("C", 1)

        log.info(f"  ND2 dimensions: P={P}, Z={Z}, C={C}")

        records = []
        for w in range(P):
            idx = w * (Z * C)  # T=0, first frame per position
            try:
                md = f.frame_metadata(idx)
                ch = getattr(md, "channels", [None])[0]
                if ch and hasattr(ch, "position"):
                    stage = ch.position.stagePositionUm
                    records.append({
                        "P": w,
                        "x_um": getattr(stage, "x", np.nan),
                        "y_um": getattr(stage, "y", np.nan)
                    })
            except Exception as e:
                log.warning(f"  Skipping P={w}: {e}")
                continue

        df = pd.DataFrame(records)
        log.info(f"  Extracted {len(df)} positions")
        return df


def _map_positions_to_wells_by_xy(
    nd2_positions: pd.DataFrame,
    ref_coordinates: pd.DataFrame,
    max_distance_um: float = 4500.0
) -> tuple[dict, dict]:
    """
    Map ND2 position indices to wells using nearest-neighbor XY matching.

    Args:
        nd2_positions: DataFrame with columns [P, x_um, y_um]
        ref_coordinates: DataFrame with columns [well, x_um, y_um]
        max_distance_um: Maximum allowed distance for match (reject if exceeded).
                         Default is ~half grid spacing (9000/2 = 4500 µm).

    Returns:
        Tuple of (mapping, diagnostics) where:
        - mapping: Dict mapping P index (0-based) to well name
        - diagnostics: Dict with 'distances', 'rejected', 'duplicates' lists
    """
    from scipy.spatial import cKDTree

    log.info("Mapping ND2 positions to wells via XY reference matching")

    # Build KD-tree from reference coordinates
    ref_xy = ref_coordinates[['x_um', 'y_um']].values
    tree = cKDTree(ref_xy)

    # Query nearest neighbor for each ND2 position
    nd2_xy = nd2_positions[['x_um', 'y_um']].values
    distances, indices = tree.query(nd2_xy, k=1)

    mapping = {}
    diagnostics = {'distances': [], 'rejected': [], 'duplicates': []}

    wells_used = set()
    for i, (p_idx, dist, ref_idx) in enumerate(zip(nd2_positions['P'], distances, indices)):
        well = ref_coordinates.iloc[ref_idx]['well']

        if dist > max_distance_um:
            diagnostics['rejected'].append({'P': int(p_idx), 'distance': float(dist), 'nearest_well': well})
            log.warning(f"  Rejected P={p_idx}: distance {dist:.1f} µm > {max_distance_um} µm (nearest: {well})")
            continue

        if well in wells_used:
            diagnostics['duplicates'].append({'P': int(p_idx), 'well': well})
            log.warning(f"  Duplicate mapping: P={p_idx} -> {well} (well already used)")

        mapping[int(p_idx)] = well
        wells_used.add(well)
        diagnostics['distances'].append(float(dist))

    # Log summary
    if diagnostics['distances']:
        log.info(f"  Matched {len(mapping)} positions")
        log.info(f"  Distance stats: min={min(diagnostics['distances']):.1f}, "
                 f"max={max(diagnostics['distances']):.1f}, "
                 f"mean={np.mean(diagnostics['distances']):.1f} µm")
    if diagnostics['rejected']:
        log.warning(f"  Rejected {len(diagnostics['rejected'])} positions (distance > {max_distance_um} µm)")
    if diagnostics['duplicates']:
        log.warning(f"  Found {len(diagnostics['duplicates'])} duplicate well mappings")

    return mapping, diagnostics


def map_series_to_wells_yx1(
    plate_metadata_csv: Path,
    scope_metadata_csv: Path,
    output_mapping_csv: Path,
    output_provenance_json: Path,
    nd2_path: Path = None,
    use_xy_reference: bool = True,
    ref_xy_csv: Path = None
) -> pd.DataFrame:
    """
    Map YX1 ND2 series numbers to well positions.

    Mapping priority (XY-based is DEFAULT):
    1. XY-based reference matching (if use_xy_reference=True, reference file exists, and ND2 path provided)
    2. Explicit mapping from plate metadata series_number_map
    3. Implicit positional mapping (fallback)

    Args:
        plate_metadata_csv: Path to plate metadata CSV
        scope_metadata_csv: Path to scope metadata CSV
        output_mapping_csv: Path to output mapping CSV
        output_provenance_json: Path to output provenance JSON
        nd2_path: Path to ND2 file for XY extraction (optional, required for XY-based mapping)
        use_xy_reference: Enable XY-based reference mapping (default True)
        ref_xy_csv: Path to custom reference XY coordinates CSV (optional)

    Returns:
        DataFrame with series-to-well mapping
    """
    log.info("Mapping YX1 series to wells")

    # Load metadata
    plate_df = pd.read_csv(plate_metadata_csv)
    scope_df = pd.read_csv(scope_metadata_csv)

    log.info(f"Loaded plate metadata: {len(plate_df)} rows")
    log.info(f"Loaded scope metadata: {len(scope_df)} rows")

    series_map = None
    mapping_method = None
    xy_diagnostics = None

    # Try XY-based reference mapping first (DEFAULT)
    if use_xy_reference and nd2_path is not None:
        ref_coordinates = _load_reference_xy_coordinates(ref_xy_csv)
        if ref_coordinates is not None:
            # Extract positions from ND2
            nd2_positions = extract_nd2_stage_positions(nd2_path)

            # Map via XY matching
            p_to_well_map, xy_diagnostics = _map_positions_to_wells_by_xy(nd2_positions, ref_coordinates)

            if p_to_well_map:
                # Convert P-based mapping to series-based mapping (series = P + 1)
                series_map = {p + 1: well for p, well in p_to_well_map.items()}
                mapping_method = "xy_reference"
                log.info(f"Successfully mapped {len(series_map)} wells via XY reference")
            else:
                log.warning("XY reference mapping produced no results, falling back to other methods")

    # Fall back to explicit mapping from plate metadata
    if series_map is None:
        series_map = _parse_series_number_map(plate_df)
        if series_map:
            mapping_method = "explicit"

    # Fall back to implicit positional mapping
    if series_map is None:
        series_map = _build_implicit_mapping(scope_df, plate_df)
        mapping_method = "implicit_positional"

    log.info(f"Using {mapping_method} mapping method")
    log.info(f"Mapped {len(series_map)} series")

    # Build mapping DataFrame
    rows = []
    for series_num, well_name in sorted(series_map.items()):
        rows.append({
            'series_number': series_num,
            'well_index': well_name,
            'mapping_method': mapping_method
        })

    mapping_df = pd.DataFrame(rows)

    # Write mapping CSV
    output_mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_mapping_csv, index=False)
    log.info(f"Wrote series mapping to {output_mapping_csv}")

    # Build provenance
    provenance = {
        'mapping_method': mapping_method,
        'total_series': len(series_map),
        'source_plate_metadata': str(plate_metadata_csv),
        'source_scope_metadata': str(scope_metadata_csv),
        'nd2_path': str(nd2_path) if nd2_path else None,
        'ref_xy_csv': str(ref_xy_csv) if ref_xy_csv else str(DEFAULT_REF_XY_PATH),
        'mapping_summary': {
            'min_series': int(min(series_map.keys())),
            'max_series': int(max(series_map.keys())),
            'wells': sorted(series_map.values())
        },
        'warnings': []
    }

    # Add XY diagnostics if available
    if xy_diagnostics:
        provenance['xy_diagnostics'] = {
            'n_matched': len(xy_diagnostics['distances']),
            'n_rejected': len(xy_diagnostics['rejected']),
            'n_duplicates': len(xy_diagnostics['duplicates']),
            'distance_stats': {
                'min': min(xy_diagnostics['distances']) if xy_diagnostics['distances'] else None,
                'max': max(xy_diagnostics['distances']) if xy_diagnostics['distances'] else None,
                'mean': float(np.mean(xy_diagnostics['distances'])) if xy_diagnostics['distances'] else None
            },
            'rejected': xy_diagnostics['rejected'],
            'duplicates': xy_diagnostics['duplicates']
        }

    # Check for gaps in series numbers
    series_nums = sorted(series_map.keys())
    expected_series = list(range(series_nums[0], series_nums[-1] + 1))
    gaps = set(expected_series) - set(series_nums)
    if gaps:
        warning = f"Series number gaps detected: {sorted(gaps)}"
        log.warning(warning)
        provenance['warnings'].append(warning)

    # Check for duplicate wells
    well_counts = pd.Series(list(series_map.values())).value_counts()
    duplicates = well_counts[well_counts > 1]
    if len(duplicates) > 0:
        warning = f"Duplicate well mappings: {duplicates.to_dict()}"
        log.warning(warning)
        provenance['warnings'].append(warning)

    # Write provenance
    with open(output_provenance_json, 'w') as f:
        json.dump(provenance, f, indent=2)
    log.info(f"Wrote provenance to {output_provenance_json}")

    return mapping_df


def load_series_mapping(mapping_csv: Path) -> dict:
    """
    Load series-to-well mapping from CSV.

    Args:
        mapping_csv: Path to mapping CSV

    Returns:
        Dictionary mapping series numbers to well names
    """
    df = pd.read_csv(mapping_csv)
    return dict(zip(df['series_number'], df['well_index']))


def map_nd2_to_wells_by_xy(
    nd2_path: Path,
    ref_xy_csv: Path = None,
    max_distance_um: float = 4500.0
) -> tuple[dict, dict]:
    """
    Standalone function to map ND2 positions to wells using XY reference.

    This is a convenience function for quick XY-based mapping without
    the full pipeline metadata requirements.

    Args:
        nd2_path: Path to ND2 file
        ref_xy_csv: Path to reference XY coordinates CSV (uses default if None)
        max_distance_um: Maximum allowed distance for match

    Returns:
        Tuple of (mapping, diagnostics) where:
        - mapping: Dict mapping P index (0-based) to well name
        - diagnostics: Dict with match statistics

    Example:
        >>> mapping, diag = map_nd2_to_wells_by_xy(Path("experiment.nd2"))
        >>> print(mapping[0])  # Well name for P=0
        'A01'
    """
    # Load reference coordinates
    ref_coordinates = _load_reference_xy_coordinates(ref_xy_csv)
    if ref_coordinates is None:
        raise FileNotFoundError(f"Reference XY coordinates not found: {ref_xy_csv or DEFAULT_REF_XY_PATH}")

    # Extract positions from ND2
    nd2_positions = extract_nd2_stage_positions(nd2_path)

    # Map via XY matching
    return _map_positions_to_wells_by_xy(nd2_positions, ref_coordinates, max_distance_um)
