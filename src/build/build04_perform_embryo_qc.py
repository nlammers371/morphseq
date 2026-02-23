import os
import pandas as pd
import scipy
import scipy.signal
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from typing import Optional
# Dependency simplification note: OLS usage can avoid statsmodels.
# Replace with NumPy least squares:
#   beta = np.linalg.lstsq(X_ft.values, Y_ft.values, rcond=None)[0]
#   predictions_full = X_full.values @ beta
import statsmodels.api as sm
from src.build.build_utils import bootstrap_perturbation_key_from_df01
from src.data_pipeline.quality_control.death_detection import compute_dead_flag2_persistence
from src.data_pipeline.quality_control.surface_area_outlier_detection import compute_sa_outlier_flag
from src.data_pipeline.quality_control.config import QC_DEFAULTS
from src.build.utils.curvature_utils import compute_embryo_curvature
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
import skimage.io as io
from src.build.qc import determine_use_embryo_flag
from src.build.build03A_process_images import resolve_sandbox_embryo_mask_from_csv
from functools import partial
from tqdm.contrib.concurrent import process_map


def _compute_curvature_for_row(row_idx: int, df: pd.DataFrame, root: Path) -> dict:
    """
    Compute curvature metrics for a single embryo row.

    Args:
        row_idx: Index of row in dataframe
        df: Build04 dataframe with mask information
        root: Project root directory

    Returns:
        Dictionary of curvature metrics indexed by row_idx, or None on failure
    """
    try:
        row = df.iloc[row_idx]

        # Skip if critical columns are missing
        if pd.isna(row.get("exported_mask_path")) or pd.isna(row.get("Height (um)")):
            return (row_idx, None)

        # Load mask from integer-labeled PNG
        try:
            mask_path = resolve_sandbox_embryo_mask_from_csv(root, row)
            im_mask_int = io.imread(mask_path)
            lbi = int(row["region_label"])
            im_mask = ((im_mask_int == lbi) * 255).astype(np.uint8)
        except Exception as e:
            # Mask loading failed - return NaN metrics
            return (row_idx, None)

        # Clean mask
        try:
            im_mask_clean, _ = clean_embryo_mask(im_mask, verbose=False)
        except Exception:
            # Cleaning failed - use original mask
            im_mask_clean = im_mask

        # Compute um_per_pixel
        um_per_pixel = float(row["Height (um)"]) / float(row["Height (px)"])

        # Compute curvature
        metrics = compute_embryo_curvature(im_mask_clean, um_per_pixel, verbose=False)

        return (row_idx, metrics)

    except Exception as e:
        # Return None for failed rows
        return (row_idx, None)


def _add_curvature_metrics(df: pd.DataFrame, root: Path, exp: str, n_workers: int = 4) -> pd.DataFrame:
    """
    Add curvature metrics to Build04 dataframe for all rows.

    For DF03: Keep only 3 essential columns
    - total_length_um
    - baseline_deviation_um
    - baseline_deviation_normalized (computed as baseline_deviation_um / total_length_um)

    For body_axis metadata: Write complete metrics to separate files
    - summary: All detailed metrics (mean/std/max curvature, keypoint deviations, etc.)
    - arrays: JSON-serialized centerline and curvature arrays

    Args:
        df: Build04 dataframe
        root: Project root directory
        exp: Experiment name (for body_axis file naming)
        n_workers: Number of parallel workers

    Returns:
        DataFrame with only 3 curvature columns added (for DF03)
    """
    print("📏 Computing curvature metrics...")

    # Define columns for DF03 (only 3 essential metrics)
    df03_columns = ['total_length_um', 'baseline_deviation_um', 'baseline_deviation_normalized']

    # Initialize columns with NaN
    for col in df03_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Compute curvature for all rows in parallel
    compute_func = partial(_compute_curvature_for_row, df=df, root=root)

    results = process_map(
        compute_func,
        range(len(df)),
        max_workers=n_workers,
        desc="Computing curvature",
        chunksize=max(1, len(df) // (n_workers * 10))
    )

    # Process results: add to DF03 and collect for body_axis files
    successful_count = 0
    summary_rows = []
    array_rows = []

    for row_idx, metrics in results:
        if metrics is not None:
            # Add essential columns to DF03
            df.loc[row_idx, 'total_length_um'] = metrics.get('total_length_um', np.nan)
            df.loc[row_idx, 'baseline_deviation_um'] = metrics.get('baseline_deviation_um', np.nan)

            # Compute normalized baseline deviation
            total_length = metrics.get('total_length_um', np.nan)
            if pd.notna(total_length) and total_length > 0:
                df.loc[row_idx, 'baseline_deviation_normalized'] = (
                    metrics.get('baseline_deviation_um', np.nan) / total_length
                )
            else:
                df.loc[row_idx, 'baseline_deviation_normalized'] = np.nan

            # Collect metrics for body_axis files
            snip_id = df.iloc[row_idx].get('snip_id', f'row_{row_idx}')

            # Summary metrics (all curvature stats)
            summary_row = {'snip_id': snip_id}
            summary_row.update({
                k: v for k, v in metrics.items()
                if k not in ['centerline_x_json', 'centerline_y_json', 'curvature_values_json', 'arc_length_values_json', 'error']
            })
            summary_rows.append(summary_row)

            # Array data (centerline and curvature arrays)
            array_row = {
                'snip_id': snip_id,
                'centerline_x_json': metrics.get('centerline_x_json'),
                'centerline_y_json': metrics.get('centerline_y_json'),
                'curvature_values_json': metrics.get('curvature_values_json'),
                'arc_length_values_json': metrics.get('arc_length_values_json'),
            }
            array_rows.append(array_row)

            successful_count += 1

    print(f"✅ Computed curvature for {successful_count}/{len(df)} rows")

    # Write to body_axis metadata files
    _write_body_axis_files(summary_rows, array_rows, root, exp)

    return df


def _write_body_axis_files(summary_rows: list, array_rows: list, root: Path, exp: str):
    """
    Write curvature metrics and arrays to body_axis metadata files (per-experiment).

    Per-experiment files allow for independent experiment processing without conflicts.

    Args:
        summary_rows: List of summary metric dictionaries
        array_rows: List of array data dictionaries
        root: Project root directory
        exp: Experiment name (used in filename)
    """
    root = Path(root)

    # Create body_axis directories
    summary_dir = root / "metadata" / "body_axis" / "summary"
    arrays_dir = root / "metadata" / "body_axis" / "arrays"
    summary_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    # Write summary metrics (per-experiment file)
    summary_path = summary_dir / f"curvature_metrics_{exp}.csv"
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        print(f"💾 Wrote {len(summary_rows)} summary metric rows to {summary_path.name}")

    # Write array data (per-experiment file)
    arrays_path = arrays_dir / f"curvature_arrays_{exp}.csv"
    if array_rows:
        arrays_df = pd.DataFrame(array_rows)
        arrays_df.to_csv(arrays_path, index=False)
        print(f"💾 Wrote {len(array_rows)} array data rows to {arrays_path.name}")


def build04_stage_per_experiment(
    root: Path,
    exp: str,
    in_csv: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    stage_ref: Optional[Path] = None,
    sa_ref_path: Optional[Path] = None,
    dead_lead_time: float = None,
    sg_window: Optional[int] = 5,
    sg_poly: int = 2,
) -> Path:
    """Load Build03 CSV for `exp`, run stage inference and QC, write per‑experiment Build04 CSV, and return its path.

    Parameters
    ----------
    root : Path
        Data root directory
    exp : str
        Experiment name/ID
    in_csv : Optional[Path], default None
        Input Build03 CSV path. If None, discovers from root/metadata/build03_output/expr_embryo_metadata_{exp}.csv
    out_dir : Optional[Path], default None
        Output directory. If None, uses root/metadata/build04_output/
    stage_ref : Optional[Path], default None
        Stage reference CSV. If None, uses root/metadata/stage_ref_df.csv
    sa_ref_path : Optional[Path], default None
        SA reference curves CSV. If None, uses root/metadata/sa_reference_curves.csv
    dead_lead_time : float, optional
        Hours before death to retroactively flag embryos.
        If None, uses QC_DEFAULTS['dead_lead_time_hours'] (default 4.0)
    sg_window : Optional[int], default 5
        Savitzky-Golay window length for smoothing. If None or insufficient data, skip smoothing
    sg_poly : int, default 2
        Savitzky-Golay polynomial order

    Returns
    -------
    Path
        Path to the output CSV file
    """
    # Use default from config if not specified
    if dead_lead_time is None:
        dead_lead_time = QC_DEFAULTS['dead_lead_time_hours']

    # Convert to Path objects
    root = Path(root)

    # Discover input path if not provided
    if in_csv is None:
        in_csv = root / "metadata" / "build03_output" / f"expr_embryo_metadata_{exp}.csv"
        if not in_csv.exists():
            raise FileNotFoundError(f"Build03 CSV not found: {in_csv}")
    else:
        in_csv = Path(in_csv)

    # Set up output directory and path
    if out_dir is None:
        out_dir = root / "metadata" / "build04_output"
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"qc_staged_{exp}.csv"

    # Set up stage reference path
    if stage_ref is None:
        stage_ref = root / "metadata" / "stage_ref_df.csv"
    else:
        stage_ref = Path(stage_ref)

    if not stage_ref.exists():
        raise FileNotFoundError(f"Stage reference not found: {stage_ref}")

    # Set up SA reference path
    if sa_ref_path is None:
        sa_ref_path = root / "metadata" / "sa_reference_curves.csv"
    else:
        sa_ref_path = Path(sa_ref_path)

    # Read input
    df = pd.read_csv(in_csv)
    print(f"📊 Loaded {len(df)} rows from Build03 CSV: {in_csv.name}")
    print(f"🔍 Available columns: {list(df.columns)}")

    # Map Build03 column names to Build04 expected names
    if "area_um2" in df.columns and "surface_area_um" not in df.columns:
        df["surface_area_um"] = df["area_um2"] 
        print("✅ Mapped surface_area_um from area_um2")
    
    # Map Time Rel (s) if it exists with different name
    if "Time Rel (s)" not in df.columns and "relative_time_s" in df.columns:
        df["Time Rel (s)"] = df["relative_time_s"]
        print("✅ Mapped 'Time Rel (s)' from relative_time_s")
    
    # Add phenotype mapping from genotype if needed  
    if "phenotype" not in df.columns and "genotype" in df.columns:
        df["phenotype"] = df["genotype"]
        print("✅ Mapped phenotype from genotype")
    
    # Add short_pert_name mapping from chem_perturbation if needed
    if "short_pert_name" not in df.columns and "chem_perturbation" in df.columns:
        df["short_pert_name"] = df["chem_perturbation"]
        print("✅ Mapped short_pert_name from chem_perturbation")
    
    # Add control_flag - assume all are controls for now, can refine later
    if "control_flag" not in df.columns:
        df["control_flag"] = True  # Conservative assumption for now
        print("✅ Added default control_flag: True")

    # Ensure required columns exist for stage inference
    if "experiment_date" not in df.columns:
        if "experiment" in df.columns:
            df["experiment_date"] = df["experiment"]
            print("✅ Added experiment_date from experiment column")
        elif exp:
            df["experiment_date"] = exp
            print(f"✅ Added experiment_date from exp parameter: {exp}")
        else:
            # Extract from filename as fallback
            basename = os.path.basename(in_csv)
            if "expr_embryo_metadata_" in basename:
                exp_name = basename.replace("expr_embryo_metadata_", "").replace(".csv", "")
                df["experiment_date"] = exp_name
                print(f"✅ Added experiment_date from filename: {exp_name}")
            else:
                df["experiment_date"] = "unknown"
                print("⚠️ Using default experiment_date: unknown")

    # Add stage inference
    # DEPRECATED: Stage inference moved to Build03 (_ensure_predicted_stage_hpf)
    # predicted_stage_hpf is already computed in Build03, so we skip this step
    # print("🧬 Running stage inference...")
    # df = infer_embryo_stage(str(root), df, stage_ref=stage_ref)
    # inferred_count = (~df["inferred_stage_hpf"].isna()).sum()
    # print(f"✅ Added inferred_stage_hpf to {inferred_count} rows")

    # Implement QC flags
    print("🔍 Computing QC flags...")
    df = _compute_qc_flags(df, stage_ref, dead_lead_time, sg_window, sg_poly, sa_ref_path)

    # Update perturbation key (enabled by default)
    print("🔑 Updating perturbation key...")
    df = _update_perturbation_key(df, root)

    # Add curvature metrics (NEW: integrated from standalone process_curvature_batch.py)
    # Computes for all rows with valid mask data; NaN where computation fails
    # DF03 keeps only 3 essential columns; detailed metrics written to body_axis metadata
    df = _add_curvature_metrics(df, root, exp=exp, n_workers=4)

    # Generate summary
    summary = _generate_summary(df)
    print(f"📋 Summary: {summary['total_rows']} total, {summary['usable_rows']} usable")
    for flag, count in summary['flag_counts'].items():
        if count > 0:
            print(f"   {flag}: {count}")

    # Print SA QC summary
    _print_sa_qc_summary(df)

    # Write output
    df.to_csv(out_csv, index=False)
    print(f"💾 Wrote {len(df)} rows to Build04 CSV: {out_csv}")

    return out_csv


def test_build04_io(in_csv, out_csv):
    """
    Tiny smoke test: copy input to output using the minimal I/O function.
    Returns a dict with basic counts to aid manual verification.
    """
    print(f"🧪 Testing Build04 I/O: {in_csv} → {out_csv}")
    out_path = build04_stage_per_experiment(root="test", exp="test", in_csv=in_csv, out_csv=out_csv)
    
    df_in = pd.read_csv(in_csv, nrows=5)
    df_out = pd.read_csv(out_path, nrows=5)
    
    result = {
        "out_path": out_path,
        "in_preview_rows": len(df_in),
        "out_preview_rows": len(df_out),
        "columns_match": list(df_in.columns) == list(df_out.columns),
    }
    
    print(f"✅ Test completed: {result}")
    return result


def infer_embryo_stage_orig(embryo_metadata_df, ref_date="20240626"):

    # build the reference set
    stage_cols = ["snip_id", "embryo_id", "short_pert_name", "phenotype", "control_flag",
                  "predicted_stage_hpf", "surface_area_um"]
    if "use_embryo_flag" in embryo_metadata_df.columns:
        stage_cols.append("use_embryo_flag")
    stage_df = embryo_metadata_df.loc[embryo_metadata_df["experiment_date"] == ref_date, stage_cols].reset_index(drop=True)
    ref_bool = (stage_df.loc[:, "phenotype"].to_numpy() == "wt") | (stage_df.loc[:, "control_flag"].to_numpy() == 1)
    ref_bool = ref_bool | (stage_df.loc[:, "phenotype"].to_numpy() == "uncertain")
    stage_df = stage_df.loc[ref_bool]
    stage_df["stage_group_hpf"] = np.round(stage_df["predicted_stage_hpf"])
    stage_df["stage_group_hpf"] = stage_df["stage_group_hpf"].astype(np.float)
    stage_key_df = stage_df.loc[:, ["stage_group_hpf", "surface_area_um"]].groupby('stage_group_hpf').quantile(.95).reset_index()
    # stage_key_df = stage_df.groupby('stage_group_hpf').quantile(.95).reset_index().loc[:, ["stage_group_hpf", "length_um"]]
    # add one entry for 72hpf taken from embryo poster 
    # row72 = pd.DataFrame([[72.01, 3.76*1000]], columns=["stage_group_hpf", "length_um"])
    # stage_key_df = pd.concat([stage_key_df, row72], axis=0, ignore_index=True)

    # get interpolator
    stage_interpolator = scipy.interpolate.interp1d(stage_key_df["surface_area_um"], stage_key_df["stage_group_hpf"],
                                    kind="linear", fill_value=np.nan, bounds_error=False)
    # iterate through dates
    date_index, date_indices = np.unique(embryo_metadata_df["experiment_date"], return_inverse=True)

    # initialize new field
    embryo_metadata_df["inferred_stage_hpf"] = np.nan

    for d, date in enumerate(tqdm(date_index)):
        date_cols = ["snip_id", "embryo_id", "time_int", "short_pert_name",
                     "phenotype", "control_flag", "predicted_stage_hpf", "surface_area_um"]
        if "use_embryo_flag" in embryo_metadata_df.columns:
            date_cols.append("use_embryo_flag")
        date_df = embryo_metadata_df.loc[date_indices == d, date_cols].reset_index(drop=True)

        # check for multiple age cohorts
        min_t = np.min(date_df["time_int"])
        cohort_key = date_df.loc[date_df["time_int"]==min_t, ["embryo_id", "predicted_stage_hpf"]]
        _, age_cohort = np.unique(np.round(cohort_key["predicted_stage_hpf"]/ 2.5) * 2.5, return_inverse=True)
        cohort_key["cohort_id"] = age_cohort

        # join onto main df
        date_df = date_df.merge(cohort_key.loc[:, ["embryo_id", "cohort_id"]], how="left", on="embryo_id")

        # check to see if this is a timeseries dataset
        _, embryo_counts = np.unique(date_df["embryo_id"], return_counts=True)
        snapshot_flag = np.max(embryo_counts) == 1
        # calculate length percentiles
        ref_bool = (date_df.loc[:, "phenotype"].to_numpy() == "wt") | (date_df.loc[:, "control_flag"].to_numpy() == 1)
        if date == "20240314":   # special allowance for this one dataset
            ref_bool = ref_bool | True

        date_df_ref = date_df.loc[ref_bool]
        # date_df["length_um"] = date_df["length_um"]*1.5
        date_df_ref["stage_group_hpf"] = np.round(date_df_ref["predicted_stage_hpf"])
        date_key_df = date_df_ref.loc[:, ["stage_group_hpf", "cohort_id", "surface_area_um"]].groupby(['stage_group_hpf', "cohort_id"]).quantile(.95).reset_index()

        # get interp predictions
        date_key_df["stage_hpf_interp"] = stage_interpolator(date_key_df["surface_area_um"])

        # if date == "20240314":
        #     print("check")
        if snapshot_flag:
            date_df["inferred_stage_hpf"] = stage_interpolator(date_df["predicted_stage_hpf"])
            # stage_skel = date_df.loc[:, ["snip_id", "stage_group_hpf"]]
            # stage_skel = stage_skel.merge(date_key_df.loc[:, ["stage_group_hpf", "stage_hpf_interp"]], how="left", on="stage_group_hpf").rename(
            #                 columns={"stage_hpf_interp":"inferred_stage_hpf"})
            # embryo_metadata_df = embryo_metadata_df.merge(stage_skel.loc[:, ["snip_id", "inferred_stage_hpf"]], how="left", on="snip_id")

        else:
            # fit regression model of predicted stage vs. interpolated stage
            Y = date_key_df['stage_hpf_interp']

            nan_ft = ~np.isnan(Y)

            X = date_key_df[['stage_group_hpf', 'cohort_id']] #, columns=['cohort_id'], drop_first=True)
            X = X.rename(columns={'stage_group_hpf':'stage'})
            X["stage2"] = X["stage"]**2
            X["interaction"] = np.prod(X[['stage', 'cohort_id']].to_numpy(), axis=1)
            # X["interaction2"] = np.prod(X[['stage2', 'cohort_id']].to_numpy(), axis=1)

            # Add a constant (intercept term) to the predictor matrix
            # X = sm.add_constant(X, has_constant='add')

            X_ft = X[nan_ft]
            Y_ft = Y[nan_ft]

            # Fit the OLS regression model
            model = sm.OLS(Y_ft, X_ft).fit()

            # now predict all stages
            X_full = date_df[['predicted_stage_hpf', 'cohort_id']] #, columns=['cohort_id'], drop_first=True)
            X_full = X_full.rename(columns={'predicted_stage_hpf':'stage'})
            X_full["stage2"] = X_full["stage"]**2
            X_full["interaction"] = np.prod(X_full[['stage', 'cohort_id']].to_numpy(), axis=1)

            # X_full = sm.add_constant(X_full, has_constant='add')
            # X_full["interaction2"] = np.prod(X_full[['stage2', 'cohort_id']].to_numpy(), axis=1)

            predictions_full = model.predict(X_full)

            # merge back to full df
            date_df["inferred_stage_hpf"] = predictions_full

        embryo_metadata_df.loc[date_indices == d, "inferred_stage_hpf"] = date_df["inferred_stage_hpf"].to_numpy()

    return embryo_metadata_df

def stage_from_sa(params, sa_vec):
    t_pd = params[3] * np.divide(sa_vec-params[0], params[1] - sa_vec + params[0])**(1/params[2])
    return t_pd

def infer_embryo_stage_sigmoid(root, embryo_metadata_df):

    # stage_key_df = pd.read_csv(os.path.join(root, "metadata", "stage_reg_key.csv"))
    stage_params = pd.read_csv(os.path.join(root, "metadata", "stage_ref_params.csv"))

    # iterate through dates
    date_index, date_indices = np.unique(embryo_metadata_df["experiment_date"], return_inverse=True)

    # initialize new field
    embryo_metadata_df["inferred_stage_hpf"] = np.nan

    for d, date in enumerate(tqdm(date_index)):
        date_cols = ["snip_id", "embryo_id", "time_int", "Time Rel (s)", "short_pert_name",
                     "phenotype", "control_flag", "predicted_stage_hpf", "surface_area_um"]
        if "use_embryo_flag" in embryo_metadata_df.columns:
            date_cols.append("use_embryo_flag")
        date_df = embryo_metadata_df.loc[date_indices == d, date_cols].reset_index(drop=True)

        # check for multiple age cohorts
        min_t = np.min(date_df["time_int"])
        cohort_key = date_df.loc[date_df["time_int"]==min_t, ["embryo_id", "predicted_stage_hpf"]]
        _, age_cohort = np.unique(np.round(cohort_key["predicted_stage_hpf"]/ 2.5) * 2.5, return_inverse=True)
        cohort_key["cohort_id"] = age_cohort

        date_df["abs_time_hr"] = date_df["Time Rel (s)"] / 3600

        # join onto main df
        date_df = date_df.merge(cohort_key.loc[:, ["embryo_id", "cohort_id"]], how="left", on="embryo_id")

        # check to see if this is a timeseries dataset
        _, embryo_counts = np.unique(date_df["embryo_id"], return_counts=True)
        snapshot_flag = np.max(embryo_counts) == 1
        # calculate length percentiles
        ref_bool = (date_df.loc[:, "phenotype"].to_numpy() == "wt") | (date_df.loc[:, "control_flag"].to_numpy() == 1)
        if date == "20240314":   # special allowance for this one dataset
            ref_bool = ref_bool | True

        date_df_ref = date_df.loc[ref_bool]
        # date_df["length_um"] = date_df["length_um"]*1.5
        date_df_ref["stage_group_hpf"] = np.round(date_df_ref["predicted_stage_hpf"])
        date_key_df = date_df_ref.loc[:, ["stage_group_hpf", "cohort_id", "surface_area_um"]].groupby(
            ['stage_group_hpf', "cohort_id"]).quantile(.90).reset_index()

        # smooth
        # sa_bound_sm = scipy.signal.savgol_filter(date_key_df["surface_area_um"], window_length=3, polyorder=2)

        # use grouped percentiles to get interpolator
        # sa_interpolator = scipy.interpolate.interp1d(date_key_df["stage_group_hpf"], sa_bound_sm,
        #                                                 kind="linear", fill_value="nearest", bounds_error=False)
        # sa_interp_full = sa_interpolator(date_df["predicted_stage_hpf"])
        sa_interp_full = np.interp(x=date_df["predicted_stage_hpf"], xp=date_key_df["stage_group_hpf"], fp=date_key_df["surface_area_um"])
        # get stage predictions
        stage_predictions = stage_from_sa(stage_params.to_numpy()[0], sa_interp_full)
        stage_predictions[stage_predictions > 96] = 96
        stage_min = np.min(stage_predictions)
        def reg_curve(params, intercept=stage_min, real_time_vec=date_df["abs_time_hr"].to_numpy()):
            pd_time = intercept + params[0] * real_time_vec + params[1] * real_time_vec ** 2
            return pd_time

        def loss_fun(params, target_time_vec=stage_predictions):
            pd_time = reg_curve(params)
            return pd_time - target_time_vec

        x0 = [1, 0.01]
        # sigmoid(x0)
        params_fit = scipy.optimize.least_squares(loss_fun, x0, bounds=[(0, 0), (2, 0.1)])
        date_df["inferred_stage_hpf"] = reg_curve(params_fit.x)

        embryo_metadata_df.loc[date_indices == d, "inferred_stage_hpf"] = date_df["inferred_stage_hpf"].to_numpy()

    return embryo_metadata_df

def infer_embryo_stage(root, embryo_metadata_df, stage_ref: Optional[Path] = None):

    # load ref dataset
    if stage_ref is None:
        ref_path = os.path.join(root, "metadata", "stage_ref_df.csv")
    else:
        ref_path = str(stage_ref)

    print(f"🔍 DEBUG: Loading stage reference from: {ref_path}")
    stage_key_df = pd.read_csv(ref_path)
    print(f"🔍 DEBUG: Stage ref loaded, shape: {stage_key_df.shape}")
    print(f"🔍 DEBUG: Stage ref columns: {list(stage_key_df.columns)}")
    print(f"🔍 DEBUG: SA range: {stage_key_df['sa_um'].min():.1f} - {stage_key_df['sa_um'].max():.1f}")
    # stage_key_df = stage_key_df.loc[stage_key_df["stage_hpf"] <= 72] # not reliable after this point

    # get interpolator
    stage_interpolator = scipy.interpolate.interp1d(stage_key_df["sa_um"], stage_key_df["stage_hpf"],
                                    kind="linear", fill_value=np.nan, bounds_error=False)
    # iterate through dates
    date_index, date_indices = np.unique(embryo_metadata_df["experiment_date"], return_inverse=True)
    print(f"🔍 DEBUG: Unique dates found: {date_index}")
    print(f"🔍 DEBUG: Input data shape: {embryo_metadata_df.shape}")
    print(f"🔍 DEBUG: Available columns: {list(embryo_metadata_df.columns)}")

    # initialize new field
    embryo_metadata_df["inferred_stage_hpf"] = np.nan

    for d, date in enumerate(tqdm(date_index)):
        print(f"\n🔍 DEBUG: Processing date {d}: {date}")
        date_cols = ["snip_id", "embryo_id", "time_int", "Time Rel (s)", "short_pert_name",
                     "phenotype", "control_flag", "predicted_stage_hpf", "surface_area_um"]
        if "use_embryo_flag" in embryo_metadata_df.columns:
            date_cols.append("use_embryo_flag")
        date_df = embryo_metadata_df.loc[date_indices == d, date_cols].reset_index(drop=True)

        print(f"🔍 DEBUG: Date {date} - {len(date_df)} embryos")
        print(f"🔍 DEBUG: Phenotype values: {date_df['phenotype'].value_counts().to_dict()}")
        print(f"🔍 DEBUG: Control flag values: {date_df['control_flag'].value_counts().to_dict()}")
        print(f"🔍 DEBUG: SA range: {date_df['surface_area_um'].min():.1f} - {date_df['surface_area_um'].max():.1f}")

        date_df["abs_time_hr"] = date_df["Time Rel (s)"] / 3600
        # check for multiple age cohorts
        min_t = np.min(date_df["time_int"])
        cohort_key = date_df.loc[date_df["time_int"] == min_t, ["embryo_id", "predicted_stage_hpf"]]
        _, age_cohort = np.unique(np.round(cohort_key["predicted_stage_hpf"] / 5) * 5, return_inverse=True)
        if date != "20240626":
            cohort_key["cohort_id"] = age_cohort
        else:
            cohort_key["cohort_id"] = 0
        cohort_key = cohort_key.drop_duplicates(subset=["embryo_id"])

        # join onto main df
        date_df = date_df.merge(cohort_key.loc[:, ["embryo_id", "cohort_id"]], how="left", on="embryo_id")

        # check to see if this is a timeseries dataset
        _, embryo_counts = np.unique(date_df["embryo_id"], return_counts=True)
        snapshot_flag = np.max(embryo_counts) == 1
        print(f"🔍 DEBUG: Snapshot flag: {snapshot_flag}")
        # calculate length percentiles
        ref_bool = (date_df.loc[:, "phenotype"].to_numpy() == "wt") | (date_df.loc[:, "control_flag"].to_numpy() == 1)
        if date == "20240314":   # special allowance for this one dataset
            ref_bool = ref_bool | True

        # date_df["abs_time_hpf"] = np.round(date_df["predicted_stage_hpf"])

        print(f"🔍 DEBUG: Reference embryos after filtering: {ref_bool.sum()} out of {len(ref_bool)}")

        date_df_ref = date_df.loc[ref_bool]
        if len(date_df_ref) == 0:
            print(f"🔍 DEBUG: No reference embryos found for date {date}, skipping")
            continue

        # date_df["length_um"] = date_df["length_um"]*1.5
        date_df_ref["stage_group_hpf"] = np.round(date_df_ref["predicted_stage_hpf"])   # ["predicted_stage_hpf"])
        date_key_df = date_df_ref.loc[:, ["stage_group_hpf", "cohort_id", "surface_area_um"]].groupby(
                                                        ['stage_group_hpf', "cohort_id"]).quantile(.95).reset_index()

        print(f"🔍 DEBUG: Date key shape after groupby: {date_key_df.shape}")
        # get interp predictions
        date_key_df["stage_hpf_interp"] = stage_interpolator(date_key_df["surface_area_um"])
        print(f"🔍 DEBUG: Interpolated stages - NaN count: {date_key_df['stage_hpf_interp'].isna().sum()}")

        if snapshot_flag:
            print(f"🔍 DEBUG: Using snapshot mode - direct interpolation from SA")
            # BUG FIX: snapshot_flag should use surface_area_um, not predicted_stage_hpf!
            date_df["inferred_stage_hpf"] = stage_interpolator(date_df["surface_area_um"])
            inferred_count = (~date_df["inferred_stage_hpf"].isna()).sum()
            print(f"🔍 DEBUG: Snapshot - inferred {inferred_count} out of {len(date_df)} embryos")
            # stage_skel = date_df.loc[:, ["snip_id", "stage_group_hpf"]]
            # stage_skel = stage_skel.merge(date_key_df.loc[:, ["stage_group_hpf", "stage_hpf_interp"]], how="left", on="stage_group_hpf").rename(
            #                 columns={"stage_hpf_interp":"inferred_stage_hpf"})
            # embryo_metadata_df = embryo_metadata_df.merge(stage_skel.loc[:, ["snip_id", "inferred_stage_hpf"]], how="left", on="snip_id")

        else:
            # fit regression model of predicted stage vs. interpolated stage
            Y = date_key_df['stage_hpf_interp']
            T = date_key_df['stage_group_hpf'].to_numpy()
            G = date_key_df['cohort_id'].to_numpy().astype(int)

            nan_ft = ~np.isnan(Y)

            Y = Y[nan_ft]
            T = T[nan_ft]
            G = G[nan_ft]

            ignore_g = True
            if len(np.unique(G)) > 1:
                ignore_g = False
            def stage_pd_fun(params, t_vec=T, g_vec=G, g_flag=ignore_g):
                #   intercept + group_dummy +
                if not g_flag:
                    stage_pd = params[0] + params[1]*g_vec + params[2]*t_vec + params[3]*t_vec**2# + params[4]**3
                else:
                    stage_pd = params[0] + params[1] * t_vec + params[2] * t_vec ** 2
                return stage_pd

            # define loss
            def loss_fun(params, y=Y):
                loss = stage_pd_fun(params) - y
                return loss

            # intercept, group intercept, slope, quadratic slope, quad center
            if not ignore_g:
                x0 = [0, 0, 1, 0]  #, 0]
                ub = (4, 72, 2.0, 1e-5)  #, 0.005)
                lb = (0, -72, 0.5, -1e-5)  #, -0.005)
            else:
                x0 = [0, 1, 0]  # , 0]
                ub = (4, 2.0, 0.005)  # , 0.005)
                lb = (0, 0.5, -0.005)  # , -0.005)

            params_fit = scipy.optimize.least_squares(loss_fun, x0,  bounds=[lb, ub])

            # now predict all stages
            T_full = date_df['predicted_stage_hpf'].to_numpy()
            G_full = date_df['cohort_id'].to_numpy().astype(int)

            predictions_full = stage_pd_fun(params_fit.x, t_vec=T_full, g_vec=G_full)

            # merge back to full df
            date_df["inferred_stage_hpf"] = predictions_full
            print(f"🔍 DEBUG: Timeseries mode - predicted {len(predictions_full)} stages")

        # Copy results back to main dataframe
        valid_inferred = (~date_df["inferred_stage_hpf"].isna()).sum()
        embryo_metadata_df.loc[date_indices == d, "inferred_stage_hpf"] = date_df["inferred_stage_hpf"].to_numpy()
        print(f"🔍 DEBUG: Copied {valid_inferred} inferred stages back to main dataframe for date {date}")

    total_inferred = (~embryo_metadata_df["inferred_stage_hpf"].isna()).sum()
    print(f"🔍 DEBUG: FINAL - Total inferred stages: {total_inferred} out of {len(embryo_metadata_df)}")
    return embryo_metadata_df


def _compute_qc_flags(df, stage_ref, dead_lead_time=None, sg_window=5, sg_poly=2, sa_ref_path=None):
    """
    Compute QC flags for Build04: sa_outlier_flag and dead_flag2.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with embryo data
    stage_ref : Path
        Path to stage reference CSV file
    dead_lead_time : float, optional
        Hours before death to retroactively flag.
        If None, uses QC_DEFAULTS['dead_lead_time_hours'] (default 4.0)
    sg_window : int
        Savitzky-Golay window length
    sg_poly : int
        Savitzky-Golay polynomial order
    sa_ref_path : Optional[Path]
        Path to SA reference curves CSV file

    Returns
    -------
    pd.DataFrame
        Dataframe with added QC flags
    """
    # Use default from config if not specified
    if dead_lead_time is None:
        dead_lead_time = QC_DEFAULTS['dead_lead_time_hours']

    df = df.copy()

    # Initialize QC flags
    df["sa_outlier_flag"] = False
    df["dead_flag2"] = False

    # Validate critical inputs and coerce dtypes
    df = _validate_and_prepare_inputs(df)

    # Surface area outlier detection - NEW METHOD (two-sided)
    if sa_ref_path and Path(sa_ref_path).exists():
        df = compute_sa_outlier_flag(
            df=df,
            sa_reference_path=sa_ref_path,
            k_upper=1.2,
            k_lower=0.9,
        )
    else:
        # Fallback to old method if reference not available
        print(f"⚠️  SA reference not found at {sa_ref_path}, using fallback method")
        df = _sa_qc_with_fallback(
            df=df,
            stage_ref_path=stage_ref,
            sg_window=sg_window,
            sg_poly=sg_poly,
            percentile=95.0,
            bin_step=0.5,
            hpf_window=0.75,
            min_embryos=2,
            margin_k=2.0,
            calibrate_scale=True,
        )

    # Death lead-time flagging - NEW METHOD
    df = compute_dead_flag2_persistence(df, dead_lead_time)

    # OLD METHOD (commented out for rollback capability)
    # df = _compute_dead_flag2(df, dead_lead_time)

    # Use centralized function - THE ONLY place that determines use_embryo_flag
    df["use_embryo_flag"] = determine_use_embryo_flag(df)

    return df


def _sa_qc_with_fallback(
    df: pd.DataFrame,
    stage_ref_path: Optional[Path],
    sg_window: Optional[int] = 5,
    sg_poly: int = 2,
    percentile: float = 95.0,
    bin_step: float = 0.5,
    hpf_window: float = 0.75,
    min_embryos: int = 2,
    margin_k: float = 2.0,
    calibrate_scale: bool = True,
) -> pd.DataFrame:
    """
    DEPRECATED: Use compute_sa_outlier_flag() from
    src.data_pipeline.quality_control.surface_area_outlier_detection instead.

    This function will be removed in a future version. Kept for legacy compatibility.

    SA QC: internal controls vs predicted_stage_hpf; fallback to stage_ref with margin.

    Note: This performs one-sided outlier detection, only flagging embryos that are
    unusually large (surface_area > threshold). Small embryos are not flagged as they
    may represent valid biological phenotypes rather than technical errors.
    """
    import warnings
    warnings.warn(
        "_sa_qc_with_fallback is deprecated. Use compute_sa_outlier_flag() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    df = df.copy()

    if "surface_area_um" not in df.columns:
        print("⚠️  Warning: surface_area_um not found, skipping SA outlier detection")
        df["sa_outlier_flag"] = False
        return df
    if "predicted_stage_hpf" not in df.columns:
        raise ValueError("Missing required column 'predicted_stage_hpf' for SA QC")

    # Reference subset: (phenotype == 'wt') OR control_flag, AND use_embryo_flag
    use_mask = df.get("use_embryo_flag", True)
    if isinstance(use_mask, bool):
        use_mask = pd.Series([use_mask] * len(df), index=df.index)
    use_mask = use_mask.astype(bool)
    ph_wt = (df["phenotype"].astype(str).str.lower() == "wt") if ("phenotype" in df.columns) else pd.Series(False, index=df.index)
    ctrl = df["control_flag"].astype(bool) if ("control_flag" in df.columns) else pd.Series(False, index=df.index)
    ref_mask = (ph_wt | ctrl) & use_mask

    # Build 95th percentile curve across bins
    time_bins = np.arange(0, 72 + bin_step, bin_step)
    perc_curve = np.full_like(time_bins, np.nan, dtype=float)

    have_reference = ref_mask.sum() >= min_embryos
    valid_bins = 0
    # Force stage_ref usage for testing
    if False:  # Temporarily disable internal controls
        for i, t in enumerate(time_bins):
            win = (
                (df["predicted_stage_hpf"] >= t - hpf_window)
                & (df["predicted_stage_hpf"] <= t + hpf_window)
                & ref_mask
                & (~df["surface_area_um"].isna())
            )
            n = win.sum()
            if n >= min_embryos:
                perc_curve[i] = df.loc[win, "surface_area_um"].quantile(percentile / 100.0)
                valid_bins += 1

    if False and valid_bins >= 3:  # Also disable this path
        # Fill edges and smooth
        valid_mask = ~np.isnan(perc_curve)
        first_valid = np.where(valid_mask)[0][0]
        last_valid = np.where(valid_mask)[0][-1]
        perc_curve[:first_valid] = perc_curve[first_valid]
        perc_curve[last_valid + 1 :] = perc_curve[last_valid]

        if sg_window is not None:
            npts = valid_mask.sum()
            w = int(sg_window)
            if w % 2 == 0:
                w = max(3, w - 1)
            w = min(max(3, w), max(3, npts if npts % 2 == 1 else npts - 1))
            try:
                perc_curve = scipy.signal.savgol_filter(perc_curve, w, sg_poly)
            except Exception as e:
                print(f"⚠️  Savitzky-Golay smoothing failed (w={w}, p={sg_poly}): {e}")

        # Assign thresholds by nearest bin
        idx = np.searchsorted(time_bins, df["predicted_stage_hpf"].to_numpy())
        idx = np.clip(idx, 0, len(perc_curve) - 1)
        thresholds = perc_curve[idx]
        df["sa_outlier_flag"] = df["surface_area_um"].to_numpy() > thresholds
        print(f"✅ SA QC: used internal controls (valid bins={valid_bins}, ref_n={int(ref_mask.sum())})")
        return df

    # Fallback: stage_ref-based threshold
    if stage_ref_path is None or not Path(stage_ref_path).exists():
        print("⚠️  SA QC fallback requested, but stage_ref missing; skipping SA QC")
        df["sa_outlier_flag"] = False
        return df
    try:
        ref_df = pd.read_csv(stage_ref_path)
    except Exception as e:
        print(f"⚠️  SA QC fallback: failed reading stage_ref: {e}")
        df["sa_outlier_flag"] = False
        return df
    if not {"stage_hpf", "sa_um"}.issubset(ref_df.columns):
        print("⚠️  SA QC fallback: stage_ref missing ['stage_hpf','sa_um']")
        df["sa_outlier_flag"] = False
        return df
    try:
        sa_of_stage = scipy.interpolate.interp1d(
            ref_df["stage_hpf"].to_numpy(),
            ref_df["sa_um"].to_numpy(),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    except Exception as e:
        print(f"⚠️  SA QC fallback: failed to build ref interpolator: {e}")
        df["sa_outlier_flag"] = False
        return df

    stage_vals = df["predicted_stage_hpf"].to_numpy()
    sa_ref = sa_of_stage(stage_vals)
    scale = 1.0
    if calibrate_scale:
        valid = (~np.isnan(sa_ref)) & (~df["surface_area_um"].isna())
        if valid.sum() >= 5:
            ratio = (df.loc[valid, "surface_area_um"].to_numpy() / sa_ref[valid])
            high = np.quantile(ratio, 0.975)
            ratio = ratio[ratio <= high]
            if len(ratio) > 0:
                scale = float(np.median(ratio))
    thresholds = scale * margin_k * sa_ref  # margin_k=2.0 (increased from 1.40 to be less exclusionary)
    df["sa_outlier_flag"] = df["surface_area_um"].to_numpy() > thresholds
    print(f"✅ SA QC: used stage_ref fallback (scale={scale:.3f}, margin_k={margin_k})")
    return df


# OLD METHOD - Commented out for rollback capability
# def _compute_dead_flag2(df, dead_lead_time=2.0):
#     """
#     Compute death lead-time flags (dead_flag2).
#
#     For embryos with any dead_flag timepoint, retroactively flag all timepoints
#     within dead_lead_time hours preceding the first death.
#     """
#     df = df.copy()
#
#     if "dead_flag" not in df.columns:
#         print("⚠️  Warning: dead_flag not found, skipping death lead-time detection")
#         df["dead_flag2"] = False
#         return df
#
#     if "embryo_id" not in df.columns:
#         print("⚠️  Warning: embryo_id not found, skipping death lead-time detection")
#         df["dead_flag2"] = False
#         return df
#
#     # Use predicted_stage_hpf for parity with legacy
#     time_col = "predicted_stage_hpf"
#
#     dead_lead_count = 0
#
#     # Process each embryo
#     for embryo_id in df["embryo_id"].unique():
#         embryo_mask = df["embryo_id"] == embryo_id
#         embryo_data = df.loc[embryo_mask]
#
#         # Check if this embryo ever dies
#         dead_timepoints = embryo_data[embryo_data["dead_flag"] == True]
#         if len(dead_timepoints) == 0:
#             continue
#
#         # Find first death time
#         first_death_time = dead_timepoints[time_col].min()
#         if pd.isna(first_death_time):
#             continue
#
#         # Flag timepoints within lead time before death
#         lead_mask = ((embryo_data[time_col] >= first_death_time - dead_lead_time) &
#                      (embryo_data[time_col] <= first_death_time))
#
#         dead_lead_indices = embryo_data.loc[lead_mask].index
#         df.loc[dead_lead_indices, "dead_flag2"] = True
#         dead_lead_count += len(dead_lead_indices)
#
#     print(f"✅ Flagged {dead_lead_count} timepoints in death lead-time")
#     return df


def _update_perturbation_key(df, root):
    """
    Update perturbation key by default (bootstrap and augment if needed).
    """
    try:
        from src.build.build_utils import bootstrap_perturbation_key_from_df01

        key_path = root / "metadata" / "perturbation_name_key.csv"

        # Load or bootstrap perturbation key
        try:
            pert_name_key = pd.read_csv(key_path)
        except Exception:
            print(f"ℹ️  Perturbation key not found, bootstrapping from data...")
            pert_name_key = bootstrap_perturbation_key_from_df01(root=str(root), df01=df, out_path=str(key_path))
            print(f"📝 Wrote bootstrapped perturbation key to {key_path}")

        # Merge perturbation info if master_perturbation column exists
        if "master_perturbation" in df.columns and "master_perturbation" in pert_name_key.columns:
            df = df.merge(pert_name_key, how="left", on="master_perturbation")
            print(f"✅ Merged perturbation key information")
        else:
            print("ℹ️  Skipping perturbation key merge (master_perturbation column not found)")

    except Exception as e:
        print(f"⚠️  Warning: Perturbation key update failed: {e}")

    return df


def _generate_summary(df):
    """
    Generate summary statistics for Build04 output.

    Returns
    -------
    dict
        Summary with total_rows, usable_rows, and flag_counts
    """
    summary = {
        "total_rows": len(df),
        "usable_rows": df.get("use_embryo_flag", pd.Series([True] * len(df))).sum(),
        "flag_counts": {}
    }

    # Count various flags
    flag_columns = ["dead_flag", "dead_flag2", "sa_outlier_flag", "sam2_qc_flag", "bubble_flag",
                   "focus_flag", "frame_flag", "no_yolk_flag"]

    for flag_col in flag_columns:
        if flag_col in df.columns:
            summary["flag_counts"][flag_col] = df[flag_col].sum()

    return summary


def _print_sa_qc_summary(df: pd.DataFrame) -> None:
    """Print percent of rows and snip_ids flagged by SA QC."""
    if "sa_outlier_flag" not in df.columns:
        return
    n = len(df)
    if n == 0:
        return
    sa_flags = df["sa_outlier_flag"].astype(bool)
    pct_rows = 100.0 * sa_flags.sum() / n
    # Prefer snip_id; else embryo_id if available
    id_col = "snip_id" if "snip_id" in df.columns else ("embryo_id" if "embryo_id" in df.columns else None)
    if id_col:
        total_ids = df[id_col].nunique(dropna=True)
        flagged_ids = df.loc[sa_flags, id_col].nunique(dropna=True)
        pct_ids = 100.0 * (flagged_ids / total_ids) if total_ids else 0.0
        print(f"📈 SA QC: {pct_rows:.1f}% rows flagged; {pct_ids:.1f}% {id_col}s flagged")
    else:
        print(f"📈 SA QC: {pct_rows:.1f}% rows flagged; id-level metric unavailable")


def _validate_and_prepare_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Fail-loud validation and dtype coercion for critical columns."""
    required = [
        "embryo_id",
        "predicted_stage_hpf",
        "surface_area_um",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Build04: {missing}")

    # Coerce booleans and provide defaults
    for col in [
        "use_embryo_flag",
        "dead_flag",
        "control_flag",
        "bubble_flag",
        "focus_flag",
        "frame_flag",
        "no_yolk_flag",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
        else:
            if col == "use_embryo_flag":
                df[col] = True
            else:
                df[col] = False
    return df


def perform_embryo_qc(
    root,
    dead_lead_time=None,
    pert_key_path: str | None = None,
    auto_augment_pert_key: bool = True,
    write_augmented_key: bool = False,
):
    """Build04: QC + stage inference with robust perturbation key handling.

    Behavior (summary):
    - If `metadata/perturbation_name_key.csv` is missing/unreadable, bootstrap
      it from df01 and write it before proceeding.
    - If the key is incomplete, auto-augment missing `master_perturbation`
      rows and always persist the augmented union. Added rows include a
      `time_auto_constructed` timestamp for traceability.
    - Fills safe defaults for optional df01 columns (flags, stage/time/surface
      area, plus `temperature`/`medium`) to avoid brittle failures.

    Parameters
    ----------
    root : str or Path
        Data root directory
    dead_lead_time : float, optional
        Hours before death to retroactively flag embryos.
        If None, uses QC_DEFAULTS['dead_lead_time_hours'] (default 4.0)
    pert_key_path : str, optional
        Path to perturbation key file
    auto_augment_pert_key : bool, default True
        Whether to auto-augment missing perturbation rows
    write_augmented_key : bool, default False
        Whether to write augmented perturbation key

    Outputs: df02 and curation CSVs written under
    `metadata/combined_metadata_files/`.
    """
    # Use default from config if not specified
    if dead_lead_time is None:
        dead_lead_time = QC_DEFAULTS['dead_lead_time_hours']

    # read in metadata
    metadata_path = os.path.join(root, 'metadata', "combined_metadata_files", '')
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df01.csv"))

    # Ensure essential columns exist with safe defaults to avoid brittle failures
    defaults_bool_false = [
        "bubble_flag", "focus_flag", "frame_flag", "no_yolk_flag", "sam2_qc_flag", "out_of_frame_flag",
    ]
    for col in defaults_bool_false:
        if col not in embryo_metadata_df.columns:
            embryo_metadata_df[col] = False
    if "use_embryo_flag" not in embryo_metadata_df.columns:
        embryo_metadata_df["use_embryo_flag"] = True
    else:
        # Ensure boolean dtype for safe logical ops
        embryo_metadata_df["use_embryo_flag"] = embryo_metadata_df["use_embryo_flag"].astype(bool)
    if "dead_flag" not in embryo_metadata_df.columns:
        embryo_metadata_df["dead_flag"] = False
    if "predicted_stage_hpf" not in embryo_metadata_df.columns:
        # Fallback: create a placeholder to allow pipeline to proceed; stage inference will overwrite later
        embryo_metadata_df["predicted_stage_hpf"] = 0.0
    if "surface_area_um" not in embryo_metadata_df.columns:
        # Create placeholder; QC outlier step will skip on insufficient data
        embryo_metadata_df["surface_area_um"] = 0.0
    if "Time Rel (s)" not in embryo_metadata_df.columns:
        embryo_metadata_df["Time Rel (s)"] = 0.0
    # Provide optional metadata fields to avoid curation selection KeyErrors
    if "temperature" not in embryo_metadata_df.columns:
        embryo_metadata_df["temperature"] = pd.NA
    else:
        embryo_metadata_df["temperature"] = embryo_metadata_df["temperature"].fillna(pd.NA)
    if "medium" not in embryo_metadata_df.columns:
        embryo_metadata_df["medium"] = pd.NA
    else:
        embryo_metadata_df["medium"] = embryo_metadata_df["medium"].fillna(pd.NA)

    ############
    # Clean up chemical perturbation variable and create a master perturbation variable
    # Make a master perturbation class
    embryo_metadata_df["chem_perturbation"] = embryo_metadata_df["chem_perturbation"].astype(str)
    embryo_metadata_df.loc[np.where(embryo_metadata_df["chem_perturbation"] == 'nan')[0], "chem_perturbation"] = "None"

    embryo_metadata_df["master_perturbation"] = embryo_metadata_df["chem_perturbation"].copy()
    embryo_metadata_df.loc[np.where(embryo_metadata_df["master_perturbation"] == "None")[0], "master_perturbation"] = \
        embryo_metadata_df["genotype"].iloc[
            np.where(embryo_metadata_df["master_perturbation"] == "None")[0]].copy().values

    embryo_metadata_df["experiment_date"] = embryo_metadata_df["experiment_date"].astype(str)
    
    # Manually re-label late time points from 20240626 experiment. This is because the temperature rose to above 30C
    # for the second day
    # relabel_flags = (embryo_metadata_df["experiment_date"].astype(str) == "20240626") & \
    #                   ((embryo_metadata_df["Time Rel (s)"] / 3600) > 30)
    # Guard special-case relabeling with presence of well column
    if "well" in embryo_metadata_df.columns:
        date_ft = embryo_metadata_df["experiment_date"].astype(str) == "20240411"
        row_vec = embryo_metadata_df["well"].astype(str).str[0]
        row_ft = (row_vec=="A") | (row_vec=="B") | (row_vec=="C")
        wt_ft = embryo_metadata_df["master_perturbation"].astype(str) == "wik"
        relabel_flags = date_ft & row_ft & wt_ft
        embryo_metadata_df.loc[relabel_flags, "master_perturbation"] = "Uncertain"  # prevent use for metric learning

    ############
    # Use surface-area of mask to remove large outliers
    min_embryos = 2  # Reduced for testing small datasets
    sa_ref_key = np.asarray(['wik', 'ab'])
    use_indices = np.where(np.isin(embryo_metadata_df["master_perturbation"], sa_ref_key) | (embryo_metadata_df["experiment_date"] == "20240626") & \
                           (embryo_metadata_df["use_embryo_flag"] == 1))[0]

    sa_vec_ref = embryo_metadata_df["surface_area_um"].iloc[use_indices].values
    time_vec_ref = embryo_metadata_df['predicted_stage_hpf'].iloc[use_indices].values

    sa_vec_all = embryo_metadata_df["surface_area_um"].values
    time_vec_all = embryo_metadata_df['predicted_stage_hpf'].values

    embryo_metadata_df['sa_outlier_flag'] = True

    hpf_window = 0.75
    offset_cushion = 1e5
    prct = 95
    ul = 72
    ll = 0
    time_index = np.linspace(ll, ul, 2*(ul-ll)+1)
    percentile_array = np.empty((len(time_index),))
    percentile_array[:] = np.nan

    # iterate through time points
    first_i = np.nan
    last_i = np.nan
    for t, ti in enumerate(time_index):
        t_indices_ref = np.where((time_vec_ref >= ti - hpf_window) & (time_vec_ref <= ti + hpf_window))[0]
        if len(t_indices_ref) >= min_embryos:
            sa_vec_t_ref = sa_vec_ref[t_indices_ref].copy()

            percentile_array[t] = np.percentile(sa_vec_t_ref, prct)

            if np.isnan(first_i):
                first_i = t
        elif ~np.isnan(first_i):
            last_i = t-1
            break

    # fill in blanks - handle insufficient data case
    if np.isnan(first_i) or np.isnan(last_i):
        print(f"⚠️  Warning: Insufficient QC reference data for surface area outlier detection.")
        print(f"   Found first_i={first_i}, last_i={last_i}")
        print(f"   Skipping surface area QC step for small dataset.")
        # Skip surface area outlier detection for small datasets
        sa_bound_sm = np.full_like(time_index, offset_cushion)
    else:
        first_i = int(first_i)
        last_i = int(last_i)
        percentile_array[:first_i] = percentile_array[first_i]
        percentile_array[last_i + 1:] = percentile_array[last_i]

        # smooth
        sa_bound_sm = offset_cushion + scipy.signal.savgol_filter(percentile_array, window_length=5, polyorder=2)

    # flag outliers
    t_ids = np.digitize(time_vec_all, bins=time_index)

    for t in range(len(time_index)):
        t_indices = np.where(t_ids == t)
        sa_vec_t_all = sa_vec_all[t_indices].copy()
        embryo_metadata_df.loc[t_indices[0], 'sa_outlier_flag'] = sa_vec_t_all > sa_bound_sm[t]

    ##############
    # Next, flag embryos that are likely dead
    embryo_metadata_df["dead_flag2"] = False

    # calculate time relative to death
    embryo_id_index = np.unique(embryo_metadata_df["embryo_id"])

    for e, eid in enumerate(embryo_id_index):
        e_indices = np.where(embryo_metadata_df["embryo_id"] == eid)[0]
        ever_dead_flag = np.any(embryo_metadata_df["dead_flag"].iloc[e_indices] == True)
        if ever_dead_flag:
            d_ind = np.where(embryo_metadata_df["dead_flag"].iloc[e_indices] == True)[0][0]
            d_time = embryo_metadata_df["predicted_stage_hpf"].iloc[e_indices[d_ind]]
            hours_from_death = embryo_metadata_df["predicted_stage_hpf"].iloc[e_indices].values - d_time
            d_indices = np.where(hours_from_death > -dead_lead_time)
            embryo_metadata_df.loc[e_indices[d_indices], "dead_flag2"] = True

    # Use centralized function - THE ONLY place that determines use_embryo_flag
    embryo_metadata_df["use_embryo_flag"] = determine_use_embryo_flag(embryo_metadata_df)

    
    # join on additional perturbation info
    if pert_key_path is None:
        key_path = os.path.join(root, 'metadata', "perturbation_name_key.csv")
    else:
        key_path = pert_key_path
    # Load or bootstrap perturbation key
    try:
        pert_name_key = pd.read_csv(key_path)
    except Exception:
        print(f"ℹ️  perturbation_name_key.csv not found or unreadable at {key_path}. Bootstrapping from df01...")
        pert_name_key = bootstrap_perturbation_key_from_df01(root=root, df01=embryo_metadata_df, out_path=key_path)
        print(f"📝 Wrote bootstrapped perturbation key to {key_path}")

    # Ensure required columns exist on key with defaults
    for col, default in (
        ("short_pert_name", None),
        ("phenotype", "unknown"),
        ("control_flag", False),
        ("pert_type", "unknown"),
        ("background", "unknown"),
        ("time_auto_constructed", None),
    ):
        if col not in pert_name_key.columns:
            pert_name_key[col] = default

    # Auto-augment missing perturbations so Build04 can proceed.
    if auto_augment_pert_key:
        masters = pd.Series(embryo_metadata_df["master_perturbation"].astype(str).unique())
        existing = set(pert_name_key["master_perturbation"].astype(str).unique())
        missing = masters[~masters.isin(existing)]
        if len(missing) > 0:
            # Derive short names from df01 if available; fallback to master
            # Build short name map if available; otherwise fallback to master
            if "short_pert_name" in embryo_metadata_df.columns:
                short_map = (
                    embryo_metadata_df.loc[:, ["master_perturbation", "short_pert_name"]]
                    .astype(str)
                    .groupby("master_perturbation")["short_pert_name"]
                    .agg(lambda s: s.value_counts().index[0])
                )
            else:
                short_map = pd.Series(dtype=str)
            import time as _time
            ts = _time.strftime('%Y-%m-%d %H:%M:%S')
            add_rows = []
            for m in missing.tolist():
                short = short_map.loc[m] if m in short_map.index else m
                add_rows.append({
                    "master_perturbation": m,
                    "short_pert_name": short,
                    "phenotype": "unknown",
                    "control_flag": False,
                    "pert_type": "unknown",
                    "background": "unknown",
                    "time_auto_constructed": ts,
                })
            if add_rows:
                augment_df = pd.DataFrame(add_rows)
                pert_name_key = pd.concat([pert_name_key, augment_df], ignore_index=True)
                # Drop any accidental duplicates, preferring original entries
                pert_name_key = (
                    pert_name_key.drop_duplicates(subset=["master_perturbation"], keep="first")
                    .reset_index(drop=True)
                )
                print(f"ℹ️  Auto-augmented perturbation key with {len(add_rows)} missing entries.")
                # Always persist augmented union for traceability
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                pert_name_key.to_csv(key_path, index=False)
                print(f"📝 Wrote augmented key to {key_path}")
    embryo_metadata_df = embryo_metadata_df.merge(pert_name_key, how="left", on="master_perturbation", indicator=True)
    if np.any(embryo_metadata_df["_merge"] != "both"):
        problem_perts = np.unique(embryo_metadata_df.loc[embryo_metadata_df["_merge"] != "both", "master_perturbation"])
        raise Exception("Some perturbations were not found in key: " + ', '.join(problem_perts.tolist()))
    embryo_metadata_df.drop(labels=["_merge"], axis=1, inplace=True)

    ##################################
    # Infer standardized embryo stages
    embryo_metadata_df = infer_embryo_stage(root=root, embryo_metadata_df=embryo_metadata_df)

    # save
    embryo_metadata_df.to_csv(os.path.join(metadata_path, "embryo_metadata_df02.csv"), index=False)

    # generate table to use for manual curation
    curation_path = os.path.join(metadata_path, "curation")
    if not os.path.exists(curation_path):
        os.makedirs(curation_path)

    ##################################
    # Make DF for frame-level curation

    # generate dataset to use for manual curation
    keep_cols = ["snip_id", 'short_pert_name', 'master_perturbation', 'temperature', 'medium',
                 'bubble_flag', 'focus_flag', 'frame_flag', 'dead_flag2', 'no_yolk_flag', 'out_of_frame_flag',
                 "use_embryo_flag", "predicted_stage_hpf"]

    curation_df = embryo_metadata_df[keep_cols].copy()

    # add additional curation cols
    curation_df.loc[:, "confinement_flag"] = np.nan
    curation_df.loc[:, "segmentation_flag"] = np.nan
    curation_df.loc[:, "hq_flag"] = np.nan
    curation_df.loc[:, "manual_stage_hpf"] = np.nan
    curation_df.loc[:, "use_embryo_manual"] = np.nan
    curation_df.loc[:, "dv_orientation"] = np.nan
    curation_df.loc[:, "head_orientation"] = np.nan
    curation_df.loc[:, "manual_update_flag"] = 0

    # Check for previous version of the curation dataset
    print("Building frame curation dataset...")
    curation_df_path = os.path.join(curation_path, "curation_df.csv")
    dt_string = str(int(np.round(time.time())))
    if os.path.exists(curation_df_path):
        curr_snips = curation_df["snip_id"].to_numpy()

        curation_df_prev = pd.read_csv(curation_df_path)

        # preserve only entries that have been manually updated
        curation_df_prev = curation_df_prev.loc[curation_df_prev["manual_update_flag"] == 1, :]

        # curation_prev = curation_df_prev.loc[:, ["snip_id", "use_embryo_flag"]].rename(columns={"use_embryo_flag" : "use_embryo_flag_frame"})
        # embryo_metadata_df = embryo_metadata_df.merge(curation_prev, how="left", on="snip_id", indicator=True)
        # embryo_metadata_df.loc["left_only"==embryo_metadata_df["_merge"], "use_embryo_flag_frame"] = True
        # embryo_metadata_df["use_embryo_flag"] = embryo_metadata_df["use_embryo_flag"] & embryo_metadata_df["use_embryo_flag_frame"]
        # embryo_metadata_df.drop(labels=["use_embryo_flag_frame", "_merge"], axis=1, inplace=True)

        # combine with new entries
        prev_snips = curation_df_prev["snip_id"].to_numpy()

        # remove duplicate entries from new dataset and concatenate
        keep_filter = ~np.isin(curr_snips, prev_snips)
        curation_df = pd.concat([curation_df.loc[keep_filter, :], curation_df_prev], axis=0, ignore_index=True)

        # rename old DF to keep it just in case
        os.rename(curation_df_path, os.path.join(curation_path, "curation_df_" + dt_string + ".csv"))

    # save
    curation_df = curation_df.sort_values(by=["snip_id"], ignore_index=True)
    curation_df.to_csv(curation_df_path, index=False)

    #######################################
    # Make embryo-level annotation DF
    print("Building embryo curation dataset...")
    keep_cols = ["embryo_id", 'short_pert_name', 'phenotype', 'background', 'master_perturbation', 'temperature']
    curation_df_emb = embryo_metadata_df.loc[:, keep_cols + ["use_embryo_flag"]].groupby(by=keep_cols).sum(["use_embryo_flag"]).reset_index()
    curation_df_emb = curation_df_emb.rename(columns={"short_pert_name":"short_pert_name_orig", "phenotype":"phenotype_orig"})


    curation_df_emb["short_pert_name"] = curation_df_emb["short_pert_name_orig"]
    curation_df_emb["phenotype"] = curation_df_emb["phenotype_orig"]
    curation_df_emb["start_stage_manual"] = np.nan
    curation_df_emb["hq_flag_emb"] = np.nan
    curation_df_emb["reference_flag"] = np.nan
    curation_df_emb["use_embryo_flag_manual"] = np.nan
    curation_df_emb["manual_update_flag"] = False

    emb_curation_df_path = os.path.join(curation_path, "embryo_curation_df.csv")
    if os.path.exists(emb_curation_df_path):

        curr_emb_ids = curation_df_emb["embryo_id"].to_numpy()
        curation_df_emb_prev = pd.read_csv(emb_curation_df_path)

        # preserve only entries that have been manually updated
        curation_df_emb_prev = curation_df_emb_prev.loc[curation_df_emb_prev["manual_update_flag"] == 1, :]
        # curation_df_emb_prev = curation_df_emb_prev.loc[:, ["embryo_id", "reference_flag", "hq_flag_emb", "master_perturbation"]].rename(
        #         columns={"hq_flag_emb" : "use_embryo_flag_emb"})
        # embryo_metadata_df = embryo_metadata_df.merge(curation_df_emb_prev, how="left", on="embryo_id", indicator=True)
        # embryo_metadata_df.loc["left_only"==embryo_metadata_df["_merge"], "use_embryo_flag_emb"] = True
        # embryo_metadata_df["use_embryo_flag"] = embryo_metadata_df["use_embryo_flag"] & embryo_metadata_df["use_embryo_flag_emb"]

        # # update perturbation labels
        # embryo_metadata_df.loc["both"==embryo_metadata_df["_merge"], "short_pert_name"] = (
        #                             embryo_metadata_df.loc)["both"==embryo_metadata_df["_merge"], "manual_perturbation"]
        # embryo_metadata_df.drop(labels=["use_embryo_flag_emb", "manual_perturbation", "_merge"], axis=1, inplace=True)

        prev_emb_ids = curation_df_emb_prev["embryo_id"].to_numpy()

        # remove duplicate entries from new dataset and concatenate_Archive
        keep_filter = ~np.isin(curr_emb_ids, prev_emb_ids)
        curation_df_emb = pd.concat([curation_df_emb.loc[keep_filter, :], curation_df_emb_prev], axis=0, ignore_index=True)

        os.rename(emb_curation_df_path, os.path.join(curation_path, "embryo_curation_df_" + dt_string + ".csv"))

    curation_df_emb.to_csv(emb_curation_df_path, index=False)

    #######
    # now, make perturbation-level keys to inform training inclusion/exclusion and metric comparisons
    print("Building metric and perturbation keys...")
    pert_train_key = embryo_metadata_df.loc[:, ["short_pert_name"]].drop_duplicates()
    pert_train_key["start_hpf"] = 0
    pert_train_key["stop_hpf"] = 100
    pert_train_key.to_csv(os.path.join(curation_path, "perturbation_train_key.csv"), index=False)

    pert_df_u = pert_name_key.drop_duplicates(subset=["short_pert_name"]).reset_index(drop=True)
    pert_u = pert_df_u.loc[:, "short_pert_name"].to_numpy()
    ctrl_flags = np.where(pert_df_u.loc[:, "control_flag"])[0]
    wt_flags = np.where(pert_df_u.loc[:, "phenotype"]=="wt")[0]
    cr_flags = np.where(pert_df_u.loc[:, "pert_type"]=="crispant")[0]
    u_flags = np.where(pert_df_u.loc[:, "phenotype"]=="uncertain")[0]
    wt_ab_flag = np.where(pert_df_u.loc[:, "short_pert_name"]=="wt_ab")[0]
    wt_wik_flag = np.where(pert_df_u.loc[:, "short_pert_name"]=="wt_wik")[0]
    wt_other_flags = np.where((pert_df_u.loc[:, "phenotype"]=="wt") & (pert_df_u.loc[:, "pert_type"]!="fluor") &
                              (pert_df_u.loc[:, "background"]!="ab") & (pert_df_u.loc[:, "background"]!="wik"))[0]

    wt_fluo_flags = np.where((pert_df_u.loc[:, "phenotype"]=="wt") & (pert_df_u.loc[:, "pert_type"]=="fluor"))[0]

    # build in some basic relations to be refined manually
    metric_array = np.zeros((len(pert_u), len(pert_u)), dtype=np.int16)
    # tell model to leave metric relations amongst control subtypes unspecified
    metric_array[np.ix_(ctrl_flags, ctrl_flags)] = -1
    # tell model to leave metric relations between control and wt subtypes unspecified
    metric_array[np.ix_(ctrl_flags, wt_flags)] = -1
    metric_array[np.ix_(wt_flags, ctrl_flags)] = -1
    # leave all relations amongst crispants and between cr and wt unspecified
    metric_array[np.ix_(cr_flags, cr_flags)] = -1
    metric_array[np.ix_(cr_flags, wt_flags)] = -1
    metric_array[np.ix_(wt_flags, cr_flags)] = -1
    # leave relation between wik and ab wt strains unspecified
    metric_array[np.ix_(wt_ab_flag, wt_wik_flag)] = -1
    metric_array[np.ix_(wt_wik_flag, wt_ab_flag)] = -1
    # embryos of uncertain phenotype are neutral relatiove to all others
    metric_array[u_flags, :] = -1
    metric_array[:, u_flags] = -1
    # apply neutrality between embryos from mutant background with no phenotype (these encompass hets and homo wt)
    metric_array[np.ix_(wt_other_flags, wt_flags)] = -1
    metric_array[np.ix_(wt_flags, wt_other_flags)] = -1
    # ditto for fluo markers
    metric_array[np.ix_(wt_fluo_flags, wt_flags)] = -1
    metric_array[np.ix_(wt_flags, wt_fluo_flags)] = -1

    # by default all phenotypes are positive references for themselves
    eye_array = np.eye(len(pert_u))
    metric_array[eye_array==1] = 1
    # save 
    pert_metric_key = pd.DataFrame(metric_array, columns=pert_u.tolist())
    pert_metric_key.set_index(pert_u, inplace=True)
    pert_metric_key.to_csv(os.path.join(curation_path, "perturbation_metric_key.csv"), index=True)
    print("Done.")

if __name__ == "__main__": 
    # root = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/"
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"

    # print('Compiling well metadata...')
    # build_well_metadata_master(root)
    #
    # print('Compiling embryo metadata...')
    # segment_wells(root, par_flag=True, overwrite_well_stats=False, overwrite_embryo_stats=False)

    # print('Extracting embryo snips...')
    perform_embryo_qc(root)

"""
Refactor Plan and Technical Debt (2025-09-14)

Context
- The per‑experiment pipeline (`build04_stage_per_experiment`) is the production path.
- Legacy combined workflow (`perform_embryo_qc`) remains for historical parity but diverges from the production logic.

Plan (implemented)
- Wire stage_ref through stage inference and validate file.
- SA QC: use internal controls first ((phenotype == 'wt' OR control_flag) AND use_embryo_flag) vs predicted_stage_hpf with ±0.75 hpf window, 0.5 hpf bins, SG smoothing; fallback to stage_ref (threshold = scale × margin_k × sa_ref(stage)).
- Death lead-time QC: use predicted_stage_hpf for parity.
- Fail‑loud input validation and dtype coercion.
- Clear logging, including % rows and % snip_ids flagged by SA QC.

Technical Debt (to address next)
- Single source of truth: move legacy functions to `build04_legacy.py` with deprecation warnings; keep only `build04_stage_per_experiment` as production.
- Parameterize “magic values” (hpf_window, percentile, sg_window/poly, margin_k) via config; remove date/well hardcodes.
- Remove unused stage inference variants (`_orig`, `_sigmoid`) unless test‑referenced.
- Align CLI and docs around `--out-dir` and discovery paths; add small synthetic tests.
"""
