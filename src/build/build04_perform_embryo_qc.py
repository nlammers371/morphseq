import os
import pandas as pd
import scipy
import numpy as np
import time
from tqdm import tqdm
import statsmodels.api as sm


def infer_embryo_stage_orig(embryo_metadata_df, ref_date="20240626"):

    # build the reference set
    stage_df = embryo_metadata_df.loc[embryo_metadata_df["experiment_date"]==ref_date, 
                    ["snip_id", "embryo_id", "short_pert_name", "phenotype", "control_flag", "predicted_stage_hpf",
                     "surface_area_um", "use_embryo_flag"]].reset_index(drop=True)
    ref_bool = (stage_df.loc[:, "phenotype"].to_numpy() == "wt") | (stage_df.loc[:, "control_flag"].to_numpy() == 1)
    ref_bool = ref_bool | (stage_df.loc[:, "phenotype"].to_numpy() == "uncertain")
    ref_bool = ref_bool & stage_df["use_embryo_flag"]
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
        date_df = embryo_metadata_df.loc[date_indices==d, ["snip_id", "embryo_id", "time_int","short_pert_name", 
                        "phenotype", "control_flag", "predicted_stage_hpf", "surface_area_um", "use_embryo_flag"]].reset_index(drop=True)

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
        if snapshot_flag:
            embryo_metadata_df["use_embryo_flag"] = True
        # calculate length percentiles
        ref_bool = (date_df.loc[:, "phenotype"].to_numpy() == "wt") | (date_df.loc[:, "control_flag"].to_numpy() == 1)
        if date == "20240314":   # special allowance for this one dataset
            ref_bool = ref_bool | True
        ref_bool = ref_bool & date_df["use_embryo_flag"]

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
        date_df = embryo_metadata_df.loc[date_indices==d, ["snip_id", "embryo_id", "time_int", "Time Rel (s)", "short_pert_name",
                        "phenotype", "control_flag", "predicted_stage_hpf", "surface_area_um", "use_embryo_flag"]].reset_index(drop=True)

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
        if snapshot_flag:
            embryo_metadata_df["use_embryo_flag"] = True
        # calculate length percentiles
        ref_bool = (date_df.loc[:, "phenotype"].to_numpy() == "wt") | (date_df.loc[:, "control_flag"].to_numpy() == 1)
        if date == "20240314":   # special allowance for this one dataset
            ref_bool = ref_bool | True
        ref_bool = ref_bool & date_df["use_embryo_flag"]

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

def infer_embryo_stage(root, embryo_metadata_df):

    # load ref dataset
    stage_key_df = pd.read_csv(os.path.join(root, "metadata", "stage_ref_df.csv"))
    # stage_key_df = stage_key_df.loc[stage_key_df["stage_hpf"] <= 72] # not reliable after this point

    # get interpolator
    stage_interpolator = scipy.interpolate.interp1d(stage_key_df["sa_um"], stage_key_df["stage_hpf"],
                                    kind="linear", fill_value=np.nan, bounds_error=False)
    # iterate through dates
    date_index, date_indices = np.unique(embryo_metadata_df["experiment_date"], return_inverse=True)

    # initialize new field
    embryo_metadata_df["inferred_stage_hpf"] = np.nan

    for d, date in enumerate(tqdm(date_index)):
        date_df = embryo_metadata_df.loc[date_indices == d, ["snip_id", "embryo_id", "time_int", "Time Rel (s)", "short_pert_name",
                        "phenotype", "control_flag", "predicted_stage_hpf", "surface_area_um", "use_embryo_flag"]].reset_index(drop=True)

        date_df["abs_time_hr"] = date_df["Time Rel (s)"] / 3600
        # check for multiple age cohorts
        min_t = np.min(date_df["time_int"])
        cohort_key = date_df.loc[date_df["time_int"] == min_t, ["embryo_id", "predicted_stage_hpf"]]
        _, age_cohort = np.unique(np.round(cohort_key["predicted_stage_hpf"] / 5) * 5, return_inverse=True)
        cohort_key["cohort_id"] = age_cohort

        # join onto main df
        date_df = date_df.merge(cohort_key.loc[:, ["embryo_id", "cohort_id"]], how="left", on="embryo_id")

        # check to see if this is a timeseries dataset
        _, embryo_counts = np.unique(date_df["embryo_id"], return_counts=True)
        snapshot_flag = np.max(embryo_counts) == 1
        if snapshot_flag:
            embryo_metadata_df.loc[date_indices == d, "use_embryo_flag"] = True

        # calculate length percentiles
        ref_bool = (date_df.loc[:, "phenotype"].to_numpy() == "wt") | (date_df.loc[:, "control_flag"].to_numpy() == 1)
        if date == "20240314":   # special allowance for this one dataset
            ref_bool = ref_bool | True

        # date_df["abs_time_hpf"] = np.round(date_df["predicted_stage_hpf"])

        ref_bool = ref_bool & date_df["use_embryo_flag"]
        date_df_ref = date_df.loc[ref_bool]
        # date_df["length_um"] = date_df["length_um"]*1.5
        date_df_ref["stage_group_hpf"] = np.round(date_df_ref["predicted_stage_hpf"])   # ["predicted_stage_hpf"])
        date_key_df = date_df_ref.loc[:, ["stage_group_hpf", "cohort_id", "surface_area_um"]].groupby(
                                                        ['stage_group_hpf', "cohort_id"]).quantile(.95).reset_index()

        # get interp predictions
        date_key_df["stage_hpf_interp"] = stage_interpolator(date_key_df["surface_area_um"])

        if snapshot_flag:
            date_df["inferred_stage_hpf"] = stage_interpolator(date_df["predicted_stage_hpf"])
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
                ub = (4, 72, 2.0, 0.005)  #, 0.005)
                lb = (0, -72, 0.5, -0.005)  #, -0.005)
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

        embryo_metadata_df.loc[date_indices == d, "inferred_stage_hpf"] = date_df["inferred_stage_hpf"].to_numpy()

    return embryo_metadata_df


def perform_embryo_qc(root, dead_lead_time=2):

    # read in metadata
    metadata_path = os.path.join(root, 'metadata', "combined_metadata_files", '')
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df01.csv"), index_col=0)

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
    relabel_flags = (embryo_metadata_df["experiment_date"].astype(str) == "20240626") & \
                      ((embryo_metadata_df["Time Rel (s)"] / 3600) > 30)
    embryo_metadata_df.loc[relabel_flags, "master_perturbation"] = "Uncertain"  # this label just prevents these time points from being used for metric learning

    ############
    # Use surface-area of mask to remove large outliers
    min_embryos = 10
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

    # fill in blanks
    percentile_array[:first_i] = percentile_array[first_i]
    percentile_array[last_i + 1:] = percentile_array[last_i]

    # smooth
    sa_bound_sm = offset_cushion + scipy.signal.savgol_filter(percentile_array, window_length=5, polyorder=2)

    # flag outliers
    t_ids = np.digitize(time_vec_all, bins=time_index)

    for t in range(len(time_index) - 1):
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

    # Update use_embryo_flag
    embryo_metadata_df.loc[:, "use_embryo_flag"] = embryo_metadata_df.loc[:, "use_embryo_flag"] & \
                                                   (~embryo_metadata_df.loc[:, "dead_flag2"]) & \
                                                   (~embryo_metadata_df.loc[:, "sa_outlier_flag"])

    
    # join on additional perturbation info
    pert_name_key = pd.read_csv(os.path.join(root, 'metadata', "perturbation_name_key.csv"))
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
                 'bubble_flag', 'focus_flag', 'frame_flag', 'dead_flag2', 'no_yolk_flag', #'out_of_frame_flag',
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