import os
import pandas as pd
import scipy
import numpy as np
import time

def perform_embryo_qc(root, dead_lead_time=2):

    # read in metadata
    metadata_path = os.path.join(root, 'metadata', '')
    embryo_metadata_df = pd.read_csv(os.path.join(metadata_path, "embryo_metadata_df.csv"), index_col=0)

    ############
    # Clean up chemical perturbation variable and create a master perturbation variable
    # Make a master perturbation class
    embryo_metadata_df["chem_perturbation"] = embryo_metadata_df["chem_perturbation"].astype(str)
    embryo_metadata_df.loc[np.where(embryo_metadata_df["chem_perturbation"] == 'nan')[0], "chem_perturbation"] = "None"

    embryo_metadata_df["master_perturbation"] = embryo_metadata_df["chem_perturbation"].copy()
    embryo_metadata_df.loc[np.where(embryo_metadata_df["master_perturbation"] == "None")[0], "master_perturbation"] = \
        embryo_metadata_df["genotype"].iloc[
            np.where(embryo_metadata_df["master_perturbation"] == "None")[0]].copy().values
    
    ############
    # Use surface-area of mask to remove large outliers
    min_embryos = 10
    sa_ref_key = 'wik'
    use_indices = np.where((embryo_metadata_df["master_perturbation"] == sa_ref_key) &\
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

    # save
    embryo_metadata_df.to_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"), index=False)

    # generate table to use for manual curation
    curation_path = os.path.join(metadata_path, "curation")
    if not os.path.exists(curation_path):
        os.makedirs(curation_path)

    ##################################
    # Make DF for frame-level curation

    # generate dataset to use for manual curation
    keep_cols = ["snip_id", 'master_perturbation', 'temperature', 'medium',
                 'bubble_flag', 'focus_flag', 'frame_flag', 'dead_flag2', 'no_yolk_flag', 'out_of_frame_flag',
                 "use_embryo_flag", "predicted_stage_hpf"]

    curation_df = embryo_metadata_df[keep_cols].copy()
    # add additional curation cols
    curation_df.loc[:, "confinement_flag"] = False
    curation_df.loc[:, "segmentation_flag"] = False
    curation_df.loc[:, "hq_flag"] = curation_df.loc[:, "use_embryo_flag"].copy()
    curation_df.loc[:, "manual_stage_hpf"] = np.nan
    curation_df.loc[:, "dv_orientation"] = np.nan
    curation_df.loc[:, "head_orientation"] = np.nan
    curation_df.loc[:, "manual_update_flag"] = 0

    # Check for previous version of the curation dataset
    curation_df_path = os.path.join(curation_path, "curation_df.csv")
    if os.path.exists(curation_df_path):
        curr_snips = curation_df["snip_id"].to_numpy()

        curation_df_prev = pd.read_csv(curation_df_path)
        # preserve only entries that have been manually updated
        curation_df_prev = curation_df_prev.loc[curation_df_prev["manual_update_flag"] == 1, :]
        prev_snips = curation_df_prev["snip_id"].to_numpy()

        # remove duplicate entries from new dataset and concatenate
        keep_filter =~ np.isin(curr_snips, prev_snips)
        curation_df = pd.concat([curation_df.loc[keep_filter, :], curation_df_prev], axis=0, ignore_index=True)

        # rename old DF to keep it just in case
        dt_string = str(np.round(time.time()))
        os.rename(curation_df_path, os.path.join(curation_path, "curation_df_" + dt_string + ".csv"))

    # save
    curation_df = curation_df.sort_values(by=["snip_id"], ignore_index=True)
    curation_df.to_csv(curation_df_path, index=False)

    #######################################
    # Make embryo-level annotation DF
    keep_cols = ["embryo_id", 'master_perturbation', 'temperature']
    curation_df_emb = embryo_metadata_df[keep_cols].drop_duplicates().reset_index(drop=True)

    curation_df_emb["start_stage_manual"] = np.nan
    curation_df_emb["hq_flag_emb"] = True
    curation_df_emb["reference_flag"] = False
    curation_df_emb["manual_update_flag"] = False

    emb_curation_df_path = os.path.join(curation_path, "embryo_curation_df.csv")
    if os.path.exists(emb_curation_df_path):
        curr_emb_ids = curation_df_emb["embryo_id"].to_numpy()
        curation_df_emb_prev = pd.read_csv(emb_curation_df_path)
        # preserve only entries that have been manually updated
        curation_df_emb_prev = curation_df_emb_prev.loc[curation_df_emb_prev["manual_update_flag"] == 1, :]
        prev_emb_ids = curation_df_emb_prev["embryo_id"].to_numpy()

        # remove duplicate entries from new dataset and concatenate
        keep_filter = ~np.isin(curr_emb_ids, prev_emb_ids)
        curation_df_emb = pd.concat([curation_df_emb.loc[keep_filter, :], curation_df_emb_prev], axis=0, ignore_index=True)

        os.rename(curation_df_path, os.path.join(curation_path, "embryo_curation_df_" + dt_string + ".csv"))

    curation_df_emb.to_csv(emb_curation_df_path, index=False)

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