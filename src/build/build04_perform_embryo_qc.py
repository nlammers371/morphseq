import os
import pandas as pd
import scipy
import numpy as np

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
    for t, time in enumerate(time_index):
        t_indices_ref = np.where((time_vec_ref >= time - hpf_window) & (time_vec_ref <= time + hpf_window))[0]
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
    embryo_metadata_df.to_csv(os.path.join(metadata_path, "embryo_metadata_df_final.csv"))


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