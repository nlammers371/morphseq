import sys
import site

user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob2 as glob
import patsy
from scipy.interpolate import interp1d
from scipy.special import factorial
from scipy.stats import poisson
from scipy.optimize import minimize
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map
import time


def run_inference(embryo_ind, ccs_df, meta_df, cov_factors, cov_col_list, spline_lookup_df, PHI, THETA, maxiter):

    # get embryo info
    embryo_id = ccs_df.index[embryo_ind]
    raw_counts = ccs_df.loc[embryo_id, :].to_numpy()
    # dis = meta_df.loc[embryo_id, "dis_protocol"]
    # expt = meta_df.loc[embryo_id, "expt"]
    cov_dict = dict({cov: meta_df.loc[embryo_id, cov] for cov in cov_factors})
    stage = meta_df.loc[embryo_id, "timepoint"]
    size_factor_log = np.log(meta_df.loc[embryo_id, "Size_Factor"])

    # construct initial covariate vec
    X0 = construct_X(20, cov_dict, cov_col_list=cov_col_list, spline_lookup_df=spline_lookup_df)

    def call_logL(params, raw_counts=raw_counts, offset=size_factor_log, X0=X0, THETA=THETA, PHI=PHI, spline_lookup_df=spline_lookup_df,
                  cov_cols=cov_col_list):
        spline_cols = [col for col in cov_cols if "ns(" in col]
        loss = calculate_PLN_logL(params, raw_counts, offset, X0, THETA, PHI, spline_lookup_df=spline_lookup_df, spline_cols=spline_cols)
        return loss

    # initialize time
    dt = np.random.normal(loc=0, scale=5)
    t0 = np.min([np.max([stage + dt, 6]), 96])
    zi0_vec = [0] * PHI.shape[0]
    params_init = np.asarray([t0] + zi0_vec)
    result = minimize(call_logL, params_init, args=(raw_counts,),  options=dict({"maxiter": maxiter}))
    # result = call_logL(params_init)


    t_hat = result.x[0]
    z_hat = result.x[1:]

    standard_errors = np.sqrt(np.diag(result.hess_inv))
    t_sig = standard_errors[0]
    z_sig = standard_errors[1:]

    # calculate latent projection (removing experiment effects and shift to bead reference
    null_dict = cov_dict
    for key in list(null_dict.keys()):
        null_dict[key] = "NA"

    X = construct_X(timepoint=t_hat, cov_dict=null_dict, cov_col_list=cov_col_list, spline_lookup_df=spline_lookup_df)
    mu = (X @ THETA.T).to_numpy().ravel()
    latents = mu + z_hat

    # save output
    latent_df = pd.DataFrame(latents[None, :], index=[embryo_id], columns=ccs_df.columns)
    latent_se_df = pd.DataFrame(z_sig[None, :], index=[embryo_id], columns=ccs_df.columns)
    time_df = pd.DataFrame(np.c_[t_hat, t_sig], index=[embryo_id], columns=["pseudostage", "pseudostage_se"])

    return time_df, latent_df, latent_se_df

# function to generate time-related covariates using spline lookup table exported from R
def get_spline_basis(new_time_vec, spline_lookup_df):
    # Create an empty dictionary to hold the interpolated values.
    out_df = pd.DataFrame(new_time_vec, columns=["timepoint"])

    # Loop through each spline column (skip the "timepoint" column).
    for col in spline_lookup_df.columns[1:]:
        # Create an interpolation function for this column.
        f_interp = interp1d(spline_lookup_df["timepoint"], spline_lookup_df[col],
                            kind='linear', fill_value="extrapolate")
        # Evaluate the interpolation at the new time value.
        out_df[col] = f_interp(new_time_vec)
    return out_df

def update_spline_cols(X, query_times, spline_lookup_df, spline_cols):
    spline_df = get_spline_basis(query_times, spline_lookup_df)
    spline_vals = spline_df.iloc[:, 1:].to_numpy()
    spline_cols_split = [col.split(":") for col in spline_cols]
    for c, col in enumerate(spline_cols):
        col_s = spline_cols_split[c]
        spline_ind = int(col_s[0][-1]) - 1
        if len(col_s) == 1:
            X.loc[:, col] = spline_vals[:, spline_ind]
        else:
            sv = spline_vals[:, spline_ind][:, None]
            iv = X.loc[:, col_s[1]].to_numpy()[:, None]
            X.loc[:, col] = np.multiply(sv, iv)

    return X, spline_vals

def construct_X(timepoint, cov_dict, cov_col_list, spline_lookup_df):

    spline_cols = [col for col in cov_col_list if "ns(" in col]

    cov_keys = list(cov_dict.keys())

    X = pd.DataFrame(np.zeros((1, len(cov_col_list))), columns=cov_col_list)
    X["Intercept"] = 1.0
    for key in cov_keys:
        val = cov_dict[key]
        vi = [i for i in range(len(cov_col_list)) if val in cov_col_list[i]]
        if len(vi) > 0:
            # print(cov_col_list[vi[0]])
            X.iloc[0, vi[0]] = 1.0

    # expt_i = [i for i in range(len(cov_col_list)) if expt in cov_col_list[i]]
    # if len(expt_i) > 0:
    #     X[cov_col_list[expt_i[0]]] = 1.0

    X, _ = update_spline_cols(X, np.asarray([timepoint]), spline_lookup_df, spline_cols)

    return X


def update_X(X, timepoint, spline_lookup_df, spline_cols):

    X, _ = update_spline_cols(X, np.asarray([timepoint]), spline_lookup_df, spline_cols)

    return X

# define key helper functions
def calc_zi(log_lambda, theta_array, X):
    Zi = log_lambda - np.matmul(X, theta_array.T)
    return Zi.to_numpy()[0]

def calc_logL_gauss(PHI, Zi): # note that the leaves out the normalization factor. If this is slow, consider simplifying for diagonal Cov
    logL = -0.5 * (Zi[None, :] @ PHI @ Zi[:, None])
    return logL[0][0]

def calc_logL_poiss(raw_counts, log_lambda, log_offset):
    # logL = np.sum(np.multiply(raw_counts, log_lambda) - np.exp(log_lambda) - factorial(raw_counts))
    log_pmf = poisson.logpmf(raw_counts, np.exp(log_lambda + log_offset))
    return np.sum(log_pmf)

def calculate_PLN_logL(params, raw_counts, log_offset, X0, THETA, PHI, spline_lookup_df, spline_cols):

    # extract params
    t = params[0]
    Zi = params[1:]

    # generate updated covariate matrix
    X = update_X(X0, t, spline_lookup_df, spline_cols)
    mu = (X @ THETA.T).to_numpy().ravel()
    # L, Zi = calculate_L(X, THETA, COV, raw_counts)

    # calculate Zi and gaussian logL
    # Zi = (L - mu)
    logL_g = calc_logL_gauss(PHI, Zi)

    # caculate Poisson logL
    L = Zi + mu
    logL_p = calc_logL_poiss(raw_counts, L, log_offset)
    if np.isnan(logL_p) or np.isinf(logL_p):
        print("wtf")

    return -(logL_g + logL_p) / len(raw_counts)

def do_latent_projections(root, model_name, max_threads=5, maxiter=300):
    # set data path and model name parameter
    fig_folder = os.path.join(root, "figures/seq_data/PLN/", model_name, "")
    os.makedirs(fig_folder, exist_ok=True)

    hooke_data_path = os.path.join(root, "seq_data/emb_projections/hooke_model_files", "")
    ccs_data_path = os.path.join(root, "seq_data/emb_projections/ccs_data_cell_type_broad", "")
    model_path = os.path.join(hooke_data_path, model_name, "")

    # make save dir
    out_dir = os.path.join(root, "seq_data", "emb_projections", "latent_projections", model_name, "")
    os.makedirs(out_dir, exist_ok=True)

    # load in model parameters
    # load full counts dataset
    hooke_counts_long = pd.read_csv(model_path + "abundance_estimates.csv", index_col=0)
    cols = list(hooke_counts_long.columns)
    cell_ind = cols.index("cell_group")
    cov_cols = cols[:cell_ind]
    hooke_counts_df = hooke_counts_long.pivot(index=cov_cols,
                                               columns=["cell_group"], values = ["log_abund"])
    hooke_counts_df.columns = ['_'.join(map(str, col)).strip('_') for col in hooke_counts_df.columns.values]
    hooke_counts_df.reset_index(inplace=True)
    new_cols = [col.replace("log_abund_", "") for col in hooke_counts_df.columns.values]
    hooke_counts_df.columns = new_cols
    sort_cols = new_cols[:cell_ind] + sorted(new_cols[cell_ind:], key=str.lower)
    hooke_counts_df = hooke_counts_df.loc[:, sort_cols]

    # make stripped-down metadata df
    meta_df = hooke_counts_df[cov_cols].copy()
    meta_df.loc[:, "dummy_response"] = 0

    # load hooke predictions (for comparison purposes)
    # latent_df = pd.read_csv(model_path + "latents.csv", index_col=0)
    time_splines = pd.read_csv(model_path + "time_splines.csv")

    # load hooke model files
    cov_array = pd.read_csv(model_path + "COV.csv", index_col=0)
    beta_array = pd.read_csv(model_path + "B.csv", index_col=0).T

    # latent_df.head()
    beta_array = beta_array.rename(columns={"(Intercept)":"Intercept"})
    cols_from = beta_array.columns
    cols_from_clean = [col.replace(" = c", "=") for col in cols_from]
    beta_array.columns = cols_from_clean
    beta_array.head()

    # model formula
    with open(model_path + "model_string.txt", "r") as file:
        formula_str = file.read().strip()
    model_desc = patsy.ModelDesc.from_formula(formula_str)
    # Extract covariate names from the right-hand side terms.
    cov_factors = []
    for term in model_desc.rhs_termlist:
        for factor in term.factors:
            # factor is a EvalFactor, convert it to string.
            cov_factors.append(str(factor).replace("EvalFactor('","").replace("')",""))
    cov_factors = np.unique([cov for cov in cov_factors if "ns(" not in cov]).tolist()

    # load in full counts table and metadata used for model inference
    mdl_counts_df = pd.read_csv(model_path + "mdl_counts_table.csv", index_col=0).T
    mdl_meta_df = pd.read_csv(model_path + "mdl_embryo_metadata.csv", index_col=0)

    ####################
    # load in ccs table

    # get list of all ccs tables
    count_suffix = "_counts_table.csv"
    meta_suffix = "_metadata.csv"

    ccs_path_list = sorted(glob.glob(ccs_data_path + "*" + count_suffix))
    ccs_name_list = [os.path.basename(p).replace(count_suffix, "") for p in ccs_path_list]

    # compile master count and metadata tables
    mdl_cell_types = mdl_counts_df.columns
    ccs_df_list = []
    meta_df_list = []
    for ccs_name in tqdm(ccs_name_list):
        ccs_temp = pd.read_csv(ccs_data_path + ccs_name + count_suffix, index_col=0).T
        ccs_temp = ccs_temp.reindex(columns=mdl_cell_types, fill_value=0)
        ccs_df_list.append(ccs_temp)
        meta_temp = pd.read_csv(ccs_data_path + ccs_name + meta_suffix, index_col=0)
        meta_df_list.append(meta_temp)

    # concatenate
    ccs_df = pd.concat(ccs_df_list, axis=0).drop_duplicates()
    meta_df = pd.concat(meta_df_list, axis=0).drop_duplicates()

    ccs_df = ccs_df.loc[~ccs_df.index.duplicated(keep='first')]
    meta_df = meta_df.loc[~meta_df.index.duplicated(keep='first')].set_index("sample")

    meta_df["pert_collapsed"] = meta_df["perturbation"].copy()
    conv_list = np.asarray(["ctrl-uninj", "reference", "novehicle"])
    meta_df.loc[np.isin(meta_df["pert_collapsed"], conv_list), "pert_collapsed"] = "ctrl"

    # keep only embryos from experiments that were included in model inference
    exp_vec = mdl_meta_df.loc[:, "expt"].unique()
    exp_filter = np.isin(meta_df["expt"], exp_vec)
    meta_df = meta_df.loc[exp_filter, :]
    ccs_df = ccs_df.loc[exp_filter, :]

    # augment ccs table to incorporate missing cell types
    # mdl_cell_types = mdl_counts_df.columns
    # ccs_df = ccs_df.reindex(columns=mdl_cell_types, fill_value=0)

    # check which ones were included in inference
    mdl_flags = np.isin(np.asarray(ccs_df.index),np.asarray(mdl_counts_df.index))
    meta_df["inference_flag"] = mdl_flags

    # flag experiments that were not included in inference
    mdl_experiments = np.unique(mdl_meta_df["expt"])
    oos_vec = ~np.isin(meta_df["expt"], mdl_experiments)
    meta_df["oos_expt_flag"] = oos_vec

    ####
    # model parameters
    ####

    # inverse cov matrix
    PHI = np.linalg.inv(cov_array)
    # COV = cov_array.to_numpy()

    # regression vars
    THETA = beta_array.copy()

    # zi0_vec = [0] * COV.shape[0]

    # covariates
    cov_col_list = beta_array.columns.tolist()


    # Then create a partially applied function that fixes the shared variables:
    run_inf_shared = partial(run_inference,
                             ccs_df=ccs_df,
                             meta_df=meta_df,
                             cov_factors=cov_factors,
                             cov_col_list=cov_col_list,
                             spline_lookup_df=time_splines, THETA=THETA, PHI=PHI, maxiter=maxiter)

    # print("Running inference")
    # run_inf_shared(10)
    # for i in tqdm(range(ccs_df.shape[0])):
    #     run_inf_shared(i)

    results = process_map(run_inf_shared, range(ccs_df.shape[0]), max_workers=max_threads,
                          chunksize=3, desc="Running Inference")

    time_dfs, latent_dfs, latent_se_dfs = zip(*results)

    # Concatenate each list of DataFrames along rows
    time_df = pd.concat(time_dfs)
    latent_df = pd.concat(latent_dfs)
    latent_se_df = pd.concat(latent_se_dfs)

    # save latent info
    latent_df.to_csv(out_dir + "latent_projections.csv")
    latent_se_df.to_csv(out_dir + "latent_projections_se.csv")

    # add time info
    time_df = time_df.merge(meta_df.loc[:, ["timepoint", "mean_nn_time",  "expt", "dis_protocol", "inference_flag", "oos_expt_flag", "temp"]],
                            how="left", left_index=True, right_index=True)
    time_df.to_csv(out_dir + "time_predictions.csv")

    # also save combined counts and metadat
    meta_df.to_csv(out_dir + "combined_metadata.csv")
    ccs_df.to_csv(out_dir + "combined_counts.csv")


    return time_df, latent_df, latent_se_df

if __name__ == "__main__":
    root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/"
    model_name = "enz_expt_linear"
    max_threads = 40
    time_df, latent_df, latent_se_df = do_latent_projections(root=root, model_name=model_name, max_threads=max_threads)
