# from src.functions.dataset_utils import *
import torch
from PIL.ImImagePlugin import split
from torch.utils.data.sampler import SubsetRandomSampler
from src.run.run_utils import initialize_model, parse_model_paths
from torchvision.utils import save_image
from pytorch_lightning import Trainer
import umap.umap_ as umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from omegaconf import OmegaConf
import pickle
from src.lightning.pl_wrappers import LitModel
from torch.utils.data import DataLoader
from src.data.dataset_configs import BaseDataConfig
from pathlib import Path
from typing import Dict, List, Tuple
torch.set_float32_matmul_precision("medium")   # good default
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as EA
import plotly.express as px


def assess_vae_results(cfg,
                       n_image_figures=50,
                       overwrite_flag=False,
                       skip_figures_flag=False,
                       batch_size=64,
                       version=None,
                       ckpt=None):

    if isinstance(cfg, str):
        # load config file
        config = OmegaConf.load(cfg)
    elif isinstance(cfg, dict):
        config = cfg
    else:
        raise Exception("cfg argument dtype is not recognized")

    # initialize
    model, model_config, train_data_config, loss_fn, train_config = initialize_model(config)

    model_dir, latest_ckpt = parse_model_paths(model_config, train_config, train_data_config, version, ckpt)

    # load train/test/eval indices
    split_path = os.path.join(model_dir, "split_indices.pkl")
    with open(split_path, 'rb') as file:
        split_dict = pickle.load(file)

    # initialize new data config for evaluation
    eval_data_config = BaseDataConfig(train_indices=np.asarray(split_dict["train"]),
                                      test_indices=np.asarray(split_dict["test"]),
                                      eval_indices=np.asarray(split_dict["eval"]),
                                      root=train_data_config.root,
                                      return_sample_names=True,
                                      transform_name="basic")
    eval_data_config.make_metadata()

    # load model
    lit_model = LitModel.load_from_checkpoint(latest_ckpt,
                                          model=model,
                                          loss_fn=loss_fn,
                                          data_cfg=eval_data_config)

    lit_model.eval()  # 1) turn off dropout / switch BN to eval
    lit_model.freeze()

    # load metadata
    embryo_metadata_df = pd.read_csv(os.path.join(train_data_config.metadata_path, "embryo_metadata_df_train.csv"),
                                     low_memory=False)
    # strip down the full dataset
    embryo_df = embryo_metadata_df[
        ["snip_id", "embryo_id", "Time Rel (s)", "experiment_date", "temperature", "medium", "short_pert_name",
         "control_flag", "phenotype", "predicted_stage_hpf", "surface_area_um",
         "length_um", "width_um"]].iloc[np.where(embryo_metadata_df["use_embryo_flag"] == 1)].copy()
    embryo_df = embryo_df.rename(columns={"Time Rel (s)": "experiment_time"})

    # embryo_df.loc[embryo_df["reference_flag"].astype(str)=="nan", "reference_flag"] = False
    embryo_df = embryo_df.reset_index()

    figure_path = os.path.join(model_dir, "figures")
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)

    prev_run_flag = os.path.isfile(os.path.join(model_dir, "embryo_stats_df.csv"))

    if prev_run_flag and overwrite_flag is False:
        print("Results already exist for: " + model_dir + ". Skipping.")
        return

    print("Evaluating model " + os.path.basename(model_dir) + ')')
    # get dictionary of dataloaders
    mode_list = ["train", "eval", "test"]
    dataset = eval_data_config.create_dataset()
    dataloader_dict = {}
    for mode in mode_list:
        # get indices for images to use for training
        load_indices = getattr(eval_data_config, f"{mode}_indices")
        sampler = SubsetRandomSampler(load_indices)

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=eval_data_config.num_workers,
            sampler=sampler,
            shuffle=False,
        )
        dataloader_dict[mode] = dl

    # look at image reconstructions
    embryo_df = assess_image_reconstructions(
                            embryo_df=embryo_df,
                            lit_model= lit_model,
                            dataloaders= dataloader_dict,  # {"train":…, "eval":…, "test":…}
                            out_dir= figure_path,
                            n_image_figs= n_image_figures,
                            device= lit_model.device,
                            skip_figures= skip_figures_flag,
                            )

    # remove any entries taht were not included in training (why would this happen?)
    zm_cols = [col for col in embryo_df.columns if "z_mu"]
    embryo_df.dropna(subset=zm_cols, inplace=True)

    # Calculate UMAPs
    embryo_df = calculate_UMAPs(embryo_df)

    print(f"Saving data...")
    # save latent arrays and UMAP
    embryo_df = embryo_df.iloc[:, 1:]
    embryo_df.to_csv(os.path.join(model_dir, "embryo_stats_df.csv"), index=False)

    # make a narrower DF with just the UMAP cols and key metadata
    emb_cols = embryo_df.columns
    umap_cols = [col for col in emb_cols if "UMAP" in col]
    umap_df = embryo_df[
        ["snip_id", "experiment_date", "medium", "short_pert_name", "predicted_stage_hpf", "train_cat",
         "recon_loss", "recon_loss_type"] + umap_cols].copy()
    umap_df.to_csv(os.path.join(model_dir, "umap_df.csv"), index=False)

    # get loss trends
    process_run(run_dir=model_dir)


def assess_image_reconstructions(
    embryo_df: pd.DataFrame,
    lit_model:  LitModel,
    dataloaders: dict[str, torch.utils.data.DataLoader],   # {"train":…, "eval":…, "test":…}
    out_dir:     str,
    n_image_figs:int = 16,
    device:      str | torch.device = "cuda",
    skip_figures: bool = False,
):
    lit_model.to(device).eval().freeze()

    # guarantee one sub-folder per split
    os.makedirs(out_dir, exist_ok=True)
    for split in dataloaders:
        os.makedirs(os.path.join(out_dir, f"{split}_images"), exist_ok=True)

    trainer = Trainer(accelerator="auto", devices=1)

    img_counter = {k: 0 for k in dataloaders}

    # loop over splits
    for split, loader in dataloaders.items():
        lit_model.current_mode = split          # ➟ predict_step can log it
        preds = trainer.predict(lit_model, dataloaders=loader)

        # concat batch dictionaries
        snip_ids = sum([list(p["snip_ids"]) for p in preds], [])
        snip_ids = [os.path.basename(s).replace(".jpg", "") for s in snip_ids]

        recon_loss = torch.cat([p["recon_loss"]      for p in preds]).numpy()
        # recon_loss_type = torch.cat([p["recon_loss_type"] for p in preds]).numpy()
        mus      = torch.cat([p["mu"]       for p in preds]).numpy()
        if np.any(np.isnan(mus)):
            raise Exception("NaN values detected in latents")
        log_var = torch.cat([p["log_var"] for p in preds])
        log_var = torch.clamp(log_var, min=-10., max=10.)  # exp() now bounded
        sigmas = torch.exp(0.5 * log_var).numpy()

        # update dataframe in one shot
        df_idx = embryo_df.index[embryo_df.snip_id.isin(snip_ids)]
        embryo_df.loc[df_idx, "train_cat"] = split
        embryo_df.loc[df_idx, "recon_loss"] = recon_loss
        embryo_df.loc[df_idx, "recon_loss_type"] = preds[0]["recon_loss_type"]

        metric_flag = False
        if hasattr(lit_model.loss_fn, "cfg"):
            metric_flag = (lit_model.loss_fn.cfg.target=="NT-Xent") or(lit_model.loss_fn.cfg.target=="Triplet")
        # latent columns
        if metric_flag:
            bio_indices = lit_model.loss_fn.cfg.biological_indices
            nbio_indices = lit_model.loss_fn.cfg.nuisance_indices
            z_mu_b_cols = [f"z_mu_b_{i:02}" for i in bio_indices]
            z_sigma_b_cols = [f"z_sigma_b_{i:02}" for i in bio_indices]
            z_mu_n_cols = [f"z_mu_n_{i:02}" for i in nbio_indices]
            z_sigma_n_cols = [f"z_sigma_n_{i:02}" for i in nbio_indices]

            z_mu_cols = z_mu_n_cols + z_mu_b_cols
            z_sigma_cols = z_sigma_n_cols + z_sigma_b_cols
        else:
            z_mu_cols    = [f"z_mu_{i:02}"    for i in range(mus.shape[1])]
            z_sigma_cols = [f"z_sigma_{i:02}" for i in range(sigmas.shape[1])]
        embryo_df.loc[df_idx, z_mu_cols]    = mus
        embryo_df.loc[df_idx, z_sigma_cols] = sigmas

        # optional figure dump
        if not skip_figures:
            for p in preds:
                if img_counter[split] >= n_image_figs:
                    break
                for i in range(p["orig"].size(0)):
                    if img_counter[split] >= n_image_figs:
                        break
                    snip_name = os.path.basename(p['snip_ids'][i]).replace(".jpg", "")
                    fpath = os.path.join(
                        out_dir, f"{split}_images",
                        f"{snip_name}_loss{int(p['recon_loss'][i]):05}.png"
                    )
                    grid = torch.stack([p["orig"][i], p["recon"][i]], dim=0)  # 2×C×H×W
                    save_image(grid, fpath, nrow=2, pad_value=1)
                    img_counter[split] += 1

    return embryo_df


def calculate_UMAPs(embryo_df):
    print(f"Calculating UMAPs...")
    zmb_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_b" in embryo_df.columns[i]]
    zmn_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_n" in embryo_df.columns[i]]
    mu_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]
    # embryo_df = embryo_df.reset_index()

    MetricFlag = len(zmb_indices) > 0

    z_mu_array = embryo_df.iloc[:, mu_indices].to_numpy()
    if MetricFlag:
        z_mu_array_b = embryo_df.iloc[:, zmb_indices].to_numpy()
        z_mu_array_n = embryo_df.iloc[:, zmn_indices].to_numpy()

    for n_components in [2, 3]:
        dim_str = str(n_components)
        # calculate 2D morphology UMAPS
        reducer = umap.UMAP(n_components=n_components)
        scaled_z_mu = StandardScaler().fit_transform(z_mu_array)
        embedding = reducer.fit_transform(scaled_z_mu)
        for n in range(n_components):
            embryo_df.loc[:, f"UMAP_{n:02}_" + dim_str] = embedding[:, n]

        if MetricFlag:
            reducer_bio = umap.UMAP(n_components=n_components)
            scaled_z_mu_bio = StandardScaler().fit_transform(z_mu_array_b)
            embedding_bio = reducer_bio.fit_transform(scaled_z_mu_bio)
            for n in range(n_components):
                embryo_df.loc[:, f"UMAP_{n:02}_bio_" + dim_str] = embedding_bio[:, n]

            reducer_n = umap.UMAP(n_components=n_components)
            scaled_z_mu_n = StandardScaler().fit_transform(z_mu_array_n)
            embedding_n = reducer_n.fit_transform(scaled_z_mu_n)
            for n in range(n_components):
                embryo_df.loc[:, f"UMAP_{n:02}_n_" + dim_str] = embedding_n[:, n]

    return embryo_df

# ----------------------------------------------------------
# Helper: read one scalar tag and return (step-list, value-list)
# ----------------------------------------------------------
def load_scalar(run_dir: Path, tag: str) -> Tuple[List[int], List[float]]:
    ea = EA.EventAccumulator(
        str(run_dir),
        size_guidance={EA.SCALARS: 10_000},   # don't truncate
    ).Reload()

    if tag not in ea.Tags()["scalars"]:
        return [], []
    evts = ea.Scalars(tag)
    steps = [e.step for e in evts]
    vals  = [e.value for e in evts]
    return steps, vals


# ----------------------------------------------------------
# Core: pull curves, make plot, build DataFrame, save CSV
# ----------------------------------------------------------
def process_run(run_dir: Path) -> None:
    tags: Dict[str, Tuple[str, str]] = {
        "total" :  ("train/loss",        "val/loss"),
        "recon" :  ("train/recon_loss",  "val/recon_loss"),
        "kld"   :  ("train/kld_loss",    "val/kld_loss"),
        "metric":  ("train/metric_loss", "val/metric_loss"),
    }

    # 1) gather into a dict of DataFrames
    curve_dict = {}
    for name, (tr_tag, va_tag) in tags.items():
        for split, tag in [("train", tr_tag), ("val", va_tag)]:
            steps, vals = load_scalar(run_dir, tag)
            if steps:
                curve_dict[(name, split)] = pd.DataFrame(
                    {"step": steps, f"{name}_{split}": vals}
                )

    if not curve_dict:        # nothing logged
        print(f"[WARN] no loss tags found in {run_dir}")
        return

    # 2) outer-join on step to get wide table
    df = None
    for subdf in curve_dict.values():
        df = subdf if df is None else df.merge(subdf, on="step", how="outer")
    df = df.sort_values("step").reset_index(drop=True)

    # 3) save CSV
    csv_path = os.path.join(run_dir, "loss_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] wrote {csv_path}")

    # 4) quick plot
    # fig = px.line(df, x="step", y="loss")
    df_list = []
    for (name, split), subdf in curve_dict.items():
        subdf = subdf.rename(columns={f"{name}_{split}":"loss"})
        subdf["split"] = split
        subdf["name"] = name
        subdf["id"] = f"{name}_{split}"
        df_list.append(subdf)

    df_plot = pd.concat(df_list)
    df_plot.to_csv(os.path.join(run_dir, "loss_history_long.csv"), index=False)

    fig = px.line(df_plot, x="step", y="loss", color="split", symbol="name")

    # fig.show()
    fig.write_image(os.path.join(run_dir, "figures", "loss_history.png"))
    # plt.title(run_dir.name)
    # plt.xlabel("global step")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


# def bio_prediction_wrapper(embryo_df, meta_df):
#     print("Training basic classifiers to test latent space information content...")
#     mu_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_" in embryo_df.columns[i]]
#
#     age_df = embryo_df.loc[:, ["snip_id", "predicted_stage_hpf", "train_cat", "master_perturbation"]].copy()
#
#     y_pd_lin, y_score_lin, y_pd_nonlin, y_score_nonlin = get_embryo_age_predictions(embryo_df, mu_indices)
#
#     age_df["stage_nonlinear_pd"] = y_pd_nonlin
#     age_df["stage_linear_pd"] = y_pd_lin
#
#     meta_df["stage_R2_nonlin_all"] = y_score_nonlin
#     meta_df["stage_R2_lin_all"] = y_score_lin
#
#     zmb_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_b" in embryo_df.columns[i]]
#     zmn_indices = [i for i in range(len(embryo_df.columns)) if "z_mu_n" in embryo_df.columns[i]]
#     if len(zmb_indices) > 0:
#         y_pd_lin_n, y_score_lin_n, y_pd_nonlin_n, y_score_nonlin_n = get_embryo_age_predictions(embryo_df,
#                                                                                                 zmn_indices)
#         y_pd_lin_b, y_score_lin_b, y_pd_nonlin_b, y_score_nonlin_b = get_embryo_age_predictions(embryo_df,
#                                                                                                 zmb_indices)
#
#         age_df["stage_nonlinear_pd_n"] = y_pd_nonlin_n
#         age_df["stage_linear_pd_n"] = y_pd_lin_n
#         age_df["stage_nonlinear_pd_b"] = y_pd_nonlin_b
#         age_df["stage_linear_pd_b"] = y_pd_lin_b
#
#         meta_df["stage_R2_nonlin_bio"] = y_score_nonlin_b
#         meta_df["stage_R2_lin_bio"] = y_score_lin_b
#         meta_df["stage_R2_nonlin_nbio"] = y_score_nonlin_n
#         meta_df["stage_R2_lin_nbio"] = y_score_lin_n
#
#     # accuracy_nonlin, accuracy_lin, perturbation_df = get_pert_class_predictions(embryo_df, mu_indices)
#     # meta_df["perturbation_acc_nonlin_all"] = accuracy_nonlin
#     # meta_df["perturbation_acc_lin_all"] = accuracy_lin
#
#     # if len(zmb_indices) > 0:
#     # accuracy_nonlin_n, accuracy_lin_n, perturbation_df_n = get_pert_class_predictions(embryo_df, zmn_indices)
#     # accuracy_nonlin_b, accuracy_lin_b, perturbation_df_b = get_pert_class_predictions(embryo_df, zmb_indices)
#     #
#     # # subset
#     # perturbation_df_n = perturbation_df_n.loc[:, ["snip_id", "class_linear_pd", "class_nonlinear_pd"]]
#     # perturbation_df_n = perturbation_df_n.rename(
#     #     columns={"class_linear_pd": "class_linear_pd_n", "class_nonlinear_pd": "class_nonlinear_pd_n"})
#     #
#     # perturbation_df_b = perturbation_df_b.loc[:, ["snip_id", "class_linear_pd", "class_nonlinear_pd"]]
#     # perturbation_df_b = perturbation_df_b.rename(
#     #     columns={"class_linear_pd": "class_linear_pd_b", "class_nonlinear_pd": "class_nonlinear_pd_b"})
#     #
#     # perturbation_df = perturbation_df.merge(perturbation_df_n, how="left", on="snip_id")
#     # perturbation_df = perturbation_df.merge(perturbation_df_b, how="left", on="snip_id")
#     #
#     # meta_df["perturbation_acc_nonlin_bio"] = accuracy_nonlin_b
#     # meta_df["perturbation_acc_lin_bio"] = accuracy_lin_b
#     #
#     # meta_df["perturbation_acc_nonlin_nbio"] = accuracy_nonlin_n
#     # meta_df["perturbation_acc_lin_nbio"] = accuracy_lin_n
#     perturbation_df = None
#
#     return age_df, perturbation_df, meta_df
#
#
# def calculate_contrastive_distances(trained_model, train_dir, device, batch_size, mode_vec=None):
#     if mode_vec is None:
#         mode_vec = ["train", "eval", "test"]
#
#     c_data_loader_vec = []
#     # n_total_samples = 0
#     for mode in mode_vec:
#         temp_dataset = MyCustomDataset(root=os.path.join(train_dir, mode),
#                                        transform=ContrastiveLearningViewGenerator(
#                                            ContrastiveLearningDataset.get_simclr_pipeline_transform(),  # (96),
#                                            2),
#                                        return_name=True
#                                        )
#         data_loader = DataLoader(
#             dataset=temp_dataset,
#             batch_size=batch_size,
#             collate_fn=collate_dataset_output,
#         )
#
#         c_data_loader_vec.append(data_loader)
#         # n_total_samples += np.min([n_contrastive_samples, len(data_loader)])
#
#     metric_df_list = []
#
#     sample_iter = 0
#
#     # contrastive_df = contrastive_df.reset_index()
#     if trained_model.model_name == "VAE":
#         trained_model.nuisance_indices = np.random.choice(range(trained_model.latent_dim), 10, replace=False)
#
#     for m, mode in enumerate(mode_vec):
#         data_loader = c_data_loader_vec[m]
#
#         for i, inputs in enumerate(tqdm(data_loader, f"Calculating {mode} contrastive differences...")):
#             inputs = set_inputs_to_device(device, inputs)
#             x = inputs["data"]
#             bs = x.shape[0]
#
#             labels = list(inputs["label"][0])
#             labels = labels * 2
#             snip_id_list = clean_path_names(labels)
#
#             # initialize temporary DF
#             metric_df_temp = pd.DataFrame(np.empty((x.shape[0] * 2, 2)),
#                                           columns=["snip_id", "contrast_id"])
#             metric_df_temp["snip_id"] = snip_id_list
#
#             # generate columns to store latent encodings
#             new_cols = []
#             for n in range(trained_model.latent_dim):
#
#                 if (trained_model.model_name == "MetricVAE") or (trained_model.model_name == "SeqVAE") or (
#                         trained_model.model_name == "MorphIAFVAE"):
#                     if n in trained_model.nuisance_indices:
#                         new_cols.append(f"z_mu_n_{n:02}")
#                     else:
#                         new_cols.append(f"z_mu_b_{n:02}")
#
#                 elif trained_model.model_name == "VAE":  # generate fake partitions
#                     if n in trained_model.nuisance_indices:
#                         new_cols.append(f"z_mu_n_{n:02}")
#                     else:
#                         new_cols.append(f"z_mu_b_{n:02}")
#                 else:
#                     raise Exception("Incompatible model type found ({trained_model.model_name})")
#
#             metric_df_temp.loc[:, new_cols] = np.nan
#
#             # latent_encodings = trained_model.encoder(inputs)
#             x0 = torch.reshape(x[:, 0, :, :, :],
#                                (x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # first set of images
#             x1 = torch.reshape(x[:, 1, :, :, :],
#                                (x.shape[0], x.shape[2], x.shape[3],
#                                 x.shape[4]))  # second set with matched contrastive pairs
#
#             encoder_output0 = trained_model.encoder(x0)
#             encoder_output1 = trained_model.encoder(x1)
#
#             mu0, log_var0 = encoder_output0.embedding, encoder_output0.log_covariance
#             mu1, log_var1 = encoder_output1.embedding, encoder_output1.log_covariance
#
#             mu0 = mu0.detach().cpu()
#             mu1 = mu1.detach().cpu()
#
#             # store
#             metric_df_temp.loc[:x.shape[0] - 1, "contrast_id"] = 0
#             metric_df_temp.loc[x.shape[0]:, "contrast_id"] = 1
#
#             metric_df_temp.loc[:x.shape[0] - 1, new_cols] = np.asarray(mu0)
#             metric_df_temp.loc[x.shape[0]:, new_cols] = np.asarray(mu1)
#
#             metric_df_list.append(metric_df_temp)
#             # sample_iter += bs
#
#     metric_df_out = pd.concat(metric_df_list, axis=0, ignore_index=True)
#
#     return metric_df_out
#
#
# def get_embryo_age_predictions(embryo_df, mu_indices):
#     train_indices = np.where((embryo_df["train_cat"] == "train") | (embryo_df["train_cat"] == "eval"))[0]
#     test_indices = np.where(embryo_df["train_cat"] == "test")[0]
#
#     # extract target vector
#     y_train = embryo_df["predicted_stage_hpf"].iloc[train_indices].to_numpy().astype(float)
#     y_test = embryo_df["predicted_stage_hpf"].iloc[test_indices].to_numpy().astype(float)
#
#     # extract predictor variables
#     X_train = embryo_df.iloc[train_indices, mu_indices].to_numpy().astype(float)
#     X_test = embryo_df.iloc[test_indices, mu_indices].to_numpy().astype(float)
#
#     ###################
#     # run MLP regressor
#     clf_age_nonlin = MLPRegressor(random_state=1, max_iter=5000).fit(X_train, y_train)
#
#     ###################
#     # Run multivariate linear regressor
#     clf_age_lin = linear_model.LinearRegression().fit(X_train, y_train)
#
#     # initialize pandas dataframe to store results
#     X_full = embryo_df.iloc[:, mu_indices].to_numpy().astype(float)
#
#     y_pd_nonlin = clf_age_nonlin.predict(X_full)
#     y_score_nonlin = clf_age_nonlin.score(X_test, y_test)
#
#     y_pd_lin = clf_age_lin.predict(X_full)
#     y_score_lin = clf_age_lin.score(X_test, y_test)
#
#     return y_pd_lin, y_score_lin, y_pd_nonlin, y_score_nonlin
#
#
# def get_pert_class_predictions(embryo_df, mu_indices):
#     train_indices = np.where((embryo_df["train_cat"] == "train"))[0]
#     test_indices = np.where((embryo_df["train_cat"] == "test") | (embryo_df["train_cat"] == "eval"))[0]
#
#     pert_df = embryo_df.loc[:, ["snip_id", "predicted_stage_hpf", "train_cat", "master_perturbation"]].copy()
#
#     ########################
#     # How well does latent space predict perturbation type?
#     pert_class_train = np.asarray(embryo_df["master_perturbation"].iloc[train_indices])
#     pert_class_test = np.asarray(embryo_df["master_perturbation"].iloc[test_indices])
#
#     # extract predictor variables
#     X_train = embryo_df.iloc[train_indices, mu_indices].to_numpy().astype(float)
#     X_test = embryo_df.iloc[test_indices, mu_indices].to_numpy().astype(float)
#
#     ###################
#     # run MLP classifier
#     ###################
#     clf = MLPClassifier(random_state=1, max_iter=5000).fit(X_train, pert_class_train)
#     accuracy_nonlin = clf.score(X_test, pert_class_test)
#
#     ###################
#     # Run multivariate logistic classifier
#     ###################
#     clf_lin = LogisticRegression(random_state=0).fit(X_train, pert_class_train)
#     accuracy_lin = clf_lin.score(X_test, pert_class_test)
#
#     class_pd_nonlin_train = clf.predict(X_train)
#     class_pd_nonlin_test = clf.predict(X_test)
#
#     class_pd_lin_train = clf_lin.predict(X_train)
#     class_pd_lin_test = clf_lin.predict(X_test)
#
#     # pert_df_train = pert_df.iloc[train_indices]
#     # pert_df_test = pert_df.iloc[test_indices]
#
#     pert_df.loc[train_indices, "class_nonlinear_pd"] = class_pd_nonlin_train
#     pert_df.loc[train_indices, "class_linear_pd"] = class_pd_lin_train
#
#     pert_df.loc[test_indices, "class_nonlinear_pd"] = class_pd_nonlin_test
#     pert_df.loc[test_indices, "class_linear_pd"] = class_pd_lin_test
#
#     return accuracy_nonlin, accuracy_lin, pert_df

if __name__ == "__main__":

    cfg = "/home/nick/projects/morphseq/src/config_files/morph_vae_test_run.yaml"
    assess_vae_results(cfg=cfg, overwrite_flag=True)




