# from src.functions.dataset_utils import *
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from src.run.run_utils import initialize_model
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
from tensorboard.backend.event_processing import event_accumulator as EA
import plotly.express as px
from glob2 import glob
import yaml
import ast
from tqdm import tqdm


def get_hydra_runs(hydra_run_path):
    # get list of runs
    run_list = sorted(glob(os.path.join(hydra_run_path, "*")))
    run_path_list = [f for f in run_list if os.path.isdir(f)]
    run_name_list = [os.path.basename(f) for f in run_path_list]

    # get override hyperparam values
    rows = []
    cfg_path_list = []
    for run_name, run_path in zip(run_name_list, run_path_list):
        # pull sweep params
        param_path = os.path.join(run_path, ".hydra", "overrides.yaml")
        overrides = yaml.safe_load(open(param_path))
        h_params = {}
        h_params["run_id"] = run_name
        h_params["run_path"] = run_path
        for o in overrides:
            # split into “full.key.path” and “value_str”
            key_path, value_str = o.split("=", 1)
            # the basename is whatever comes after the last “.”
            base_name = key_path.rsplit(".", 1)[-1]
            # parse the right-hand side into a Python literal (int, float, list, etc.)
            try:
                value = ast.literal_eval(value_str)
            except:
                value = value_str
            h_params[base_name] = value

        rows.append(h_params)

        # pull cfg path for later
        cfg_path = os.path.join(run_path, ".hydra", "config.yaml")
        cfg_path_list.append(cfg_path)

    # write hyperaram df
    hyper_df = pd.DataFrame(rows)

    return hyper_df, cfg_path_list

def parse_hydra_paths(run_path, version=None, ckpt=None):

    if version is None:  # find most recent version
        all_versions = glob(os.path.join(run_path, "*"))
        all_versions = sorted([v for v in all_versions if os.path.isdir(v)])
        model_dir = all_versions[-1]
    else:
        model_dir = os.path.join(run_path, version, "")
    # get checkpoint
    ckpt_dir = os.path.join(model_dir, "checkpoints", "")
    # find all .ckpt files
    if ckpt is None:
        all_ckpts = glob(os.path.join(ckpt_dir, "*.ckpt"))
        # pick the one with the latest modification time
        latest_ckpt = max(all_ckpts, key=os.path.getmtime)
    else:
        latest_ckpt = ckpt_dir + ckpt + ".ckpt"

    return model_dir, latest_ckpt


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
        # df_idx = embryo_df.index[embryo_df.snip_id.isin(snip_ids)]
        # build your mapping once
        mapping = dict(zip(embryo_df['snip_id'].values, embryo_df.index.values))
        # then map your list of snip_ids in O(1) each
        df_idx = np.array([mapping[snip] for snip in snip_ids])
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
        "pixel" : ("train/pixel_loss", "val/pixel_loss"),
        "pips"  : ("train/pips_loss", "val/pips_loss"),
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



def assess_hydra_results(hydra_run_path,
                           n_image_figures=50,
                           overwrite_flag=False,
                           skip_figures_flag=False,
                           batch_size=256):

    hyper_df, cfg_path_list = get_hydra_runs(hydra_run_path)

    hyper_df.to_csv(os.path.join(hydra_run_path, "hyperparam_df.csv"), index=False)

    for cfg in tqdm(cfg_path_list, "Processing training runs..."):

        if isinstance(cfg, str):
            # load config file
            config = OmegaConf.load(cfg)
        elif isinstance(cfg, dict):
            config = cfg
        else:
            raise Exception("cfg argument dtype is not recognized")

        # initialize
        model, model_config, train_data_config, loss_fn, train_config = initialize_model(config)

        run_path = os.path.join(os.path.dirname(os.path.dirname(cfg)), "lightning_logs")
        model_dir, latest_ckpt = parse_hydra_paths(run_path=run_path)

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


if __name__ == "__main__":

    hydra_path = "/media/nick/hdd021/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/training_data/20241107_ds/hydra_outputs/ntxent_test_20250503_170712/"
    assess_hydra_results(hydra_run_path=hydra_path, overwrite_flag=True)

    # hydra_path = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/20241107_ds/hydra_outputs/nets_pip05_20250502_170746/"
    # assess_hydra_results(hydra_run_path=hydra_path, overwrite_flag=True)




