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
                       out_path,
                       n_image_figures=25,
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

    # # load metadata
    # embryo_metadata_df = pd.read_csv(os.path.join(train_data_config.metadata_path, "embryo_metadata_df_train.csv"),
    #                                  low_memory=False)
    # # strip down the full dataset
    # embryo_df = embryo_metadata_df[
    #     ["snip_id", "embryo_id", "Time Rel (s)", "experiment_date", "temperature", "medium", "short_pert_name",
    #      "control_flag", "phenotype", "predicted_stage_hpf", "surface_area_um",
    #      "length_um", "width_um"]].iloc[np.where(embryo_metadata_df["use_embryo_flag"] == 1)].copy()
    # embryo_df = embryo_df.rename(columns={"Time Rel (s)": "experiment_time"})

    # # embryo_df.loc[embryo_df["reference_flag"].astype(str)=="nan", "reference_flag"] = False
    # embryo_df = embryo_df.reset_index()

    figure_path = os.path.join(out_path, "figures")
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)

    print("Evaluating model " + os.path.basename(model_dir) + ')')
    # get dictionary of dataloaders
    dataset = eval_data_config.create_dataset()

    # get indices for images to use for training
    load_indices = getattr(eval_data_config, "test_indices")
    sampler = SubsetRandomSampler(load_indices)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=eval_data_config.num_workers,
        sampler=sampler,
        shuffle=False,
    )

    # look at image reconstructions
    embryo_df = assess_image_reconstructions(
                            lit_model= lit_model,
                            dataloader= dataloader,  # {"train":…, "eval":…, "test":…}
                            out_dir= figure_path,
                            n_image_figs= n_image_figures,
                            device= lit_model.device,
                            skip_figures= skip_figures_flag,
                            )

  


def assess_image_reconstructions(
    lit_model:  LitModel,
    dataloader: torch.utils.data.DataLoader,   # {"train":…, "eval":…, "test":…}
    out_dir:     str,
    n_image_figs:int = 16,
    device:      str | torch.device = "cuda",
):
    lit_model.to(device).eval().freeze()

    trainer = Trainer(accelerator="auto", devices=1)

    # loop over splits         # ➟ predict_step can log it
    preds = trainer.predict(lit_model, dataloaders=dataloader)

    # concat batch dictionaries
    snip_ids = sum([list(p["snip_ids"]) for p in preds], [])
    snip_ids = [os.path.basename(s).replace(".jpg", "") for s in snip_ids]

    # make im fig
    for p in preds:
        for i in range(p["orig"].size(0)):
            if img_counter[split] >= n_image_figs:
                break
            snip_name = os.path.basename(p['snip_ids'][i]).replace(".jpg", "")
            fpath = os.path.join(
                out_dir, f"{split}_images", f"{snip_name}_loss{int(p['recon_loss'][i]):05}.png"
            )
            grid = torch.stack([p["orig"][i], p["recon"][i]], dim=0)  # 2×C×H×W
            save_image(grid, fpath, nrow=2, pad_value=1)
            img_counter[split] += 1

