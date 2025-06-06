# from src.functions.dataset_utils import *
import torch
from PIL.ImImagePlugin import split
from torch.utils.data.sampler import SubsetRandomSampler
from src.run.run_utils import initialize_model, parse_model_paths
from torchvision.utils import save_image
from pytorch_lightning import Trainer
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import pickle
from src.lightning.pl_wrappers import LitModel
from torch.utils.data import DataLoader
from src.data.dataset_configs import BaseDataConfig
from src.assess.assess_hydra_results import get_hydra_runs, initialize_model_to_asses, parse_hydra_paths

torch.set_float32_matmul_precision("medium")   # good default



def recon_wrapper(hydra_run_path,
                       run_type,
                       out_path,
                       n_image_figures=64):

    hyper_df, cfg_path_list = get_hydra_runs(hydra_run_path, run_type)

    for cfg in tqdm(cfg_path_list, "Processing training runs..."):

        if isinstance(cfg, str):
            # load config file
            config = OmegaConf.load(cfg)
        elif isinstance(cfg, dict):
            config = cfg
        else:
            raise Exception("cfg argument dtype is not recognized")


        # initialize
        try:
            model, model_config = initialize_model_to_asses(config)
        except:
            continue
        loss_fn = model_config.lossconfig.create_module()
        run_path = os.path.dirname(os.path.dirname(cfg))
        latest_ckpt = parse_hydra_paths(run_path=run_path)
        if latest_ckpt is None:
            continue
        # load train/test/eval indices
        split_path = os.path.join(run_path, "split_indices.pkl")
        with open(split_path, 'rb') as file:
            split_dict = pickle.load(file)

        # initialize new data config for evaluation
        eval_data_config = BaseDataConfig(train_indices=np.asarray(split_dict["train"]),
                                          test_indices=np.asarray(split_dict["test"]),
                                          eval_indices=np.asarray(split_dict["eval"]),
                                          root=os.path.dirname(os.path.dirname(os.path.dirname(run_path))),
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

        print("Evaluating model " + os.path.basename(run_path) + ')')
        # get dictionary of dataloaders
        dataset = eval_data_config.create_dataset()

        # get indices for images to use for training
        load_indices = getattr(eval_data_config, "test_indices")
        sampler = SubsetRandomSampler(load_indices)

        dl = DataLoader(
            dataset,
            batch_size=n_image_figures,
            num_workers=eval_data_config.num_workers,
            sampler=sampler,
            shuffle=False,
        )

        # construct out path
        folder_name = os.path.basename(os.path.dirname(run_path))
        mdl_name = model_config.ddconfig.name
        pips_wt = model_config.lossconfig.pips_weight
        gan_wt = model_config.lossconfig.gan_weight

        out_name = f"{mdl_name}_p{int(10*pips_wt)}_g{int(10*gan_wt)}_{folder_name}"
        mdl_folder = os.path.join(out_path, out_name)
        os.makedirs(mdl_folder, exist_ok=True)
        # look at image reconstructions
        assess_image_reconstructions(
                                lit_model= lit_model,
                                dataloader= dl,  # {"train":…, "eval":…, "test":…}
                                out_dir=mdl_folder,
                                device= lit_model.device
                                )



def assess_image_reconstructions(
    lit_model:  LitModel,
    dataloader: torch.utils.data.DataLoader,   # {"train":…, "eval":…, "test":…}
    out_dir:     str,
    device:      str | torch.device = "cuda",
):
    lit_model.to(device).eval().freeze()

    trainer = Trainer(accelerator="auto", devices=1, limit_predict_batches=1)
    lit_model.current_mode = "test"
    preds = trainer.predict(lit_model, dataloaders=dataloader)

    # concat batch dictionaries
    # snip_ids = sum([list(p["snip_ids"]) for p in preds], [])
    # snip_ids = [os.path.basename(s).replace(".jpg", "") for s in snip_ids]

    # make im fig
    for p in preds:
        for i in range(p["orig"].size(0)):
            snip_name = os.path.basename(p['snip_ids'][i]).replace(".jpg", "")
            fpath = os.path.join(
                out_dir, f"{snip_name}_loss{int(p['recon_loss'][i]):05}.png"
            )
            grid = torch.stack([p["orig"][i], p["recon"][i]], dim=0)  # 2×C×H×W
            save_image(grid, fpath, nrow=2, pad_value=1)
