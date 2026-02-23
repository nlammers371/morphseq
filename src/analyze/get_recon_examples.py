import torch
from torch.utils.data.sampler import SubsetRandomSampler
# from src.run.run_utils import initialize_model, parse_model_paths
from torchvision.utils import save_image
from pytorch_lightning import Trainer
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from omegaconf import OmegaConf
from vae.models.auto_model import AutoModel
from typing import Union
import pickle
from lightning.pl_wrappers import LitModel
from torch.utils.data import DataLoader
from data.dataset_configs import BaseDataConfig, EvalDataConfig
import shutil

torch.set_float32_matmul_precision("medium")   # good default


def copy_config_to_outfolder(cfg_path, out_folder):
    """
    Copies the Hydra config file to the output folder for provenance.
    If the config is a dictionary (in-memory), saves it as YAML.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    if isinstance(cfg_path, str) and os.path.isfile(cfg_path):
        # Config is a file on disk
        shutil.copy2(cfg_path, out_folder / "config.yaml")
    else:
        # Config is a dict or OmegaConf object
        import yaml
        config_path = out_folder / "config.yaml"
        with open(config_path, "w") as f:
            if hasattr(cfg_path, "to_container"):
                # OmegaConf object
                yaml.safe_dump(cfg_path.to_container(resolve=True), f)
            else:
                yaml.safe_dump(cfg_path, f)


def recon_wrapper(
    hydra_run_path,
    run_type,
    out_path,
    model_class="hydra",
    n_image_figures=64,
    random_seed=42,        # <-- ensure deterministic sampling
):
    # ----- Load paths -----
    if model_class == "legacy":
        cfg_path_list = [hydra_run_path]  # direct path to legacy model folder
    else:
        from analyze.assess_hydra_results import get_hydra_runs, initialize_model_to_asses, parse_hydra_paths
        _, cfg_path_list = get_hydra_runs(hydra_run_path, run_type)

    for cfg in tqdm(cfg_path_list, "Processing training runs..."):
        if model_class == "legacy":
            # ===== LEGACY LOADING =====
            model_dir = Path(cfg)
            lit_model = AutoModel.load_from_folder(model_dir)
            lit_model.eval()

            # data_root = Path(cfg).parents[3]  # assumes /models/legacy/model_name structure
            # input_size = (288, 128)
            # eval_data_config = EvalDataConfig(
            #     experiments=None,  # will pull from legacy metadata
            #     root=data_root,
            #     return_sample_names=True,
            #     transforms="basic"
            # )
        else:
            # ===== HYDRA LOADING =====
            config = OmegaConf.load(cfg)
            model, model_config = initialize_model_to_asses(config)
            loss_fn = model_config.lossconfig.create_module()
            run_path = os.path.dirname(os.path.dirname(cfg))
            latest_ckpt = parse_hydra_paths(run_path=run_path)
            if latest_ckpt is None:
                continue

            # load split indices
            split_path = os.path.join(run_path, "split_indices.pkl")
            with open(split_path, 'rb') as file:
                split_dict = pickle.load(file)

            eval_data_config = BaseDataConfig(
                train_indices=np.asarray(split_dict["train"]),
                test_indices=np.asarray(split_dict["test"]),
                eval_indices=np.asarray(split_dict["eval"]),
                root=os.path.dirname(os.path.dirname(os.path.dirname(run_path))),
                return_sample_names=True,
                transform_name="basic"
            )
            eval_data_config.make_metadata()

            lit_model = LitModel.load_from_checkpoint(
                latest_ckpt,
                model=model,
                loss_fn=loss_fn,
                data_cfg=eval_data_config
            )
            lit_model.eval().freeze()

        # print("Evaluating model " + os.path.basename(run_path) + ')')
        if model_class == "legacy":
            run_path = cfg

            # load split indices
            split_path = os.path.join(run_path, "split_indices.pkl")
            with open(split_path, 'rb') as file:
                split_dict = pickle.load(file)

            eval_data_config = BaseDataConfig(
                train_indices=np.asarray(split_dict["train"]),
                test_indices=np.asarray(split_dict["test"]),
                eval_indices=np.asarray(split_dict["eval"]),
                root=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(run_path))), "training_data", "models"),
                return_sample_names=True,
                transform_name="basic"
            )
            eval_data_config.make_metadata()

        dataset = eval_data_config.create_dataset()

        # deterministic random selection
        # if model_class == "legacy":
        #     load_indices = np.arange(len(dataset))
        # else:
        load_indices = getattr(eval_data_config, "test_indices")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        shuffled_indices = np.random.permutation(load_indices)
        selected_indices = shuffled_indices[:n_image_figures]
        sampler = SubsetRandomSampler(selected_indices)

        dl = DataLoader(
            dataset,
            batch_size=n_image_figures,
            num_workers=eval_data_config.num_workers,
            sampler=sampler,
            shuffle=False,
        )

        out_stub = Path(cfg).name if model_class == "legacy" else Path(run_path).name
        mdl_folder = os.path.join(out_path, out_stub)
        os.makedirs(mdl_folder, exist_ok=True)

        if model_class != "legacy":
            copy_config_to_outfolder(cfg, mdl_folder)

        assess_image_reconstructions(
            lit_model=lit_model,
            dataloader=dl,
            out_dir=mdl_folder,
            device=lit_model.device,
            model_class=model_class
        )


def assess_image_reconstructions(
    lit_model,
    dataloader: torch.utils.data.DataLoader,
    out_dir: str,
    device: Union[str, torch.device] = "cuda",
    model_class: str = "hydra",
):
    os.makedirs(out_dir, exist_ok=True)

    if model_class == "legacy":
        lit_model.to(device).eval()
        for batch in dataloader:
            x = batch["data"].to(device)
            # recon = lit_model.dec(lit_model.enc(x))
            encoder_output = lit_model.encoder(x)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            z0, _ = lit_model._sample_gauss(mu, std)
            recon = lit_model.decoder(z0)["reconstruction"]

            for i in range(x.size(0)):
                snip_name = os.path.basename(batch['label'][0][i]).replace(".jpg", "")
                fpath = os.path.join(out_dir, f"{snip_name}.png")
                grid = torch.stack([x[i], recon[i]], dim=0)
                save_image(grid, fpath, nrow=2, pad_value=1)
    else:
        lit_model.to(device).eval().freeze()
        trainer = Trainer(accelerator="auto", devices=1, limit_predict_batches=1)
        lit_model.current_mode = "test"
        preds = trainer.predict(lit_model, dataloaders=dataloader)

        for p in preds:
            for i in range(p["orig"].size(0)):
                snip_name = os.path.basename(p['snip_ids'][i]).replace(".jpg", "")
                fpath = os.path.join(out_dir, f"{snip_name}_loss{int(p['recon_loss'][i]):05}.png")
                grid = torch.stack([p["orig"][i], p["recon"][i]], dim=0)
                save_image(grid, fpath, nrow=2, pad_value=1)

