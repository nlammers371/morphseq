import sys
from pathlib import Path

# Path to the project *root* (the directory that contains the `src/` folder)
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# Put `src/` at the *front* of sys.path so Python finds top-level packages
sys.path.insert(0, str(SRC_ROOT))

# from src.functions.dataset_utils import *
import torch
from vae.models.auto_model import AutoModel
from PIL.ImImagePlugin import split
from torch.utils.data.sampler import SubsetRandomSampler
# from src.run.run_utils import initialize_model, parse_model_paths
from torchvision.utils import save_image
from pytorch_lightning import Trainer
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import pickle
from typing import List, Literal, Optional, Any, Union
from lightning.pl_wrappers import LitModel
from torch.utils.data import DataLoader
from data.data_transforms import basic_transform
from data.dataset_configs import EvalDataConfig
# from src.analyze.assess_hydra_results import initialize_model_to_asses, parse_hydra_paths
from models.factories import build_from_config
import pytorch_lightning as pl
import io
import pandas as pd
import importlib
import glob2 as glob
# from src.build.pipeline_objects import Experiment

torch.set_float32_matmul_precision("medium")   # good default

def parse_hydra_paths(run_path, version=None, ckpt=None):

    # if version is None:  # find most recent version
    #     all_versions = glob(os.path.join(run_path, "*"))
    #     all_versions = sorted([v for v in all_versions if os.path.isdir(v)])
    #     model_dir = all_versions[-1]
    # else:
    #     model_dir = os.path.join(run_path, version, "")
    # get checkpoint
    ckpt_dir = os.path.join(run_path, "checkpoints", "")
    # find all .ckpt files
    if ckpt is None:
        latest_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if not os.path.isfile(latest_ckpt):
            all_ckpts = glob(os.path.join(ckpt_dir, "*.ckpt"))
            # pick the one with the latest modification time
            if len(all_ckpts) > 0:
                latest_ckpt = max(all_ckpts, key=os.path.getmtime)
            else:
                latest_ckpt = None
    else:
        latest_ckpt = ckpt_dir + ckpt + ".ckpt"

    return latest_ckpt

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def initialize_model_to_asses(config):

    # initialize the model
    config_full = config.copy()
    model_dict = config.pop("model", OmegaConf.create())
    target = model_dict["config_target"]
    model_config = get_obj_from_str(target)
    model_config = model_config.from_cfg(cfg=config_full)

    # parse dataset related options and merge with defaults as needed
    # data_config = model_config.dataconfig
    # data_config.make_metadata()

    # initialize model
    model = build_from_config(model_config)

    return model, model_config


class LegacyWrapper(pl.LightningModule):  # ✓ looks like Lightning
    def __init__(self, enc, dec=None):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        with torch.no_grad():
            return self.enc(x)

# 3. Create a custom Unpickler that maps all CUDA -> CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def calculate_morph_embeddings(data_root: Union[str, Path],
                  model_name: str,
                  model_class: Literal["legacy"],
                  experiments: List[str],
                  cfg: Optional[Any] = None,
                  batch_size: int = 64,
                  ):

    # Environment switching logic for Python 3.9 legacy models
    import os
    import sys
    import subprocess
    
    if (model_class == "legacy" and
        sys.version_info[:2] != (3, 9)):
        
        print(f"⚠️  Legacy model requires Python 3.9, current version: {sys.version_info[0]}.{sys.version_info[1]}")
        print("🔄 Switching to vae-env-cluster environment...")
        print(f"🔍 Current CONDA_EXE: {os.environ.get('CONDA_EXE', 'Not set')}")
        print(f"🔍 Current PATH contains conda: {'conda' in os.environ.get('PATH', '')}")
        
        # Build command to run this function in Python 3.9 environment
        current_script = __file__
        repo_root = Path(current_script).parent.parent.parent  # Go up to morphseq root
        
        # Use dedicated Python 3.9 script to avoid import compatibility issues
        script_path = repo_root / "src" / "run_morphseq_pipeline" / "services" / "generate_embeddings_py39.py"
        experiments_json = str(experiments).replace("'", '"')  # Convert to JSON format
        
        # Try different conda/mamba paths
        conda_paths = [
            "/net/trapnell/vol1/home/mdcolon/software/miniconda3/bin/conda",
            "/net/trapnell/vol1/home/mdcolon/software/miniconda3/condabin/conda", 
            "/usr/bin/conda",
            "conda"
        ]
        
        mamba_paths = [
            "/net/trapnell/vol1/home/mdcolon/software/miniconda3/bin/mamba",
            "/usr/bin/mamba", 
            "mamba"
        ]
        
        # Try to find working conda/mamba command
        cmd_base = None
        for conda_path in conda_paths:
            try:
                subprocess.run([conda_path, "--version"], capture_output=True, check=True)
                cmd_base = [conda_path, "run", "-p", "/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster", "--no-capture-output"]
                print(f"✅ Found conda at: {conda_path}")
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        if cmd_base is None:
            for mamba_path in mamba_paths:
                try:
                    subprocess.run([mamba_path, "--version"], capture_output=True, check=True)
                    cmd_base = [mamba_path, "run", "-p", "/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster", "--no-capture-output"]
                    print(f"✅ Found mamba at: {mamba_path}")
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        
        if cmd_base is None:
            raise RuntimeError("❌ Could not find conda or mamba executable. Please ensure conda/mamba is installed and accessible.")
        
        cmd = cmd_base + ["python", str(script_path), str(data_root), model_name, model_class, experiments_json, str(batch_size)]
        
        try:
            # Ensure subprocess inherits conda environment variables
            env = os.environ.copy()
            # Add conda paths if they're not already there
            if 'CONDA_EXE' not in env and os.path.exists("/net/trapnell/vol1/home/mdcolon/software/miniconda3/condabin/conda"):
                env['CONDA_EXE'] = "/net/trapnell/vol1/home/mdcolon/software/miniconda3/condabin/conda"
            if 'CONDA_PREFIX' not in env:
                env['CONDA_PREFIX'] = "/net/trapnell/vol1/home/mdcolon/software/miniconda3"
            
            # Run subprocess from repo root so script can use sys.path.insert(0, ".")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=repo_root)
            print("✅ Python 3.9 subprocess completed successfully")
            print(result.stdout)
            return  # Exit early - the subprocess handled the work
        except subprocess.CalledProcessError as e:
            print(f"❌ Subprocess failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise

    legacy = model_class == "legacy"
    data_root = Path(data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if legacy:
        # ---- load the old encoder (and decoder if you need it) ----
        model_dir = data_root / "models" / model_class / model_name 
        # model_dir = Path(model_name) if Path(model_name).is_dir() else Path(model_name).parent
        # enc_path = model_dir / "encoder.pkl"
        # dec_path = model_dir / "decoder.pkl"  # optional

        lit_model = AutoModel.load_from_folder(model_dir)

        input_size = (288, 128)

    else:
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
            raise Exception(f"Error loading config file for {model_name}")

        loss_fn = model_config.lossconfig.create_module()
        run_path = os.path.dirname(os.path.dirname(cfg))
        latest_ckpt = parse_hydra_paths(run_path=run_path)

        raise Exception("Need to add way to assess image size")

    transform = basic_transform(target_size=input_size)

    # initialize new data config for evaluation
    eval_data_config = EvalDataConfig(experiments=experiments,
                                      root=data_root,
                                      return_sample_names=True, 
                                      transforms=transform)
    
    if not legacy:
        lit_model = LitModel.load_from_checkpoint(latest_ckpt,
                                                  model=model,
                                                  loss_fn=loss_fn,
                                                  data_cfg=eval_data_config)
        
        

    lit_model.eval()  # 1) turn off dropout / switch BN to eval
    # lit_model.freeze()

    # print("Generating embeddings using model " + os.path.basename(run_path) + ')')

    # get dictionary of dataloaders
    dataset = eval_data_config.create_dataset()

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=eval_data_config.num_workers,
        shuffle=False,
    )

    # construct out path
    # folder_name = os.path.basename(os.path.dirname(run_path))
    # mdl_name = model_config.ddconfig.name
    # pips_wt = model_config.lossconfig.pips_weight
    # gan_wt = model_config.lossconfig.gan_weight
    # attn = model_config.ddconfig.dec_use_local_attn
    # out_name = f"{mdl_name}_p{int(10*pips_wt)}_g{int(np.ceil(100*gan_wt))}_attn{attn}_GAN{model_config.lossconfig.gan_net}_{folder_name}"
    # mdl_folder = os.path.join(out_path, out_name)
    # os.makedirs(mdl_folder, exist_ok=True)

    # look at image reconstructions
    if legacy:
        latent_df = extract_embeddings_legacy(lit_model=lit_model,
                                  dataloader=dl,
                                  device=device)
        
    print("Saving embeddings...")
    save_root = data_root / "analysis" / "latent_embeddings" / model_class / model_name 
    
    for exp in experiments:
        exp_df = latent_df.loc[latent_df["experiment_date"]==exp]
        # out_path = save_root / exp 
        os.makedirs(save_root, exist_ok=True)
        exp_df.to_csv(save_root / f"morph_latents_{exp}.csv", index=False)
    
    return {}


import logging


def extract_embeddings_legacy(
                        lit_model:  Any,
                        dataloader: torch.utils.data.DataLoader,   # {"train":…, "eval":…, "test":…}
                        device:      Union[str, torch.device] = "cuda",
                        ):

    lit_model.to(device).eval()

    logger = logging.getLogger(__name__)
    raw_nuisance = getattr(lit_model, 'nuisance_indices', None)
    nuisance_set = None
    if raw_nuisance is not None:
        try:
            if hasattr(raw_nuisance, 'tolist'):
                nuisance_set = {int(v) for v in raw_nuisance.tolist()}
            else:
                nuisance_set = {int(v) for v in raw_nuisance}
        except Exception:
            nuisance_set = None
    nuisance_len = len(nuisance_set) if nuisance_set is not None else 0
    logger.info(
        "extract_embeddings_legacy metadata: "
        f"model_name={getattr(lit_model, 'model_name', None)}, "
        f"latent_dim={getattr(lit_model, 'latent_dim', None)}, "
        f"nuisance_len={nuisance_len}, "
        f"nuisance_type={type(raw_nuisance)}"
    )

    new_mu_cols = []
    new_sigma_cols = []
    
    for n in range(lit_model.latent_dim):

        if (lit_model.model_name == "MetricVAE") or (lit_model.model_name == "SeqVAE"):
            if nuisance_set is not None and n in nuisance_set:
                new_mu_cols.append(f"z_mu_n_{n:02}")
                new_sigma_cols.append(f"z_sigma_n_{n:02}")
            else:
                new_mu_cols.append(f"z_mu_b_{n:02}")
                new_sigma_cols.append(f"z_sigma_b_{n:02}")
        else:
            new_mu_cols.append(f"z_mu_{n:02}")
            new_sigma_cols.append(f"z_sigma_{n:02}")
            
    # embryo_df.loc[:, new_mu_cols] = np.nan
    # embryo_df.loc[:, new_sigma_cols] = np.nan

    df_list = []

    for n, inputs in enumerate(tqdm(dataloader, f"Getting embeddings...")):

        # inputs = inputs.to(device)
        x = inputs["data"].to(device)

        labels = list(inputs["label"][0])
        snip_id_vec = np.asarray([os.path.basename(lb)[:-4] for lb in labels])
        emb_id_vec = np.asarray(["_".join(os.path.basename(lb).split("_")[:-1]) for lb in labels])
        exp_id_vec = np.asarray(["_".join(os.path.basename(lb).split("_")[:-3]) for lb in labels])

        encoder_output = lit_model.encoder(x)
        # mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        # std = torch.exp(0.5 * log_var)

        # z0, eps = lit_model._sample_gauss(mu, std)

        # add latent encodings
        zm_array = np.asarray(encoder_output[0].detach().cpu())
        zs_array = np.asarray(encoder_output[1].detach().cpu())

        temp_df = pd.DataFrame(np.concatenate((exp_id_vec[:, None], emb_id_vec[:, None], snip_id_vec[:, None]), axis=1), columns=["experiment_date", "embryo_id", "snip_id"])
        temp_df[new_mu_cols + new_sigma_cols] = np.concatenate((zm_array, zs_array), axis=1)
        # temp_df[new_sigma_cols] = zs_array
        temp_df = temp_df.copy()
        df_list.append(temp_df)

    latent_df = pd.concat(df_list, ignore_index=True)
    return latent_df



def extract_embeddings_pl(
    lit_model:  LitModel,
    dataloader: torch.utils.data.DataLoader,   # {"train":…, "eval":…, "test":…}
    out_dir:     str,
    device:      Union[str, torch.device] = "cuda",
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


def assess_image_reconstructions(
    lit_model:  LitModel,
    dataloader: torch.utils.data.DataLoader,   # {"train":…, "eval":…, "test":…}
    out_dir:     str,
    device:      Union[str, torch.device] = "cuda",
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

if __name__ == "__main__":
    data_root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/"
    model_name = "20241107_ds_sweep01_optimum"
    experiments = ["20250703_chem3_35C_T00_1101"]
    calculate_morph_embeddings(data_root=data_root, model_name=model_name, experiments=experiments)
