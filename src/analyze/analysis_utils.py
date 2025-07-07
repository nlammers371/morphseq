# from src.functions.dataset_utils import *
import torch
from PIL.ImImagePlugin import split
from torch.utils.data.sampler import SubsetRandomSampler
from src.run.run_utils import initialize_model, parse_model_paths
from torchvision.utils import save_image
from pytorch_lightning import Trainer
from pathlib import Path, Optional, Any
import numpy as np
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import pickle
from typing import List, Literal
from src.lightning.pl_wrappers import LitModel
from torch.utils.data import DataLoader
from src.data.dataset_configs import EvalDataConfig
from src.analyze.assess_hydra_results import get_hydra_runs, initialize_model_to_asses, parse_hydra_paths
import pytorch_lightning as pl

from src.build.pipeline_objects import Experiment

torch.set_float32_matmul_precision("medium")   # good default


class LegacyWrapper(pl.LightningModule):  # ✓ looks like Lightning
    def __init__(self, enc, dec=None):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        with torch.no_grad():
            return self.enc(x)



def recon_wrapper(data_root: str | Path,
                  model_name: str,
                  experiments: List[str],
                  out_path: str | Path,
                  batch_size: int = 64,
                  cfg: Optional[Any] = None):

    legacy = cfg is None

    if legacy:
        # ---- load the old encoder (and decoder if you need it) ----
        model_dir = Path(model_name) if Path(model_name).is_dir() else Path(model_name).parent
        enc_path = model_dir / "encoder.pkl"
        dec_path = model_dir / "decoder.pkl"  # optional

        encoder = torch.load(enc_path, map_location="cpu")
        encoder.eval()

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


    # initialize new data config for evaluation
    eval_data_config = EvalDataConfig(experiments=experiments,
                                      root=data_root,
                                      return_sample_names=True)

    eval_data_config.make_metadata()
    if legacy:
        lit_model = LegacyWrapper(encoder)
    else:
        # load model
        lit_model = LitModel.load_from_checkpoint(latest_ckpt,
                                                  model=model,
                                                  loss_fn=loss_fn,
                                                  data_cfg=eval_data_config)

    lit_model.eval()  # 1) turn off dropout / switch BN to eval
    lit_model.freeze()

    print("Generating embeddings using model " + os.path.basename(run_path) + ')')

    # get dictionary of dataloaders
    dataset = eval_data_config.create_dataset()

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=eval_data_config.num_workers,
        shuffle=False,
    )

    # construct out path
    folder_name = os.path.basename(os.path.dirname(run_path))
    mdl_name = model_config.ddconfig.name
    pips_wt = model_config.lossconfig.pips_weight
    gan_wt = model_config.lossconfig.gan_weight
    attn = model_config.ddconfig.dec_use_local_attn
    out_name = f"{mdl_name}_p{int(10*pips_wt)}_g{int(np.ceil(100*gan_wt))}_attn{attn}_GAN{model_config.lossconfig.gan_net}_{folder_name}"
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