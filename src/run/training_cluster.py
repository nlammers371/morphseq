from omegaconf import OmegaConf, DictConfig
import hydra
from src.run.run_utils import train_vae
import torch
from hydra import initialize, compose

torch.set_float32_matmul_precision("medium")   # good default

# @hydra.main(version_base="1.1",
#             config_path="/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/hydra_configs",
#             config_name="vae_base_cluster")


def sequential_sweep():

    # 2) Define your hyperparam grid however you like
    sweeps = [
        {"model.lossconfig.kld_weight": 5.0,  "model.lossconfig.pips_weight": 0.5},
        {"model.lossconfig.kld_weight": 10.0, "model.lossconfig.pips_weight": 0.5},
    ]

    # 3) Loop over each override set, compose a fresh cfg, and run
    for idx, overrides in enumerate(sweeps):
        # 1) Tell Hydra where your config lives and give a job name for wandering working dirs
        with initialize(
            config_path="src/hydra_configs",
            job_name="vae_sweep",
            version_base="1.1"
        ):
            # 2) Compose your base config + this override
            cfg = compose(
                config_name="vae_base",
                overrides=[f"{k}={v}" for k, v in overrides.items()]
            )
            # 3) Convert and call your existing train function
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            train_vae(cfg_dict)

def main(cfg: DictConfig):
    # immediately turn it into a plain dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    train_vae(cfg_dict)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("fork", force=True)
    sequential_sweep()



