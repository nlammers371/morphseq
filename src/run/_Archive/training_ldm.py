from omegaconf import OmegaConf, DictConfig
import hydra
from src.run.run_utils import train_vae
import torch

torch.set_float32_matmul_precision("medium")   # good default

@hydra.main(version_base="1.1",
            config_path="/src/hydra_configs",
            config_name="vae_ldm_base")

def main(cfg: DictConfig):
    # immediately turn it into a plain dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    train_vae(cfg_dict)


if __name__ == "__main__":
    main()



