from omegaconf import OmegaConf, DictConfig
import hydra
from src.run.run_utils import train_vae, collect_results_recursive
import torch
import os
from pathlib import Path
import warnings
import wandb

# 1) Silence all FutureWarning / UserWarning from torchvision, lpips, torch, etc.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_float32_matmul_precision("medium")   # good default


def _abs_ancestor(path: str, levels: str) -> str:
    p = Path(path).resolve()                   # make it absolute
    idx = int(levels) - 1                      # levels= "2" → index 1
    return str(p.parents[idx])

OmegaConf.register_new_resolver("ancestor", _abs_ancestor)

@hydra.main(version_base="1.1",
            config_path="/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/hydra_configs",
            config_name="base_cluster")

def main(cfg: DictConfig):
    # immediately turn it into a plain dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    train_vae(cfg_dict)
    results_dir = os.path.join(cfg.model.dataconfig.root, "training_outputs", "")
    collect_results_recursive(results_dir=results_dir)


if __name__ == "__main__":
    main()






