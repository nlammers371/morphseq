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
    # 1) Tell Hydra where your config lives and give a job name for wandering working dirs
    initialize(
        config_path="/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/hydra_configs",
        job_name="vae_sweep",
    )

    # 2) Define your hyperparam grid however you like
    sweeps = [
        {"model.lossconfig.kld_weight": 5.0,  "model.lossconfig.pips_weight": 0.5},
        {"model.lossconfig.kld_weight": 10.0, "model.lossconfig.pips_weight": 0.5},
    ]

    # 3) Loop over each override set, compose a fresh cfg, and run
    for idx, overrides in enumerate(sweeps):
        # Hydra will create a new output dir for each compose() call
        override_args = [f"{k}={v}" for k, v in overrides.items()]
        print(f"\n\n==== Starting sweep #{idx} with {override_args} ====\n\n")

        cfg: DictConfig = compose(config_name="vae_base_cluster", overrides=override_args)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Pass straight into your existing train_vae; it will pick up
        # HydraConfig.get().runtime.output_dir automatically.
        train_vae(cfg_dict)

def main(cfg: DictConfig):
    # immediately turn it into a plain dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    train_vae(cfg_dict)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("fork", force=True)
    main()



