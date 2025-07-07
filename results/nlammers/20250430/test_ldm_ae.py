from src.run.training import train_vae
from src.analyze.assess_vae_results import assess_vae_results

if __name__ == "__main__":

    cfg = "/home/nick/projects/morphseq/src/config_files/ldm_ae/ldmAE_test.yaml"
    train_vae(cfg=cfg)
    assess_vae_results(cfg=cfg, overwrite_flag=True)
