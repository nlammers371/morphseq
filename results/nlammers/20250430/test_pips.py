from src.run.training import train_vae
from src.assess.assess_vae_results import assess_vae_results

if __name__ == "__main__":

    cfg0 = "/home/nick/projects/morphseq/src/config_files/legacy/vae_pips_test.yaml"
    train_vae(cfg=cfg0)
    assess_vae_results(cfg=cfg0, overwrite_flag=True)

    # cfg1 = "/home/nick/projects/morphseq/src/config_files/legacy/vae_pips_high_beta_test.yaml"
    # train_vae(cfg=cfg1)
    # assess_vae_results(cfg=cfg1, overwrite_flag=True)
