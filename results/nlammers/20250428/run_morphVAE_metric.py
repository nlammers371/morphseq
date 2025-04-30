from src.run.training import train_vae
from src.assess.assess_vae_results import assess_vae_results

if __name__ == "__main__":

    # # beta = 0.1
    # cfg0 = "/home/nick/projects/morphseq/src/config_files/legacy/morph_vae_low_beta_high_m.yaml"
    # train_vae(cfg=cfg0)
    # assess_vae_results(cfg=cfg0, overwrite_flag=True)
    #
    # # # beta = 1
    # cfg1 = "/home/nick/projects/morphseq/src/config_files/legacy/morph_vae_low_beta_z128.yaml"
    # train_vae(cfg=cfg1)
    # assess_vae_results(cfg=cfg1, overwrite_flag=True)
    #
    # # beta = 0.1 ; bio_kld = True
    # cfg2 = "/home/nick/projects/morphseq/src/config_files/legacy/morph_vae_low_beta_high_m_z128.yaml"
    # train_vae(cfg=cfg2)
    # assess_vae_results(cfg=cfg2, overwrite_flag=True)

    # cfg3 = "/home/nick/projects/morphseq/src/config_files/legacy/morph_vae_beta_bio.yaml"
    cfg3 = "/home/nick/projects/morphseq/src/config_files/legacy/initial_tests/morph_vae_low_beta.yaml"
    train_vae(cfg=cfg3)
    assess_vae_results(cfg=cfg3, overwrite_flag=True)
