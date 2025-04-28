from src.run.training import train_vae

if __name__ == "__main__":

    cfg0 = "/home/nick/projects/morphseq/src/config_files/morph_vae_low_beta.yaml"
    train_vae(cfg=cfg0)

    cfg1 = "/home/nick/projects/morphseq/src/config_files/morph_vae_baseline.yaml"
    train_vae(cfg=cfg1)
