from src.models.legacy_models import VAE, morphVAE
from src.models.ldm_models import AutoencoderKLModel
from dataclasses import asdict
from src.models.model_components.legacy_components import (
    EncoderConvVAE, DecoderConvVAE
)


def build_from_config(cfg):
    if cfg.name == "VAE":
        encoder = EncoderConvVAE(cfg.ddconfig)
        decoder = DecoderConvVAE(cfg.ddconfig)
        model = VAE(cfg, encoder=encoder, decoder=decoder)

    elif cfg.name == "morphVAE":
        encoder = EncoderConvVAE(cfg.ddconfig)
        decoder = DecoderConvVAE(cfg.ddconfig)
        model = morphVAE(cfg, encoder=encoder, decoder=decoder)

    elif cfg.name == "ldmAEkl":
        model = AutoencoderKLModel(ddconfig=asdict(cfg.ddconfig), embed_dim=cfg.ddconfig.embed_dim)

    else:
        raise NotImplementedError

    return model