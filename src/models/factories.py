from src.models.legacy_models import VAE, morphVAE
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

    else:
        raise NotImplementedError

    return model