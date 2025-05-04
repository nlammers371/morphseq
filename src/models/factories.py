from src.models.legacy_models import VAE, morphVAE
from src.models.ldm_models import AutoencoderKLModel
from dataclasses import asdict
from src.models.model_components.legacy_components import (
    EncoderConvVAE, DecoderConvVAE, DecoderConvVAEUpsamp
)
from src.models.model_components.ldm_components_ae import (WrappedLDMDecoder,
                                                           WrappedLDMEncoderPool
                                                           )


def build_from_config(cfg):
    if cfg.name == "VAE":
        if "convAE" in cfg.ddconfig.name:
            encoder = EncoderConvVAE(cfg.ddconfig)
            decoder = DecoderConvVAEUpsamp(cfg.ddconfig)
        else:
            raise NotImplementedError
        model = VAE(cfg, encoder=encoder, decoder=decoder)

    elif cfg.name == "morphVAE":
        if "convAE" in cfg.ddconfig.name:
            encoder = EncoderConvVAE(cfg.ddconfig)
            decoder = DecoderConvVAEUpsamp(cfg.ddconfig)
        else:
            raise NotImplementedError
        model = morphVAE(cfg, encoder=encoder, decoder=decoder)

    elif cfg.name == "VAEFancy":
        if "ldmVAE" in cfg.ddconfig.name:
            encoder = WrappedLDMEncoderPool(asdict(cfg.ddconfig))
            decoder = WrappedLDMDecoder(asdict(cfg.ddconfig))
        else:
            raise NotImplementedError
        model = VAE(cfg, encoder=encoder, decoder=decoder)

    elif cfg.name == "morphVAEFancy":
        if "ldmVAE" in cfg.ddconfig.name:
            encoder = WrappedLDMEncoderPool(asdict(cfg.ddconfig))
            decoder = WrappedLDMDecoder(asdict(cfg.ddconfig))
        else:
            raise NotImplementedError

        model = morphVAE(cfg, encoder=encoder, decoder=decoder)
    # elif cfg.name == "ldmAEkl":
    #     model = AutoencoderKLModel(ddconfig=asdict(cfg.ddconfig), embed_dim=cfg.ddconfig.embed_dim)

    else:
        raise NotImplementedError


    return model