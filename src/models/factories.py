from src.models.legacy_models import VAE, morphVAE
from src.models.ldm_models import AutoencoderKLModel
from dataclasses import asdict
from src.models.model_components.legacy_components import (
    EncoderConvVAE, DecoderConvVAE, DecoderConvVAEUpsamp
)
from src.models.model_components.timm_components import TimmEncoder, UniDecLite
from src.models.model_components.ldm_components_ae import (WrappedLDMDecoder,
                                                           WrappedLDMEncoderPool
                                                           )


def build_from_config(cfg):
    if cfg.name == "VAE":
        if "convVAE" in cfg.ddconfig.name:
            encoder = EncoderConvVAE(cfg.ddconfig)
            decoder = DecoderConvVAEUpsamp(cfg.ddconfig)
        elif "ldmVAE" in cfg.ddconfig.name:
            encoder = WrappedLDMEncoderPool(asdict(cfg.ddconfig))
            decoder = WrappedLDMDecoder(asdict(cfg.ddconfig))
        elif cfg.ddconfig.is_timm_arch:
            encoder = TimmEncoder(cfg.ddconfig)
            decoder = UniDecLite(cfg=cfg.ddconfig, enc_ch_last=encoder.embed_dim)
        else:
            raise NotImplementedError
        model = VAE(cfg, encoder=encoder, decoder=decoder)

    elif cfg.name == "morphVAE":
        if "convVAE" in cfg.ddconfig.name:
            encoder = EncoderConvVAE(cfg.ddconfig)
            decoder = DecoderConvVAEUpsamp(cfg.ddconfig)
        elif "ldmVAE" in cfg.ddconfig.name:
            encoder = WrappedLDMEncoderPool(asdict(cfg.ddconfig))
            decoder = WrappedLDMDecoder(asdict(cfg.ddconfig))
        elif cfg.ddconfig.is_timm_arch:
            encoder = TimmEncoder(cfg.ddconfig)
            decoder = UniDecLite(cfg=cfg.ddconfig, enc_ch_last=encoder.embed_dim)
        else:
            raise NotImplementedError
        model = morphVAE(cfg, encoder=encoder, decoder=decoder)

    else:
        raise NotImplementedError

    return model