from src.core.models.legacy_models import VAE, metricVAE
from dataclasses import asdict
from src.core.models.model_components.legacy_components import (
    EncoderConvVAE, DecoderConvVAEUpsamp
)
from src.core.models.model_components.timm_components import TimmEncoder, UniDecLite
from src.core.models.model_components.ldm_components_ae import (WrappedLDMDecoder,
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

    elif cfg.name == "metricVAE":
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
        model = metricVAE(cfg, encoder=encoder, decoder=decoder)

    else:
        raise NotImplementedError

    return model