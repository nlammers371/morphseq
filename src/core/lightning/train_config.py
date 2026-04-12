from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class LitTrainConfig:
    """Training hyperparameters for LitModel.

    Learning-rate hierarchy
    -----------------------
    All LRs derive from ``lr_base``.  The multipliers are chosen to avoid
    disrupting pretrained backbone features while still allowing the decoder
    and VAE heads to train at full speed:

    +-----------------+------------------------+-----------------------------+
    | Module          | Multiplier             | Rationale                   |
    +=================+========================+=============================+
    | Encoder trunk   | lr_base * encoder_lr_scale | Pretrained weights need |
    |                 | (default 0.1)          | slow fine-tuning            |
    +-----------------+------------------------+-----------------------------+
    | Decoder         | lr_base (1×)           | Trained from scratch        |
    +-----------------+------------------------+-----------------------------+
    | Embedding / var | lr_base * 2 (2×)       | Linear heads converge fast  |
    | heads           |                        |                             |
    +-----------------+------------------------+-----------------------------+
    | Discriminator   | lr_base / 2 or /4      | Hinge GAN: D trains slower  |
    |                 | (depends on gan_net)   | than G to avoid mode-drop   |
    +-----------------+------------------------+-----------------------------+

    Gradient clipping
    -----------------
    Set ``grad_clip_norm > 0`` (recommended: 1.0) to clip the global L2
    gradient norm before each optimiser step.  This stabilises GAN training
    and is especially helpful when switching between loss-weight ramps.
    A value of 0.0 (the default) disables clipping entirely.
    """

    benchmark: bool = True          # run initial cuDNN conv benchmark sweep
    accumulate_grad_batches: int = 2
    max_epochs: int = 100
    lr_base: float = 1e-4
    save_every_n: int = 50
    eval_gpu_flag: bool = True
    save_epochs: list[int] = field(default_factory=list)

    # --- Encoder LR scale ---
    # Set to 1.0 to give the encoder backbone the same LR as the decoder.
    # Values < 1.0 preserve pretrained feature representations; 0.1 is a
    # standard starting point for Timm/ImageNet-pretrained backbones.
    encoder_lr_scale: float = 0.1

    # --- Gradient clipping ---
    # Maximum L2 norm of the combined gradient vector before each step.
    # 0.0 disables clipping.  1.0 is a conservative default for VAE+GAN.
    grad_clip_norm: float = 0.0

    # will be overwritten by lossconfig.gan_net at config-assembly time
    gan_net: str = "patch"

    # -------------------------------------------------------------------
    # Derived LRs (read-only properties)
    # -------------------------------------------------------------------

    @property
    def lr_encoder(self) -> float:
        return self.lr_base * self.encoder_lr_scale

    @property
    def lr_decoder(self) -> float:
        return self.lr_base

    @property
    def lr_head(self) -> float:
        return self.lr_base * 2

    @property
    def lr_gan(self) -> float:
        # StyleGAN-2 discriminator is more powerful → needs a lower LR
        if "style" in self.gan_net:
            return self.lr_base / 4
        else:
            return self.lr_base / 2
