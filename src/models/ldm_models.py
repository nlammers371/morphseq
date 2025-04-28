import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Dict, Any
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pythae.data.datasets import BaseDataset
from src.models.model_components.ldm_components_ae import Encoder, Decoder, DiagonalGaussianDistribution


class AutoencoderKLModel(nn.Module):
    """
    Pure PyTorch model: encoder + quantization + decoder.
    """
    def __init__(
        self,
        ddconfig: Dict[str, Any],
        embed_dim: int,
        colorize_nlabels: Optional[int] = None,
        ckpt_path: Optional[str] = None,
        ignore_keys: Optional[List[str]] = None,
        image_key: str = "image",
    ):
        super().__init__()
        self.image_key = image_key
        # core submodules
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig.get("double_z", False), "ddconfig[double_z] must be True"
        z_ch = ddconfig["z_channels"]
        self.quant_conv = nn.Conv2d(2 * z_ch, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_ch, 1)
        self.embed_dim = embed_dim
        # optional color buffer
        if colorize_nlabels is not None:
            buf = torch.randn(3, colorize_nlabels, 1, 1)
            self.register_buffer("colorize", buf)
        # load pretrained weights
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys or [])

    def init_from_ckpt(self, path: str, ignore_keys: List[str]) -> None:
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                sd.pop(k)
        self.load_state_dict(sd, strict=False)

    def encode(self, x: Tensor) -> Any:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: Tensor, sample: bool = True) -> Any:
        posterior = self.encode(x)
        z = posterior.sample() if sample else posterior.mode()
        recon = self.decode(z)
        return recon, posterior

    def get_last_layer(self) -> nn.Parameter:
        return self.decoder.conv_out.weight

    def get_input(self, batch: Dict[str, Any]) -> Tensor:
        x = batch[self.image_key]
        if x.ndim == 3:
            x = x[..., None]
        # BCHW
        return x.permute(0, 3, 1, 2).contiguous().float()