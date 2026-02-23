from pydantic.dataclasses import dataclass # as pydantic_dataclass
from dataclasses import field
from typing import Any

@dataclass
class LitTrainConfig:

    benchmark: bool=True  # Perform initial conv optimization sweep?
    accumulate_grad_batches: int=2
    max_epochs: int=100
    lr_base: float=1e-4
    save_every_n: int=50
    eval_gpu_flag: bool=True
    save_epochs: list[int] = field(default_factory=list)

    gan_net: str="patch" # this will be overwritten by whatever is passed to lossconfig

    @property
    def lr_encoder(self) -> float:
        return self.lr_base / 10

    @property
    def lr_decoder(self) -> float:
        return self.lr_base

    @property
    def lr_head(self) -> float:
        return self.lr_base * 2

    @property
    def lr_gan(self) -> float:
        if "style" in self.gan_net:
            return self.lr_base / 4
        else:
            return self.lr_base / 2
