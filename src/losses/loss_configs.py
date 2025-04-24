from typing    import Literal
from pydantic.dataclasses import dataclass
from importlib import import_module

@dataclass
class BasicLoss:
    target_path: Literal[
        "src.losses.legacy_loss_functions.VAELossBasic"
    ] = "src.losses.legacy_loss_functions.VAELossBasic"
    kld_weight: float = 1.0
    reconstruction_loss: str = "mse"

    def create_module(self):
        # dynamically import the module & class
        module_name, class_name = self.target_path.rsplit(".", 1)
        mod       = import_module(module_name)
        loss_cls  = getattr(mod, class_name)
        # instantiate with your validated kwargs
        return loss_cls(
            kld_weight=self.kld_weight,
            reconstruction_loss=self.reconstruction_loss,
        )

    # Pydantic v2 looks for `model_config`, not `Config`
    # model_config = ConfigDict(arbitrary_types_allowed=True)