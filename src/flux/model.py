import torch.nn as nn
import torch
from typing import Sequence


# base neural vector field (NVF) model
class MLPVelocityField(nn.Module):
    """Simple MLP that maps latent embedding z → velocity f_θ(z)."""

    def __init__(self, dim: int, hidden: Sequence[int] = (256, 128, 128), activation=nn.GELU):
        super().__init__()
        layers = []
        last = dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

# neural vector field (NVF) model with gamma scaling for experiment-specific scaling
# class NVF_with_gamma(nn.Module):
#     def __init__(self, d, num_exp):
#         super().__init__()
#         self.vf = NVF(d)
#         self.log_gamma = nn.Parameter(torch.zeros(num_exp))  # learns exp-wise scaling

#     def forward(self, z, exp_idx):
#         gamma = torch.exp(self.log_gamma[exp_idx])  # strictly positive scale factor
#         return self.vf(z) * gamma