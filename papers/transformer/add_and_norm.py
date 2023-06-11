import torch
from torch import nn

class AddAndNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init_()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(x + residual)
