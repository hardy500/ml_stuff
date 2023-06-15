import torch
from torch import nn
import torch.nn.functional as F

class PositionWiseFF(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
      super().__init__()
      self.fc1 = nn.Linear(d_model, d_ff)
      self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
      return self.fc2(F.relu(self.fc1(x)))