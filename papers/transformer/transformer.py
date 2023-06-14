import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder


class Tranformer(nn.Module):
  def __init__(self, d_model: int, n_blocks: int, n_heads: int, d_ff: int):
    super().__init__()
    # TODO: put every together
    pass

