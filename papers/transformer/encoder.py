import torch
from torch import nn

from mha_attention import MHA
from positionwise_ff import PositionWiseFF
from add_and_norm import AddAndNorm

from typing import Optional

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
      super().__init__()

      self.QW = nn.Linear(d_model, d_model)
      self.KW = nn.Linear(d_model, d_model)
      self.VW = nn.Linear(d_model, d_model)

      self.mha = MHA(d_model, n_heads)
      self.add_and_norm1 = AddAndNorm(d_model)

      self.ffn = PositionWiseFF(d_model, d_ff)
      self.add_and_norm2 = AddAndNorm(d_model)


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
      # Linear projection of Q, K, V
      Q = self.QW(x)  # (batch_size, seq_len, d_model)
      K = self.KW(x)
      V = self.VW(x)

      mha_out, _ = self.mha(Q, K, V, mask)
      mha_out = self.add_and_norm1(x, mha_out)

      ffn_out = self.ffn(mha_out)
      ffn_out = self.add_and_norm2(mha_out, ffn_out)
      return ffn_out

class Encoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_blocks: int):
      super().__init__()
      self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)])

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor]=None) -> torch.Tensor:


      if x.dtype != mask:
        x = x.type(torch.float32)
      for block in self.encoder_blocks:
        out = block(x, mask)
      return out

if __name__ == "__main__":
  token_id = torch.randn(1, 6)
  pad_mask = (token_id != 0).unsqueeze(-2).unsqueeze(0)
  src_emb = torch.randn(1, 6, 512)

  encoder = Encoder(2, 512, 8, 22)
  print(encoder(src_emb, pad_mask).shape)