import torch
from torch import nn

from mha_attention import MHA
from add_and_norm import AddAndNorm
from positionwise_ff import PositionWiseFF

class DecoderBlock(nn.Module):
  def __init__(self, d_model: int, n_heads: int, d_ff: int):
    super().__init__()

    self.QW_1 = nn.Linear(d_model, d_model)
    self.KW_1 = nn.Linear(d_model, d_model)
    self.VW_1 = nn.Linear(d_model, d_model)

    self.mha_1 = MHA(d_model, n_heads)
    self.add_and_norm1 = AddAndNorm(d_model)

    # -----------------------------------------------------------------------

    self.QW_2 = nn.Linear(d_model, d_model)
    self.KW_2 = nn.Linear(d_model, d_model)
    self.VW_2 = nn.Linear(d_model, d_model)

    self.mha_2 = MHA(d_model, n_heads)
    self.add_and_norm2 = AddAndNorm(d_model)

    # -----------------------------------------------------------------------

    self.ffn = PositionWiseFF(d_model, d_ff)
    self.add_and_norm3 = AddAndNorm(d_model)

  def forward(self,
              x: torch.Tensor,
              encoder_out: torch.Tensor,
              source_mask: torch.Tensor,
              target_mask: torch.Tensor) -> torch.Tensor:

    q_1 = self.QW_1(x)
    k_1 = self.KW_1(x)
    v_1 = self.VW_1(x)
    mha_1_out = self.add_and_norm1(self.mha_1(q_1, k_1, v_1, target_mask)[0], x)

    # -----------------------------------------------------------------------

    q_2 = self.QW_2(mha_1_out)
    k_2 = self.KW_2(encoder_out)
    v_2 = self.VW_2(encoder_out)
    mha_2_out = self.add_and_norm1(self.mha_1(q_2, k_2, v_2, source_mask)[0], mha_1_out)

    # -----------------------------------------------------------------------

    ffn_out = self.add_and_norm3(self.ffn(mha_2_out), mha_2_out)

    return ffn_out

class Decoder(nn.Module):
  def __init__(self, d_model: int, n_heads: int, d_ff: int, n_blocks: int):
    super().__init__()

    self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)])

  def forward(self,
              x: torch.Tensor,
              encoder_out: torch.Tensor,
              source_mask: torch.Tensor,
              target_mask: torch.Tensor) -> torch.Tensor:

    if x.dtype != encoder_out.dtype:
      x = x.type(torch.float32)
    for block in self.blocks:
      x = block(x, encoder_out, source_mask, target_mask)
    return x

if __name__ == "__main__":
    x = torch.randn(1, 1, 512)
    encoder_output = torch.randn(1, 6, 512)
    src_mask = torch.randn(1, 1, 1, 6)
    trg_mask = torch.randn(1, 1, 1, 1)

    decoder = Decoder(512, 8, 20, 2)
    out = decoder(x, encoder_output, src_mask, trg_mask)
    print(out.shape)