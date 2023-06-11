import torch
from torch import nn

from mha_attention import MHA
from positionwise_ff import PositionWiseFF
from add_and_norm import AddAndNorm

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


    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
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
    def __init__(self, n_blocks: int, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, mask=None):
        for block in self.encoder_blocks:
            out = block(x, mask)
        return out

if __name__ == "__main__":
    D_MODEL = 512
    N_BLOCKS = 5
    N_HEADS = 8
    BS = 30
    SEQ_LEN = 200
    D_FF = 2048
    N_LAYERS = 5
    HEAD_DIM = D_MODEL // N_HEADS

    x = torch.randn((BS, SEQ_LEN, D_MODEL)) # includes positional encoding

    encoder = Encoder(N_BLOCKS, D_MODEL, N_HEADS, D_FF)
    out = encoder(x)
    print(out.shape)