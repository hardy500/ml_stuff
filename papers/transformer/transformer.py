import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder
from positional_encoding import PositionalEncoding

import math

class Tranformer(nn.Module):
  def __init__(self,
               src_vocab_size: int,
               trg_vocab_size: int,
               d_model: int,
               n_blocks: int,
               n_heads: int,
               d_ff: int):

    super().__init__()
    self.d_model = d_model

    # encoder
    self.src_emb = nn.Embedding(src_vocab_size, d_model)
    self.src_pos_emb = PositionalEncoding(500, d_model)
    self.encoder = Encoder(d_model, n_heads, d_ff, n_blocks)

    # decoder
    self.trg_emb = nn.Embedding(trg_vocab_size, d_model)
    self.trg_pos_emb = PositionalEncoding(500, d_model)
    self.decoder = Decoder(d_model, n_heads, d_ff, n_blocks)

    # linear mapping
    self.linear = nn.Linear(d_model, trg_vocab_size)

    # sharing weight
    self.src_emb.weight = self.trg_emb.weight
    self.linear.weight = self.trg_emb.weight

  def encode(self,
             src_token_ids: torch.Tensor,
             src_mask: torch.Tensor) -> torch.Tensor:

    src_emb = self.src_emb(src_token_ids) * math.sqrt(self.d_model)
    src_emb = self.src_pos_emb(src_emb)
    return self.encoder(src_emb, src_mask)

  def decode(self, trg_token_ids: torch.Tensor,
             encoder_out: torch.Tensor,
             src_mask: torch.Tensor,
             trg_mask: torch.Tensor) -> torch.Tensor:

    trg_emb = self.trg_emb(trg_token_ids) * math.sqrt(self.d_model)
    trg_emb = self.trg_pos_emb(trg_emb)
    return self.linear(self.decoder(trg_emb, encoder_out, src_mask, trg_mask))

  def forward(self,
              src_token_ids: torch.Tensor,
              trg_token_ids: torch.Tensor,
              src_mask: torch.Tensor,
              trg_mask: torch.Tensor):

    encoder_out = self.encode(src_token_ids, src_mask)
    decoder_out = self.decode(trg_token_ids, encoder_out, src_mask, trg_mask)
    return decoder_out

if __name__ == "__main__":
  #src_token_ids = torch.randint(0, 60_000, size=(1, 6))
  # expect: (1, 5, 512)

  trg_token_ids = torch.randint(0, 60_000, size=(1, 1))
  encoder_out = torch.rand(1, 5, 512)
  src_mask = torch.randn(1, 1, 1, 6)
  trg_mask = torch.randn(1, 1, 1, 1)
  # expect: (1, 1, 60_000)

  d_model = 512
  n_blocks = 6
  vocab_size = 60_000
  d_ff = 2048
  n_heads = 8

  transformer = Tranformer(vocab_size, vocab_size, d_model, n_blocks, n_heads, d_ff)
  print(transformer.decode(trg_token_ids, encoder_out, src_mask, trg_mask).shape)