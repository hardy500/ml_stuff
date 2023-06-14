#%%
import torch
from torch import nn

max_seq_len = 40
d_model = 512

class PositionalEncoding(nn.Module):
  def __init__(self, max_seq_len: int, d_model: int):
    super().__init__()

    self.max_seq_len = max_seq_len
    self.d_model = d_model
    self.pe = self.get_pe()

  def get_pe(self):
    even_i = torch.arange(0, self.d_model, 2).float()

    denominator = torch.pow(10e3, even_i/d_model)
    position = torch.arange(self.max_seq_len, dtype=float).unsqueeze(1)

    even_PE = torch.sin(position/denominator)
    odd_PE = torch.cos(position/denominator)

    stacked = torch.stack([even_PE, odd_PE], dim=2)
    pe = torch.flatten(stacked, start_dim=1, end_dim=2).unsqueeze(0)
    return pe

  def forward(self, embeddings_batch: torch.Tensor):
      seq_len = embeddings_batch.shape[1]
      pe_batch = self.pe[:, :seq_len].clone().detach()
      return embeddings_batch + pe_batch

if __name__ == "__main__":
    embeddings_batch = torch.randn(1, 6, 512)
    positional_encoding = PositionalEncoding(500, 512)
    print(positional_encoding(embeddings_batch).shape)
