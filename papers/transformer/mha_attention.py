import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Optional

def attention(Q: torch.Tensor,
              K: torch.Tensor,
              V: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

  d_k = Q.shape[-1]
  attn = torch.matmul(Q, K.transpose(-2, -1))
  scaled_attn = attn/torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

  if mask:
    scaled_attn = scaled_attn.masked_fill(mask==0, float('-inf'))

  attn_weights = F.softmax(scaled_attn, dim=-1)

  output = torch.matmul(attn_weights, V)
  return output, attn_weights

class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
      super().__init__()

      self.d_model = d_model
      self.n_heads = n_heads
      self.head_dim = d_model//n_heads

      self.output_linear = nn.Linear(d_model, d_model)

    def _split_into_heads(self,
                          Q: torch.Tensor,
                          K: torch.Tensor,
                          V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      # Split heads
      batch_size = Q.shape[0]
      Q = Q.view(batch_size, -1, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, heads_dim)
      K = K.view(batch_size, -1, self.n_heads, self.head_dim)
      V = K.view(batch_size, -1, self.n_heads, self.head_dim)

      # Tranpose dimension for matmul
      Q = Q.transpose(1, 2)  # (batch_size, n_heads, seq_len, heads_dim)
      K = K.transpose(1, 2)
      V = V.transpose(1, 2)
      return Q, K, V

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                mask=None) -> Tuple[torch.Tensor, torch.Tensor]:

      batch_size = Q.shape[0]

      # Calculate attention for each head
      Q, K, V = self._split_into_heads(Q, K, V)
      attn, attn_weight = attention(Q, K, V)   # (batch_size, n_heads, seq_len, heads_dim)

      # Concatenate and linear transform the attened values
      attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
      output = self.output_linear(attn)
      return output, attn_weight