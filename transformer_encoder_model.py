import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from positional_encoder import PositionalEncoding

class Encoder(nn.Module):
  def __init__(self,
               vocab_size,
               max_len,
               d_k,
               d_model,
               n_heads,
               n_layers,
               n_classes,
               dropout_prob):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
    transformer_blocks = [
        TransformerBlock(
            d_k,
            d_model,
            n_heads,
            dropout_prob
        ) for _ in range(n_layers)]

    self.transformer_blocks = nn.Sequential(*transformer_blocks) #encapsulate in sequential
    self.ln = nn.LayerNorm(d_model)
    self.fc = nn.Linear(d_model, n_classes) #outputs n_classes


  def forward(self, x, mask=None):
    x = self.embedding(x)
    x = self.pos_encoding(x)
    for block in self.transformer_blocks:
      x = block(x, mask)

    #many-to-one (x has the shape N x T x D)
    x = x[:, 0, :] #x: (N x T x D) --> x: (N X D) single output

    x = self.ln(x)
    x = self.fc(x)

    return x