import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_prob)
    #equations
    #PE(pos,2i) = sin(pos/10000^2i/dmodel)
    #PE(pos, 2i + 1) = cos(pos/10000^2i/dmodel)

    #arange goes from 0 to max lenght
    position = torch.arange(max_len).unsqueeze(1) #Pos
    exp_term = torch.arange(0, d_model, 2) #2i
    div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model)) #10000^-2i/dmodel
    pe = torch.zeros(1, max_len, d_model) #(1, T, D) to brodcast to (N, T, D)
    pe[0, :, 0::2] = torch.sin(position * div_term) #PE(pos,2i) = sin(pos/10000^2i/dmodel)
    pe[0, :, 1::2] = torch.cos(position * div_term) #PE(pos, 2i + 1) = cos(pos/10000^2i/dmodel)
    self.register_buffer("pe", pe) # save and load correctly register and does not required gradient

  def forward(self, x):
    # x.shape: N x T x D
    x = x + self.pe[:, :x.size(1), :] #accessing register buffer
    return self.dropout(x)
