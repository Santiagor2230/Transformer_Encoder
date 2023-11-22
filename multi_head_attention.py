import math

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
  def __init__(self, d_k, d_model, n_heads):
    super().__init__()

    # Assume d_v=d_k
    self.d_k = d_k
    self.n_heads = n_heads

    # ALL the attentions all at once with n_heads
    self.key = nn.Linear(d_model, d_k*n_heads)
    self.query = nn.Linear(d_model, d_k*n_heads)
    self.value = nn.Linear(d_model, d_k*n_heads)

    # final linear layer
    self.fc = nn.Linear(d_k * n_heads, d_model)

  def forward(self, q, k, v, mask=None):
    q = self.query(q) # N x T x (hd_k)
    k = self.key(k) # N x T x (hd_k)
    v = self.value(v) # N x T x (hd_v)

    N = q.shape[0] #sample
    T = q.shape[1] #sequence

    # change the shape to:
    # (N, T, h, d_k) -> (N, h, T, d_k)
    # in order for matrix multiply to work properly
    q = q.view(N, T, self.n_heads, self.d_k).transpose(1,2)
    k = k.view(N, T, self.n_heads, self.d_k).transpose(1,2)
    v = v.view(N, T, self.n_heads, self.d_k).transpose(1,2)

    #compute attention weights
    #(N, h, T, d_k) x (N, h, d_k, T) --> (N, h, T, T)
    # scaling score  = query * key Transpose/ square root of(dimension)
    attn_scores = q @ k.transpose(-2,-1)/ math.sqrt(self.d_k)

    #we mask
    if mask is not None:
      attn_scores = attn_scores.masked_fill(
          #mask:(N,T)-> mask[:, None, None, :] -> mask:(N,1,1,T)
          #this allows us to broadcast correctly
          mask[:, None, None, :] == 0, float('-inf')
      )

    #attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)

    #compute attention weights-weighted values
    # (N, h, T, T) X (N, h, T, d_k) --> (N, h, T, d_k)
    A = attn_weights @ v

    #reshape it back before final linear layer
    A = A.transpose(1,2) # (N, h, T, d_k) --> (N, T, h, d_k)
    #contiguous allows us to set our values correctly in memory
    A = A.contiguous().view(N, T, self.d_k * self.n_heads) #(N, T, h*d_k)

    #projection
    return self.fc(A)