
# The code here is inspired by:
#https://github.com/dreamgonfly/BERT-pytorch


import torch
from torch import nn


class TransformerLayer(nn.Module):
  def __init__(self, dim: int, heads: int, intermediate_dim: int):
    super(TransformerLayer, self).__init__()
    self.attention_layer = MHA(dim=dim,n_heads=n_heads)
    self.fcn_layer = FCN(intermediate_dim=intermediate_dim)

  def forward(self, input):
    pass


class MHA(nn.Module):
  def __init__(self, dim: int, heads: int, dropout_prob: float):
    super(MHA, self).__init__()

    assert dim % n_heads == 0, "dim must be divisble with n_heads"
    self.d_head = dim // heads
    self.heads = heads

    self.query_proj = nn.Linear(dim,dim)
    self.key_proj = nn.Linear(dim,dim)
    self.value_proj = nn.Linear(dim,dim)
    self.dropout = nn.Dropout(dropout_prob)
    self.softmax = nn.Softmax(dim=3)

  def forward(self, input):

    bs, seq_len, d_model = input.size()
    d_head = d_model // self.heads

    assert d_head == self.d_head

    query = self.query_proj(input)
    key = self.key_proj(input)
    value = self.value_proj(input)

