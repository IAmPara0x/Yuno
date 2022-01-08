
# The code here is directly taken from:
# https://github.com/codertimo/BERT-pytorch
#

import torch
from torch import nn
import torch.nn.functional as F

import math


class Attention(nn.Module):
  """
  Compute 'Scaled Dot Product Attention
  """

  def forward(self, query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
  """
  A residual connection followed by a layer norm.
  Note for code simplicity the norm is first as opposed to last.
  """

  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.sequential = nn.Sequential(nn.LayerNorm(size),
                                    nn.Dropout(dropout))

  def forward(self, x, sublayer):
    "Apply residual connection to any sublayer with the same size."
    return x + self.sequential(sublayer(x))


class MultiHeadedAttention(nn.Module):
  """
  Take in model size and number of heads.
  """

  def __init__(self, h, d_model, dropout=0.1):
    super().__init__()
    assert d_model % h == 0

    # We assume d_v always equals d_k
    self.d_k = d_model // h
    self.h = h

    self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
    self.output_linear = nn.Linear(d_model, d_model)
    self.attention = Attention()

    self.dropout = nn.Dropout(p=dropout)

  def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)

    # 1) Do all the linear projections in batch from d_model => h x d_k
    query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                         for l, x in zip(self.linear_layers, (query, key, value))]

    # 2) Apply attention on all the projected vectors in batch.
    x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

    # 3) "Concat" using a view and apply a final linear.
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

    return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
  "Implements FFN equation."

  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.encoder = nn.Sequential(nn.Linear(d_model, d_ff),
                                 nn.GELU(),
                                 nn.Linear(d_ff, d_model)
                                 )

  def forward(self, x):
    return self.encoder(x)


class TransformerBlock(nn.Module):
  """
  Bidirectional Encoder = Transformer (self-attention)
  Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
  """

  def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
    """
    :param hidden: hidden size of transformer
    :param attn_heads: head sizes of multi-head attention
    :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
    :param dropout: dropout rate
    """

    super().__init__()
    self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
    self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
    self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
    self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x, mask):
    x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
    x = self.output_sublayer(x, self.feed_forward)
    return self.dropout(x)

# Embeddings


class TokenEmbedding(nn.Embedding):
  def __init__(self, vocab_size, embed_size=512):
    super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):

  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    return self.pe[:, :x.size(1)]


class SegmentEmbedding(nn.Embedding):
  def __init__(self, embed_size=512):
    super().__init__(3, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
  """
  BERT Embedding which is consisted with under features
      1. TokenEmbedding : normal embedding matrix
      2. PositionalEmbedding : adding positional information using sin, cos
      2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

      sum of all these features are output of BERTEmbedding
  """

  def __init__(self, vocab_size, embed_size, dropout=0.1):
    """
    :param vocab_size: total vocab size
    :param embed_size: embedding size of token embedding
    :param dropout: dropout rate
    """
    super().__init__()
    self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
    self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
    self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
    self.dropout = nn.Dropout(p=dropout)
    self.embed_size = embed_size

  def forward(self, sequence, segment_label):
    x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
    return self.dropout(x)
