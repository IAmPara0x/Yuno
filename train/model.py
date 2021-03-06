from typing import List, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from .config import Config

Tensor = torch.Tensor
Triplet = Tuple[Tensor, Tensor, Tensor]


class FCN(nn.Module):
  def __init__(self, layer_dims: List[int], dropout: float):
    super().__init__()

    fcn = []
    for dim1, dim2 in zip(layer_dims, layer_dims[1:]):
      fcn.extend([
          nn.Linear(dim1, dim2),
          nn.Dropout(dropout, inplace=False),
          nn.Tanh(),
      ])
    fcn.extend([
        nn.Linear(layer_dims[-1], layer_dims[-1]),
    ])
    self.encoder = nn.Sequential(*fcn)

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()

  def forward(self, x: Tensor) -> Tensor:
    return self.encoder(x)


class Model(nn.Module):
  def __init__(self, config: Config):
    super().__init__()

    config_name = f"{self.name()}_config"
    model_config = getattr(config, config_name)
    for name, val in zip(model_config._fields, model_config.__iter__()):
      setattr(self, name, val)

    self.transformer = RobertaModel.from_pretrained(self.pretrained,
                                                    output_hidden_states=True)
    self.tanh = nn.Tanh()
    self.encoder = FCN(self.embedding_layers, self.dropout)
    self.feats = self.transformer.pooler.dense.out_features

  def forward(self, x: Union[Tensor, Triplet]) -> Union[Tensor, Triplet]:
    if isinstance(x, tuple) and len(x) == 3:
      return (self.forward_once(x[0]), self.forward_once(x[1]),
              self.forward_once(x[2]))
    elif isinstance(x, torch.Tensor):
      return self.forward_once(x)

  def forward_once(self, x):
    outputs = self.transformer(x)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix + 1):
      hmix.append(hidden_states[-i][:, 0].reshape((-1, 1, self.feats)))

    hmix_tensor = torch.cat(hmix, 1)
    pool_tensor = torch.mean(hmix_tensor, 1)
    pool_tensor = self.tanh(pool_tensor)
    embeddings = self.encoder(pool_tensor)
    return embeddings

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()
