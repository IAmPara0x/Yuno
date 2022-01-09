from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class Distill(nn.Module):

  def __init__(self, num_layers: int, d_modelT: int,
               d_modelS: int, layermap: Callable[[int], int],
               attn_headmap: Callable[[int], int],
               ):
    super(Distill, self).__init__()

    self.num_layers = num_layers
    self.d_modelT = d_modelT
    self.d_modelS = d_modelS
    self.layermap = layermap
    self.attn_headmap = attn_headmap

    self.hidden_proj = [nn.Linear(d_modelS, d_modelT)
                        for _ in range(num_layers)]

  def forward(self, outputT, outputS):
    hidden_statesT = outputT.hidden_states
    hidden_statesS = outputS.hidden_states

    attentionsT = outputT.attentions
    attentionsS = outputS.attentions

    layers_loss = []

    for i in range(self.num_layers):
      g = self.layermap(i)

      hidden_stateS = self.hidden_proj[i](hidden_statesS[i])
      hidden_stateT = hidden_statesT[g]

      if i != 0:
        mha_matS = attentionsS[i]
        mha_matT = attentionsT[g]

        layer_loss = (self.attention_loss(mha_matT, mha_matS) +
                      self.hidden_loss(hidden_stateT, hidden_stateS))
      else:
        layer_loss = self.hidden_loss(hidden_stateT, hidden_stateS)

      layers_loss.append(layer_loss)
    return torch.mean(torch.vstack(layers_loss))

  def attention_loss(self, mha_matT, mha_matS):

    n_headsS = mha_matS.size(1)

    matT = mha_matT.transpose(0, 1)  # (n_heads, batch_size, seq_len, seq_len)
    matS = mha_matS.transpose(0, 1)  # (n_heads', batch_size, seq_len, seq_len)

    attn_loss = []

    for idxS in range(n_headsS):
      idxT = self.attn_headmap(idxS)
      attn_loss.append(F.mse_loss(matS[idxS], matT[idxT]))

    loss = torch.mean(torch.vstack(attn_loss).cuda())
    return loss

  def hidden_loss(self, hidden_stateT, hidden_stateS):
    return F.mse_loss(hidden_stateT, hidden_stateS)
