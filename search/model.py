from typing import List
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel # type: ignore


class FCN(nn.Module):
  def __init__(self, input_dim:int, embedding_dim:int, n_hidden_layers:int, n_hidden_units:List[int], dropout_prob:float):
    super(FCN, self).__init__()

    assert n_hidden_layers != 0
    assert len(n_hidden_units) + 1 == n_hidden_layers

    encoder_layers = []
    for i in range(n_hidden_layers):
      if i == n_hidden_layers - 1:
        out_dim = embedding_dim
        encoder_layers.extend(
          [
            nn.Linear(input_dim, out_dim),
          ])
        continue
      else:
        out_dim = n_hidden_units[i]

      encoder_layers.extend( # type: ignore
        [
          nn.Linear(input_dim, out_dim), # type: ignore
          nn.Tanh(),# type: ignore
          nn.Dropout(dropout_prob, inplace=False),# type: ignore
        ]
      )
      input_dim = out_dim
    self.encoder = nn.Sequential(*encoder_layers)

  def forward(self, x_array: torch.Tensor) -> torch.Tensor:
      return self.encoder(x_array)


class EmbeddingModel(nn.Module):
  def __init__(self, model_name: str) -> None:
    super().__init__()
    self.roberta = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
    self.hid_mix = 5
    self.feats = self.roberta.pooler.dense.out_features
    self.embedding_layers = FCN(self.feats, 1280, 4, [896, 1024, 1152], 0.1)
    self.tanh = nn.Tanh()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    outputs = self.roberta(x)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:,0].reshape((-1,1,self.feats)))

    hmix_tensor = torch.cat(hmix,1)
    pool_tensor = torch.mean(hmix_tensor,1)
    pool_tensor = self.tanh(pool_tensor)
    embeddings = self.embedding_layers(pool_tensor)
    return embeddings

class Model:
  def __init__(self, model_name:str, model_weight_path:str) -> None:
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.embedding_model = EmbeddingModel(model_name).to(self.device)
    self.embedding_model.load_state_dict(torch.load(model_weight_path))
    self.embedding_model.eval()
    self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

  def __call__(self,text:str) -> np.ndarray:
    tokenized_text = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
    with torch.no_grad():
      return self.embedding_model(tokenized_text).squeeze().cpu().numpy()
