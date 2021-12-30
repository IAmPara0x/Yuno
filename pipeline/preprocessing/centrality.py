from typing import Callable, List, Tuple
from dataclasses import dataclass
from cytoolz.curried import reduce
import torch

Tensor = torch.Tensor


@dataclass
class CentralityBase:
  model: Callable[[List[str]], Tensor]
  prob_threshold: float
  batch_size: int


@dataclass
class Centrality(CentralityBase):

  def __call__(self, texts: List[str]) -> List[str]:

    def cum_prob(data:Tuple[float,List[str]],
                 input: Tuple[float,str]
                 ):
      cum_p,itexts = data
      p,text = input
      if p < self.prob_threshold:
        cum_p += p
        itexts.append(text)

      return (cum_p,itexts)

    embds = []
    for idx in range(0, len(texts), self.batch_size):
      b_input = texts[idx:idx+self.batch_size]
      b_embds = self.model(b_input)
      embds.append(b_embds)

    embds = torch.vstack(embds)
    state_vec = self.eig_centrality(embds)
    _, new_texts = reduce(cum_prob,
                          sorted(zip(state_vec,texts),reverse=True),
                          (0,[]))
    return new_texts

  @staticmethod
  def eig_centrality(mat: Tensor) -> List[float]:
    adj_mat = torch.cosine_similarity(mat.unsqueeze(1), mat, dim=-1)
    B = adj_mat / torch.sum(adj_mat,dim=1)
    e,v = torch.eig(B,eigenvectors=True)

    state_vec = v[:,0]/torch.sum(v[:,0])
    return state_vec.tolist()
