from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
from cytoolz.curried import reduce  # type: ignore
import torch

Tensor = torch.Tensor
Output = Tuple[List[int], List[str]]
Prob = float


@dataclass
class CentralityBase:
  model: Callable[[List[str]], Tensor]
  prob_threshold: float
  batch_size: int


@dataclass
class Centrality(CentralityBase):

  def __call__(self, texts: List[str], prob_threshold: Optional[float] = None) -> Output:

    if prob_threshold is None:
      prob_threshold = self.prob_threshold

    embds = []
    for idx in range(0, len(texts), self.batch_size):
      b_input = texts[idx:idx+self.batch_size]
      b_embds = self.model(b_input)
      embds.append(b_embds)

    embds = torch.vstack(embds)

    assert embds.shape[0] == len(texts)

    state_vec = self.eig_centrality(embds, self.batch_size)
    datas = sorted(zip(state_vec, [*enumerate(texts)]), reverse=True)

    acc_prob: float = 0
    output: Output = ([],[])

    for data in datas:
      prob,idx,text = data[0],data[1]

      if acc_prob < self.prob_threshold:
        output[0].append(idx)
        output[1].append(text)
        acc_prob += prob
      else:
        break

    return output

  @staticmethod
  def eig_centrality(mat: Tensor, batch_size: int) -> List[Prob]:

    adj_mat = []

    for idx in range(0, mat.shape[0], batch_size):
      padj_mat = torch.cosine_similarity(mat[idx:idx+batch_size].unsqueeze(1),
                                         mat, dim=-1)
      adj_mat.append(padj_mat)

    adj_mat = torch.vstack(adj_mat)

    assert (adj_mat.shape[0] == mat.shape[0] and
            adj_mat.shape[1] == mat.shape[0])

    B = adj_mat / torch.sum(adj_mat, dim=1)
    e, v = torch.eig(B, eigenvectors=True)

    state_vec = v[:, 0]/torch.sum(v[:, 0])
    return state_vec.tolist()
