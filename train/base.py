from typing import Tuple, Dict, List, Callable, Union, Any
from enum import Enum, auto
from dataclasses import dataclass
from tqdm import tqdm
from cytoolz.curried import concat, compose

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
Triplet = Tuple[Tensor, Tensor, Tensor]
DataUid = int


class SampleMetric(Enum):
  l2_norm = auto()
  l1_norm = auto()
  cosine_similarity = auto()


@dataclass(init=True)
class Data:
  uid: DataUid
  neg_uids: List[DataUid]
  pos_uids: List[DataUid]
  anchors: List[str]
  positives: List[str]

  @staticmethod
  def _sample(data: Union[List[DataUid], List[str]], size):
    return np.random.choice(data, size, replace=False).tolist()

  def sample_data(self, size: int, type: str) -> List[str]:
    assert type in ["pos", "neg", "anc"]

    if type == "anc":
      data = self._sample(self.anchors, size)
    else:
      data = self._sample(self.positives, size)
    return data

  def sample_uid(self, size: int, type: str) -> List[DataUid]:
    assert type in ["pos", "neg"]

    if type == "pos":
      data = self._sample(self.pos_uids, size)
    else:
      data = self._sample(self.neg_uids, size)
    return data


@dataclass(init=True)
class Sampler:
  model: Callable[[List[str]], Tuple[Tensor, Tensor]]
  sample_metric: SampleMetric
  sample_cls_size: int
  data_size: int
  all_data: Dict[DataUid, Data]
  train_uids: List[DataUid]
  test_uids: List[DataUid]

  def __call__(self, data: str = "train") -> Triplet:
    assert data == "train" or data == "test"

    if data == "train":
      sample_uid = np.random.choice(self.train_uids)
    else:
      sample_uid = np.random.choice(self.test_uids)

    return self.sample_hard(sample_uid)

  def sample_hard(self, uid: DataUid) -> Triplet:
    data = self.all_data[uid]
    pos_uids = data.sample_uid(size=1, type="pos")
    neg_uids = data.sample_uid(size=self.sample_cls_size, type="neg")

    anc_data = self._sample_data([uid], type="pos")
    pos_data = self._sample_data(pos_uids, type="pos")
    neg_data = self._sample_data(neg_uids, type="neg")

    tokenized_data, embds = self.model(anc_data + pos_data + neg_data)

    tanc_data = tokenized_data[:self.data_size]
    tpos_data = tokenized_data[self.data_size:self.data_size*2]
    tneg_data = tokenized_data[self.data_size*2:]

    anc_embds = embds[:self.data_size]
    pos_embds = embds[self.data_size:self.data_size*2]
    neg_embds = embds[self.data_size*2:]

    pos_scores = self.pairwise_metric((anc_embds, pos_embds))
    hard_positives = torch.max(pos_scores, 1).indices

    if anc_embds.shape == neg_embds.shape:
      neg_scores = self.pairwise_metric((anc_embds, neg_embds))
    else:
      anc_size, neg_size = anc_embds.size(0), neg_embds.size(0)
      h, w = (neg_size - anc_size), anc_embds.shape[1]
      padding = torch.zeros((h, w)).to(anc_embds.device)
      padded_anchors = torch.cat((anc_embds, padding))
      neg_scores = self.pairwise_metric((padded_anchors, neg_embds))
      neg_scores = neg_scores[:anc_size]

    hard_negatives = torch.min(neg_scores, 1).indices

    hard_triplets = [(tanc_data[a], tpos_data[p], tneg_data[n])
                     for a, p, n in zip(range(self.data_size),
                                        hard_positives,
                                        hard_negatives)
                     ]

    anchors = torch.vstack([i[0] for i in hard_triplets])
    hpos_data = torch.vstack([i[1] for i in hard_triplets])
    hneg_data = torch.vstack([i[2] for i in hard_triplets])

    return anchors, hpos_data, hneg_data

  def pairwise_metric(self, input: Tuple[Tensor, Tensor]) -> Tensor:

    assert isinstance(input, tuple)
    assert isinstance(input[0], torch.Tensor)
    assert input[0].shape == input[1].shape

    m1, m2 = input
    if self.sample_metric == SampleMetric.l2_norm:
      squared_norm1 = torch.matmul(m1, m1.T).diag()
      squared_norm2 = torch.matmul(m2, m2.T).diag()
      middle = torch.matmul(m2, m1.T)

      scores_mat = (squared_norm1.unsqueeze(0) - 2 * middle +
                    squared_norm2.unsqueeze(1)).T

    elif self.sample_metric == SampleMetric.l1_norm:
      diff_mat = torch.abs(m1.unsqueeze(1) - m2)
      scores_mat = torch.sum(diff_mat, dim=-1)

    elif self.sample_metric == SampleMetric.cosine_similarity:
      scores_mat = F.cosine_similarity(m1.unsqueeze(1), m2, dim=-1)

    else:
      raise Exception("sample_metric should be in SampleMetric enum.")
    return scores_mat

  def _sample_data(self, uids: List[DataUid], type: str) -> List[str]:
    return compose(list,
                   concat)([self.all_data[uid].sample_data(size=self.data_size,
                                                           type=type,)
                            for uid in uids
                            ])


@dataclass(init=True)
class Trainer:
  sampler: Sampler
  tokenizer: Callable[[List[str]], Tensor]
  model: nn.Module
  loss_fn: Callable[[Triplet], Tensor]
  optimizer: Any
  batch_size: int
  acc_steps: int
  train_steps: int
  test_steps: int

  def train(self):
    self.model.train()
    pass
