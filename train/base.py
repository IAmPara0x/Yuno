from typing import Tuple, Dict, List, Callable, Union
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
Model = Callable[[List[str]], Tuple[Tensor, Tensor]]
Tokenizer = Callable[..., Dict[str, Tensor]]


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
  sample_metric: SampleMetric
  sample_cls_size: int
  data_size: int
  all_data: Dict[DataUid, Data]
  train_uids: List[DataUid]
  eval_uids: List[DataUid]

  def __call__(self, model: Model, mode: str = "train") -> Triplet:
    assert mode == "train" or mode == "eval"

    if mode == "train":
      sample_uid = np.random.choice(self.train_uids)
    else:
      sample_uid = np.random.choice(self.eval_uids)

    return self.sample_hard(sample_uid, model)

  def sample_hard(self, uid: DataUid, model: Model) -> Triplet:
    data = self.all_data[uid]
    pos_uids = data.sample_uid(size=1, type="pos")
    neg_uids = data.sample_uid(size=self.sample_cls_size, type="neg")

    anc_data = self._sample_data([uid], type="pos")
    pos_data = self._sample_data(pos_uids, type="pos")
    neg_data = self._sample_data(neg_uids, type="neg")

    tokenized_data, embds = model(anc_data + pos_data + neg_data)

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

    return (anchors, hpos_data, hneg_data)

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
  tokenizer: Tokenizer
  model: nn.Module
  loss_fn: Callable[[Triplet], Tensor]
  optimizer: torch.optim.AdamW
  batch_size: int
  acc_steps: int
  train_steps: int
  eval_steps: int
  device: str

  def train(self) -> None:
    self.model.train()

    avg_loss, acc_loss = [], []
    step, acc_step = 0, 0
    tbar = tqdm(total=self.train_steps)

    while True:

      if step == self.train_steps:
        tbar.close()
        break

      anc, pos, neg = self.batch_sample()
      anc_embds, pos_embds, neg_embds = self.model(anc=anc, pos=pos, neg=neg)
      loss = self.loss_fn((anc_embds, pos_embds, neg_embds))
      tbar.set_description(f"AVG_LOSS: {np.average(avg_loss):.5f},\
                             LOSS:{loss.item():.5f},\
                             STEP: {step}")
      loss /= self.acc_steps
      loss.backward()
      acc_loss.append(loss.item())
      acc_step += 1

      if acc_step % self.acc_steps == 0:
        step += 1
        self.optimizer.step()
        self.model.zero_grad()
        avg_loss.append(sum(acc_loss))
        tbar.update(1)
        acc_loss = []

  def eval(self) -> None:
    self.model.eval()
    avg_loss, step = [], 0
    tbar = tqdm(total=self.eval_steps)

    while True:
      if step == self.eval_steps:
        tbar.close()
        break
      anc, pos, neg = self.batch_sample(mode="eval")

      with torch.no_grad():
        anc_embds, pos_embds, neg_embds = self.model(anc=anc, pos=pos, neg=neg)
        loss = self.loss_fn((anc_embds, pos_embds, neg_embds))
        avg_loss.append(loss.item())
        tbar.set_description(f"AVG_LOSS: {np.average(avg_loss):.5f},\
                               LOSS:{loss.item():.5f},\
                               STEP: {step}")
      tbar.update(1)
      step += 1

  def batch_sample(self, batch_size=None, mode="train") -> Triplet:

    if batch_size is None:
      batch_size = self.batch_size

    b_anc, b_pos, b_neg = [], [], []
    for _ in range(batch_size):
      anc, pos, neg = self.sampler(self._offline_sample, mode=mode)
      b_anc.append(anc)
      b_pos.append(pos)
      b_neg.append(neg)

    return (torch.vstack(b_anc),
            torch.vstack(b_pos),
            torch.vstack(b_neg))

  def _offline_sample(self, texts: List[str]) -> Tuple[Tensor, Tensor]:
    ttexts = self.tokenizer(texts, padding=True,
                            truncation=True,
                            return_tensors="pt"
                            )["input_ids"].to(self.device)
    with torch.no_grad():
      embds = self.model(ttexts=ttexts)
    return (ttexts, embds)
