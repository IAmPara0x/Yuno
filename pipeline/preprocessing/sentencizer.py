from typing import Any, Callable, List
from dataclasses import dataclass
from itertools import zip_longest
from cytoolz.curried import reduce
import torch

Tensor = torch.Tensor

@dataclass(init=True)
class SentencizerBase:
  nlp: Callable[[str], List[str]]
  model: Callable[[List[str]], Tensor]
  config: Config

@dataclass(init=True)
class Config:
  tolerance: float
  threshold: Callable[[Tensor], float]
  terminate: Callable[[Tensor], bool]
  batch_size: int
  max_sent_lenght: int


@dataclass(init=True)
class Sentencizer(SentencizerBase):

  def __call__(self, texts: List[str]) -> List[str]:

    def acc(data, sents):
      data[0].extend(sents)
      data[1].append(data[-1]+len(sents))
      return data

    b_sents,b_sents_idx = self.create_sents(texts)
    all_sents, pos_idxs = reduce(acc, b_sents, [[],[0]])
    all_embds = self.model(all_sents)
    batch_texts = self.filter_texts(all_sents, all_embds, pos_idxs)

    new_texts = []

    for idx in range(len(texts)):
      new_texts.append(batch_texts.pop(0))
      b_sents_idx.pop(0)

      while b_sents_idx[0] == idx:
        new_texts[-1] += " " + batch_texts.pop(0)
        b_sents_idx.pop(0)

    return new_texts

  def create_sents(self, texts: List[str]):

    def group_sents(sents):
      grp_sents = []
      curr_sent = []
      curr_sent_len = 0

      for sent in sents:
        sent_len = len(sent.split())
        self.curr_sent_len += sent_len
        if self.curr_sent_len >= self.max_sent_lenght:
          grp_sents.append(curr_sent)
          curr_sent = [sent]
          curr_sent_len = sent_len
        else:
          curr_sent.appned(sent)

      if len(grp_sents) > 1 and curr_sent_len < 64:
        grp_sents[-2].extend(grp_sents.pop())

      return grp_sents

    batch_sents_idx = []
    batch_sents = []

    for idx, text in enumerate(texts):
      sents = group_sents([sent.text for sent in self.nlp(text)])
      batch_sents.extend(sents)
      batch_sents_idx.extend([idx]*len(sents))

    return batch_sents, batch_sents_idx

  def filter_texts(self, all_sents: List[str],
                   all_embds: Tensor, pos_idxs: List[int]
                   ) -> List[str]:

    def acc_text(n_texts, idxs):
      s_idx,e_idx = idxs
      n_text = self.filter_text(all_sents[s_idx:e_idx],
                                all_embds[s_idx:e_idx])
      n_texts.append(n_text)
      return n_texts

    new_texts = reduce(acc_text, zip_longest(pos_idxs,pos_idxs[1:]), [])
    return new_texts

  def filter_text(self, sents: List[str], embds: Tensor) -> str:
    q, qe = sents[0], embds[0]
    sents, sents_e = sents[1:], embds[1:]

    contrib = l2_approx(qe,sents_e.T, sents_e)
    neg_idxs = torch.where(contrib < 0)[0].tolist()
    filtered_sents = [sent for idx, sent in enumerate(sents)
                      if idx not in neg_idxs]

    return " ".join([filtered_sents])

