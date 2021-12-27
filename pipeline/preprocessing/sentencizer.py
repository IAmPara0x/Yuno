from typing import Any, Callable, List
from dataclasses import dataclass
from itertools import zip_longest
from cytoolz.curried import reduce, compose, map, concat
import torch

Tensor = torch.Tensor


@dataclass(init=True)
class SentencizerConfig:
  tolerance: float
  threshold: Callable[[Tensor], float]
  terminate: Callable[[Tensor], bool]
  batch_size: int
  max_sent_length: int
  tol_sent_len: int


@dataclass(init=True)
class SentencizerBase:
  nlp: Callable[[str], List[str]]
  model: Callable[[List[str]], Tensor]
  cfg: SentencizerConfig
  l2_approx: Callable[[Tensor,Tensor,Tensor],Tensor]


@dataclass(init=True)
class Sentencizer(SentencizerBase):

  def __call__(self, texts: List[str]) -> List[str]:

    def acc(data, sents):
      data[0].extend(sents)
      data[1].append(data[-1]+len(sents))
      return data

    b_sents,bs_len = self.create_sents(texts)
    all_sents, pos_idxs = reduce(acc, b_sents, [[],[0]])
    all_embds = self.model(all_sents)
    batch_texts = self.filter_texts(all_sents, all_embds, pos_idxs)

    new_texts = []

    # IMPROVE: replace the loop
    for b_len in bs_len:
      new_texts.append(" ".join(batch_texts[:b_len]))
      del batch_texts[:b_len]

    return new_texts

  def create_sents(self, texts: List[str]):

    def acc_sents(text: str):
      sents = self.group_sents([sent.text for sent in self.nlp(text).sents])
      return sents

    bs = compose(list,map(acc_sents))(texts)
    bs_len = compose(list,map(len))(bs)
    return concat(bs),bs_len

  def group_sents(self, sents: List[str]) -> List[List[str]]:

    def acc(data,input_data):
      idx,sent_len = input_data
      curr_sent_len = torch.sum(sents_len[data[-1]])

      if (curr_sent_len + sent_len) >= self.cfg.max_sent_length:
        data.append([idx])
      else:
        data[-1].append(idx)
      return data

    sents_len = compose(torch.tensor,map)(lambda x: len(x.split()),sents)
    b_idxs = reduce(acc,enumerate(sents_len[1:],start=1),[[0]])

    if (len(b_idxs) > 1 and
        torch.sum(sents_len[b_idxs[-1]]) <= self.cfg.tol_sent_len):
      b_idxs[-2].extend(b_idx.pop(-1))

    batch_sents = compose(list,map(lambda s: [sents[i] for i in s]))(b_idxs)

    return batch_sents

  def filter_texts(self, all_sents: List[str],
                   all_embds: Tensor, pos_idxs: List[int]
                   ) -> List[str]:

    def acc_text(n_texts, idxs):
      s_idx,e_idx = idxs
      sents,embds = all_sents[s_idx:e_idx], all_embds[s_idx:e_idx]
      n_text = self.filter_text(sents)
      n_texts.append(n_text)
      return n_texts

    new_texts = reduce(acc_text, zip_longest(pos_idxs,pos_idxs[1:]), [])
    return new_texts

  def filter_text(self, sents: List[str], embds: Tensor) -> str:
    q, qe = sents[0], embds[0]
    sents, sents_e = sents[1:], embds[1:]

    contrib = self.l2_approx(qe,sents_e.T, sents_e)
    neg_idxs = torch.where(contrib < 0)[0].tolist()
    filtered_sents = [sent for idx, sent in enumerate(sents)
                      if idx not in neg_idxs]

    return " ".join([filtered_sents])

