from enum import Enum, auto
from typing import NamedTuple, Callable, NewType, Optional
import numpy as np
from dataclasses import dataclass
from toolz.curried import reduce  # type: ignore
import operator

from .base import Scores


class TagIdxingMethod(Enum):
  all = auto()
  per_category = auto()


class TagIdxingMetric(Enum):
  cosine_similarity = auto()
  l2norm = auto()


class TagIdxrCfg(NamedTuple):
  indexing_method: TagIdxingMethod
  indexing_metric: TagIdxingMetric


class AccIdxingMetric(Enum):
  add = auto()
  multiply = auto()


class AccIdxrCfg(NamedTuple):
  score_fn: Callable[[Scores], float]


class SearchCfg(NamedTuple):
  embedding_dim: int
  top_k: int
  weight: float


class TagSimIdxrCfg(NamedTuple):
  use_negatives: bool
  use_sim: bool
  weight: float

class NodeIdxrCfg(NamedTuple):
  weight: float


@dataclass(frozen=True)
class Config:
  search_cfg: Optional[SearchCfg]
  accindexer_cfg: Optional[AccIdxrCfg]
  tagsimindexer_cfg: Optional[TagSimIdxrCfg]
  nodeindexer_cfg: Optional[NodeIdxrCfg]


def inv(x: np.ndarray) -> Scores:
  return Scores(1/x)


def acc_sum(scores: Scores) -> float:
  return reduce(operator.add, scores, 0)


class DefaultCfg():
  search_cfg = SearchCfg(1280, 256, 1.25)
  accindexer_cfg = AccIdxrCfg(acc_sum)
  tagsimindexer_cfg = TagSimIdxrCfg(True,True,2)
  nodeindexer_cfg = NodeIdxrCfg(1.0)
