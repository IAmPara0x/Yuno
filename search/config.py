from enum import Enum, auto
from typing import NamedTuple, Callable, Optional
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


@dataclass(frozen=True)
class TagIdxrCfg:
  indexing_method: TagIdxingMethod
  indexing_metric: TagIdxingMetric


class AccIdxingMetric(Enum):
  add = auto()
  multiply = auto()


@dataclass(frozen=True)
class AccIdxrCfg:
  """
   config for AccIdxr
  """
  score_fn: Callable[[Scores], float]


@dataclass(frozen=True)
class SearchCfg:
  embedding_dim: int
  top_k: int
  weight: float


@dataclass(frozen=True)
class TagSimIdxrCfg:
  use_negatives: bool
  use_sim: bool
  weight: float


@dataclass(frozen=True)
class NodeIdxrCfg:
  weight: float
  device: str = "cpu"


@dataclass(frozen=True)
class ContextIdxrCfg:
  sim_thres: float
  cutoff_sim: float
  topk: int
  device: str = "cpu"


@dataclass(frozen=True)
class Config:
  search_cfg: Optional[SearchCfg]
  accindexer_cfg: Optional[AccIdxrCfg]
  tagsimindexer_cfg: Optional[TagSimIdxrCfg]
  nodeindexer_cfg: Optional[NodeIdxrCfg]
  contextidxr_cfg: Optional[ContextIdxrCfg]


def inv(x: np.ndarray) -> Scores:
  return Scores(1/x)


def acc_sum(scores: Scores) -> float:
  return reduce(operator.add, scores, 0)


@dataclass(frozen=True)
class DefaultCfg:
  search_cfg = SearchCfg(1280, 256, 1.25)
  accindexer_cfg = AccIdxrCfg(acc_sum)
  tagsimindexer_cfg = TagSimIdxrCfg(True, False, 2)
  nodeindexer_cfg = NodeIdxrCfg(1.0, "cuda")
  contextidxr_cfg = ContextIdxrCfg(0.65,0.7,50,"cuda")
