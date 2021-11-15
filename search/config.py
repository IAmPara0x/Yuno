from enum import Enum, auto
from typing import NamedTuple, Callable, Optional
import numpy as np
from dataclasses import dataclass
from cytoolz.curried import reduce  # type: ignore
import operator


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
   AccIdxrCfg config class for AccIdxr.

   ...

   Parameters
   ----------
   score_fn : Callable[[ndarray], float]
  """
  score_fn: Callable[[np.ndarray], float]


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
  stride: int
  sim_threshold: float
  device: str = "cpu"


@dataclass(frozen=True)
class TopkIdxrCfg:
  topk: int
  tag_thres: float


@dataclass(init=True,frozen=True)
class Config:
  search_cfg: Optional[SearchCfg]
  accindexer_cfg: Optional[AccIdxrCfg]
  tagsimindexer_cfg: Optional[TagSimIdxrCfg]
  nodeindexer_cfg: Optional[NodeIdxrCfg]
  topkindexer_cfg: Optional[TopkIdxrCfg]
  # contextidxr_cfg: Optional[ContextIdxrCfg]


def acc_sum(scores: np.ndarray) -> float:
  return reduce(operator.add, scores, 0)


search_cfg = SearchCfg(1280, 256, 1.25)
accindexer_cfg = AccIdxrCfg(acc_sum)
tagsimindexer_cfg = TagSimIdxrCfg(True, False, 2)
nodeindexer_cfg = NodeIdxrCfg(1.0, "cuda")
topkindexer_cfg = TopkIdxrCfg(5,0.65)
# contextidxr_cfg = ContextIdxrCfg(0.65, 0.7,"cuda")

default_cfg = Config(search_cfg,
                     accindexer_cfg,
                     tagsimindexer_cfg,
                     nodeindexer_cfg,
                     topkindexer_cfg
                     )
