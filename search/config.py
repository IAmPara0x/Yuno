from enum import Enum, auto
from typing import NamedTuple, Callable, NewType
import numpy as np
from dataclasses import dataclass


Scores = NewType("Scores",np.ndarray)


class TagIndexingMethod(Enum):
  all = auto()
  per_category = auto()


class TagIndexingMetric(Enum):
  cosine_similarity = auto()
  l2norm = auto()


class TagIndexerConfig(NamedTuple):
  indexing_method: TagIndexingMethod
  indexing_metric: TagIndexingMetric


class AccIndexingMetric(Enum):
  add = auto()
  multiply = auto()


class AccIndexerConfig(NamedTuple):
  acc_fn: Callable[[Scores], float]


class SearchConfig(NamedTuple):
  embedding_dim: int
  top_k: int
  dist_fn: Callable[[np.ndarray], Scores]


@dataclass(init=True)
class Config:
  search_config: SearchConfig
  tagindexer_config: TagIndexerConfig
  accindexer_config: AccIndexerConfig

