from enum import Enum, auto
from typing import NamedTuple, Callable, NewType
import numpy as np
from dataclasses import dataclass
from toolz.curried import reduce  # type: ignore
import operator

from .base import Scores


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


@dataclass(frozen=True)
class Config:
  search_config: SearchConfig
  tagindexer_config: TagIndexerConfig
  accindexer_config: AccIndexerConfig


def inv(x: np.ndarray) -> Scores:
  return Scores(1/x)


def acc_sum(scores: Scores) -> float:
  return reduce(operator.add, scores, 0)


@dataclass(frozen=True)
class DefaultConfig(Config):
  search_config = SearchConfig(1280, 128, inv)
  tagindexer_config = TagIndexerConfig(TagIndexingMethod.per_category,
                                       TagIndexingMetric.cosine_similarity)
  accindexer_config = AccIndexerConfig(acc_sum)
