from enum import Enum,auto
from typing import NamedTuple,Callable
from .base_classes import Scores

class TagIndexingMethod(Enum):
  all = auto()
  per_category = auto()


class TagIndexingMetric(Enum):
  cosine_similarity = auto()
  l2norm = auto()


class TagReIndexerConfig(NamedTuple):
  indexing_method: TagIndexingMethod
  indexing_metric: TagIndexingMetric


class AccIndexingMetric(Enum):
  add = auto()
  multiply = auto()


class AccReIndexerConfig(NamedTuple):
  acc_fn: Callable[[Scores],float]


class SearchConfig(NamedTuple):
  embedding_dim: int
  top_k:int


class Config:
  @classmethod
  def add_config(cls,name,obj):
    setattr(cls,name,obj)


# class DefaultConfig(Config):
#   search_config = SearchConfig(1280,128)
#   tagreindexer_config = TagReIndexerConfig(TagIndexing.per_category,TagIndexingMetric.cosine_similarity)
#   accreindexer_config = AccReIndexerConfig(AccIndexingMetric.add)
