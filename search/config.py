
from enum import Enum,auto
from typing import NamedTuple

class TagIndexing(Enum):
  all = auto()
  per_category = auto()

class TagIndexingMetric(Enum):
  cosine_similarity = auto()
  l2norm = auto()

class TagIndexerConfig(NamedTuple):
  tag_indexing_method: TagIndexing
  tag_indexing_metric: TagIndexingMetric

class SearchConfig(NamedTuple):
  embedding_dim: int
  top_k:int

class Config:
  @classmethod
  def add_config(cls,name,obj):
    setattr(cls,name,obj)

class DefaultConfig(Config):
  search_config = SearchConfig(1280,128)
  tagindexer_config = TagIndexerConfig(TagIndexing.per_category,TagIndexingMetric.cosine_similarity)
