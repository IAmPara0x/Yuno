
from enum import Enum,auto

class TagIndexing(Enum):
  ALL = auto()
  PER_CATEGORY = auto()

class TagIndexingMetric(Enum):
  CosineSimilarity = auto()
  L2Norm = auto()

class TagIndexerConfig(NamedTuple):
  TAG_INDEXING_METHOD: TagIndexing
  TAG_INDEXING_METRIC: TagIndexingMetric

class SearchConfig(NamedTuple):
  EMBEDDING_DIM: int
  TOP_K:int

class Config:
  @classmethod
  def add_config(cls,name,obj):
    setattr(cls,name,obj)

class DefaultConfig(Config):
  search_config = SearchConfig(1280,128)
  tagindexer_config = TagIndexerConfig(TagIndexing.PER_CATEGORY,TAG_INDEXING_METRIC.CosineSimilarity)
