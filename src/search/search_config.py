
from enum import Enum,auto

class TagIndexing(Enum):
  ALL = auto()
  PER_CATEGORY = auto()

class TagIndexingMetric(Enum):
  CosineSimilarity = auto()
  L2Norm = auto()

class SearchConfig(NamedTuple):
  EMBEDDING_DIM: int
  TOP_K:int
  TAG_INDEXING_METHOD: TagIndexing
  TAG_INDEXING_METRIC: TagIndexingMetric


DEFAULT_SEARCH_CONFIG = SearchConfig(1280,128,TagIndexing.PER_CATEGORY,TagIndexingMetric.CosineSimilarity)
