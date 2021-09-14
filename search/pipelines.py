from typing import List
from .base_classes import ReIndexingPipelineBase
from .reindexers import Search,TagReIndexer,SearchBase,AccReIndexer
from .config import Config

class DefaultPipleline(ReIndexingPipelineBase):
  def __init__(self, search_base:SearchBase, config: Config) -> None:
    self.add_reindexer("knn_search", Search(search_base,config))
    self.add_reindexer("acc_indexer", AccReIndexer(search_base,config))
    self.add_reindexer("tag_reindexer", TagReIndexer(search_base,config))

