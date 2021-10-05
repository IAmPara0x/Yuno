from typing import List
from dataclasses import dataclass
from .base import SearchPipelineBase, SearchBase, Query
from .indexers import Search, TagSimIdxr, AccIdxr
from .config import DefaultConfig


def id_query(query: Query) -> Query:
  return query


@dataclass(frozen=True)
class DefaultPipleline(SearchPipelineBase):
  @staticmethod
  def new(search_base: SearchBase, config: DefaultCfg) -> "DefaultPipleline":
    query_processor_pipeline = [id_query]
    knn_search = Search.new(search_base, config.search_config)
    indexer_pipeline = [TagSimIdxr.new(search_base, config.tagsimindexer_config),
                        AccIdxr.new(search_base, config.accindexer_config)]
    return DefaultPipleline(search_base,
                            query_processor_pipeline,
                            knn_search,
                            indexer_pipeline)
