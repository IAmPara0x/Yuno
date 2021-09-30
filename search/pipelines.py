from typing import List
from dataclasses import dataclass
from .base import SearchPipelineBase, SearchBase, Query
from .indexers import Search, TagSimIndexer, AccIndexer
from .config import DefaultConfig


def id_query(query: Query) -> Query:
  return query


@dataclass(frozen=True)
class DefaultPipleline(SearchPipelineBase):
  @staticmethod
  def new(search_base: SearchBase, config: DefaultConfig) -> "DefaultPipleline":
    query_processor_pipeline = [id_query]
    knn_search = Search.new(search_base, config.search_config)
    indexer_pipeline = [AccIndexer.new(search_base, config.accindexer_config),
                        TagSimIndexer.new(search_base, config.tagsimindexer_config)]
    return DefaultPipleline(search_base,
                            query_processor_pipeline,
                            knn_search,
                            indexer_pipeline)
