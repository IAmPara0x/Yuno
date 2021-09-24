from typing import List
from dataclasses import dataclass
from .base_classes import SearchPipelineBase, SearchBase, Query
from .reindexers import Search, TagIndexer, AccIndexer
from .config import Config


def id_query(query: Query) -> Query:
  return query


@dataclass(frozen=True)
class DefaultPipleline(SearchPipelineBase):
  @staticmethod
  def new(search_base: SearchBase, config: Config) -> "DefaultPipleline":
    query_processor_pipeline = [id_query]
    knn_search = Search.new(search_base, config.search_config)
    indexer_pipeline = [AccIndexer.new(search_base, config.accindexer_config),
                        TagIndexer.new(search_base, config.tagindexer_config)]

    return DefaultPipleline(query_processor_pipeline, knn_search, indexer_pipeline)
