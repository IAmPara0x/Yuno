from dataclasses import dataclass
from .base import SearchPipelineBase, SearchBase, Query
from .indexers import Search, TagSimIdxr, AccIdxr, NodeIdxr, TopkIdxr
from .config import Config


def id_query(query: Query) -> Query:
  return query


#FIXME: note type checking config.
@dataclass(frozen=True)
class DefaultPipleline(SearchPipelineBase):
  @staticmethod
  def new(search_base: SearchBase, config: Config) -> "DefaultPipleline":
    query_processor_pipeline = [id_query]
    knn_search = Search.new(search_base, config.search_cfg)
    indexer_pipeline = [
        NodeIdxr.new(search_base, config.nodeindexer_cfg),
        TagSimIdxr.new(search_base, config.tagsimindexer_cfg),
        AccIdxr.new(search_base, config.accindexer_cfg),
        TopkIdxr.new(search_base, config.topkindexer_cfg)
    ]
    return DefaultPipleline(search_base, query_processor_pipeline, knn_search,
                            indexer_pipeline)
