# from typing import List
# from dataclasses import dataclass
# from .base_classes import SearchPipelineBase,SearchBase,Query
# from .reindexers import Search,TagReIndexer,AccReIndexer
# from .config import Config

# def id_query(query:Query) -> Query:
#   return query

# @dataclass
# class DefaultPipleline(SearchPipelineBase):
#   search_base: SearchBase

#   def _post_init__(self):
#     self.add_query_processor(id_query)
#     self.add_query_processor(id_query)


