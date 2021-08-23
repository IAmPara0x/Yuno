from typing import NamedTuple, List, Callable, Any
from enum import Enum
from config import Config
from abc import ABCMeta, abstractmethod


#TODO: create TagCategory enum for different tags category
#TODO: create Genre enum for different genre


class Tag(NamedTuple):
  uid: int
  name: str
  category: TagCategory
  description: str
  embedding: np.ndarray


class Anime(NamedTuple):
  uid: int
  name: str
  genres: List[Genre]
  tags: List[Tag]
  tags_score: List[int]


class SearchResult(NamedTuple):
  query_embedding: np.ndarray
  result_embeddings: np.ndarray
  result_indexs: np.ndarray
  scores: np.ndarray
  anime_infos: List[Anime]


class SearchBase(NamedTuple):
  MODEL: Any
  INDEX : Any
  ALL_ANIME: dict
  ALL_TAGS: dict
  ALL_TAGS_CATEGORY: List[TagCategory]
  ALL_GENRE: List[Genre]
  EMBEDDINGS: np.ndarray
  LABELS: np.ndarray
  TEXTS: List[str]


class ReIndexerBase(metaclass=ABCMeta):
  def __init__(self,search_base:SearchBase,config:Config):
    for name,val in zip(search_base._fields,search_base.__iter__()):
      setattr(self,name,val)

    config_name = f"{Class.__name__.lower()}_config" #FIXME: find a way to get the class name dynamically
    reindexer_config = getattr(config,config_name)

    for name,val in zip(search_config._fields,reindexer_config.__iter__()):
      setattr(self,name,val)

  @abstractmethod
  def __call__(self, search_result:SearchResult) -> SearchResult:
    pass


class ReIndexingPipeline:
  _reindexer_names = []

  @classmethod
  def add_reindexer(cls, name: str, reindexer: ReIndexerBase):
    self._reindexer_names(name)
    setattr(cls,name,reindexer)

  @classmethod
  def reindex_all(cls, search_result: SearchResult) -> SearchResult:
    return
