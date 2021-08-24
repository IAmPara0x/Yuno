from typing import NamedTuple, List, Callable, Any, Dict
from enum import Enum
from config import Config
from abc import ABCMeta, abstractmethod
import numpy as np


class Genre(NamedTuple):
  uid: int
  name: str


class Tag(NamedTuple):
  uid: int
  name: str
  description: str
  embedding: np.ndarray
  category_uid: int


class TagCategory(NamedTuple):
  uid: int
  name: str
  description: str
  tags_uid: List[int]

  def tag_pos(self, tag_uid) -> int:
    return self.tags_uid.index(tag_uid)


class Anime(NamedTuple):
  uid: int
  name: str
  genres_uid: List[int]
  tags_uid: List[int]
  tags_score: List[int]


class Query(NamedTuple):
  text: str
  embedding: np.ndarray


class SearchResult(NamedTuple):
  query: Query
  result_embeddings: np.ndarray
  result_indexs: np.ndarray
  scores: np.ndarray
  anime_infos: List[Anime]


class SearchBase(NamedTuple):
  MODEL: Any
  INDEX : Any
  ALL_ANIME: Dict[int,Anime]
  ALL_TAGS: Dict[int,Tag]
  ALL_TAGS_CATEGORY: Dict[int,TagCategory]
  ALL_GENRE: Dict[int,Genre]
  EMBEDDINGS: np.ndarray
  LABELS: np.ndarray
  TEXTS: List[str]


class ReIndexerBase(metaclass=ABCMeta):
  def __init__(self,search_base:SearchBase, config:Config):
    for name,val in zip(search_base._fields,search_base.__iter__()):
      setattr(self,name,val)

    config_name = f"{self.name}_config"
    reindexer_config = getattr(config,config_name)

    for name,val in zip(reindexer_config._fields,reindexer_config.__iter__()):
      setattr(self,name,val)

  @abstractmethod
  def __call__(self, search_result:SearchResult) -> SearchResult:
    pass

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()

  def new_search_result(self, prev_result: SearchResult, **kwargs) -> SearchResult:
    remaining_fields = set(prev_result._fields) - set(kwargs.keys())
    kwargs.update({field_name: getattr(prev_result,field_name) for field_name in remaining_fields})
    return SearchResult(**kwargs)


class ReIndexingPipeline:
  _reindexer_names: List[str] = []

  @classmethod
  def add_reindexer(cls, name: str, reindexer: ReIndexerBase) -> None:
    cls._reindexer_names.append(name)
    setattr(cls,name,reindexer)

  @classmethod
  def reindex_all(cls, search_result: SearchResult) -> SearchResult:
    pass
