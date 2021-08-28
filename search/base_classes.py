
import numpy as np
from enum import Enum
from typing import NamedTuple, List, Callable, Any, Dict, Union, Tuple
from abc import ABCMeta, abstractmethod
from functools import wraps

from .model import Model
from .config import Config


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
  tags_uid: List[int]

  def tag_pos(self, tag_uid) -> int:
    return self.tags_uid.index(tag_uid)


class Anime(NamedTuple):
  uid: int
  name: str
  genres_uid: List[int]
  tags_uid: List[int]
  tags_score: List[int]

  def __eq__(self, anime: object) -> bool:
    if not isinstance(anime, Anime):
      return NotImplemented
    return anime.uid == self.uid


class Query(NamedTuple):
  text: str
  embedding: np.ndarray


class SearchResult(NamedTuple):
  query: Query
  result_embeddings: np.ndarray
  result_indexs: np.ndarray
  scores: np.ndarray
  anime_infos: List[Anime]

  def get_result(self,idx: int) -> Tuple[np.ndarray,int,int,Anime]:
    return (self.result_embeddings[idx],self.result_indexs[idx],self.scores[idx],self.anime_infos[idx])

  @staticmethod
  def new_search_result(prev_result: "SearchResult", **kwargs) -> "SearchResult":
    remaining_fields = set(prev_result._fields) - set(kwargs.keys())
    kwargs.update({field_name: getattr(prev_result,field_name) for field_name in remaining_fields})
    return SearchResult(**kwargs)


class SearchBase(NamedTuple):
  MODEL: Model
  INDEX : Any
  ALL_ANIME: Dict[int,Anime]
  ALL_TAGS: Dict[int,Tag]
  ALL_TAGS_CATEGORY: Dict[int,TagCategory]
  ALL_GENRE: Dict[int,Genre]
  EMBEDDINGS: np.ndarray
  LABELS: np.ndarray
  TEXTS: List[str]


def sort_search(f):

  @wraps(f)
  def _impl(self, *args, **kwargs) -> SearchResult:

    def sort(values):
      return [value  for _, value in sorted(zip(search_result.scores,values),reverse=True)]

    search_result = f(self,*args)
    result_embeddings = sort(search_result.result_embeddings)
    result_indexs = sort(search_result.result_indexs)
    anime_infos = sort(search_result.anime_infos)

    return SearchResult.new_search_result(search_result,result_embeddings=result_embeddings,
                                          result_indexs=result_indexs,anime_infos=anime_infos)

  return _impl


def normalize(f):

  @wraps(f)
  def _impl(self, *args, **kwargs) -> SearchResult:

    def sigmoid(x: np.ndarray) -> np.ndarray:
      return 1/(1+np.exp(-x))

    def rescale_scores(scores:np.ndarray) -> np.ndarray:
      return scores/np.linalg.norm(scores)

    search_result = f(self,*args)

    if kwargs.get("sigmoid") == True:
      scores = sigmoid(search_result.scores)
    else:
      scores = rescale_scores(search_result.scores)
    return SearchResult.new_search_result(search_result,scores=scores)

  return _impl


class ReIndexerBase(metaclass=ABCMeta):
  def __init__(self,search_base:SearchBase, config:Config) -> None:
    for name,val in zip(search_base._fields,search_base.__iter__()):
      setattr(self,name,val)

    config_name = f"{self.name()}_config"
    reindexer_config = getattr(config,config_name)
    if reindexer_config is not None:
      for name,val in zip(reindexer_config._fields,reindexer_config.__iter__()):
        setattr(self,name,val)

  @normalize
  @sort_search
  @abstractmethod
  def __call__(self, search_result:SearchResult) -> SearchResult:
    pass

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()


class ReIndexingPipelineBase:
  _reindexer_names: List[str] = []

  @classmethod
  def add_reindexer(cls, name: str, reindexer: ReIndexerBase) -> None:
    cls._reindexer_names.append(name)
    setattr(cls,name,reindexer)

  @classmethod
  def reindex_all(cls, input) -> SearchResult:
    for name in cls._reindexer_names:
      reindexer = getattr(cls,name)
      input = reindexer(input)
    return input

