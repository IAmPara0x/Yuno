
from functools import wraps
from toolz.curried import curry,reduce,map,apply,compose
import numpy as np
from enum import Enum
from typing import NamedTuple, List, Callable, Any, Dict, Union, Tuple
from abc import ABCMeta, abstractmethod

from .model import Model
from .config import Config
from . import utils


class Genre(NamedTuple):
  uid: int
  name: str

  def __eq__(self, other_genre: object) -> bool:
    if not isinstance(other_genre, Tag):
      return NotImplemented
    return other_genre.uid == self.uid


class Tag(NamedTuple):
  uid: int
  name: str
  description: str
  embedding: np.ndarray
  category_uid: int

  def __eq__(self, other_tag: object) -> bool:
    if not isinstance(other_tag, Tag):
      return NotImplemented
    return other_tag.uid == self.uid


class TagCategory(NamedTuple):
  uid: int
  name: str
  tags_uid: List[int]

  def tag_pos(self, tag_uid) -> int:
    return self.tags_uid.index(tag_uid)

  def __eq__(self, other_category: object) -> bool:
    if not isinstance(other_category, TagCategory):
      return NotImplemented
    return other_category.uid == self.uid


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

  def __hash__(self):
    return hash(self.uid)


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
      return [value  for _, value in sorted(zip(search_result.scores,values),
                      key=lambda x: x[0],reverse=True)]

    search_result = f(self,*args)
    result_embeddings = compose(np.array,sort)(search_result.result_embeddings)
    result_indexs = compose(np.array,sort)(search_result.result_indexs)
    anime_infos = sort(search_result.anime_infos)
    scores = compose(np.array,sorted)(search_result.scores,reverse=True)

    return SearchResult.new_search_result(search_result,result_embeddings=result_embeddings,
                                          result_indexs=result_indexs,anime_infos=anime_infos,scores=scores)

  return _impl


def normalize(**kwargs):

  def wrapper(f):

    @wraps(f)
    def _impl(self, *args) -> SearchResult:

      search_result = f(self,*args)

      if kwargs.get("sigmoid") == True:
        scores = utils.sigmoid(search_result.scores)
      else:
        scores = utils.rescale_scores(search_result.scores)
      return SearchResult.new_search_result(search_result,scores=scores)

    return _impl

  return wrapper


class ReIndexerBase(metaclass=ABCMeta):
  def __init__(self,search_base:SearchBase, config:Config) -> None:
    for name,val in zip(search_base._fields,search_base.__iter__()):
      setattr(self,name,val)

    config_name = f"{self.name()}_config"
    reindexer_config = getattr(config,config_name,None)
    if reindexer_config is not None:
      for name,val in zip(reindexer_config._fields,reindexer_config.__iter__()):
        setattr(self,name,val)

  @abstractmethod
  def __call__(self, search_result:SearchResult) -> SearchResult:
    pass

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()


class ReIndexingPipelineBase:
  _reindexer_names: List[str] = []

  def __call__(self, input:str) -> SearchResult:
    return self.reindex_all(input)

  def add_reindexer(self, name: str, reindexer: ReIndexerBase) -> None:
    self._reindexer_names.append(name)
    setattr(self,name,reindexer)

  def reindex_all(self, input) -> SearchResult:
    f = partial(getattr,self)
    output = reduce(lambda input,name: f(name)(input),self._reindexer_names,input)
    return output
