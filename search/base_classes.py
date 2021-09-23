from typing import List, Callable, Any, Dict, Union, Tuple, NewType, Optional
from dataclasses import dataclass, field
from returns.maybe import Some,Nothing,Maybe
from functools import wraps
from toolz.curried import reduce,map,compose,concat# type: ignore
import numpy as np
from abc import ABCMeta, abstractmethod

from .model import Model
from .config import Config
from . import utils


GenreId = NewType("GenreId", int)
TagId = NewType("TagId", int)
TagCategoryId = NewType("TagCategoryId", int)
AnimeId = NewType("AnimeId", int)
Scores = NewType("Scores",np.ndarray)


@dataclass(init=True,repr=True,eq=True,order=False,frozen=True)
class Genre:
  uid: GenreId
  name: str = field(compare=False)


@dataclass(init=True,repr=True,eq=True,order=False,frozen=True)
class Tag:
  uid: TagId
  name: str = field(compare=False)
  description: str = field(compare=False)
  category_uid: int = field(compare=False)
  embedding: np.ndarray = field(compare=False,repr=False)


@dataclass(init=True,repr=True,eq=True,order=False,frozen=True)
class TagCategory:
  uid: TagCategoryId
  name: str = field(compare=False)
  tags_uid: List[TagId] = field(compare=False)


@dataclass(init=True,repr=True,eq=True,order=False,frozen=True)
class Anime:
  uid: AnimeId
  name: str = field(compare=False)
  genres_uid: List[GenreId] = field(compare=False)
  tags_uid: List[TagId] = field(compare=False)
  tags_score: np.ndarray = field(compare=False)


@dataclass(init=True,repr=True,eq=True,order=False,frozen=True)
class Query:
  text: str
  embedding: Optional[np.ndarray] = field(compare=False)


@dataclass(init=True,repr=False,eq=True,order=True,frozen=True)
class SearchResult:
  query: Query = field(compare=False)
  result_embeddings: np.ndarray = field(compare=False)
  result_indexs: np.ndarray = field(compare=False)
  scores: Scores = field(compare=True)
  anime_infos: List[Anime] = field(compare=False)

  def __getitem__(self,idx:int) -> Tuple[np.ndarray,int,int,Anime]:
    return self.get_result(idx)

  def get_result(self,idx: int) -> Tuple[np.ndarray,int,int,Anime]:
    data = (self.result_embeddings,self.result_indexs,self.scores,self.anime_infos)
    return compose(tuple,map)(lambda x: x[idx], data)

  @staticmethod
  def new_search_result(prev_result: "SearchResult", **kwargs) -> "SearchResult":
    remaining_fields = set(prev_result.__dict__.keys()) - set(kwargs.keys())
    kwargs.update({field_name: getattr(prev_result,field_name) for field_name in remaining_fields})
    return SearchResult(**kwargs)


@dataclass(init=True,repr=False,eq=False,order=False,frozen=True)
class SearchData:
  _embeddings: np.ndarray
  _labels: np.ndarray
  _texts: np.ndarray

  def __getitem__(self,idx: int) -> Tuple[np.ndarray,AnimeId,str]:
    data = (self._embeddings,self._labels,self._texts)
    return compose(tuple,map)(lambda x: x[idx], data)


@dataclass(init=True,repr=False,eq=False,order=False,frozen=True)
class SearchBase:
  model: Model
  index : Any
  search_data: SearchData
  _animes: Dict[AnimeId,Anime]
  _tags: Dict[TagId,Tag]
  _tags_categories: Dict[TagCategoryId,TagCategory]
  _genres: Dict[GenreId,Genre]

  def get_tag(self,id:TagId) -> Tag:
    return self._tags[id]

  def get_tagcategory(self,id:TagCategoryId) -> TagCategory:
    return self._tags_categories[id]

  def get_genre(self,id:GenreId) -> Genre:
    return self._genres[id]

  def get_anime(self,id:AnimeId) -> Anime:
    return self._animes[id]

  def get_searchdata(self, idx) -> Tuple[np.ndarray,int,str]:
    return self.search_data[idx]


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


def normalize(norm_f: Optional[Callable[[Scores],Scores]]=None):
  def wrapper(f: Callable[[SearchResult],SearchResult]):
    @wraps(f)
    def _impl(self, *args) -> SearchResult:

      search_result = f(self,*args)

      scores = Maybe.from_optional(norm_f
                ).bind_optional(
                  lambda fn: fn(search_result.scores)
                ).or_else_call(
                  lambda: search_result.scores
                )

      return SearchResult.new_search_result(search_result,scores=scores)
    return _impl
  return wrapper


@dataclass
class ReIndexerBase:
  search_base: SearchBase

  def __call__(self, search_result:SearchResult) -> SearchResult:
    raise NotImplementedError


@dataclass
class QueryProcessorBase:
  def __call__(self, search_result:Query) -> Query:
    raise NotImplementedError


@dataclass(init=True,repr=False,eq=False,order=False,frozen=True)
class SearchPipelineBase(metaclass=ABCMeta):
  query_processor_pipeline: List[Callable[[Query],Query]]
  search: Callable[[Query],SearchResult]
  reindexer_pipeline: List[Callable[[SearchResult],SearchResult]]

  def add_query_processor(self,f:Callable[[Query],Query]):
    self.query_processor_pipeline.append(f)

  def add_reindxer(self,f:Callable[[SearchResult],SearchResult]):
    self.reindexer_pipeline.append(f)

  def __call__(self, query:Query) -> SearchResult:
    return reduce(lambda f,x: f(x),
                  concat([self.query_processor_pipeline,[self.search],self.reindexer_pipeline]),
                  query)

