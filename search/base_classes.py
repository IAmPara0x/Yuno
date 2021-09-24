from typing import Sequence, List, Callable, Any, Dict, Union, Tuple, NewType, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from returns.maybe import Maybe
from functools import wraps
from toolz.curried import reduce, map, compose, concat, pipe, nth  # type: ignore
import numpy as np
from abc import ABCMeta, abstractmethod

from .model import Model
from .config import Config
from . import utils


GenreId = NewType("GenreId", int)
TagId = NewType("TagId", int)
TagCatId = NewType("TagCatId", int)
AnimeId = NewType("AnimeId", int)
DataId = NewType("DataId", int)
Scores = NewType("Scores", np.ndarray)
Embedding = NewType("Embedding", np.ndarray)


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class Genre:
  uid: GenreId
  name: str = field(compare=False)


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class Tag:
  uid: TagId
  name: str = field(compare=False)
  description: str = field(compare=False)
  cat_uid: "TagCatId" = field(compare=False)
  embedding: np.ndarray = field(compare=False, repr=False)


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class TagCat:
  uid: TagCatId
  name: str = field(compare=False)
  tag_uids: List[TagId] = field(compare=False)


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class Anime:
  uid: AnimeId
  name: str = field(compare=False)
  genre_uids: List[GenreId] = field(compare=False)
  tag_uids: List[TagId] = field(compare=False)
  tag_scores: np.ndarray = field(compare=False)

  def tags(self, search_base: "SearchBase") -> List[Tag]:
    return [search_base.get_tag(uid) for uid in self.tag_uids]

  def genres(self, search_base: "SearchBase") -> List[Genre]:
    return [search_base.get_genre(uid) for uid in self.genre_uids]


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class Query:
  text: str
  embedding: Optional[np.ndarray] = field(compare=False)


class DataType(Enum):
  long = auto()
  short = auto()
  recs = auto()


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class Data:
  data_uid: DataId = field(repr=False)
  anime_uid: AnimeId
  embedding: Embedding = field(repr=False, compare=False)
  text: str = field(compare=False)
  rating: int = field(compare=False)
  type: DataType = field(compare=False, repr=False)


@dataclass(init=True, repr=False, eq=False, order=False, frozen=True)
class SearchResult:
  query: Query
  data: List[Data]
  scores: Scores

  @property
  def embeddings(self) -> np.ndarray:
    return compose(np.vstack, list, map)(lambda x: x.embedding, self.data)

  @property
  def texts(self) -> List[str]:
    return compose(list, map)(lambda x: x.text, self.data)

  @staticmethod
  def new(prev_result: "SearchResult", **kwargs) -> "SearchResult":
    remaining_fields = set(prev_result.__dict__.keys()) - set(kwargs.keys())
    kwargs.update({field_name: getattr(prev_result, field_name)
                  for field_name in remaining_fields})
    return SearchResult(**kwargs)

  def __getitem__(self, idx: int) -> Tuple[float, Data]:
    return self._result(idx)

  def _result(self, idx: int) -> Tuple[float, Data]:
    data = (self.scores, self.data)
    return compose(tuple, map)(nth(idx), data)

  def animes(self, search_base: "SearchBase") -> List[Anime]:
    return compose(list, map)(lambda x: search_base.get_anime(x.anime_uid), self.data)


@dataclass(init=True, repr=False, eq=False, order=False, frozen=True)
class SearchBase:
  model: Model
  index: Any
  _search_data: List[Data]
  _animes: Dict[AnimeId, Anime]
  _tags: Dict[TagId, Tag]
  _tag_cats: Dict[TagCatId, TagCat]
  _genres: Dict[GenreId, Genre]

  @property
  def tag_cats(self) -> List[TagCat]:
    return list(self._tag_cats.values())

  @property
  def tags(self) -> List[Tag]:
    return list(self._tags.values())

  @property
  def animes(self) -> List[Anime]:
    return list(self._animes.values())

  @property
  def datas(self) -> List[Data]:
    return self._search_data

  def get_tag(self, id: TagId) -> Tag:
    return self._tags[id]

  def get_tagcat(self, id: TagCatId) -> TagCat:
    return self._tag_cats[id]

  def get_genre(self, id: GenreId) -> Genre:
    return self._genres[id]

  def get_anime(self, id: AnimeId) -> Anime:
    return self._animes[id]

  def get_searchdata(self, idx: int) -> Data:
    return self._search_data[idx]


def sort_search(f):
  @wraps(f)
  def _impl(self, *args, **kwargs) -> SearchResult:

    def sort(values):
      return [value for _, value in sorted(zip(search_result.scores, values),
                                           key=lambda x: x[0], reverse=True)]

    search_result = f(self, *args)

    data = [data for _, data in sorted(zip(search_result.scores, search_result.data),
                                       key=lambda x: x[0], reverse=True)]
    scores = compose(np.array, sorted)(search_result.scores, reverse=True)
    return SearchResult.new(search_result, data=data, scores=scores)
  return _impl


def normalize(norm_f: Optional[Callable[[Scores], Scores]] = None):
  def wrapper(f):
    @wraps(f)
    def _impl(self, *args) -> SearchResult:

      search_result = f(self, *args)

      scores = Maybe.from_optional(norm_f
                                   ).bind_optional(
          lambda fn: fn(search_result.scores)
      ).or_else_call(
          lambda: search_result.scores
      )

      return SearchResult.new(search_result, scores=scores)
    return _impl
  return wrapper


@dataclass(frozen=True)
class IndexerBase:
  search_base: SearchBase

  @staticmethod
  def new(search_base: SearchBase, config):
    raise NotImplementedError

  def model(self, text: str) -> np.ndarray:
    return self.search_base.model(text)

  def knn_search(self, q_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    return self.search_base.index.search(q_embedding, top_k)

  def __call__(self, search_result: SearchResult) -> SearchResult:
    raise NotImplementedError


@dataclass
class QueryProcessorBase:
  def __call__(self, search_result: Query) -> Query:
    raise NotImplementedError


@dataclass(init=True, repr=False, eq=False, order=False, frozen=True)
class SearchPipelineBase(metaclass=ABCMeta):
  query_processor_pipeline: Sequence[Callable[[Query], Query]]
  search: Callable  # NOTE: Bug with mypy thinks self is also an arg
  # actual type:  search: Callable[[Query], SearchResult]
  indexer_pipeline: Sequence[Callable[[SearchResult],SearchResult]]

  @staticmethod
  def new(search_base: SearchBase, config:Config):
    raise NotImplementedError

  def __call__(self, query: Query) -> SearchResult:
    return pipe(query, *concat([self.query_processor_pipeline, [self.search], self.indexer_pipeline]))
