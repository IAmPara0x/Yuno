from typing import Sequence, List, Callable, Any, Dict, Union, Tuple, NewType, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from returns.maybe import Maybe
from functools import wraps
from toolz.curried import reduce, map, compose, concat, pipe, nth  # type: ignore
import numpy as np
from abc import ABCMeta, abstractmethod


class GenreUid(int): pass
class TagUid(int): pass
class TagCatUid(int): pass
class AnimeUid(int): pass
class DataUid(int): pass
class Scores(np.ndarray): pass
class Embedding(np.ndarray): pass
class AllData(object): pass

from . import utils
from .model import Model
from .config import Config

@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class Genre:
  uid: GenreUid
  name: str = field(compare=False)


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class Tag:
  uid: TagUid
  name: str = field(compare=False)
  description: str = field(compare=False)
  cat_uid: "TagCatUid" = field(compare=False)
  embedding: np.ndarray = field(compare=False, repr=False)


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class TagCat:
  uid: TagCatUid
  name: str = field(compare=False)
  tag_uids: List[TagUid] = field(compare=False)


@dataclass(init=True, repr=True, eq=True, order=False, frozen=True)
class Anime:
  uid: AnimeUid
  name: str = field(compare=False)
  genre_uids: List[GenreUid] = field(compare=False)
  tag_uids: List[TagUid] = field(compare=False)
  tag_scores: np.ndarray = field(compare=False)


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
  data_uid: DataUid = field(repr=False)
  anime_uid: AnimeUid
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


@dataclass(init=True, repr=False, eq=False, order=False, frozen=True)
class SearchBase:
  model: Model
  index: Any
  _search_data: List[Data]
  _animes: Dict[AnimeUid, Anime]
  _tags: Dict[TagUid, Tag]
  _tag_cats: Dict[TagCatUid, TagCat]
  _genres: Dict[GenreUid, Genre]


Uids = Union[GenreUid, TagUid, TagCatUid, AnimeUid, DataUid, int]
Datas = Union[Genre, Tag, TagCat, Anime, Data]


@dataclass(frozen=True)
class Impl:
  search_base: SearchBase

  def uid_data(self, uid: Uids):
    data_funcs = [self._get_genre,
                  self._get_tag,
                  self._get_tagcat,
                  self._get_anime,
                  self._get_searchdata,
                  self._get_searchdata]
    uids = [GenreUid, TagUid, TagCatUid, AnimeUid, DataUid, int]
    return self.dispatcher(uids, data_funcs)

  def tags(self, d_type: Union[AllData, AnimeUid, TagCatUid]) -> List[Tag]:
    data_funcs = [self._all_tags, self._anime_tags, self._tagcat_tags]
    data_types = [AllData, AnimeUid, TagCatUid]
    return self.dispatcher(data_types, data_funcs)

  def tag_cats(self, d_type: Union[AllData]) -> List[TagCat]:
    data_funcs = [self._all_cats]
    data_types = [AllData]
    return self.dispatcher(data_types, data_funcs)

  def animes(self, d_type: Union[AllData, SearchResult]) -> List[Anime]:
    data_funcs = [self._all_animes,self._searchres_animes]
    data_types = [AllData,SearchResult]
    return self.dispatcher(data_types, data_funcs)

  def datas(self, d_type: Union[AllData,SearchResult]) -> List[Data]:
    data_funcs = [self._all_datas,self._searchres_datas]
    data_types = [AllData,SearchResult]
    return self.dispatcher(data_types, data_funcs)

  def texts(self,d_type:Union[SearchResult]) -> List[str]:
    data_funcs = [self._searchres_texts]
    data_types = [SearchResult]
    return self.dispatcher(data_types, data_funcs)

  @staticmethod
  def dispatcher(types, funcs):
    for d_type, f in zip(types, funcs):
      if isinstance(d_type, uid_type):
        return f(d_type)
    raise NotImplementedError

  def _get_genre(self, id: GenreUid) -> Genre:
    return self.search_base._genres[id]

  def _get_tag(self, id: TagUid) -> Tag:
    return self.search_base._tags[id]

  def _get_tagcat(self, id: TagCatUid) -> TagCat:
    return self.search_base._tag_cats[id]

  def _get_anime(self, id: AnimeUid) -> Anime:
    return self.search_base._animes[id]

  def _get_searchdata(self, idx: int) -> Data:
    return self.search_base._search_data[idx]

  def _all_tags(self, _: AllData) -> List[Tag]:
    return list(self.search_base._tags.values())

  def _anime_tags(self, a_uid: AnimeUid) -> List[Tag]:
    anime = self.uid_data(a_uid)
    return compose(list, map)(self.uid_data, anime.tag_uids)

  def _tagcat_tags(self, c_uid: TagCatUid) -> List[Tag]:
    cat = self.uid_data(c_uid)
    return compose(list, map)(self.uid_data, cat.tag_uids)

  def _all_cats(self, _: AllData) -> List[TagCat]:
    return list(self.search_base._tag_cats.values())

  def _searchres_animes(self,search_result:SearchResult) -> List[Anime]:
    return compose(list,map)(lambda x: self.uid_data(x.anime_uid), search_result.data)

  def _all_animes(self,_:AllData) -> List[Anime]:
    return list(self.search_base._animes.values())

  def _all_datas(self,_:AllData) -> List[Data]:
    return self.search_base._search_data

  def _searchres_datas(self, search_result:SearchResult) -> List[Data]:
    return search_result.data

  def _searchres_texts(self,search_result:SearchResult) -> List[str]:
    return compose(list,map)(lambda x: x.text,search_result.data)

@dataclass(frozen=True)
class IndexerBase(Impl):
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
  indexer_pipeline: Sequence[Callable[[SearchResult], SearchResult]]

  @staticmethod
  def new(search_base: SearchBase, config: Config):
    raise NotImplementedError

  def __call__(self, query: Query) -> SearchResult:
    return pipe(query, *concat([self.query_processor_pipeline, [self.search], self.indexer_pipeline]))


def sort_search(f):
  @wraps(f)
  def _impl(self, *args, **kwargs) -> SearchResult:

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
