from typing import Sequence, List, Callable, Any, Dict, Union, Tuple, NewType, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from returns.maybe import Maybe
from functools import wraps, singledispatch, update_wrapper
from toolz.curried import reduce, map, compose, concat, pipe, nth  # type: ignore
import numpy as np


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
  embedding: np.ndarray = field(compare=False)


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
  datas: List[Data]
  scores: np.ndarray

  @staticmethod
  def new(prev_result: "SearchResult", **kwargs) -> "SearchResult":
    remaining_fields = set(prev_result.__dict__.keys()) - set(kwargs.keys())
    kwargs.update({field_name: getattr(prev_result, field_name)
                  for field_name in remaining_fields})
    return SearchResult(**kwargs)


@dataclass(init=True, repr=False, eq=False, order=False, frozen=True)
class SearchBase:
  model: Model
  index: Any
  _search_data: List[Data]
  _animes: Dict[AnimeUid, Anime]
  _tags: Dict[TagUid, Tag]
  _tag_cats: Dict[TagCatUid, TagCat]
  _genres: Dict[GenreUid, Genre]


def singledispatchmethod(func):
  dispatcher = singledispatch(func)

  def wrapper(*args, **kw):
    return dispatcher.dispatch(args[1].__class__)(*args, **kw)
  wrapper.register = dispatcher.register
  update_wrapper(wrapper, func)
  return wrapper


@dataclass(frozen=True)
class ImplUidData:
  search_base: SearchBase

  @singledispatchmethod
  def uid_data(self, uid):
    raise NotImplementedError

  @uid_data.register(GenreUid)
  def _get_genre(self, instance: GenreUid) -> Genre:
    return self.search_base._genres[instance]

  @uid_data.register(TagUid)
  def _get_tag(self, instance: TagUid) -> Tag:
    return self.search_base._tags[instance]

  @uid_data.register(TagCatUid)
  def _get_tagcat(self, instance: TagCatUid) -> TagCat:
    return self.search_base._tag_cats[instance]

  @uid_data.register(AnimeUid)
  def _get_anime(self, instance: AnimeUid) -> Anime:
    return self.search_base._animes[instance]

  @uid_data.register(int)
  def _get_searchdata(self, instance: int) -> Data:
    return self.search_base._search_data[instance]


@dataclass(frozen=True)
class ImplTags(ImplUidData):
  search_base: SearchBase

  @singledispatchmethod
  def get_tags(self, d_type) -> List[Tag]:
    raise NotImplementedError

  @get_tags.register(AllData)
  def _all_tags(self, _: AllData) -> List[Tag]:
    return list(self.search_base._tags.values())

  @get_tags.register(Anime)
  def _anime_tags(self, instance: Anime) -> List[Tag]:
    return [self.uid_data(tag_uid) for tag_uid in instance.tag_uids]

  @get_tags.register(TagCat)
  def _tagcat_tags(self, instance: TagCat) -> List[Tag]:
    return [self.uid_data(tag_uid) for tag_uid in instance.tag_uids]


@dataclass(frozen=True)
class ImplTagCats:
  search_base: SearchBase

  @singledispatchmethod
  def get_tagcats(self, d_type) -> List[TagCat]:
    raise NotImplementedError

  @get_tagcats.register(AllData)
  def _all_cats(self, _: AllData) -> List[TagCat]:
    return list(self.search_base._tag_cats.values())


@dataclass(frozen=True)
class ImplAnimes(ImplUidData):
  search_base: SearchBase

  @singledispatchmethod
  def get_animes(self, d_type) -> List[Anime]:
    raise NotImplementedError

  @get_animes.register(SearchResult)
  def _searchres_animes(self, instance: SearchResult) -> List[Anime]:
    return [self.uid_data(data.anime_uid) for data in instance.datas]

  @get_animes.register(AllData)
  def _all_animes(self, _: AllData) -> List[Anime]:
    return list(self.search_base._animes.values())


@dataclass(frozen=True)
class ImplDatas:
  search_base: SearchBase

  @singledispatchmethod
  def get_datas(self, d_type) -> List[Data]:
    raise NotImplementedError

  @get_datas.register(AllData)
  def _all_datas(self, _: AllData) -> List[Data]:
    return self.search_base._search_data

  @get_datas.register(SearchResult)
  def _searchres_datas(self, instance: SearchResult) -> List[Data]:
    return instance.datas


@dataclass(frozen=True)
class ImplTexts:
  search_base: SearchBase

  @singledispatchmethod
  def get_texts(self, d_type) -> List[str]:
    raise NotImplementedError

  @get_texts.register(SearchResult)
  def _searchres_texts(self, instance: SearchResult) -> List[str]:
    return [data.text for data in instance.datas]


@dataclass(frozen=True)
class ImplEmbeddings:
  search_base: SearchBase

  @singledispatchmethod
  def get_embeddings(self, d_type) -> np.ndarray:
    raise NotImplementedError

  @get_embeddings.register(SearchResult)
  def _searchres_embeddings(self, instance: SearchResult) -> np.ndarray:
    return np.vstack([data.embedding for data in instance.datas])

  @get_embeddings.register(Query)
  def _query_embedding(self, instance: Query) -> np.ndarray:
    return instance.embedding

  @get_embeddings.register(Data)
  def _data_embedding(self, instance: Data) -> np.ndarray:
    return instance.embedding


class Impl(ImplTags, ImplTagCats, ImplAnimes, ImplDatas, ImplTexts, ImplEmbeddings): pass


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
class SearchPipelineBase(Impl):
  query_processor_pipeline: Sequence[Callable[[Query], Query]]
  search: Callable  # NOTE: Bug with mypy thinks self is also an arg
  # actual type:  search: Callable[[Query], SearchResult]
  indexer_pipeline: Sequence[Callable[[SearchResult], SearchResult]]

  @staticmethod
  def new(search_base: SearchBase, config):
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


def process_result(norm_f: Optional[Callable[[np.ndarray], np.ndarray]] = None):
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
