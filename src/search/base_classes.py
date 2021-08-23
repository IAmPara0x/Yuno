from typing import NamedTuple, List, Callable
from enum import Enum
from search_config import SearchConfig


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
  distances: np.ndarray
  anime_infos: List[Anime]

class SearchBase(NamedTuple):
  MODEL: str
  INDEX : str
  ALL_ANIME: dict
  ALL_TAGS: dict
  EMBEDDINGS: np.ndarray
  LABELS: np.ndarray
  TEXTS: List[str]

class Search:
  def __init__(self,search_base:SearchBase,search_config:SearchConfig):
    for name,val in zip(search_base._fields,search_base.__iter__()):
      setattr(self,name,val)

    for name,val in zip(search_config._fields,search_config.__iter__()):
      setattr(self,name,val)

  def knn_search(self,text:str) -> SearchResult:
    q_embedding = self.model(text)
    distances,n_id = self.INDEX.search(q_embedding,self.TOP_K)
    distances = distances.squeeze()
    n_id = n_id.squeeze()

    n_embeddings = self.EMBEDDINGS[n_id]

    n_anime_uids = self.LABELS[n_id]
    n_anime = map(lambda uid: self.ALL_ANIME[uid], n_anime_uids)

    return SearchResult(q_embedding,n_embeddings,n_id,distances,list(n_anime))


class ReIndexer:
  _reindexer_names = []
  def __init__(self, search_config:SearchConfig):
    for name,val in zip(search_config._fields,search_config.__iter__()):
      setattr(self,name,val)

  @classmethod
  def add_reindexer(cls, name: str, func:Callable[[SearchResult],SearchResult]):
    cls._reindexer_names.append(name)
    setattr(cls,name,staticmethod(func))
