from typing import NamedTuple, List
from enum import Enum


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


class Search(SearchBase):
  def __init__(self):
    pass
