from typing import Union, Tuple, List, Callable, Optional
from toolz.curried import compose, flip, map, unique, partial, first, curry  # type: ignore
from dataclasses import dataclass
import numpy as np
import operator

from .base import (Anime,
                   Tag,
                   Query,
                   Data,
                   Scores,
                   AllData,
                   IndexerBase,
                   SearchResult,
                   SearchBase,
                   process_result,
                   sort_search)

from .config import (SearchConfig,
                     AccIndexerConfig,
                     TagIndexerConfig,
                     TagSimIndexerConfig,
                     TagIndexingMethod,
                     TagIndexingMetric)

from .model import Model
from .utils import rescale_scores, cos_sim


@dataclass(init=True, frozen=True)
class Search(IndexerBase):
  embedding_dim: int
  top_k: int

  @staticmethod
  def new(search_base: SearchBase, config: SearchConfig) -> "Search":
    return Search(search_base, config.embedding_dim, config.top_k)

  @sort_search
  def __call__(self, query: Query) -> SearchResult:
    q_embedding = compose(
        flip(np.expand_dims, 0),
        self.model
    )(query.text)

    dist, n_idx = compose(
        map(np.squeeze),
        self.knn_search
    )(q_embedding, self.top_k)

    result_data = [self.uid_to_data(int(idx)) for idx in n_idx]
    query = Query(query.text, q_embedding)
    scores = np.array([cos_sim(q_embedding, data.embedding)
                      for data in result_data])
    return SearchResult(query, result_data, scores)


@dataclass(init=True, frozen=True)
class AccIndexer(IndexerBase):
  acc_fn: Callable  # NOTE: Bug with mypy thinks self is also an arg
  # actual type:  acc_fn: Callable[[Scores], float]

  @staticmethod
  def new(search_base: SearchBase, config: AccIndexerConfig) -> "AccIndexer":
    return AccIndexer(search_base, config.acc_fn)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    anime_uids = [
        anime.uid for anime in self.get_animes(search_result)]
    unique_uids = unique(anime_uids)
    uids_idxs = compose(list, map)(lambda eq:
                                   [idx for idx, uid in enumerate(
                                       anime_uids) if eq(uid)],
                                   map(curry(operator.eq), unique_uids))

    scores = compose(list, map
                     )(lambda uid_idxs: np.sum(search_result.scores[uid_idxs]).item(), uids_idxs)
    scores = np.array(scores, dtype=np.float32)
    result_data = [search_result.data[idx] for idx in map(first, uids_idxs)]
    return SearchResult.new(search_result, data=result_data, scores=scores)


@dataclass(init=True, frozen=True)
class TagSimIndexer(IndexerBase):
  use_negatives: bool
  use_sim: bool

  @staticmethod
  def new(search_base: SearchBase, config: TagSimIndexerConfig) -> "TagSimIndexer":
    return TagSimIndexer(search_base, config.use_negatives, config.use_sim)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    q_embd = search_result.query.embedding

    tag_scores = []
    for anime in self.get_animes(search_result):
      mat = np.vstack([tag.embedding for tag in self.get_tags(anime.uid)]).T
      tag_scores.append(self.linear_approx(mat, q_embd.squeeze()))
    scores = np.array(tag_scores) * search_result.scores
    return SearchResult.new(search_result, scores=scores)

  def linear_approx(self, mat: np.ndarray, x: np.ndarray) -> float:
    y = (mat.T @ mat) @ mat.T @ x
    if not self.use_negatives and len(np.where(y < 0)[0]) > 0:
      mat = mat.T[np.where(y > 0)].T
      return self.linear_approx(mat, x)
    else:
      return cos_sim(mat@y, x).item()


@dataclass(init=True, frozen=True)
class TagIndexer(IndexerBase):
  indexing_method: TagIndexingMethod
  indexing_metric: TagIndexingMetric

  @staticmethod
  def new(search_base: SearchBase, config: TagIndexerConfig) -> "TagIndexer":
    return TagIndexer(search_base, config.indexing_method, config.indexing_metric)

  @process_result(rescale_scores(t_min=1, t_max=8, inverse=False))
  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    query_mat = self.tags_mat(search_result.query)

    if self.indexing_method == TagIndexingMethod.per_category:
      similarity_scores = compose(list, map)(self.per_category_indexing(
          query_mat), self.get_animes(search_result))
    elif self.indexing_method == TagIndexingMethod.all:
      query_mat = query_mat.reshape(-1)
      similarity_scores = compose(list, map)(self.all_category_indexing(
          query_mat), self.get_animes(search_result))
    else:
      raise Exception(f"{self.indexing_method} is not a corret type.")

    similarity_scores = rescale_scores(
        t_min=1, t_max=3, inverse=False)(np.array(similarity_scores, dtype=np.float32))
    similarity_scores *= search_result.scores
    return SearchResult.new(search_result, scores=similarity_scores)

  def tags_mat(self, x: Union[Anime, Query]) -> np.ndarray:
    tag_cats = self.get_tagcats(AllData())
    rows, cols = len(tag_cats), compose(max, map)(
        lambda cat: len(cat.tag_uids), tag_cats)
    tags_mat = np.zeros((rows, cols))

    def tag_pos(tag: Tag) -> Tuple[int, int]:
      i = [idx for idx, cat in enumerate(
          tag_cats) if cat.uid == tag.cat_uid][0]
      j = [idx for idx, tag_uid in enumerate(
          tag_cats[i].tag_uids) if tag_uid == tag.uid][0]
      return (i, j)

    if isinstance(x, Anime):
      anime_tags = self.get_tags(x.uid)
      i_s, j_s = zip(*map(tag_pos, anime_tags))
      tags_mat[(i_s, j_s)] = x.tag_scores
    elif isinstance(x, Query):
      all_tags = self.get_tags(AllData())
      i_s, j_s = zip(*map(tag_pos, all_tags))
      scores = [cos_sim(x.embedding, tag.embedding).item()
                for tag in all_tags]
      tags_mat[(i_s, j_s)] = scores
    else:
      raise Exception(
          f"Only supported types are Anime and Query but {type(x)} is None of them")
    return tags_mat

  @curry
  def per_category_indexing(self, query_mat: np.ndarray, anime_info: Anime) -> int:
    anime_mat = self.tags_mat(anime_info)
    x = compose(np.diag, np.dot)(anime_mat, query_mat.T)
    y = compose(np.diag, np.dot)(anime_mat, anime_mat.T)
    return np.dot(x, y).item()

  @curry
  def all_category_indexing(self, query_mat: np.ndarray, anime_info: Anime) -> int:
    anime_mat = self.tags_mat(anime_info)
    anime_mat = anime_mat.reshape(-1)
    return cos_sim(anime_mat, query_mat).item()
