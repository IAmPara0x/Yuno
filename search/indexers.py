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

from .config import (SearchCfg,
                     AccIdxrCfg,
                     TagIdxrCfg,
                     TagSimIdxrCfg,
                     TagIdxingMethod,
                     TagIdxingMetric)

from .model import Model
from .utils import rescale_scores, cos_sim


@dataclass(init=True, frozen=True)
class Search(IndexerBase):
  embedding_dim: int
  top_k: int
  weight: float

  @staticmethod
  def new(search_base: SearchBase, config: SearchCfg) -> "Search":
    return Search(search_base, config.embedding_dim, config.top_k, config.weight)

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

    result_data = map(compose(self.uid_data, int), n_idx)
    query = Query(query.text, q_embedding)
    scores = np.fromiter(map(cos_sim(q_embedding), result_data),
                         dtype=np.float32)
    return SearchResult(query, result_data, self.weight*scores)


@dataclass(init=True, frozen=True)
class AccIdxr(IndexerBase):
  acc_fn: Callable

  @staticmethod
  def new(search_base: SearchBase, config: AccIdxrCfg) -> "AccIdxr":
    return AccIdxr(search_base, config.acc_fn)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    anime_uids = [
        anime.uid for anime in self.get_animes(search_result)]
    unique_uids = unique(anime_uids)

    uids_idxs = map(lambda eq:
                    [idx for idx, uid in enumerate(anime_uids) if eq(uid)],
                    map(curry(operator.eq), unique_uids))

    scores = np.fromiter(
        map(lambda uid_idxs: self.acc_fn(search_result.scores[uid_idxs]),
            uids_idxs),
        dtype=np.float32)
    result_data = [search_result.datas[idx] for idx in map(first, uids_idxs)]
    return SearchResult.new(search_result, data=result_data, scores=scores)


@dataclass(init=True, frozen=True)
class TagSimIdxr(IndexerBase):
  use_negatives: bool
  use_sim: bool
  weight: float

  @staticmethod
  def new(search_base: SearchBase, config: TagSimIdxrCfg) -> "TagSimIdxr":
    return TagSimIdxr(search_base, config.use_negatives, config.use_sim, config.weight)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    q_embd = search_result.query.embedding
    approx_f = self.linear_approx(q_embd.squeeze())

    tag_scores = []
    for anime in self.get_animes(search_result):
      mat = np.vstack([tag.embedding for tag in self.get_tags(anime)]).T
      tag_scores.append(approx_f(mat))
    scores =  self.weight * np.array(tag_scores) + search_result.scores
    return SearchResult.new(search_result, scores=scores)

  @curry
  def linear_approx(self, x: np.ndarray, mat: np.ndarray) -> float:
    y = (mat.T @ mat) @ mat.T @ x
    if not self.use_negatives and len(np.where(y < 0)[0]) > 0:
      mat = mat.T[np.where(y > 0)].T
      return self.linear_approx(mat, x)
    else:
      return cos_sim(mat@y, x).item()


#NOTE: This indexer doesn't score very well
@dataclass(init=True, frozen=True)
class TagIdxr(IndexerBase):
  indexing_method: TagIdxingMethod
  indexing_metric: TagIdxingMetric

  @staticmethod
  def new(search_base: SearchBase, config: TagIdxrCfg) -> "TagIdxr":
    return TagIdxr(search_base, config.indexing_method, config.indexing_metric)

  @process_result(rescale_scores(t_min=1, t_max=8, inverse=False))
  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    query_mat = self.tags_mat(search_result.query)

    if self.indexing_method == TagIdxingMethod.per_category:
      similarity_scores = compose(list, map)(self.per_category_indexing(
          query_mat), self.get_animes(search_result))
    elif self.indexing_method == TagIdxingMethod.all:
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
      anime_tags = self.get_tags(x)
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
