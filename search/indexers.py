from typing import Union, Tuple, List, Callable, Dict
from toolz.curried import (compose,   # type: ignore
                            flip,
                            map,
                            unique,
                            curry,
                            remove,
                            reduce,
                            nth,
                            groupby)  # type: ignore
from dataclasses import dataclass
from returns.maybe import Maybe, Nothing
import numpy as np
import operator

from .base import (AnimeUid,
                   Anime,
                   Tag,
                   Query,
                   ProcessedQuery,
                   Data,
                   DataType,
                   Scores,
                   AllData,
                   IndexerBase,
                   SearchResult,
                   SearchBase,
                   process_result,
                   sort_search)

from .config import (SearchCfg,
                     AccIdxrCfg,
                     NodeIdxrCfg,
                     TagIdxrCfg,
                     TagSimIdxrCfg,
                     TagIdxingMethod,
                     TagIdxingMetric)

from .model import Model
from .utils import rescale_scores, cos_sim

getattrn = compose(curry,flip)(getattr)
fst = nth(0)
snd = nth(1)

@dataclass(init=True, frozen=True)
class Search(IndexerBase):
  cfg: SearchCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: SearchCfg) -> "Search":
    return Search(search_base, cfg)

  @process_result(rescale_scores(t_min=0.5, t_max=3, inverse=False))
  @sort_search
  def __call__(self, query: Query) -> SearchResult:

    cfg = self._get_config(query.config,
                           self.cfg,
                          "search_cfg")

    q_embd = compose(flip(np.expand_dims, 0),
                     self.model)(query.text)

    data_sim = compose(cos_sim(q_embd),
                       getattrn("embedding"))

    def acc_fn(datas,idx):
      data = compose(self.uid_data,int)(idx)
      if data.type == DataType.recs:
        datas[0].extend(map(lambda a_uid:
                              Data.new(data,
                                       anime_uid=a_uid,
                                       type=DataType.short),
                            data.anime_uid))
        datas[1].extend([data_sim(data)*cfg.weight]*2)
      else:
        datas[0].append(data)
        datas[1].append(data_sim(data))
      return datas


    result_data,scores = reduce(acc_fn,
                                compose(snd,
                                        map(np.squeeze),
                                        self.knn_search)(q_embd,cfg.top_k),
                                [[],[]])

    p_query = ProcessedQuery(query.text,
                             q_embd)

    return SearchResult(p_query,
                        result_data,
                        np.array(scores,dtype=np.float32).squeeze(),
                        query.config)


@dataclass(init=True, frozen=True)
class AccIdxr(IndexerBase):
  cfg: AccIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: AccIdxrCfg) -> "AccIdxr":
    return AccIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:

    cfg = self._get_config(search_result.config,
                           self.cfg,
                           "accindexer_cfg")

    datas = groupby(compose(getattrn("anime_uid"),
                            fst)
                    ,zip(search_result.datas,
                         search_result.scores)
                    ).values()

    def acc_fn(acc_data,a_datas):
      data = compose(fst,
                     fst)(a_datas)
      score = compose(cfg.score_fn,
                      list,
                      map(snd))(a_datas)

      acc_data[0].append(data)
      acc_data[1].append(score)
      return acc_data

    res_data, new_scores = reduce(acc_fn,
                                  datas,
                                  [[],[]])

    return SearchResult.new(search_result,
                            datas=res_data,
                            scores=np.array(new_scores,dtype=np.float32).squeeze())


@dataclass(init=True, frozen=True)
class TagSimIdxr(IndexerBase):
  cfg: TagSimIdxrCfg
  @staticmethod
  def new(search_base: SearchBase, cfg: TagSimIdxrCfg) -> "TagSimIdxr":
    return TagSimIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:

    cfg = self._get_config(search_result.config,
                            self.cfg,
                            "tagsimindexer_cfg")

    approx_f = compose(self.linear_approx,
                        np.squeeze,
                        getattrn("embedding"),
                        getattrn("query"))(search_result)

    def acc_fn(result_data, p_data):
      data,score = p_data
      anime = self.uid_data(data.anime_uid)
      result_data[0].append(data)

      if not anime.tag_uids:
        result_data[1].append(score * cfg.weight * 2)
      else:
        mat = np.vstack([tag.embedding
                         for tag in self.get_tags(anime)]
                        ).T
        result_data[1].append(score
                              + approx_f(mat,cfg.use_negatives) * cfg.weight)
      return result_data

    result_data,new_scores = reduce(acc_fn,
                                    zip(search_result.datas,
                                        search_result.scores),
                                    [[],[]])

    return SearchResult.new(search_result,
                          scores=np.array(new_scores,dtype=np.float32).squeeze())

  @curry
  def linear_approx(self, x: np.ndarray,
                          mat: np.ndarray,
                          use_negatives: bool) -> float:
    y = np.linalg.inv(mat.T @ mat) @ mat.T @ x
    if not use_negatives and len(np.where(y < 0)[0]) > 0:
      mat = mat.T[np.where(y > 0)].T
      return self.linear_approx(mat, x)
    else:
      return cos_sim(mat@y, x).item()


@dataclass(init=True, frozen=True)
class NodeIdxr(IndexerBase):
  cfg: NodeIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: NodeIdxrCfg) -> "NodeIdxr":
    return NodeIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    cfg = self._get_config(search_result.config,
                            self.cfg,
                            "nodeindexer_cfg")

    uids = [data.anime_uid for data in
              unique(search_result.datas,
                       key=lambda data: data.anime_uid)]

    result_datas: Dict[AnimeUid,List[Data]] = {uid: compose(
                                                      list,
                                                      remove(lambda data: not data.type == DataType.long),
                                                      self.get_datas,
                                                      self.uid_data)(uid)
                                                    for uid in uids}
    def helper(rank_scores,data):

      if not len(result_datas[data.anime_uid]):
        rank_scores.append(1)
      else:
        mat = np.vstack([x.embedding
                          for x in result_datas[data.anime_uid]])
        v = data.embedding

        if data.type == DataType.long:
          rank_scores.append(self.node_rank(v,mat))
        else:
          max_idx = np.argmax((mat @ v)/ #TODO: will speed this up.
                              (np.linalg.norm(mat,axis=1)*np.linalg.norm(v)))
          rank_scores.append(self.node_rank(mat[max_idx],mat))

      return rank_scores

    rank_scores = compose(np.array,
                          reduce)(helper,
                                  self.get_datas(search_result),
                                  [])
    new_scores = search_result.scores * rank_scores * cfg.weight

    return SearchResult.new(search_result,
                            scores=new_scores)

  @staticmethod
  def node_rank(v: np.ndarray, mat: np.ndarray) -> float:
    sims = (mat @ v)/ (np.linalg.norm(mat,axis=1)*np.linalg.norm(v))
    return np.average(sims).item()

