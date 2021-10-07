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
import numpy as np
import operator

from .base import (AnimeUid,
                   Anime,
                   Tag,
                   Query,
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

getattr = compose(curry,flip)(getattr)
snd = nth(1)
fst = nth(0)

@dataclass(init=True, frozen=True)
class Search(IndexerBase):
  embedding_dim: int
  top_k: int
  weight: float

  @staticmethod
  def new(search_base: SearchBase, config: SearchCfg) -> "Search":
    return Search(search_base, config.embedding_dim, config.top_k, config.weight)

  @process_result(rescale_scores(t_min=0.5, t_max=3, inverse=False))
  @sort_search
  def __call__(self, query: Query) -> SearchResult:

    q_embedding = compose(flip(np.expand_dims, 0),
                          self.model)(query.text)

    data_sim = compose(cos_sim(q_embedding),getattr("embedding"))

    def acc_fn(datas,idx):
      data = compose(self.uid_data,int)(idx)
      if data.type == DataType.recs:
        datas[0].extend(map(lambda a_uid: Data.new(data,anime_uid=a_uid,type=DataType.short),
                          data.anime_uid))
        datas[1].extend([data_sim(data)*self.weight]*2)
      else:
        datas[0].append(data)
        datas[1].append(data_sim(data))
      return datas

    result_data,scores = reduce(acc_fn,
                                compose(snd,map(np.squeeze),
                                  self.knn_search)(q_embedding, self.top_k),
                                [[],[]])
    query = Query(query.text, q_embedding)
    return SearchResult(query, result_data, np.array(scores,dtype=np.float32).squeeze())


@dataclass(init=True, frozen=True)
class AccIdxr(IndexerBase):
  score_fn: Callable

  @staticmethod
  def new(search_base: SearchBase, config: AccIdxrCfg) -> "AccIdxr":
    return AccIdxr(search_base, config.acc_fn)

  @sort_search
  def __call__(self, srch_res: SearchResult) -> SearchResult:

    datas = groupby(compose(getattr("anime_uid"),fst)
                    ,zip(srch_res.datas,srch_res.scores)).values()

    def acc_fn(acc_data,a_datas):
      data = compose(fst,fst)(a_datas)
      score = compose(self.score_fn,list,map(snd))(a_datas)
      acc_data[0].append(data)
      acc_data[1].append(score)
      return acc_data

    res_data,new_scores = reduce(acc_fn,datas,[[],[]])
    return SearchResult.new(srch_res, datas=res_data,
                        scores=np.array(new_scores,dtype=np.float32).squeeze())


@dataclass(init=True, frozen=True)
class TagSimIdxr(IndexerBase):
  use_negatives: bool
  use_sim: bool
  weight: float

  @staticmethod
  def new(search_base: SearchBase, config: TagSimIdxrCfg) -> "TagSimIdxr":
    return TagSimIdxr(search_base, config.use_negatives, config.use_sim, config.weight)

  @sort_search
  def __call__(self, srch_res: SearchResult) -> SearchResult:
    approx_f = self.linear_approx(compose(np.squeeze,
                                          getattr("embedding"),
                                          getattr("query"))(srch_res))
    def acc_fn(result_data,p_data):
      data,score = p_data
      anime = self.uid_data(data.anime_uid)
      result_data[0].append(data)
      if not anime.tag_uids:
        result_data[1].append(score*self.weight*2)
      else:
        mat = np.vstack([tag.embedding for tag in self.get_tags(anime)]).T
        result_data[1].append(score+approx_f(mat)*self.weight)
      return result_data

    result_data,new_scores = reduce(acc_fn,
                                    zip(srch_res.datas,srch_res.scores),
                                    [[],[]])
    return SearchResult.new(srch_res, scores=np.array(new_scores,dtype=np.float32).squeeze())

  @curry
  def linear_approx(self, x: np.ndarray, mat: np.ndarray) -> float:
    y = (mat.T @ mat) @ mat.T @ x
    if not self.use_negatives and len(np.where(y < 0)[0]) > 0:
      mat = mat.T[np.where(y > 0)].T
      return self.linear_approx(mat, x)
    else:
      return cos_sim(mat@y, x).item()


@dataclass(init=True, frozen=True)
class NodeIdxr(IndexerBase):
  weight: float

  @staticmethod
  def new(search_base: SearchBase, config: NodeIdxrCfg) -> "NodeIdxr":
    return NodeIdxr(search_base, config.weight)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    uids = [data.anime_uid for data in
            unique(search_result.datas,key=lambda data: data.anime_uid)]

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
        mat = np.vstack([x.embedding for x in result_datas[data.anime_uid]])
        v = data.embedding
        if data.type == DataType.long:
          rank_scores.append(self.node_rank(v,mat))
        else:
          max_idx = np.argmax((mat @ v)/
                              (np.linalg.norm(mat,axis=1)*np.linalg.norm(v)))
          rank_scores.append(self.node_rank(mat[max_idx],mat))
      return rank_scores

    rank_scores = compose(np.array,reduce)(helper,self.get_datas(search_result),[])
    new_scores = search_result.scores * rank_scores * self.weight
    return SearchResult.new(search_result, scores=new_scores)

  def node_rank(self,v: np.ndarray, mat: np.ndarray) -> float:
    sims = (mat @ v)/ (np.linalg.norm(mat,axis=1)*np.linalg.norm(v))
    return np.average(sims).item()
