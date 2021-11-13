from typing import (
  List,
  Dict,
  TypeVar,
  Optional,
  Tuple,
)
from cytoolz.curried import (  # type: ignore
  compose,
  flip,
  map,
  curry,
  filter,
  reduce,
  groupby,
  concat,
  valmap,
)
from dataclasses import dataclass
import numpy as np
import torch

from .base import (
  AnimeUid,
  Query,
  ProcessedQuery,
  Data,
  DataType,
  IndexerBase,
  SearchResult,
  SearchBase,
  process_result,
  sort_search,
)
from .config import (
  SearchCfg,
  AccIdxrCfg,
  NodeIdxrCfg,
  TagSimIdxrCfg,
  ContextIdxrCfg,
)
from .utils import (
  Result,
  getattr,
  fst,
  snd,
  rescale_scores,
  cos_sim,
  get_config,
  group_data,
  ungroup_data,
  pair_sim,
  from_vstack,
  l2_approx,
  datas_filter,
)

Tensor = torch.Tensor


@dataclass(init=True, frozen=True)
class Search(IndexerBase):
  """
    Search class inherits from IndexerBase uses knn search to find top_k data and scores
    them using cosine similarity.

    ...

    Attributes:
    ----------
      cfg: SearchCfg
        contains all the default parameters that is need for __call__

    Methods:
    ----------
      new(search_base: SearchBase, cfg: SearchCfg) -> Search
        returns new instance of Search class.

      __call__(query:Query) -> SearchResult
        calls base_idxr with given query and default cfg or given cfg.

      base_idxr(query: Query, cfg: SearchCfg) -> SearchResult
        uses knn search to get top_k data and scores them according to cosine similarity
        with query.

  """
  cfg: SearchCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: SearchCfg) -> "Search":
    """
      creates new instance of Search class.

      Parameters
      ----------
        search_base: SearchBase
        cfg: SearchCfg

      Returns
      ----------
        Search
    """

    return Search(search_base, cfg)

  @process_result(rescale_scores(t_min=0.5, t_max=3, inverse=False))
  @sort_search
  def __call__(self, query: Query) -> SearchResult:
    """
      calls method base_idxr with cfg argument if provided in query or else
      uses default cfg attribute. The datas returned by base_idxr is
      then sorted according to their scores and then the score is
      linearly rescaled between min=0.5, max=3.

      Parameters
      ----------
        query: Query

      Returns
      ----------
        SearchResult
    """

    cfg = get_config(query.config, self.cfg, "search_cfg")
    return self.base_idxr(query, cfg)

  def base_idxr(self, query: Query, cfg: SearchCfg) -> SearchResult:
    """
      Uses knn search to get top_k data similar to query and uses cosine similarity
      where top_k is the parameter present in cfg. It also multiply the score by some weight
      to increase/decrease the score some data by that factor.

      Parameters
      ----------
        query: Query
        cfg: SearchCfg

      Returns
      ----------
        SearchResult

    """

    def acc_fn(datas, idx):
      data = compose(self.uid_data, int)(idx)

      if data.type == DataType.recs:
        datas[0].extend(
            map(
                lambda a_uid: Data.new(
                    data, anime_uid=a_uid, type=DataType.short),
                data.anime_uid), )
        datas[1].extend([data_sim(data) * cfg.weight] * 2)
      else:
        datas[0].append(data)
        datas[1].append(data_sim(data))

      return datas

    q_embd = compose(flip(np.expand_dims, 0), self.model)(query.text)
    data_sim = compose(cos_sim(q_embd), getattr("embedding"))
    result_data, scores = reduce(
        acc_fn,
        compose(snd, map(np.squeeze), self.knn_search)(q_embd, cfg.top_k),
        [[], []],
    )
    p_query = ProcessedQuery(query.text, q_embd.squeeze())

    return SearchResult(p_query, result_data,
                        np.array(scores, dtype=np.float32).squeeze(),
                        query.config)


@dataclass(init=True, frozen=True)
class AccIdxr(IndexerBase):
  """
    AccIdxr class inherits from IndexerBase.
    Accumulates all the data of same anime_uid scores
    them according to the function in AccIdxrCfg.

    ...

    Attributes:
    ----------
      cfg: AccIdxrCfg
        contains all the default parameters that is need for __call__

    Methods:
    ----------
      new(search_base: SearchBase, cfg: AccIdxrCfg) -> AccIdxr
        returns new instance of AccIdxr class

      __call__(search_result: SearchResult) -> SearchResult
        calls acc_idxr with given search_result and default cfg or given cfg

      acc_idxr(search_result: SearchResult, cfg: AccIdxrCfg) -> SearchResult
        Accumulates all the data of same anime_uid and scores them according to the function in AccIdxrCfg.
  """

  cfg: AccIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: AccIdxrCfg) -> "AccIdxr":
    """
      creates new instance of AccIdxr class.

      Parameters
      ----------
        search_base: SearchBase
        cfg: AccIdxrCfg

      Returns
      ----------
        AccIdxr
    """

    return AccIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    """
      calls method acc_idxr with cfg argument if provided in
      search_result or else uses default cfg attribute.
      The datas returned by acc_idxr is then
      sorted according to their scores.

      Parameters
      ----------
        search_result: SearchResult

      Returns
      ----------
        SearchResult
    """

    cfg = get_config(search_result.config, self.cfg, "accindexer_cfg")
    return self.acc_idxr(search_result, cfg)

  def acc_idxr(self, search_result: SearchResult,
               cfg: AccIdxrCfg) -> SearchResult:
    """
      Accumulates all the data of same anime_uid and scores
      them according to the function provided in cfg. Due to this
      all the data in the returned search_result has unique anime_uid.

      Parameters
      ----------
        search_result: SearchResult
        cfg: AccIdxrCfg

      Returns
      ----------
        SearchResult
    """

    def acc_fn(acc_data, a_datas):
      data = a_datas[0][0]
      score = compose(cfg.score_fn, list, map(snd))(a_datas)
      acc_data[0].append(data)
      acc_data[1].append(score)

      return acc_data

    grp_datas = group_data("anime_uid", search_result.datas,
                           search_result.scores).values()
    res_data, new_scores = reduce(acc_fn, grp_datas, [[], []])

    return SearchResult.new(search_result,
                            datas=res_data,
                            scores=np.array(new_scores,
                                            dtype=np.float32).squeeze())


@dataclass(init=True, frozen=True)
class TagSimIdxr(IndexerBase):
  """
    TagSimIdxr class inherits from IndexerBase.
    Uses linear L2 approximation between the query and the tags of animes and then scores them.

    ...

    Attributes:
    ----------
      cfg: TagSimIdxrCfg
        contains all the default parameters that is need for __call__.


    Methods:
    ----------
      new(search_base: SearchBase, cfg: TagSimIdxrCfg) -> TagSimIdxr
        returns new instance of TagSimIdxr class.

      __call__(search_result: SearchResult) -> SearchResult
        calls tagsim_idxr with given search_result and default cfg or given cfg.

      tagsim_idxr(search_result: SearchResult, cfg: TagSimIdxrCfg) -> SearchResult
        scores the data according to score returned by method linear_approx.

      linear_approx(x: Tensor, mat: Tensor, use_negatives: bool, use_sim: bool) -> float
        returns the score according to L2 linear approximation.
  """

  cfg: TagSimIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: TagSimIdxrCfg) -> "TagSimIdxr":
    """
      creates new instance of TagSimIdxr class.

      Parameters
      ----------
        search_base: SearchBase
        cfg: TagSimIdxr

      Returns
      ----------
        TagSimIdxr
    """

    return TagSimIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    """
      calls method tagsim_idxr with cfg argument if provided in search_result or else
      uses default cfg attribute. The datas returned by tagsim_idxr is
      then sorted according to according to their scores.

      Parameters
      ----------
        search_result: SearchResult

      Returns
      ----------
        SearchResult
    """

    cfg = get_config(search_result.config, self.cfg, "tagsimindexer_cfg")
    return self.tagsim_idxr(search_result, cfg)

  def tagsim_idxr(self, search_result: SearchResult, cfg: TagSimIdxrCfg) -> SearchResult:
    """
      scores the data in search_result according to the score returned
      by method linear_approx times weight ie. provided in TagSimIdxrCfg.

      Parameters
      ----------
        search_result: SearchResult
        cfg: TagSimIdxrCfg

      Returns
      ----------
        SearchResult
    """

    def approx_map(item):
      a_uid, datas, scores = item[0], *map(list, zip(*item[1]))  # type: ignore
      scores, a_tags = np.array(scores), self.get_tags(a_uid)

      if not len(a_tags):
        scores *= 2 * cfg.weight
      else:
        mat = from_vstack([tag.embedding for tag in a_tags]).T
        scores += approx_f(mat, cfg.use_negatives, cfg.use_sim) * cfg.weight

      pairs = compose(list, zip)(datas, scores)
      return (a_uid, pairs)

    approx_f = self.linear_approx(
        torch.from_numpy(search_result.query.embedding))
    grp_datas: Dict[AnimeUid, Result] = group_data("anime_uid", search_result.datas,
                                         search_result.scores)
    result_data, new_scores = ungroup_data(approx_map, grp_datas)

    return SearchResult.new(search_result,
                            datas=result_data,
                            scores=np.array(new_scores,
                                            dtype=np.float32).squeeze())

  @curry
  def linear_approx(self, x: Tensor, mat: Tensor, use_negatives: bool,
                    use_sim: bool) -> float:
    """
      Uses L2 linear approximation to find the best approximation between
      given a set of vectors and the target vector and then returns the score
      according to parameter use_sim.


      Parameters
      ----------
        x: Tensor
          target vector that is being approximated.

        mat: Tensor
          vectors that is used to approximate the target vector x.

        use_negatives: bool
           If False solution in found under positive contraint
           else no contraint is used.

        use_sim: bool
          if True then the cosine similarity between the approximated vector and the
          target vector is returned as the score, else the mean of coefficients of the
          approximation of the vectors is returned as score.

      Returns
      ----------
        float
          the score between target vector x and mat.
    """

    y = l2_approx(x, mat, mat.T)
    if not use_negatives and len(np.where(y < 0)[0]) > 0:

      mat = mat.T[torch.where(y > 0)].T
      return self.linear_approx(x, mat, use_negatives, use_sim)
    else:

      if use_sim:
        return torch.cosine_similarity(mat @ y, x).item()
      else:
        return torch.mean(y).item()


@dataclass(init=True, frozen=True)
class NodeIdxr(IndexerBase):
  """
    NodeIdxr class inherits from IndexerBase.
    Uses the authority score of the data in the search_result to score to datas.
    ...

    Attributes:
    ----------
      cfg: NodeIdxrCfg
        contains all the default parameters that is need for __call__.

    Methods:
    ----------
      new(search_base: SearchBase, cfg: NodeIdxrCfg) -> NodeIdxr
        returns new instance of NodeIdxr class.

      __call__(search_result: SearchResult) -> SearchResult
        calls node_idxr with given search_result and default cfg or given cfg.

      node_idxr(search_result: SearchResult, cfg: NodeIdxrCfg) -> SearchResult
        scores the data according to the score returned by method node_rank.

      node_rank(v: Tensor, mat: Tensor) -> float
        returns the authority score of the data ie the mean of the cosine similarity
        between v and mat.
  """

  cfg: NodeIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: NodeIdxrCfg) -> "NodeIdxr":
    """
      creates new instance of NodeIdxr class.

      Parameters
      ----------
        search_base: SearchBase
        cfg: NodeIdxrCfg

      Returns
      ----------
        NodeIdxr
    """

    return NodeIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    """
      calls method node_idxr with cfg argument if provided in search_result or else
      uses default cfg attribute. The datas returned by node_idxr is
      then sorted according to according to their scores.

      Parameters
      ----------
        search_result: SearchResult

      Returns
      ----------
        SearchResult
    """

    cfg = get_config(search_result.config, self.cfg, "nodeindexer_cfg")
    return self.node_idxr(search_result, cfg)

  def node_idxr(self, search_result: SearchResult, cfg: NodeIdxrCfg) -> SearchResult:
    """
      scores the data in search_result according to the score returned
      by method node_rank times weight ie. provided in NodeIdxrCfg.

      Parameters
      ----------
        search_result: SearchResult
        cfg: NodeIdxrCfg

      Returns
      ----------
        SearchResult
    """

    def noderank_map(item):
      a_uid, grp_types = item
      a_ds = datas_filter(lambda data: data.type == DataType.long,
                          self.get_datas(a_uid))

      if not len(a_ds):
        pairs = compose(list, concat)(grp_types.values())
        rank_scores = np.ones(len(pairs)) * cfg.weight
      else:
        mat = from_vstack([a_d.embedding for a_d in a_ds]).to(cfg.device)
        embds = []

        if DataType.long in grp_types:
          _embds = from_vstack([d.embedding for d, _ in grp_types[DataType.long]
                               ]).to(cfg.device)
          embds.extend(_embds)

        if DataType.short in grp_types:
          _embds = from_vstack([d.embedding for d, _ in grp_types[DataType.short]
                               ]).to(cfg.device)
          max_idxs = torch.argmax(pair_sim(_embds, mat), dim=-1)
          embds.extend(mat[max_idxs])

        rank_scores = self.node_rank(torch.vstack(embds),
                                     mat).cpu().numpy() * cfg.weight

        pairs = concat([
            v for k, v in sorted(grp_types.items(), key=lambda t: t[0].value)
        ])

      return (a_uid,
              [(pair[0], pair[1] * rank_score)
              for pair, rank_score in zip(pairs, rank_scores)],)

    grp_datas = valmap(
        groupby(compose(getattr("type"), fst)),
        group_data("anime_uid", search_result.datas, search_result.scores),)
    datas, new_scores = ungroup_data(noderank_map, grp_datas)

    return SearchResult.new(search_result,
                            datas=datas,
                            scores=np.array(new_scores).squeeze())

  @staticmethod
  def node_rank(v: Tensor, mat: Tensor) -> Tensor:
    """
      calculates the authority of the node in the graph ie
      mean of cosine similarity between all the vectors.

      Parameters
      ----------
        v: Tensor
          vectors whose authority score is need to be calculated.
        mat: Tensor
          vectors that are related to target vector v.

      Returns
      ----------
        Tensor
          authority scores of the target vectors v.
    """

    assert len(v.shape) == 2, "The dim of v must be 2"
    sims = pair_sim(v, mat)
    return torch.mean(sims, dim=1)


@dataclass(init=True, frozen=True)
class ContextIdxr(IndexerBase):
  """
    ContextIdxr class inherits from IndexerBase.
    uses brute force search with some pruning to search
    combinations of different sentences that best
    linearly approximate query.

    ...

    Attributes:
    ----------
      cfg: ContextIdxrCfg
        contains all the default parameters that is need for __call__.

    Methods:
    ----------
      new(search_base: SearchBase, cfg: ContextIdxrCfg) -> ContextIdxr
        returns new instance of ContextIdxr class.

      __call__(search_result: SearchResult) -> SearchResult
        calls context_idxr with given search_result and default cfg or given cfg.

      context_idxr(search_result: SearchResult, cfg: ContextIdxrCfg) -> SearchResult
        scores the data according to the score returned by method context_score.

      context_score(q: Tensor, mat: Tensor, combinations: int,
          cutoff_sim: float, device: str) -> Tuple[float, List[int]]

        uses brute force search with pruning that searches for best combination of
        vectors that best linearly approximate the q vector.
  """

  cfg: ContextIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: ContextIdxrCfg) -> "ContextIdxr":
    """
      creates new instance of ContextIdxr class.

      Parameters
      ----------
        search_base: SearchBase
        cfg: ContextIdxrCfg

      Returns
      ----------
        ContextIdxr
    """

    return ContextIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    """
      calls method context_idxr with cfg argument if provided in search_result or else
      uses default cfg attribute. The datas returned by context_idxr is
      then sorted according to according to their scores.

      Parameters
      ----------
        search_result: SearchResult

      Returns
      ----------
        SearchResult
    """

    cfg = get_config(search_result.config, self.cfg, "nodeindexer_cfg")
    return self.context_idxr(search_result, cfg)

  def context_idxr(self, search_result: SearchResult, cfg: ContextIdxrCfg) -> SearchResult:
    """
      scores the data in search_result according to the score returned
      by method context_score which is then linearly scaled between min=0.5,max=3.

      Parameters
      ----------
        search_result: SearchResult
        cfg: ContextIdxrCfg

      Returns
      ----------
        SearchResult
    """

    def contextscore_acc(acc_data, data):
      a_uid = data.anime_uid
      a_datas = datas_filter(lambda data: data.type == DataType.recs,
                             self.get_datas(a_uid))

      if len(a_datas):
        embds = from_vstack([data.embedding
                             for data in a_datas]).to(cfg.device)
        idxs = compose(sorted, map(snd), sorted,
                       filter(lambda x: x[0] >= cfg.sim_thres), zip)(
                           torch.cosine_similarity(q, embds),
                           range(embds.shape[0]),
                       )[-cfg.topk:]

        if len(idxs):
          new_score, max_idxs = self.context_score(q, embds[idxs], 2,
                                                   cfg.cutoff_sim, cfg.device)
          new_datas = [
              Data.new(a_datas[idxs[midx]],
                       anime_uid=a_uid,
                       type=DataType.short) for midx in max_idxs
          ]
        else:
          new_datas = [data]
          new_score = 0.5

      else:
        new_datas = [data]
        new_score = cfg.sim_thres
      acc_data.append((new_datas, new_score))
      return acc_data


    q = torch.from_numpy(search_result.query.embedding).unsqueeze(0).to(
        cfg.device)
    new_datas, new_scores = zip(
        *reduce(contextscore_acc, search_result.datas, []))
    new_scores = (compose(rescale_scores(0.5, 3), np.array)(new_scores) *
                  search_result.scores)
    new_scores = compose(list, concat)(
        [[score] * len(data) for data, score, in zip(new_datas, new_scores)])

    return SearchResult.new(search_result,
                            datas=compose(list, concat)(new_datas),
                            scores=np.array(new_scores).squeeze())

  def context_score(self, q: Tensor, mat: Tensor, combinations: int,
      cutoff_sim: float, device: str) -> Tuple[float, List[int]]:
    """
      uses brute force search with pruning to find a combination
      of vectors that best linearly approximate the target vector q.

      Parameters
      ----------
        q: Tensor
          the target vector which is being approximated.
        mat: Tensor
          set of vectors that will be used to find approximation of the target vector q.
        combinations: int
          number of vectors to combine to approximate the target vector.
        cutoff_sim: int
          pruning some combinations of vector from set of vectors mat that are too similar to
          each other.
        device: str
          "cpu" or "cuda".

      Returns
      ----------
        Tuple[float,List[int]]
          float: context score between the best combination of the vectors and
                 target vector
          List[int]: indexs of vector that formed the best combination.
    """

    if combinations > mat.shape[0]:
      sims = torch.cosine_similarity(q, mat)
      return torch.mean(sims).item(), [int(torch.argmax(sims))]

    else:
      _adjmat = pair_sim(mat, mat)
      comb_idxs = compose(
          list,
          filter(lambda idx: _adjmat[tuple(idx)] <= cutoff_sim),
          torch.combinations,
      )(torch.arange(mat.shape[0]), combinations)

      if len(comb_idxs):
        q_mat = (torch.vstack([q for _ in range(len(comb_idxs))
                               ]).unsqueeze(-1).to(device))
        mat = torch.vstack([mat[c_idx].T.unsqueeze(0) for c_idx in comb_idxs])

        mat_t = torch.movedim(mat.T, -1, 0).to(device)
        r = l2_approx(q_mat, mat, mat_t)
        max_idx = torch.argmax(torch.sum(r, 1))
        return (
            torch.cosine_similarity(
                torch.movedim(mat[max_idx] @ r[max_idx], 1, 0), q).item(),
            comb_idxs[max_idx],
        )
      else:
        r = l2_approx(q.squeeze(), mat.T, mat)
        return torch.cosine_similarity((mat.T @ r).unsqueeze(0),
                                       q).item(), [int(torch.argmax(r))]
