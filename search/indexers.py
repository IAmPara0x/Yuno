from typing import (List,
                    Dict,
                    TypeVar,
                    Optional,
                    Tuple,)
from toolz.curried import (compose,  # type: ignore
                           flip,
                           map,
                           curry,
                           filter,
                           reduce,
                           groupby,
                           concat,
                           valmap,)
from dataclasses import dataclass
import numpy as np
import torch

from .base import (AnimeUid,
                   Query,
                   ProcessedQuery,
                   Data,
                   DataType,
                   IndexerBase,
                   SearchResult,
                   SearchBase,
                   process_result,
                   sort_search,)
from .config import (SearchCfg,
                     AccIdxrCfg,
                     NodeIdxrCfg,
                     TagSimIdxrCfg,
                     ContextIdxrCfg,)
from .utils import (A,
                    Result,
                    getattr,
                    fst,
                    snd,
                    rescale_scores,
                    cos_sim,
                    get_config,
                    group_data,
                    ungroup_data,
                    pair_sim,)


@dataclass(init=True, frozen=True)
class Search(IndexerBase):
  cfg: SearchCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: SearchCfg) -> "Search":
    return Search(search_base, cfg)

  @process_result(rescale_scores(t_min=0.5, t_max=3, inverse=False))
  @sort_search
  def __call__(self, query: Query) -> SearchResult:
    cfg = get_config(query.config, self.cfg, "search_cfg")
    return self.base_idxr(query, cfg)

  def base_idxr(self, query: Query, cfg: SearchCfg) -> SearchResult:


    def acc_fn(datas, idx):
      data = compose(self.uid_data, int)(idx)
      if data.type == DataType.recs:
        datas[0].extend(map(lambda a_uid: Data.new(data, anime_uid=a_uid, type=DataType.short),
                            data.anime_uid),)
        datas[1].extend([data_sim(data) * cfg.weight] * 2)
      else:
        datas[0].append(data)
        datas[1].append(data_sim(data))
      return datas


    q_embd = compose(flip(np.expand_dims, 0), self.model)(query.text)
    data_sim = compose(cos_sim(q_embd), getattr("embedding"))
    result_data, scores = reduce(acc_fn,
                                 compose(snd, map(np.squeeze), self.knn_search)(q_embd, cfg.top_k),
                                 [[], []],)
    p_query = ProcessedQuery(query.text, q_embd.squeeze())

    return SearchResult(p_query, result_data,
                        np.array(scores, dtype=np.float32).squeeze(),
                        query.config)


@dataclass(init=True, frozen=True)
class AccIdxr(IndexerBase):
  cfg: AccIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: AccIdxrCfg) -> "AccIdxr":
    return AccIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    cfg = get_config(search_result.config, self.cfg, "accindexer_cfg")
    return self.acc_idxr(search_result, cfg)

  def acc_idxr(self, search_result: SearchResult, cfg: AccIdxrCfg) -> SearchResult:


    def acc_fn(acc_data, a_datas):
      data = a_datas[0][0]
      score = compose(cfg.score_fn, list, map(snd))(a_datas)
      acc_data[0].append(data)
      acc_data[1].append(score)

      return acc_data


    grp_datas = group_data("anime_uid", search_result.datas, search_result.scores).values()
    res_data, new_scores = reduce(acc_fn, grp_datas, [[], []])

    return SearchResult.new(search_result,
                            datas=res_data,
                            scores=np.array(new_scores, dtype=np.float32).squeeze())


@dataclass(init=True, frozen=True)
class TagSimIdxr(IndexerBase):
  cfg: TagSimIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: TagSimIdxrCfg) -> "TagSimIdxr":
    return TagSimIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    cfg = get_config(search_result.config, self.cfg, "tagsimindexer_cfg")
    return self.tagsim_idxr(search_result, cfg)

  def tagsim_idxr(self, search_result: SearchResult, cfg: TagSimIdxrCfg) -> SearchResult:


    def approx_map(item):
      a_uid, datas, scores = item[0], *map(list, zip(*item[1]))
      scores, a_tags = np.array(scores), self.get_tags(a_uid)

      if not len(a_tags):
        scores *= 2 * cfg.weight
      else:
        mat = np.vstack([tag.embedding for tag in a_tags]).T
        scores += approx_f(mat, cfg.use_negatives, cfg.use_sim) * cfg.weight

      pairs = compose(list, zip)(datas, scores)
      return (a_uid, pairs)


    approx_f = self.linear_approx(search_result.query.embedding)
    grp_datas: Dict[AnimeUid,Result] = group_data("anime_uid",
                                                                  search_result.datas,
                                                                  search_result.scores)
    result_data, new_scores = ungroup_data(approx_map, grp_datas)

    return SearchResult.new(search_result,
                            datas=result_data,
                            scores=np.array(new_scores, dtype=np.float32).squeeze())

  @curry
  def linear_approx(self, x: np.ndarray, mat: np.ndarray,
                    use_negatives: bool, use_sim: bool) -> float:

    y = np.linalg.inv(mat.T @ mat) @ mat.T @ x
    if not use_negatives and len(np.where(y < 0)[0]) > 0:

      mat = mat.T[np.where(y > 0)].T
      return self.linear_approx(x, mat, use_negatives, use_sim)
    else:

      if use_sim:
        return cos_sim(mat @ y, x).item()
      else:
        return np.average(y)


@dataclass(init=True, frozen=True)
class NodeIdxr(IndexerBase):
  cfg: NodeIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: NodeIdxrCfg) -> "NodeIdxr":
    return NodeIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    cfg = get_config(search_result.config, self.cfg, "nodeindexer_cfg")
    return self.node_idxr(search_result, cfg)

  def node_idxr(self, search_result: SearchResult, cfg: NodeIdxrCfg) -> SearchResult:


    def noderank_map(item):
      a_uid, grp_types = item
      a_ds = compose(list, filter(lambda data: data.type == DataType.long),
                     self.get_datas)(a_uid)

      if not len(a_ds):
        pairs = compose(list, concat)(grp_types.values())
        rank_scores = np.ones(len(pairs)) * cfg.weight
      else:
        mat = compose(torch.from_numpy, np.vstack)([a_d.embedding for a_d in a_ds]).to(cfg.device)
        embds = []

        if DataType.long in grp_types:
          _embds = compose(torch.from_numpy, np.vstack)(
            [d.embedding for d, _ in grp_types[DataType.long]]).to(cfg.device)
          embds.extend(_embds)

        if DataType.short in grp_types:
          _embds = compose(torch.from_numpy, np.vstack)(
            [d.embedding for d, _ in grp_types[DataType.short]]).to(cfg.device)

          max_idxs = torch.argmax(pair_sim(_embds, mat))
          embds.extend(mat[max_idxs])

        rank_scores = self.node_rank(torch.vstack(embds), mat).cpu().numpy() * cfg.weight
        pairs = concat([v for k, v in sorted(grp_types.items(), key=lambda t: t[0].value)])

      return (a_uid,
              [(pair[0], pair[1] * rank_score) for pair, rank_score in zip(pairs, rank_scores)],)


    grp_datas = valmap(groupby(compose(getattr("type"), fst)),
                       group_data("anime_uid", search_result.datas, search_result.scores),)
    datas, new_scores = ungroup_data(noderank_map, grp_datas)

    return SearchResult.new(search_result, datas=datas,
                            scores=np.array(new_scores).squeeze())

  @staticmethod
  def node_rank(v: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    assert len(v.shape) == 2, "The dim of v must be 2"
    sims = pair_sim(v, mat)
    return torch.mean(sims, dim=1)


@dataclass(init=True, frozen=True)
class ContextIdxr(IndexerBase):
  cfg: ContextIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: ContextIdxrCfg) -> "ContextIdxr":
    return ContextIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    cfg = get_config(search_result.config, self.cfg, "nodeindexer_cfg")
    return self.context_idxr(search_result, cfg)

  def context_idxr(self, search_result: SearchResult, cfg: ContextIdxrCfg) -> SearchResult:


    def contextscore_map(item):
      a_uid, datas, scores = item[0], *map(list, zip(*item[1]))
      a_datas = compose(list, filter(lambda data: data.type == DataType.recs),
                        )(self.get_datas(a_uid))
      t_score = sum(scores)

      if not len(a_datas):
        return (a_uid,
                [(datas[0], t_score)])

      else:
        q = torch.from_numpy(search_result.query.embedding).to(cfg.device)
        embds = compose(torch.from_numpy, np.vstack,)(
            [data.embedding for data in a_datas]).to(cfg.device)

        idxs = compose(sorted, map(snd), sorted, filter(lambda x: x[0] >= cfg.sim_thres), zip)(
            torch.cosine_similarity(q.unsqueeze(0), embds).cpu(),
            range(embds.shape[0]),)[-cfg.topk:]

        if len(idxs) == 0:
          return (a_uid,
                  [(datas[0], t_score)],)

        elif len(idxs) == 1:
          cntxt_scr = torch.cosine_similarity(q.unsqueeze(0),embds[idxs])
          return (a_uid, [(datas[0], t_score * cntxt_scr)])

        else:
          embds = embds[idxs]
          a_datas = [data for i, data in enumerate(a_datas) if i in idxs]

          _adjmat = pair_sim(embds, embds)
          comb_idxs = compose(list, filter(lambda idx: _adjmat[tuple(idx)] <= cfg.cutoff_sim),
                              torch.combinations)(torch.arange(embds.shape[0]).to(cfg.device), 2)

          mat = torch.vstack([embds[c_idx].T.unsqueeze(0) for c_idx in comb_idxs])
          cntxt_scr, max_idx = self.context_score(q, mat, cfg.device)

          return (a_uid,
                  [(a_datas[idx], t_score * cntxt_scr) for idx in max_idx])



    grp_datas: Dict[AnimeUid, Result] = group_data("anime_uid",
                                                   search_result.datas,
                                                   search_result.scores)
    datas, new_scores = ungroup_data(contextscore_map, grp_datas)

    return SearchResult.new(search_result, datas=datas,
                            scores=np.array(new_scores).squeeze())

  def context_score(self, q, mat, device):

    mat_t = torch.movedim(mat.T, -1, 0).to(device)
    q_mat = torch.vstack([q for _ in range(mat.shape[0])]).unsqueeze(-1).to(device)
    r = torch.inverse(mat_t @ mat) @ mat_t @ q_mat
    max_idx = torch.argmax(torch.sum(r, 1))

    cntxt_scr = torch.cosine_similarity(
        (mat[max_idx] @ r[max_idx]).unsqueeze(0), q.unsqueeze(0))
    return cntxt_scr, max_idx
