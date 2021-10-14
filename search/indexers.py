from typing import (List,
                    Dict,
                    TypeVar,
                    Optional,
                    )
from toolz.curried import (compose,  # type: ignore
                           flip,
                           map,
                           curry,
                           filter,
                           reduce,
                           nth,
                           groupby,
                           concat,
                           itemmap,
                           valmap,
                           )
from dataclasses import dataclass
from returns.maybe import Maybe, Nothing
import numpy as np
import torch

from .base import (Query,
                   ProcessedQuery,
                   Data,
                   DataType,
                   IndexerBase,
                   SearchResult,
                   SearchBase,
                   process_result,
                   sort_search,
                   )
from .config import (Config,
                     SearchCfg,
                     AccIdxrCfg,
                     NodeIdxrCfg,
                     TagSimIdxrCfg,
                     )
from .utils import rescale_scores, cos_sim


getattr = compose(curry, flip)(getattr)
fst = nth(0)
snd = nth(1)

A = TypeVar("A")


def get_config(config: Optional[Config], default_cfg: A, name: str) -> A:
  m_cfg: Maybe[A] = Maybe.from_optional(config).bind_optional(
      lambda cfg: getattr(name, cfg))
  if m_cfg == Nothing:
    cfg = default_cfg
  else:
    cfg = m_cfg.unwrap()
  return cfg


def group_data(attr: str, datas: List[Data], scores: np.ndarray) -> Dict:
  return groupby(compose(getattr(attr), fst), zip(datas, scores))


def ungroup_data(fn, grp_datas):
  datas, scores = map(list,
                      zip(*concat(itemmap(fn, grp_datas).values(),))
                      )
  return datas, scores


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
        scores += approx_f(mat, cfg.use_negatives) * cfg.weight

      pairs = compose(list, zip)(datas, scores)
      return (a_uid, pairs)


    approx_f = self.linear_approx(search_result.query.embedding)
    grp_datas = group_data("anime_uid", search_result.datas, search_result.scores)
    result_data, new_scores = ungroup_data(approx_map, grp_datas)

    return SearchResult.new(search_result,
                            datas=result_data,
                            scores=np.array(new_scores, dtype=np.float32).squeeze())

  @curry
  def linear_approx(self, x: np.ndarray, mat: np.ndarray, use_negatives: bool) -> float:
    y = np.linalg.inv(mat.T @ mat) @ mat.T @ x
    if not use_negatives and len(np.where(y < 0)[0]) > 0:
      mat = mat.T[np.where(y > 0)].T
      return self.linear_approx(mat, x)
    else:
      return cos_sim(mat @ y, x).item()


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
        pairs = concat(grp_types.values())
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

          max_idxs = torch.argmax(torch.cosine_similarity(_embds.unsqueeze(1), mat, dim=-1),
                                  dim=-1)
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
    sims = torch.cosine_similarity(v.unsqueeze(1), mat, dim=-1)
    return torch.mean(sims, dim=1)
