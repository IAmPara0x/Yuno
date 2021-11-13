from typing import (Callable, Optional, List, TypeVar, Tuple, Dict)
from cytoolz.curried import (  # type: ignore
    curry,
    compose,
    flip,
    nth,
    concat,
    itemmap,
    groupby,
    filter)
from returns.maybe import Maybe, Nothing

import numpy as np
import torch

from .config import Config
from .base import Data

A = TypeVar("A")
Result = List[Tuple[Data, float]]
Tensor = torch.Tensor

getattr = compose(curry, flip)(getattr)
fst = nth(0)
snd = nth(1)


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))


def get_config(config: Optional[Config], default_cfg: A, name: str) -> A:
  m_cfg: Maybe[A] = Maybe.from_optional(config).bind_optional(
      lambda cfg: getattr(name, cfg))
  if m_cfg == Nothing:
    cfg = default_cfg
  else:
    cfg = m_cfg.unwrap()
  return cfg


def datas_filter(pred, datas):
  return compose(list, filter(pred))(datas)


def group_data(attr: str, datas: List[Data],
               scores: np.ndarray) -> Dict[A, Result]:
  return groupby(compose(getattr(attr), fst), zip(datas, scores))


def ungroup_data(fn, grp_datas):
  datas, scores = map(list, zip(*concat(itemmap(fn, grp_datas).values(), )))
  return datas, scores


def pair_sim(mat1, mat2):
  return torch.cosine_similarity(mat1.unsqueeze(1), mat2, dim=-1)


def from_vstack(mat):
  return compose(torch.from_numpy, np.vstack)(mat)


def l2_approx(x: Tensor, mat: Tensor, mat_t: Tensor) -> Tensor:
  return torch.inverse(mat_t @ mat) @ mat_t @ x


def rescale_scores(
    t_min: float = 1,
    t_max: float = 2,
    inverse: bool = False) -> Callable[[np.ndarray], np.ndarray]:
  def dispatch(scores: np.ndarray) -> np.ndarray:
    r_min, r_max = min(scores), max(scores)

    if inverse:
      scaled_scores = (r_min - scores) / (r_max - r_min)
      return scaled_scores * (t_max - t_min) + t_max
    else:
      scaled_scores = (scores - r_min) / (r_max - r_min)
      return scaled_scores * (t_max - t_min) + t_min

  return dispatch


@curry
def cos_sim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
