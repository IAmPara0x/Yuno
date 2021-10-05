from typing import Callable, NewType, Optional
import numpy as np
from toolz.curried import curry # type: ignore


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1/(1+np.exp(-x))


def rescale_scores(t_min: int = 1, t_max: int = 2, inverse: bool = False) -> Callable[[np.ndarray], np.ndarray]:
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
  return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
