import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1/(1+np.exp(-x))

def rescale_scores(scores:np.ndarray, t_min: int=1, t_max: int=2, inverse:bool=True) -> np.ndarray:
  r_min,r_max = min(scores),max(scores)

  if inverse:
    scaled_scores = (r_min - scores) / (r_max - r_min)
    return scaled_scores * (t_max - t_min) + t_max
  else:
    scaled_scores = (scores - r_min) / (r_max - r_min)
    return scaled_scores * (t_max - t_min) + t_min
