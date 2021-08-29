
def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1/(1+np.exp(-x))

def rescale_scores(scores:np.ndarray) -> np.ndarray:
  return scores/np.linalg.norm(scores)
