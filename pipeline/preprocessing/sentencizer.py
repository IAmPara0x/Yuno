from typing import Any, Callable, List
from dataclasses import dataclass
import torch

Tensor = torch.Tensor

@dataclass(init=True)
class SentencizerBase:
  nlp: Callable[[str], List[str]]
  model: Callable[[List[str]], Tensor]
  config: Config

@dataclass(init=True)
class Config:
  tolerance: float
  threshold: Callable[[Tensor], float]
  terminate: Callable[[Tensor], bool]
  batch_size: int


@dataclass(init=True)
class Sentencizer(SentencizerBase):

  def __call__(self, texts: List[str]) -> List[str]:
    pass
