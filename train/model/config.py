from typing import NamedTuple, List
from enum import Enum, auto
import numpy as np

class SampleMetric(Enum):
  l1 = auto()
  l2 = auto()
  cosine_similarity = auto()

class Anime(NamedTuple):
  uid: int
  neg_uid: List[int]
  sentences: List[str]

class TrainParameters(NamedTuple):
  lr: float
  steps: int

class SampleTripletsConfig(NamedTuple):
  sample_metric: SampleMetric

class TrainConfig:
  sampletriplets_config: SampleTripletsConfig

