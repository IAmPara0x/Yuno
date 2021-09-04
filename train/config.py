from typing import NamedTuple, List, Callable
from enum import Enum, auto
import torch

class SampleMetric(Enum):
  l1_norm = auto()
  l2_norm = auto()
  cosine_similarity = auto()


class SampleTripletsConfig(NamedTuple):
  sample_metric: SampleMetric


class SampleDataConfig(NamedTuple):
  sample_class_size: int
  sample_data_size: int
  device: str


class ModelConfig(NamedTuple):
  train_steps: int
  test_steps: int
  acc_steps: int
  batch_size: int
  lr: float
  loss_fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor], float]
  pretrained_model_path: str
  device: str


class TrainConfig:
  sampletriplets_config: SampleTripletsConfig
  sampledata_config: SampleDataConfig
  model_config: ModelConfig

