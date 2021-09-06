from typing import NamedTuple, List, Callable, Any
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
  pretrained: str
  hid_mix: int
  embedding_layers: List[int]
  dropout: float

class TrainConfig(NamedTuple):
  loss_fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],float]
  lr: float
  optimizer: Any
  batch_size: int
  accumulation_steps: int
  train_steps: int
  test_steps: int = None
  save_info: bool = False
  save_info_path: str = None
  pretrained_weights: str = None
  device: str

class Config:
  @classmethod
  def add_config(cls,name,obj):
    setattr(cls,name,obj)

class DefaultConfig(Config):
  sampletriplets_config = SampleTripletsConfig(SampleMetric.l2_norm)
  sampledata_config = SampleDataConfig(12,4,"cuda")
  model_config = ModelConfig("/kaggle/input/review-emddings-pt2/roberta_base_anime_finetuned.h5",
                            5, [768,896,1024,1152,1280],0.1)
  train_config = DefaultPipelineConfig(lambda x: x,1,8,1e-5,1500,1000,"cuda")

