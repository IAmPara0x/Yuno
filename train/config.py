from typing import NamedTuple, List, Callable, Any, Union, Tuple
from enum import Enum, auto
import torch
import torch.nn.functional as F

Tensor = torch.Tensor
Triplet = Tuple[Tensor, Tensor, Tensor]


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
  loss_fn: Callable[[Triplet], float]
  step_fn: Callable[[], None]
  model: Any
  batch_size: int
  accumulation_steps: int
  train_steps: int
  device: str
  test_steps: Union[int, None] = None
  save_info: bool = False
  save_info_path: Union[str, None] = None


class Config:
  @classmethod
  def add_config(cls, name, obj):
    setattr(cls, name, obj)


class DefaultConfig(Config):
  sampletriplets_config = SampleTripletsConfig(SampleMetric.l2_norm)
  sampledata_config = SampleDataConfig(12, 4, "cuda")
  model_config = ModelConfig(
      "/kaggle/input/review-emddings-pt2/roberta_base_anime_finetuned.h5", 5,
      [768, 896, 1024, 1152, 1280], 0.1)
  train_config = TrainConfig(F.triplet_margin_loss, 1e-5, torch.optim.AdamW, 1,
                             8, 1500, "cuda", 1500, True, "/kaggle/working")
