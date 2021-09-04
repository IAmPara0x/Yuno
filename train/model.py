import torch
import torch.nn as nn
from .config import TrainConfig

class Model(nn.Module):
  def __init__(self, config: TrainConfig):

    config_name = f"{self.name()}_config"
    model_config = getattr(config,config_name,None)
    for name,val in zip(model_config._fields,model_config.__iter__()):
      setattr(self,name,val)

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()
