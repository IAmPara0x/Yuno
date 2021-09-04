from typing import Tuple
from tqdm import tqdm

from .base_classes import TrainBase, SampleData, SampleTriplets
from .model import Model
from .config import Config, DefaultConfig

class Train:
  def __init__(self,train_base: TrainBase,config: Config):
    self.sample_data = SampleData(train_base,config)
    self.sample_triplets = SampleTriplets(config)
    self.model = Model(config)

    config_name = f"{self.name()}_config"
    train_config = getattr(config,config_name,None)
    for name,val in zip(train_config._fields,train_config.__iter__()):
      setattr(self,name,val)

    self.optim = self.optimizer(self.model.parameters())

  def _batch_sample(self,sample_test:bool=False) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    b_anchors,b_pos,b_negs = [],[],[]
    for _ in range(self.batch_size):
      anchors,pos_data,neg_data = self.sample_data()
      a,p,n = self.sample_triplets(anchors,pos_data,neg_data,self.model)
      b_anchors.append(a)
      b_pos.append(p)
      b_negs.append(n)
    return torch.vstack(b_anchors),torch.vstack(b_pos),torch.vstack(b_negs)

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()
