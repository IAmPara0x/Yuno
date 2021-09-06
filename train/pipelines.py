from typing import Tuple
from tqdm import tqdm
import pickle
import numpy as np
import torch

from .base_classes import TrainBase, SampleData, SampleTriplets
from .model import Model
from .config import Config, DefaultConfig

class Train:
  def __init__(self,train_base: TrainBase,config: Config):
    self.sample_data = SampleData(train_base,config)
    self.sample_triplets = SampleTriplets(config)
    self.model = Model(config)

    config_name = f"{self.name()}_config"
    train_config = getattr(config,config_name)
    for name,val in zip(train_config._fields,train_config.__iter__()):
      setattr(self,name,val)

    self.model = self.model.to(self.device)
    self.optim = self.optimizer(self.model.parameters(), lr=self.lr)

    if self.pretrained_weights:
      self.model.load_state_dict(torch.load(self.pretrained_weights))

  def _batch_sample(self,sample_test:bool=False) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    b_anchors,b_pos,b_negs = [],[],[]
    batch_size = self.batch_size*4 if sample_test else self.batch_size

    for _ in range(batch_size):
      anchors,pos_data,neg_data = self.sample_data(sample_test)
      a,p,n = self.sample_triplets(anchors,pos_data,neg_data,self.model)
      b_anchors.append(a)
      b_pos.append(p)
      b_negs.append(n)
    return torch.vstack(b_anchors),torch.vstack(b_pos),torch.vstack(b_negs)

  def start_training(self):
    self.model.train()
    avg_loss,acc_loss = [],[]
    i,step = 1,0
    tbar = tqdm(total=self.train_steps)

    while True:
      if step == tbar.total:
        tbar.close()
        if self.save_info:
          print("saving weights")
          torch.save(self.model.state_dict(), f"{self.save_info_path}/model_weights.h5")
          with open(f"{self.save_info_path}/train_avg_loss.pkl","wb") as f:
            pickle.dump(avg_loss,f)
        break

      b_anchors,b_pos,b_negs = self._batch_sample()
      a_embds,p_embds,n_embds = self.model.get_triplets(b_anchors,b_pos,b_negs)
      loss = self.loss_fn(a_embds,p_embds,n_embds)
      tbar.set_description(f"AVG_LOSS: {np.average(avg_loss):.5f}, LOSS:{loss.item():.5f}, STEP: {step}")

      loss /= self.accumulation_steps
      loss.backward()
      acc_loss.append(loss.item())

      if i % self.accumulation_steps == 0:
        step += 1
        self.optim.step()
        self.optim.zero_grad()
        avg_loss.append(sum(acc_loss))
        tbar.update(1)
        acc_loss = []

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()
