from typing import Tuple
from cytoolz.curried import map,reduce,compose
from tqdm import tqdm
import pickle
import numpy as np
import torch

from .base import TrainBase, SampleData, SampleTriplets
from .model import Model
from .config import Config, DefaultConfig


Tensor = torch.Tensor
Triplet = Tuple[Tensor, Tensor, Tensor]


class Train:
  def __init__(self,train_base: TrainBase,config: Config):
    self.sample_data = SampleData(train_base,config)
    self.sample_triplets = SampleTriplets(config)

    config_name = f"{self.name()}_config"
    train_config = getattr(config,config_name)
    for name,val in zip(train_config._fields,train_config.__iter__()):
      setattr(self,name,val)

  def _batch_sample(self,sample_test:bool=False) -> Triplet:
    batch_size = self.batch_size*4 if sample_test else self.batch_size

    def acc_data(data,_):
      b_anchors,b_pos,b_negs = data
      a,p,n = self.sample_triplets(self.sample_data(sample_test),self.model)
      b_anchors.append(a)
      b_pos.append(p)
      b_negs.append(n)
      return data

    b_anchors, b_pos, b_negs = compose(map(torch.vstack), reduce)(acc_data,range(batch_size),([],[],[]))

    return b_anchors, b_pos, b_negs

  def start_training(self):
    self.model.train()
    train_loss = self._iterate(is_test = False)

    print("==================")
    print("Entering Eval Mode")
    print("==================")
    self.model.eval()
    test_loss = self._iterate(is_test = True)

    if self.save_info:
      with open(f"{self.save_info_path}/losses.pkl", "wb") as f:
        data = {"train_loss": train_loss, "test_loss": test_loss}
        pickle.dump(data,f)

  def _iterate(self,is_test: bool = False):
    if is_test:
      acc_steps = 1
      total_steps = self.test_steps
      tbar = tqdm(total=total_steps)
    else:
      acc_steps = self.accumulation_steps
      total_steps = self.train_steps
      tbar = tqdm(total=total_steps)

    def forward_pass():
      if is_test:
        with torch.no_grad():
          loss = compose(self.loss_fn,self.model,self._batch_sample)(is_test)
      else:
        loss = compose(self.loss_fn,self.model,self._batch_sample)(is_test)
      return loss

    def helper(avg_loss,i):
      loss = forward_pass()
      avg_loss.append(loss.item())
      tbar.set_description(f"AVG_LOSS: {np.average(avg_loss):.5f}, LOSS:{loss.item():.5f}, STEP: {tbar.n}")
      loss /= acc_steps
      if not is_test:
        loss.backward()
        if i % acc_steps == 0:
          tbar.update(1)
          self.step_fn()
      return avg_loss

    avg_loss = reduce(helper, range(1,acc_steps*total_steps), [])
    tbar.close()
    return avg_loss

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()
