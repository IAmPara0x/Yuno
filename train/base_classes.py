
from typing import Tuple, NamedTuple, Dict, Union, List
import numpy as np
import torch
import torch.nn.functional as F
from .config import Config, SampleMetric
from .model import Model


class Data(NamedTuple):
  uid: int
  neg_uids: List[int]
  tokenized_sents: torch.Tensor
  tokenized_qs: Union[torch.Tensor, None] = None


class TrainBase(NamedTuple):
  ALL_DATA: Dict[int,Data]
  TRAIN_DATA_UIDS: List[int]
  TEST_DATA_UIDS: List[int] = None


class SampleData:
  def __init__(self, train_base: TrainBase, config: Config):

    for name,val in zip(train_base._fields,train_base.__iter__()):
      setattr(self,name,val)

    config_name = f"{self.name()}_config"
    sampledata_config = getattr(config,config_name,None)
    for name,val in zip(sampledata_config._fields,sampledata_config.__iter__()):
      setattr(self,name,val)

  def __call__(self,sample_test:bool=False):
    return self.sample(sample_test)

  def sample(self,sample_test:bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if sample_test:
      sampled_uid = np.random.choice(self.TEST_DATA_UIDS)
    else:
      sampled_uid = np.random.choice(self.TRAIN_DATA_UIDS)

    sampled_data = self.ALL_DATA[sampled_uid]
    anchors,pos_data = self._get_triplet_data(sampled_data)

    sampled_neg_uids = np.random.choice(sampled_data.neg_uids,self.sample_class_size,replace=False)
    neg_data = torch.cat([self._get_triplet_data(self.ALL_DATA[uid], is_neg_data=True)
                          for uid in sampled_neg_uids],0)
    return (anchors,pos_data,neg_data)

  def _sample_idxs(self, xs: torch.Tensor, size: int = None) -> torch.Tensor:

    if size is None: size = self.sample_data_size
    idxs = np.random.choice(len(xs), size, replace=False)
    return xs[idxs]

  def _get_triplet_data(self, data: Data,
                    is_neg_data: bool = False) -> Union[Tuple[torch.Tensor,torch.Tensor],torch.Tensor]:

    if is_neg_data:
      neg_data = self._sample_idxs(data.tokenized_sents)
      return neg_data.to(self.device)
    else:
      if data.tokenized_qs is not None:
        anchors = self._sample_idxs(data.tokenized_qs)
        pos_data = self._sample_idxs(data.tokenized_sents)
      else:
        anchors = self._sample_idxs(data.tokenized_sents)
        pos_data = anchors.detach().clone()
      return (anchors.to(self.device), pos_data.to(self.device))

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()


class SampleTriplets:
  def __init__(self, config: Config):

    config_name = f"{self.name()}_config"
    sampletriplets_config = getattr(config,config_name)
    for name,val in zip(sampletriplets_config._fields,sampletriplets_config.__iter__()):
      setattr(self,name,val)

  def __call__(self, anchors: torch.Tensor, pos_data: torch.Tensor,
                     neg_data: torch.Tensor, model: Model) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    return self.hard_sample(anchors,pos_data,neg_data,model)

  def hard_sample(self, anchors: torch.Tensor, pos_data: torch.Tensor,
                        neg_data: torch.Tensor, model: Model) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    def assert_data(anchors,pos_data,neg_data):
      assert anchors.shape == pos_data.shape
      assert anchors.shape[1] == pos_data.shape[1] == neg_data.shape[1]
      assert type(anchors) == type(pos_data) == type(neg_data) == torch.Tensor

    assert_data(anchors,pos_data,neg_data)

    a_size = anchors.size(0)
    p_size = pos_data.size(0)
    n_size = neg_data.size(0)

    mini_batch_data = torch.cat((anchors,pos_data,neg_data),0)
    with torch.no_grad():
      embeddings = model(mini_batch_data)

    a_embds = embeddings[:a_size]
    p_embds = embeddings[a_size:p_size]
    n_embds = embeddings[p_size:]

    pos_distances = self.pairwise_metric((a_embds,p_embds))
    hard_positives = torch.max(pos_distances,1).indices

    if a_embds.shape == n_embds:
      neg_distances = self.pairwise_metric((a_embds,n_embds))
    else:
      h,w = (n_size - a_size), a_embds.shape[1]
      padding = torch.zeros((h,w)).to(a_embds.device)
      padded_anchors = torch.cat((a_embds,padding))
      neg_distances = self.pairwise_metric((a_embds,n_embds))
      neg_distances = neg_distances[:a_size]

    hard_negatives = torch.min(neg_distances,1).indices
    hard_triplets = [(anchors[i],pos_data[j],neg_data[k])
                      for i,j,k in zip(range(a_size), hard_positives,hard_negatives)]

    anchors = torch.vstack([i[0] for i in hard_triplets])
    h_pos = torch.vstack([i[1] for i in hard_triplets])
    h_neg = torch.vstack([i[2] for i in hard_triplets])

    return anchors,h_pos,h_neg

  def pairwise_metric(self, input: Tuple[torch.Tensor,torch.Tensor]) -> torch.Tensor:

    assert isinstance(input,tuple)
    assert isinstance(input[0],torch.Tensor)
    assert input[0].shape == input[1].shape

    m1,m2 = input
    if self.sample_metric == SampleMetric.l2_norm:
      squared_norm1 = torch.matmul(m1,m1.T).diag()
      squared_norm2 = torch.matmul(m2,m2.T).diag()
      middle = torch.matmul(m2,m1.T)

      scores_mat = (squared_norm1.unsqueeze(0) - 2 * middle + squared_norm2.unsqueeze(1)).T

    elif self.sample_metric == SampleMetric.l1_norm:
      diff_mat = torch.abs(m1.unsqueeze(1) - m2)
      scores_mat = torch.sum(diff_mat,dim=-1)

    elif self.sample_metric == SampleMetric.cosine_similarity:
      scores_mat = F.cosine_similarity(m1.unsqueeze(1),m2,dim=-1)

    else:
      raise Exception("sample_metric should be in SampleMetric enum.")
    return scores_mat

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()

