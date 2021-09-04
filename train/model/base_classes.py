from typing import Tuple
import torch
from .config import TrainConfig


class SampleTriplets:

  def __init__(self, config: TrainConfig):
    config_name = f"{self.name()}_config"
    sampledata_config = getattr(config,config_name)
    for name,val in zip(reindexer_config._fields,reindexer_config.__iter__()):
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
    hard_triplets = [(anchors[i],pos[j],neg[k]) for i,j,k in zip(range(a_size), hard_positives,hard_negatives)]

    return hard_triplets

  def pairwise_metric(self, input: Tuple[torch.Tensor,torch.Tensor]) -> torch.Tensor:

    assert isinstance(input,tuple)
    assert isinstance(input[0],torch.Tensor)
    assert input[0].shape == input[1].shape

    m1,m2 = input
    if self.sample_metric == SampleMetric.l2:
      squared_norm1 = torch.matmul(m1,m1.T).diag()
      squared_norm2 = torch.matmul(m2,m2.T).diag()
      middle = torch.matmul(m2,m1.T)

      scores = (squared_norm1.unsqueeze(0) - 2 * middle + squared_norm2.unsqueeze(1)).T

    elif self.sample_metric == SampleMetric.l1:
      diff_mat = torch.abs(m1.unsqueeze(1) - m2)
      scores = torch.sum(diff_mat,dim=-1)

    elif self.sample_metric = SampleMetric.cosine_similarity:
      pass

    else:
      raise Exception("sample_metric should be in SampleMetric enum.")

    return scores






