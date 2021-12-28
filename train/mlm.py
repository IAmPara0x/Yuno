from typing import List, Callable, Any, Tuple
from dataclasses import dataclass
from cytoolz.curried import reduce, compose
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch

Tensor = torch.Tensor

@dataclass(init=True)
class MLM:
  model: nn.Module
  tokenizer: Any
  special_token_ids: List[int]
  total_steps: int
  batch_size: int
  accumulation_steps: int
  device: str

  def __post_init__(self):
    self.mask_token_id = self.tokenizer.mask_token_id
    self.vocab_len = len(self.tokenizer.get_vocab())

  def mask_tokens(self, tokens: Tensor) -> Tuple[Tensor,Tensor]:

    def mask(labels, input):
      idx, token = input

      if token in self.special_token_ids:
        labels.append(-100)
        return labels

      prob = torch.rand(1).item()

      if prob <= 0.15:
        prob /= 0.15
        labels.append(token.item())

        if prob < 0.8:
          tokens[idx] = self.mask_token_id
        elif prob < 0.9:
          tokens[idx] = torch.randint(0,self.vocab_len,(1,)).item()

      else:
        labels.append(-100)
      return labels

    labels = compose(torch.tensor)(reduce(mask, enumerate(tokens),[]))
    return (tokens,labels)

  def train(self, sents: List[str]):

    input, labels = self._get_data(sents)

    self.model.train()
    optim = torch.optim.AdamW(self.model.parameters(), lr=1e-5, betas=(0.9,0.98))

    tbar = tqdm(range(self.total_steps))
    step,avg_loss,acc_loss = 1, [], []

    # IMPROVE: remove while loop.
    while True:

      if step == self.total_steps:
        break
      else:
        idxs = torch.randint(0,input.shape[0], (self.batch_size,))
        x, y = input[idxs].to(self.device), labels[idxs].to(self.device)
        outputs = self.model(x)
        loss = F.cross_entropy(outputs.logits.view(-1, self.vocab_len), y.view(-1))
        tbar.set_description(f"loss:{loss.item()} avg_loss: {torch.mean(torch.tensor(avg_loss))}")
        loss /= self.accumulation_steps
        loss.backward()
        acc_loss.append(loss.item())
        step += 1
        if step % self.accumulation_steps == 0:
          optim.step()
          optim.zero_grad()
          avg_loss.append(sum(acc_loss))
          acc_loss = []
          tbar.update(1)


  def eval(self, sents: List[str], eval_steps: int):

    input,labels = self._get_data(sents)
    self.model.eval()

    tbar = tqdm(range(eval_steps))
    avg_loss = []

    for i in tbar:
      idxs = torch.randint(0,input.shape[0], (self.batch_size,))
      x,y = input[idxs].to(self.device),labels[idxs].to(self.device)
      with torch.no_grad():
        outputs = self.model(x)
        loss = F.cross_entropy(outputs.logits.view(-1, self.vocab_len), y.view(-1))
        avg_loss.append(loss.item())
      tbar.set_description(f"eval loss:{loss.item()} avg_eval_loss: {torch.mean(torch.tensor(avg_loss))}")

  def _get_data(self, sents: List[str]) -> Tuple[Tensor,Tensor]:

    t_sents  = self.tokenizer(sents, return_tensors="pt",
                              truncation=True, padding=True
                              )["input_ids"]

    t_labels = []

    for t_sent in t_sents:
      tokens,labels = self.mask_tokens(t_sent)
      t_sent = tokens
      t_labels.append(labels)

    return (t_sents,torch.vstack(t_labels))
