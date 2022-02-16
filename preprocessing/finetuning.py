import random
from cytoolz.curried import reduce
from typing import List
from abc import ABCMeta, abstractmethod
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, AdamW


class FineTuningBase(metaclass=ABCMeta):
  def __init__(self, modelObj, tokenizerObj, model_name):
    self.model = modelObj.from_pretrained(model_name)
    self.tokenizer = tokenizerObj.from_pretrained(model_name)
    self.model_name = model_name

    self.vocab_dict = tokenizer.get_vocab()
    self.vocab_list = list(self.vocab_dict.keys())
    self.len_vocab_list = len(self.vocab_list)
    self.pad_token_id = self.tokenizer.pad_token_id
    self.mask_token_id = self.tokenizer.mask_token_id

  @abstractmethod
  def mask_words(self, tokens: List[int]):
    pass

  @abstractmethod
  def _create_training_data(self, sents):
    pass

  @abstractmethod
  def train(self, sents):
    pass


class FineTuningModel(FineTuningBase):
  PROB_MERGE_SENTS = 0.1
  MAX_SEQ_LENGTH = 256
  BATCH_SIZE = 32
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  LR = 1e-5

  def mask_words(self, tokens: List[int]) -> List[List[int], List[int]]:
    """
      Mask random tokens for language modelling with probabilities as in BERT paper.
    """
    def mask_token(data, token):
      if token == self.pad_token_id:
        data[-1].append(-100)
        data[0].append(token)

      prob = random.random()
      if prob < 0.15:
        prob /= 0.15
        data[-1].append(token)

        if prob < 0.8:
          data[0].append(self.mask_token_id)
        elif prob < 0.9:
          data[0].append(random.randrange(self.len_vocab_list))
        else:
          data[0].append(token)

      else:
        data[-1].append(-100)
        data[0].append(token)
      return data

    return reduce(mask_token, tokens, [[], []])

  def _create_training_data(self, sents: List[List[str]]) -> List[str]:
    def acc_sents(p_sents, sent):
      if random.random() < PROB_MERGE_SENTS:
        p_sents[-1] += " " + sent
      else:
        p_sents.append(sent)
      return p_sents

    def merge(data, x):
      data.extend(reduce(acc_sents, x[1:], [x[0]]))
      return data

    input_sents = reduce(merge, sents, [])

    return input_sents

  def train(self, sents):
    pass
