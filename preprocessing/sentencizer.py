import re
from cytoolz.curried import reduce
from typing import List
from abc import ABCMeta, abstractmethod
import spacy


class SentencizerBase(metaclass=ABCMeta):
  def __init__(self, nlp):
    self.nlp = nlp

  @staticmethod
  @abstractmethod
  def format_text(text: str) -> str:
    pass

  @abstractmethod
  def sents(self, doc: str) -> List[str]:
    pass


class ReviewSentencizer(SentencizerBase):
  """
    Divides long review into sentences of desired length.
    Using greedy sentence filling algorithm.
  """

  MIN_SENTENCE_LENGTH: int = 96
  MAX_SENTENCE_LENGTH: int = 256

  def __init__(self, nlp):
    self.nlp = nlp

  @staticmethod
  def format_text(text: str) -> str:
    """
    Preprocess the review to remove text elements.
    Like consecutive spaces, blank lines etc.
    """

    emoji_regex = re.compile(
        pattern="["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE)
    text = emoji_regex.sub(r'', text)
    text = re.sub(r"\r|\n", "", text)
    text = re.sub(r"\xa0|&nbsp", " ", text)
    text = re.sub(" +", " ", text)

    if text[:10] == " more pics": text = text[11:]
    if text[-8:] == "Helpful ": text = text[:-8]
    text = re.sub(
        r"Overall \d+ Story \d+ Animation \d+ Sound \d+ Character \d+ Enjoyment \d+ ",
        "", text)
    return text[:-1]

  def sents(self, doc: str) -> List[str]:
    """
      Greedy Sentence filling algorithm to make sure
      that sentences are of desired length.
    """

    doc = self.format_text(doc)
    doc_sents = [i.text for i in list(self.nlp(doc).sents)]

    def greedy_sentence_filling(x, y):
      if len(x[-1].split()) >= self.MAX_SENTENCE_LENGTH:
        x.append(y)
      else:
        x[-1] += " " + y
      return x

    doc_sents = reduce(greedy_sentence_filling, doc_sents, [""])

    if len(doc_sents[-1].split()) >= self.MIN_SENTENCE_LENGTH or len(
        doc_sents) == 1:
      return doc_sents
    else:
      doc_sents[-2] += " " + doc_sents[-1]
      return doc_sents[:-1]
