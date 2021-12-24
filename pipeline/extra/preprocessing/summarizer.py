
import re
from typing import List
import torch
from transformers import pipeline


class Summarizer:
  """
    Class to 'summarize' reviews of the anime.
    This process results in getting only important part of review
    ie. parts of reviews related to anime.
  """

  MAX_SUMMARIZATION_LENGTH: int = 192
  TOP_K: int = 96
  TOP_P: int = 0.95
  NUM_BEAMS: int = 4
  SAMPLE: bool = True
  EARLY_STOPPING: bool = False
  TRUNCATION: bool = True
  MODEL_NAME: str = "sshleifer/distilbart-cnn-12-6"
  DEVICE: int = 0 if torch.cuda.is_available() else -1
  LENGTH_PENALTY:int = 2.0
  summarizer_model = pipeline("summarization", model=MODEL_NAME, device=DEVICE)
  BATCH_SIZE:int = 24

  def __init__(self):
    pass

  def summarize(self, docs: List[str]) -> List[str]:
        summarized_docs = self.summarizer_model(docs, max_length=self.MAX_SUMMARIZATION_LENGTH,
                                                  do_sample=self.SAMPLE,
                                                  top_k=self.TOP_K,
                                                  top_p=self.TOP_P,
                                                  num_beams=self.NUM_BEAMS,
                                                  early_stopping=self.EARLY_STOPPING,
                                                  length_penalty=self.LENGTH_PENALTY,
                                                  truncation=self.TRUNCATION)
    summarized_docs = [doc["summary_text"] for doc in summarized_docs]
    return summarized_docs

