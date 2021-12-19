# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

# useful for handling different item types with a single interface
import json
import os
import numpy as np
from functools import singledispatch, update_wrapper
from pymongo import MongoClient
import pickle

from crawler.items import AnimeItem, ReviewItem


def singledispatchmethod(func):
  dispatcher = singledispatch(func)

  def wrapper(*args, **kw):
    return dispatcher.dispatch(args[1].__class__)(*args, **kw)

  wrapper.register = dispatcher.register
  update_wrapper(wrapper, func)
  return wrapper

class ProcessPipeline:

  def process_item(self, item, spider):
    return self._process_item_dispatcher(item)

  @singledispatchmethod
  def _process_item_dispatcher(self, item):
    raise NotImplemented

  @_process_item_dispatcher.register(AnimeItem)
  def process_anime(self, item: AnimeItem) -> AnimeItem:
    if 'N/A' in item['score']:
      item['score'] = np.nan
    else:
      item['score'] = float(item['score'].replace("\n", "").strip())

    if item['rank'] == 'N/A':
      item['rank'] = np.nan
    else:
      item['rank']     = int(item['rank'].replace("#", "").strip())

    item['popularity'] = int(item['popularity'].replace("#", "").strip())
    item['uid'] = int(item['uid'])

    return item

  @_process_item_dispatcher.register(ReviewItem)
  def process_review(item: ReviewItem) -> ReviewItem:
    pass


class SaveMongoPipeline(object):
  def __init__(self, mongodb_url = ""):
    self.mongodb_url = mongodb_url

  @classmethod
  def from_crawler(cls, crawler):
    settings = crawler.settings
    return cls(settings.get('mongodb_url'))

  @property
  def is_configured(self):
    return (self.mongodb_url is not None)

  def open_spider(self, spider) -> None:
    if self.is_configured:
      self.client  = MongoClient(self.mongodb_url)
      self.db      = self.client['yuno']
      self.collection = {}
      self.collection['AnimeItem']   = self.db.animes
    else:
      raise Exception("MONGODB_URL not provided.")

  def close_spider(self, spider):
    self.client.close()

  def process_item(self, item, spider):
    self.save_item_dispatcher(item)
    return item

  @singledispatchmethod
  def save_item_dispatcher(self, item):
    raise NotImplemented

  @save_item_dispatcher.register(AnimeItem)
  def save_item_dispatcher(self, item: AnimeItem) -> None:
    self.collection["AnimeItem"].replace_one({"uid": dict(item)["uid"]}, dict(item), upsert=True)
