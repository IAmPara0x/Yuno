# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

# useful for handling different item types with a single interface
from typing import NoReturn
import json
import os
import numpy as np
from pymongo import MongoClient
import pickle

from crawler.items import AnimeItem, ReviewItem, RecItem
from .utils import singledispatchmethod


class ProcessPipeline:

  def process_item(self, item, spider):
    return self._process_item_dispatcher(item)

  @singledispatchmethod
  def _process_item_dispatcher(self, item):
    raise NotImplemented

  @_process_item_dispatcher.register(AnimeItem)
  def process_anime(self, item: AnimeItem) -> AnimeItem:
    if item['score'] is None:
      item['score'] = np.nan
    else:
      item['score'] = float(item['score'].replace("\n", "").strip())

    if item['rank'] == None:
      item['rank'] = np.nan
    else:
      item['rank'] = int(item['rank'].replace("#", "").strip())

    item['popularity'] = int(item['popularity'].replace("#", "").strip())
    item['uid'] = int(item['uid'])

    return item

  @_process_item_dispatcher.register(ReviewItem)
  def process_review(self, item: ReviewItem) -> ReviewItem:
    return item

  @_process_item_dispatcher.register(RecItem)
  def process_review(self, item: RecItem) -> RecItem:
    return item


class SaveMongoPipeline(object):

  def __init__(self, mongodb_url):
    self.mongodb_url = mongodb_url

  @classmethod
  def from_crawler(cls, crawler):
    settings = crawler.settings
    return cls(settings.get('mongodb_url'))

  @property
  def is_configured(self):
    return (self.mongodb_url is not None)

  def open_spider(self, spider) -> NoReturn:
    if self.is_configured:
      self.client = MongoClient(self.mongodb_url)
      self.db = self.client['yuno']
      self.collection = {}
      self.collection['animes'] = self.db.animes
      self.collection['reviews'] = self.db.reviews
      self.collection['recs'] = self.db.recs
    else:
      raise Exception("MONGODB_URL not provided.")

  def close_spider(self, spider):
    self.client.close()

  def process_item(self, item, spider):
    self.save_item_dispatcher(item)
    return item

  @singledispatchmethod
  def save_item_dispatcher(self, item) -> NoReturn:
    raise NotImplemented

  @save_item_dispatcher.register(AnimeItem)
  def _save_anime(self, item: AnimeItem) -> NoReturn:
    item = dict(item)
    self.collection["animes"].replace_one({"uid": item["uid"]}, item, upsert=True)

  @save_item_dispatcher.register(ReviewItem)
  def _save_review(self, item: ReviewItem) -> NoReturn:
    item = dict(item)
    self.collection["reviews"].replace_one({"uid": item["uid"]}, item, upsert=True)

  @save_item_dispatcher.register(RecItem)
  def _save_review(self, item: RecItem) -> NoReturn:
    item = dict(item)
    self.collection["recs"].replace_one({"link": item["link"]}, item, upsert=True)
