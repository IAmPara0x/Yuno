from typing import List, Dict
import json
import pathlib

import scrapy
from scrapy.selector.unified import Selector

from crawler.utils import AnimeInfoSelector as AnimeInfoSel
from crawler.items import TagItem, AnimeInfoItem, AnimeUid

FILEPATH = pathlib.Path(__file__).parent.parent.parent.resolve()

class AnimeInfoCrawler(scrapy.Spider):
  name: str = "animeinfo-crawler"
  allowed_domains: List[str] = ["https://graphql.anilist.co"]

  def __init__(self, anime_uids: List[AnimeUid]):
    self.anime_uids = anime_uids

  @classmethod
  def from_crawler(cls, crawler, *args, **kwargs) -> "IndexCrawler":
    settings = crawler.settings
    anime_uids = cls.get_anime_uids()
    spider = cls(anime_uids, *args, **kwargs)
    spider._set_crawler(crawler)
    return spider

  @staticmethod
  def get_anime_uids() -> List[AnimeUid]:
    with open(f"{FILEPATH}/data/crawl_index_uids.txt", "r") as f:
      lines = f.readlines()
      f.seek(0)
      anime_uids = [AnimeUid(uid) for uid in lines]
    return anime_uids

  def start_requests(self):

    for anime_uid in self.anime_uids:
      variables = {"id": anime_uid}
      yield scrapy.Request(url=self.allowed_domains[0],
                           body = json.dumps({"query":AnimeInfoSel.query,
                                   "variables": variables }),
                           method='POST',
                           headers={'Content-Type':'application/json'}
                           )

  def parse(self, response):
    body = json.loads(response.text)["data"]["Media"]
    animeinfo_item = AnimeInfoSel.animeinfo_sel(body)
    tag_items = AnimeInfoSel.tags_sel(body)

    yield animeinfo_item

    for tag_item in tag_items:
      yield tag_item
