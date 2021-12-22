import pathlib
from typing import List, Dict
from cytoolz.curried import reduce, compose, map, filter

from pymongo import MongoClient
import scrapy
from scrapy.selector.unified import Selector

from crawler.items import ReviewItem, RecItem, AnimeUid, ReviewUid
from crawler.utils import ReviewSelectors as RevSels
from crawler.utils import RecSelectors as RecSels


FILEPATH = pathlib.Path(__file__).parent.parent.parent.resolve()


class IndexCrawler(scrapy.Spider):
  name: str = "index-crawler"
  allowed_domains: List[str] = ["myanimelist.net"]

  def __init__(self, cache_data: Dict[AnimeUid, List[ReviewUid]], mongodb_url: str,
               parse_recs: str):

    self.mongodb_url = mongodb_url
    self.cache_data = cache_data

    self.cookies = {"reviews_sort": "recent", "reviews_inc_preliminary": "1"}
    self.parse_recs = parse_recs
    self.new_cache_data: Dict[AnimeUid, List[ReviewUid]] = {}
    self.hit_limit = 10

  @classmethod
  def from_crawler(cls, crawler, *args, **kwargs) -> "IndexCrawler":
    settings = crawler.settings
    mongodb_url = settings.get("mongodb_url")
    parse_recs = settings.get("parse_recs")

    anime_uids = cls.get_anime_uids()
    cache_data = cls.get_cache_data(anime_uids, mongodb_url)
    spider = cls(cache_data, mongodb_url, *args, **kwargs)
    spider._set_crawler(crawler)
    return spider

  @staticmethod
  def close(spider, reason):

    new_cache_data = spider.new_cache_data
    mongodb_url = spider.mongodb_url

    client = MongoClient(mongodb_url)
    cache_col = client["yuno"].reviews_cache

    for anime_uid, review_cache in new_cache_data.items():
      item = {"anime_uid": anime_uid, "review_uids": review_cache}
      cache_col.replace_one({"anime_uid": anime_uid}, item, upsert=True)

    client.close()

  @staticmethod
  def get_anime_uids() -> List[AnimeUid]:
    with open(f"{FILEPATH}/data/crawl_index_uids.txt", "r") as f:
      lines = f.readlines()
      f.seek(0)
      anime_uids = [AnimeUid(uid) for uid in lines]
    return anime_uids

  @staticmethod
  def get_cache_data(anime_uids: List[AnimeUid],
                     mongodb_url: str) -> Dict[AnimeUid, List[ReviewUid]]:

    def acc_cache(cache_data, anime_uid):
      n = cache_col.count_documents({"anime_uid": anime_uid})
      if n == 0:
        cache_data[anime_uid] = []
      else:
        doc = cache_col.find({"anime_uid": anime_uid})[0]
        cache_data[anime_uid] = doc["review_uids"]
      return cache_data

    client = MongoClient(mongodb_url)
    cache_col = client["yuno"].reviews_cache
    cache_data = reduce(acc_cache, anime_uids, {})
    client.close()

    return cache_data

  def start_requests(self):

    for anime_uid in self.cache_data.keys():
      base_url = f"https://myanimelist.net/anime/{anime_uid}/YUNOGASAI/"

      yield scrapy.Request(
          f"{base_url}/reviews?p=1",
          callback=self.parse_list_reviews,
          # meta={"dont_use_proxy": True},
          cookies=self.cookies,
      )

      if self.parse_recs == "True":
        yield scrapy.Request(
            f"{base_url}/userrecs", callback=self.parse_userrecs)

  def parse_list_reviews(self, response):

    def review_fn(sel: Selector) -> ReviewItem:
      attr = {}
      attr["anime_uid"] = anime_uid
      attr["text"] = RevSels.text_sel(sel)
      attr["score"] = RevSels.score_sel(sel)
      attr["link"] = RevSels.link_sel(sel)
      attr["helpful"] = RevSels.helpful_sel(sel)
      uid = ReviewUid(attr["link"].split("=")[-1])
      attr["uid"] = uid

      return ReviewItem(**attr)

    def acc_hit(hit_rem:int , review: ReviewItem) -> int:
      if review["uid"] in self.cache_data[anime_uid]:
        return hit_rem - 1
      else:
        return hit_rem

    next_page = RevSels.next_page_sel(response)
    curr_page = int(response.url.split("=")[-1])
    anime_uid = AnimeUid(response.url.split("/")[4])

    reviews = compose(list,
                      map(review_fn),
                      RevSels.reviews_sel
                      )(response)

    if curr_page == 1:
      self.new_cache_data[anime_uid] = [review["uid"] for review in reviews]

    hit_rem = reduce(acc_hit, reviews, self.hit_limit)

    for review in reviews:
      yield review

    if hit_rem <= 0:
      print(
        f"{'='*40}\n HIT LIMIT REACHED ABORTING EARLY AT PAGE: {curr_page}\n{'='*40}"
      )
      next_page = None

    if next_page is not None:
      yield scrapy.Request(
          next_page, callback=self.parse_list_reviews, cookies=self.cookies)

  def parse_userrecs(self, response):

    def rec_fn(sel: Selector) -> RecItem:
      attr = {}
      attr["texts"] = RecSels.text_sel(sel)
      url = RecSels.link_sel(sel)
      attr["link"] = url
      attr["anime_uid1"], attr["anime_uid2"] = RecSels.animeuids_sel(url)
      return RecItem(**attr)

    recs = compose(list,
                   map(rec_fn),
                   RecSels.recs_sel
                   )(response)

    for rec in recs:
      yield rec
