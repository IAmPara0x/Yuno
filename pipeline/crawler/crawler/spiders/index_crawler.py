import pathlib
from typing import List, Dict

from pymongo import MongoClient
import scrapy
from scrapy.selector.unified import Selector

from crawler.items import ReviewItem,RecsItem, AnimeUid, ReviewUid, ReviewCache
from crawler.utils import ReviewSelectors as RevSels


FILEPATH = pathlib.Path(__file__).parent.parent.parent.resolve()

# IMPOVE: REMOVE ALL THE FOR LOOPS

class IndexCrawler(scrapy.Spider):
  name: str = "index-crawler"
  allowed_domains: List[str] = ["myanimelist.net"]

  def __init__(self, cache_data: Dict[AnimeUid, ReviewCache],
               mongodb_url: str, parse_recs: str):

    self.mongodb_url = mongodb_url
    self.cache_data = cache_data

    self.cookies = {"reviews_sort": "recent", "reviews_inc_preliminary": "1"}
    self.parse_recs = parse_recs
    self.new_cache_data = {}
    self.hit_limit = 1

  @classmethod
  def from_crawler(cls, crawler, *args, **kwargs):
    settings = crawler.settings
    mongodb_url = settings.get("mongodb_url")

    with open(f"{FILEPATH}/data/crawl_index_uids.txt", "r") as f:
      lines = f.readlines()
      f.seek(0)
      anime_uids = [AnimeUid(uid) for uid in lines]

    cache_data = cls.get_cache_data(anime_uids, mongodb_url)

    spider = cls(cache_data, mongodb_url, *args, **kwargs)
    spider._set_crawler(crawler)
    return spider

  @staticmethod
  def close(spider, reason):
    closed = getattr(spider, 'closed', None)

    new_cache_data = spider.new_cache_data
    mongodb_url = spider.mongodb_url

    client = MongoClient(mongodb_url)
    cache_col = client['yuno'].reviews_cache

    for anime_uid, review_cache in new_cache_data.items():
      item = {"anime_uid": anime_uid,
              "review_uids": review_cache.review_uids}
      cache_col.replace_one({"anime_uid": anime_uid}, item, upsert=True)

    client.close()

    if callable(closed):
      return closed(reason)

  @staticmethod
  def get_cache_data(anime_uids: List[AnimeUid], mongodb_url: str) -> Dict[AnimeUid,ReviewCache]:
    client = MongoClient(mongodb_url)
    cache_col = client['yuno'].reviews_cache
    cache_data = {}

    for anime_uid in anime_uids:
      res = cache_col.count_documents({"anime_uid": anime_uid})

      if res == 0:
        cache_data[anime_uid] = ReviewCache(anime_uid=anime_uid,
                                            review_uids=[])
      else:
        doc = cache_col.find({"anime_uid": anime_uid})[0]
        cache_data[anime_uid] = ReviewCache(anime_uid=anime_uid,
                                            review_uids=doc["review_uids"])

    client.close()

    return cache_data

  def start_requests(self):

    for anime_uid in self.cache_data.keys():
      base_url = f"https://myanimelist.net/anime/{anime_uid}/YUNOGASAI/"

      yield scrapy.Request(f"{base_url}/reviews?p=1",
                           callback=self.parse_list_reviews,
                           meta={"dont_use_proxy": True},
                           cookies = self.cookies
                          )

      if self.parse_recs == "True":
        yield scrapy.Request(f"{base_url}/userrecs",
                             callback=self.parse_userrecs
                             )

  def parse_list_reviews(self, response):
    next_page = RevSels.next_page_sel(response)
    reviews_sel = RevSels.reviews_sel(response)
    anime_uid = AnimeUid(response.url.split("/")[4])
    curr_page = int(response.url.split("=")[-1])

    hit_rem = self.hit_limit

    if curr_page == 1:
      review_cache = ReviewCache(**{"anime_uid": anime_uid, "review_uids": []})


    for review_sel in reviews_sel:
      attr = {}
      attr["anime_uid"] = anime_uid
      attr["text"] = RevSels.text_sel(review_sel)
      attr["score"] = RevSels.score_sel(review_sel)
      attr["link"] = RevSels.link_sel(review_sel)
      attr["helpful"] = RevSels.helpful_sel(review_sel)

      review_uid = ReviewUid(attr["link"].split("=")[-1])
      attr["uid"] = review_uid

      if curr_page == 1:
        review_cache.review_uids.append(review_uid)

      if review_uid in self.cache_data[anime_uid].review_uids:
        hit_rem -= 1

      yield ReviewItem(**attr)

    if curr_page == 1:
      self.new_cache_data[anime_uid] = review_cache

    if hit_rem <= 0:
      print(f"{'='*40}\n HIT LIMIT REACHED ABORTING EARLY AT PAGE: {curr_page}\n{'='*40}")
      next_page = None

    if next_page is not None:
      yield scrapy.Request(next_page,
                           callback=self.parse_list_reviews,
                           cookies=self.cookies
                           )


  # def parse_userrecs(self, response):
  #   recs = response.css("div.borderClass > table")
  #   link = recs[0].css("div > span > a::attr(href)").extract_first()
  #   texts = [recs[2].css("div.borderClass.bgColor2 div.spaceit_pad.detail-user-recs-text *::text").extract(),recs[2].css("div.borderClass.bgColor1 div.spaceit_pad.detail-user-recs-text *::text").extract()]


