from typing import List

import json
import pickle
import sys
import scrapy

from cytoolz.curried import (
    compose,
    map,
    filter,
    reduce,
)

from .selectors import Selectors as Sels
from crawler.items import AnimeItem


class AnimeCrawler(scrapy.Spider):
  name: str = "anime-crawler"
  allowed_domains: List[str] = ["myanimelist.net"]

  def start_requests(self):

    for k in range(int(self.start), int(self.end), 50):
      yield scrapy.Request(
          f"https://myanimelist.net/topanime.php?limit={k}",
          callback=self.parse,
      )

  def parse(self, response):
    try:
      urls = compose(
          filter(lambda x: x.startswith("https://myanimelist.net/anime")),
          filter(lambda x: not x.endswith("/video")), set,
          Sels.topk_anime_urls_sel)(
              response)
      for url in urls:
        yield scrapy.Request(url, callback=self.parse_anime)

    except Exception as e:
      raise Exception(
          f"{e}\nsomething went wrong while using following css selector:\
                      `topk_anime_urls_selector`")

  def parse_anime(self, response):
    attr = {}
    attr["uid"] = Sels.extract_anime_uid(response)
    attr["title"] = Sels.anime_title_sel(response)
    attr["synopsis"] = Sels.anime_synopsis_sel(response)
    attr["link"] = response.url
    attr["score"] = Sels.anime_score_sel(response)
    attr["rank"] = Sels.anime_rank_sel(response)
    attr["popularity"] = Sels.anime_popularity_sel(response)
    attr["genres"] = Sels.anime_genres_sel(response)

    yield AnimeItem(**attr)


class IndexCrawler(scrapy.Spider):
  name: str = "index-crawler"
  allowed_domains: List[str] = ["myanimelist.net"]
  pass

def print_formatted_text(text: str) -> None:
  border = "=" * 25
  print(border)
  print(text)
  print(border)
