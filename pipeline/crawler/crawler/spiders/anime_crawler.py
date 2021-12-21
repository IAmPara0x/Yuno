from typing import List

import json
import pickle
import sys
import scrapy
import re

from cytoolz.curried import (
    compose,
    map,
    filter,
    reduce,
)

from crawler.utils import AnimeSelectors as ASels
from crawler.items import AnimeItem


class AnimeCrawler(scrapy.Spider):
  name: str = "anime-crawler"
  allowed_domains: List[str] = ["myanimelist.net"]
  anime_urlre: re.Pattern = re.compile("https://myanimelist.net/anime/\d+/")

  def start_requests(self):

    for k in range(int(self.start), int(self.end), 50):
      yield scrapy.Request(
          f"https://myanimelist.net/topanime.php?limit={k}",
          callback=self.parse,
          meta={"dont_use_proxy": True})

  def parse(self, response):
    try:
      urls = compose(set,
                     map(lambda url: self.anime_urlre.search(url).group(0)),
                     ASels.topk_anime_urls_sel)(response)

      for url in urls:
        yield scrapy.Request(url, callback=self.parse_anime)

    except Exception as e:
      raise Exception(
          f"{e}\nsomething went wrong while using following css selector:\
                      `topk_anime_urls_selector`")

  def parse_anime(self, response):
    attr = {}
    attr["uid"] = ASels.extract_anime_uid(response)
    attr["title"] = ASels.anime_title_sel(response)
    attr["synopsis"] = ASels.anime_synopsis_sel(response)
    attr["link"] = response.url
    attr["score"] = ASels.anime_score_sel(response)
    attr["rank"] = ASels.anime_rank_sel(response)
    attr["popularity"] = ASels.anime_popularity_sel(response)
    attr["genres"] = ASels.anime_genres_sel(response)

    yield AnimeItem(**attr)


