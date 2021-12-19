# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from typing import List, Dict

import scrapy


class AnimeUid(int): pass


class AnimeItem(scrapy.Item):
  uid: AnimeUid = scrapy.Field()
  title: str = scrapy.Field()
  synopsis: str = scrapy.Field()
  link: str = scrapy.Field()
  score: float = scrapy.Field()
  rank: int = scrapy.Field()
  popularity: int = scrapy.Field()
  genres: List[str] = scrapy.Field()


class ReviewItem(scrapy.Item):
  uid: AnimeUid = scrapy.Field()
  profile: str = scrapy.Field()
  anime_uid: int = scrapy.Field()
  text: str = scrapy.Field()
  score: float = scrapy.Field()
  cat_scores: Dict[str, float] = scrapy.Field()
  link: str = scrapy.Field()
  helpful: int = scrapy.Field()
