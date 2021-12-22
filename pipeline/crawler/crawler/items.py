# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from typing import List, Dict

from dataclasses import dataclass
import scrapy


class AnimeUid(int):
  pass

class ReviewUid(int):
  pass

class AnimeItem(scrapy.Item):
  uid: AnimeUid     = scrapy.Field()
  title: str        = scrapy.Field()
  synopsis: str     = scrapy.Field()
  link: str         = scrapy.Field()
  score: float      = scrapy.Field()
  rank: int         = scrapy.Field()
  popularity: int   = scrapy.Field()
  genres: List[str] = scrapy.Field()


class ReviewItem(scrapy.Item):
  uid: ReviewUid               = scrapy.Field()
  anime_uid: AnimeUid          = scrapy.Field()
  text: str                    = scrapy.Field()
  score: float                 = scrapy.Field()
  link: str                    = scrapy.Field()
  helpful: int                 = scrapy.Field()


class RecItem(scrapy.Item):
  anime_uid1: AnimeUid    = scrapy.Field()
  anime_uid2: AnimeUid    = scrapy.Field()
  texts: List[str]  = scrapy.Field()
  link: str         = scrapy.Field()

