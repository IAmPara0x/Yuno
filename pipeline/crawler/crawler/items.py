# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from typing import List, Dict, Tuple

from dataclasses import dataclass
import scrapy


class AnimeUid(int): pass
class ReviewUid(int): pass
class TagUid(int): pass


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
  uid: ReviewUid        = scrapy.Field()
  anime_uid: AnimeUid   = scrapy.Field()
  text: str             = scrapy.Field()
  score: float          = scrapy.Field()
  link: str             = scrapy.Field()
  helpful: int          = scrapy.Field()


class RecItem(scrapy.Item):
  anime_uid1: AnimeUid  = scrapy.Field()
  anime_uid2: AnimeUid  = scrapy.Field()
  texts: List[str]      = scrapy.Field()
  link: str             = scrapy.Field()


class AnimeInfoItem(scrapy.Item):
  uidMal: int                     = scrapy.Field()
  uidAnilist: int                 = scrapy.Field()
  title: Dict[str,str]            = scrapy.Field()
  synonyms: List[str]             = scrapy.Field()
  description: str                = scrapy.Field()
  img_url: str                    = scrapy.Field()
  characters: List[Dict[str,str]] = scrapy.Field()
  tags: List[Tuple[TagUid,float]] = scrapy.Field()


class TagItem(scrapy.Item):
  uid: TagUid      = scrapy.Field()
  name: str        = scrapy.Field()
  description: str = scrapy.Field()
  category: str    = scrapy.Field()
