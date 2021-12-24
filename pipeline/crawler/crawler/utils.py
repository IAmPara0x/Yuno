from typing import List, NewType, Optional, Dict, Tuple
from functools import singledispatch, update_wrapper
import re
from cytoolz.curried import curry,map,filter,compose

from scrapy.http.response.html import HtmlResponse
from scrapy.selector.unified import SelectorList, Selector

from crawler.items import AnimeInfoItem, TagItem

Response = NewType("Response", HtmlResponse)

def singledispatchmethod(func):
  dispatcher = singledispatch(func)

  def wrapper(*args, **kw):
    return dispatcher.dispatch(args[1].__class__)(*args, **kw)

  wrapper.register = dispatcher.register
  update_wrapper(wrapper, func)
  return wrapper

class AnimeSelectors:

  @staticmethod
  def topk_anime_urls_sel(res: Response) -> List[str]:
    _sel = "td.title.al.va-t.word-break a::attr(href)"
    return res.css(_sel).extract()

  @staticmethod
  def extract_anime_uid(res: Response) -> str:
    return res.url.split("/")[4]

  @staticmethod
  def anime_title_sel(res: Response) -> str:
    _sel = "p.title-english.title-inherit::text"
    return res.css(_sel).extract_first()

  @staticmethod
  def anime_synopsis_sel(res: Response) -> str:
    _sel = "p[itemprop='description']::text"
    return " ".join(res.css(_sel).extract()[:-1])

  @staticmethod
  def anime_score_sel(res: Response) -> str:
    _sel = "div.score ::Text"
    return res.css(_sel).extract_first()

  @staticmethod
  def anime_rank_sel(res: Response) -> str:
    _sel = "span.ranked strong ::Text"
    return res.css(_sel).extract_first()

  @staticmethod
  def anime_popularity_sel(res: Response) -> str:
    _sel = "span.popularity strong ::Text"
    return res.css(_sel).extract_first()

  @staticmethod
  def anime_genres_sel(res: Response) -> str:
    _sel = "div span[itemprop='genre'] ::text"
    return res.css(_sel).extract()


class ReviewSelectors:

  @staticmethod
  def next_page_sel(res: Response) -> Optional[str]:
    _sel = "div.mt4 a::attr(href)"
    urls = res.css(_sel).extract()

    curr_page = int(res.url.split("=")[-1])

    if len(urls) == 2:
      return urls[1]
    elif curr_page == 1 and len(urls) == 1:
      return urls[0]
    else:
      return None

  @staticmethod
  def reviews_sel(res: Response) -> SelectorList:
    _sel = "div.borderDark"
    return res.css(_sel)

  @staticmethod
  def text_sel(sel: Selector) -> str:
    _sel = "div.spaceit.textReadability.word-break.pt8.mt8 ::text"
    texts = compose(TextFilters.apply_all_filters,
                    lambda s: s.css(_sel).extract())(sel)
    return " ".join(texts)

  @staticmethod
  def helpful_sel(sel: Selector) -> int:
    _sel = "div.spaceit div.lightLink.spaceit strong span::text"
    return int(sel.css(_sel).extract_first())

  @staticmethod
  def score_sel(sel: Selector) -> int:
    _sel = "div.spaceit div.mb8 div ::text"
    sel_str = sel.css(_sel).extract()[-1]
    score = re.search(r"\d+",sel_str).group(0)
    return int(score)

  @staticmethod
  def link_sel(sel: Selector) -> str:
    _sel = "div.mt12.pt4.pb4.pl0.pr0.clearfix div a::attr(href)"
    url = sel.css(_sel).extract_first()
    return url

class RecSelectors:

  @staticmethod
  def recs_sel(res: Response) -> SelectorList:
    _sel = "div.borderClass > table"
    return res.css(_sel)

  def link_sel(sel: Selector) -> str:
    _sel = "div > span > a::attr(href)"
    return sel.css(_sel).extract_first()

  def text_sel(sel: Selector) -> List[str]:
    _sel1 = "div.borderClass.bgColor2 div.spaceit_pad.detail-user-recs-text *::text"
    _sel2 = "div.borderClass.bgColor1 div.spaceit_pad.detail-user-recs-text *::text"

    texts = compose(list,TextFilters.apply_all_filters,
                    lambda s: s.css(f"{_sel1},{_sel2}").extract())(sel)
    return texts

  def animeuids_sel(url: str) -> Tuple[int,int]:

    uids = compose(list,map(int),
                   lambda x: x.split("-")
                   )(url.split("/")[-1])

    return (uids[0],uids[1])


# IMPROVE: make it like search indexing pipeline.
class TextFilters:
  special_char_re = re.compile("\r\n|\n|\r|\xa0")
  comp_ws_re = re.compile("^\s+$")
  cont_ws_re = re.compile("\s+")
  trail_or_start_ws_re = re.compile("^\s|\s$")
  special_words_re = re.compile(r"^Helpful$|"
                                r"^read more$|"
                                r"^Overall$|"
                                r"^Story$|"
                                r"^Animation$|"
                                r"^Sound$|"
                                r"^Enjoyment$|"
                                r"^Character$|"
                                r"^&nbsp$|"
                                r"^\xa0$")

  comp_num_re = re.compile("^\d+$")

  @staticmethod
  def apply_all_filters(texts: List[str]) -> List[str]:

    @curry
    def apply_filter(filter, sub_char, str):
      return re.sub(filter, sub_char, str)

    texts = compose(
        filter(lambda str: len(str) > 1),
        map(apply_filter(TextFilters.comp_num_re,"")),
        map(apply_filter(TextFilters.special_words_re,"")),
        map(apply_filter(TextFilters.trail_or_start_ws_re,"")),
        map(apply_filter(TextFilters.cont_ws_re, " ")),
        map(apply_filter(TextFilters.comp_ws_re,"")),
        map(apply_filter(TextFilters.special_char_re, " ")),
        )(texts)

    return texts

class AnimeInfoSelector:
  query: str = '''
                query ($id: Int) {
                  Media (idMal: $id, type: ANIME) {
                    id
                    idMal
                    title {
                      romaji
                      english }
                    description
                    coverImage{
                      large }
                    synonyms
                    siteUrl
                    tags{
                      id
                      name
                      description
                      category
                      rank }
                    characters(role: MAIN) {
                     nodes{
                       name {
                        first
                        last
                        full
                        alternative }
                       gender } } } }
                '''

  def animeinfo_sel(body: Dict) -> AnimeInfoItem:
    attr = {}
    attr["uidMal"]      = body["idMal"]
    attr["uidAnilist"]         = body["id"]
    attr["title"]       = body["title"]
    attr["synonyms"]    = body["synonyms"]
    attr["description"] = body["description"]
    attr["img_url"]     = body["coverImage"]["large"]
    attr["characters"]  = body["characters"]["nodes"]
    attr["tags"]        = [(tag["id"],tag["rank"]) for tag in body["tags"]]
    return AnimeInfoItem(**attr)

  def tags_sel(body: Dict) -> List[TagItem]:
    tag_items = []

    for tag in body["tags"]:
      attr = {}
      attr["uid"]         = tag["id"]
      attr["name"]        = tag["name"]
      attr["description"] = tag["description"]
      attr["category"]    = tag["category"]
      tag_items.append(TagItem(**attr))
    return tag_items
