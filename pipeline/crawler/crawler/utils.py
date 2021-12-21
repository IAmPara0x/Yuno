from typing import Callable, List, NewType, Optional, Dict
from functools import singledispatch, update_wrapper
import re
from cytoolz.curried import curry,map,filter,compose

from scrapy.http.response.html import HtmlResponse
from scrapy.selector.unified import SelectorList, Selector

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
    sel = "td.title.al.va-t.word-break a::attr(href)"
    return res.css(sel).extract()

  @staticmethod
  def extract_anime_uid(res: Response) -> str:
    return res.url.split("/")[4]

  @staticmethod
  def anime_title_sel(res: Response) -> str:
    sel = "p.title-english.title-inherit::text"
    return res.css(sel).extract_first()

  @staticmethod
  def anime_synopsis_sel(res: Response) -> str:
    sel = "p[itemprop='description']::text"
    return " ".join(res.css(sel).extract()[:-1])

  @staticmethod
  def anime_score_sel(res: Response) -> str:
    sel = "div.score ::Text"
    return res.css(sel).extract_first()

  @staticmethod
  def anime_rank_sel(res: Response) -> str:
    sel = "span.ranked strong ::Text"
    return res.css(sel).extract_first()

  @staticmethod
  def anime_popularity_sel(res: Response) -> str:
    sel = "span.popularity strong ::Text"
    return res.css(sel).extract_first()

  @staticmethod
  def anime_genres_sel(res: Response) -> str:
    sel = "div span[itemprop='genre'] ::text"
    return res.css(sel).extract()


class ReviewSelectors:

  @staticmethod
  def next_page_sel(res: Response) -> Optional[str]:
    sel = "div.mt4 a::attr(href)"
    urls = res.css(sel).extract()

    curr_page = int(res.url.split("=")[-1])

    if len(urls) == 2:
      return urls[1]
    elif curr_page == 1:
      return urls[0]
    else:
      return None

  @staticmethod
  def reviews_sel(res: Response) -> SelectorList:
    sel = "div.borderDark"
    return res.css(sel)

  @staticmethod
  def text_sel(sel: Selector) -> str:
    _sel = "div.spaceit.textReadability.word-break.pt8.mt8 ::text"
    texts = sel.css(_sel).extract()

    @curry
    def apply_filter(filter, sub_char, str):
      return re.sub(filter, sub_char, str)

    texts = compose(
        filter(lambda str: len(str) > 1),
        map(apply_filter(TextFilters.special_words_re,"")),
        map(apply_filter(TextFilters.trail_or_start_ws_re,"")),
        map(apply_filter(TextFilters.cont_ws_re, " ")),
        map(apply_filter(TextFilters.comp_ws_re,"")),
        map(apply_filter(TextFilters.special_char_re, " ")),
        )(texts)

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


# IMPROVE: make it like search indexing pipeline.
class TextFilters:
  special_char_re = re.compile("\r\n|\n|\r")
  comp_ws_re = re.compile("^\s+$")
  cont_ws_re = re.compile("\s+")
  trail_or_start_ws_re = re.compile("^\s|\s$")
  special_words_re = re.compile(r"""^Helpful$|^read more$|^Overall$|^Story$|^Animation$|^Sound$|^Enjoyment$|^Character$""")

