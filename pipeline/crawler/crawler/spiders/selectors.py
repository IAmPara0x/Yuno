from typing import Callable, List, NewType

from scrapy.http.response.html import HtmlResponse

Response = NewType("Response", HtmlResponse)


class Selectors:
  topk_anime_urls_sel: Callable[[Response], List[str]] = (lambda res:
    res.css( "td.title.al.va-t.word-break a::attr(href)").extract())

  extract_anime_uid: Callable[[Response], str] = lambda res: res.url.split("/")[4]

  anime_title_sel: Callable[[Response], str] = (lambda res:
    res.css("p.title-english.title-inherit::text").extract_first())

  anime_synopsis_sel: Callable[[Response], str] = (lambda res:
    " ".join(res.css("p[itemprop='description']::text").extract()[:-1]))

  anime_score_sel: Callable[[Response], str] = (lambda res:
    res.css("div.score ::Text").extract_first())

  anime_rank_sel: Callable[[Response], str] = (lambda res:
    res.css("span.ranked strong ::Text").extract_first())

  anime_popularity_sel: Callable[[Response], str] = (lambda res:
    res.css("span.popularity strong ::Text").extract_first())

  anime_genres_sel: Callable[[Response], List[str]] = (lambda res:
    res.css("div span[itemprop='genre'] ::text").extract())

