import re
from typing import List, NamedTuple, Callable
from cytoolz.curried import reduce, compose, map # type: ignore
from enum import Enum
import inspect


class SpecialToken:
  """
  special tokens that will be used with transformers to replace normal text.
  """
  anime_name_token = "[ANIME_NAME]"
  main_char_token = "[MAIN_CHAR]"
  male_char_token = "[MALE_CHAR]"
  female_char_token = "[FEMALE_CHAR]"


class Gender(Enum):
  MALE = SpecialToken.male_char_token
  FEMALE = SpecialToken.female_char_token
  OTHER = SpecialToken.main_char_token


class Character(NamedTuple):
  names: List[str]
  gender: Gender


class AnimeInfo(NamedTuple):
  uid: int
  names: List[str]
  characters: List[Character]


class FilterText:
  """
    filters the text/review with of a particular anime with the special tokens.
    custom filters can be added using 'add_filter' method that replace desired text with
    desired special token.
  """
  _filter_names = ["filter_anime_names", "filter_character_names", "filter_special_chars"]

  def __init__(self, anime_infos: List[AnimeInfo]):
    self.anime_infos = {
        anime_info.uid: anime_info
        for anime_info in anime_infos
    }

  @staticmethod
  def filter_anime_names(anime_info: AnimeInfo, texts: List[str]) -> List[str]:
    if len(anime_info.names):
      anime_names = r"\b|\b".join([re.escape(name) for name in anime_info.names])
      regexp = lambda text: re.sub(rf"(?i){anime_names}",
                                   f"{SpecialToken.anime_name_token}", text)
      return compose(list, map)(regexp, texts)
    return texts

  @staticmethod
  def filter_character_names(anime_info: AnimeInfo,
                             texts: List[str]) -> List[str]:
    characters = anime_info.characters

    def sub_char_name(char, texts):
      if len(char.names):
        char_names = r"\b|\b".join([re.escape(name) for name in char.names])
        return [
          re.sub(rf"(?i){char_names}", f"{char.gender.value}", text)
          for text in texts
        ]
      else:
        return texts


    def names_filter(chars, texts):
      if not chars: return texts
      return names_filter(chars[1:], sub_char_name(chars[0], texts))

    return names_filter(characters, texts)

  @staticmethod
  def filter_special_chars(_, texts: List[str]) -> List[str]:
    special_chars_re = re.compile(r"[^a-zA-Z0-9\.\,\?\'\"\[\]\_]+")
    cont_ws_re = re.compile(r"\s+")
    non_ascii_re = re.compile(r"[^\x00-\x7F]+")

    texts = compose(list,map(lambda text: re.sub(non_ascii_re, "", text)),
                    map(lambda text: re.sub(cont_ws_re, " ", text)),
                    map(lambda text: re.sub(special_chars_re, " ", text))
                    )(texts)
    return texts


  @classmethod
  def add_filter(cls, name: str, func: Callable[[AnimeInfo, List[str]],
                                                List[str]]):
    """
    add custom filter that will replace particular text with special tokens.

    Parameters
    ---------
    name: string
      name of the filter this name will be used to create a new staticmethod for FilterText.
    func: Callable[[AnimeInfo, List[str]],List[str]]
      a callable function that will be used as filter for the text.

    Returns
    --------
    None
    """
    cls._filter_names.append(name)
    setattr(cls, name, staticmethod(func))

  def filter(self, uid: int, texts: List[str], filter_name: str):
    anime_info = self.anime_infos[uid]
    dispatch = getattr(self, filter_name)
    assert inspect.isfunction(dispatch) == True
    return dispatch(anime_info, texts)

  def filter_all(self, uid: int, texts: List[str]) -> List[str]:
    for filter_name in self._filter_names:
      texts = self.filter(uid, texts, filter_name)
    return texts
