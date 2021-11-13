from dataclasses import dataclass, field
from typing import List, Dict

from yuno.search.base import AnimeUid

@dataclass(init=True, repr=True, frozen=True)
class AnimeInfo:
  names: List[str] = field(repr=True)
  img: np.array = field(repr=False)
  synopsis: str = field(repr=True)
  anilist_url: str = field(repr=True)
  mal_url: str = field(repr=True)

@dataclass(init=True, repr=False, frozen=True)
class InfoBase:
  _anime_infos: Dict[AnimeUid,AnimeInfo]
