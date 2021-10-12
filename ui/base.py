from dataclasses import dataclass, field
from typing import List, Dict

from search.base import AnimeUid

@dataclass(init=True, repr=True, frozen=True)
class AnimeInfo:
  name: List[str] = field(repr=True)
  img_path: str = field(repr=False)
  synopsis: str = field(repr=True)
  anilist_url: str = field(repr=True)
  mal_url: str = field(repr=True)

@dataclass(init=True, repr=True, frozen=True)
class InfoBase:
  _anime_infos: Dict[AnimeUid,AnimeInfo]
