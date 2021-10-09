
@dataclass(init=True, repr=True, eq=False, order=False, frozen=True)
class Info:
  name: List[str] = field(compare=False,repr=True)
  img_path: str = field(compare=False,repr=False)
  synopsis: str = field(compare=False,repr=True)
  anilist_url: str = field(compare=False,repr=True)
  mal_url: str = field(compare=False,repr=True)
