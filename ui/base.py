from dataclasses import dataclass, field
from typing import List, Dict,Callable
import numpy as np
from cytoolz import curry
import ast

from ipywidgets import (Layout,Box,HTML,Output,Text,Button)

#NOTE: not recommended use `pipenv install`.
# But using it alternative will take lot to time to install all the packages required

from ..search.base import AnimeUid, Data, Query
from ..search.pipelines import SearchPipelineBase
from .templates import Templates


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

  def __getitem__(self,anime_uid: AnimeUid) -> AnimeInfo:
    return self._anime_infos[anime_uid]


class LayoutState:
  def __init__(self, canvas: Output, **kwargs):
    self.style = Layout(**kwargs)
    self.states = []
    self.canvas = canvas

  def __call__(self,clear=False):
    with self.canvas:
      if clear:
        self.canvas.clear_output()
      display(self.box())

  def __len__(self):
    return len(self.states)

  def box(self):
    return Box([state() if callable(state) else state for state in self.states],
                layout=self.style)

  def add(self,item) -> None:
    self.states.append(item)

  def pop(self):
    self.states.pop()

  def clear_states(self) -> None:
    self.states = []


@dataclass
class BaseWidget:
  main_layout: LayoutState
  canvas: Output
  templates = Templates()


@dataclass
class SearchWidget(BaseWidget):
  style: Layout
  search: Callable

  def __post_init__(self):
    self.search_btn = self.templates.search_btn
    self.search_btn.on_click(self.process)
    self.search_bar = self.templates.search_bar

  def __call__(self):
    return Box([self.search_bar,self.search_btn])

  def process(self, _):
    if len(self.main_layout) == 3:
      self.main_layout.pop()
    self.main_layout.add(self.templates.loading_widget)
    with self.canvas:
      self.main_layout(clear=True)
      result = self.search(self.search_bar.value)
      self.main_layout.pop()
      self.main_layout.add(result)
      self.main_layout(clear=True)

@dataclass
class ItemWidget(BaseWidget):
  data: Data
  info: AnimeInfo

  def __post_init__(self):
    self.name = self.info.names[0]
    self.url = self.info.mal_url

    #TODO: very bad way to taking out tags will improve it later.

    try:
      self.tags = ast.literal_eval(self.data.text[-1])
      self.texts = self.data.text[:-1]
    except ValueError:
      self.tags = []
      self.texts = self.data.text

    self.info_area = self.templates.item_template(self.name, self.tags, self.url)
    self.info_btn = self.templates.info_btn
    self.info_btn.on_click(self.process)

    self.back_btn = self.templates.back_btn

  def __call__(self):
    return Box([self.info_area,self.info_btn],
              layout=Layout(display="flex", flex_flow="row nowrap", align_items="stretch", width="100%"))

  def process(self,_):

    prev_states = self.main_layout.states.copy()
    self.main_layout.clear_states()
    self.main_layout.add(self.templates.info_template(self.name,self.texts))
    self.back_btn.on_click(self.revert(prev_states))
    self.main_layout.add(self.back_btn)

    with self.canvas:
      self.main_layout(clear=True)

  @curry
  def revert(self,prev_states,_):
    self.main_layout.states = prev_states
    with self.canvas:
      self.main_layout(clear=True)

@dataclass
class ResultWidget(BaseWidget):
  search_engine: SearchPipelineBase
  info_base: InfoBase
  style: Layout

  def __call__(self, text: str):
    search_result = self.search_engine(Query(text, None))
    items = []

    for data in search_result.datas:
      item = ItemWidget(self.main_layout,self.canvas,data,self.info_base._anime_infos[data.anime_uid])
      items.append(item())
    return Box(items,layout=self.style)


