from ipywidgets import Button,HTML,Layout,Box,Text
from typing import List
from pathlib import Path

path = Path(__file__).parent


class Colors:
  black = "#1F1D36"
  navy = "#3F3351"
  purple = "#864879"
  peach = "#E9A6A6"
  peach2 = "#FDD2BF"
  cold = "#916BBF"
  white = "#F9F9F9"
  pink = "#FFBCBC"
  light_blue = "#88FFF7"
  light_red = "#E98580"
  red = "#DF5E5E"
  light_green = "#CDF0EA"
  light_purple = "#BEAEE2"


class Templates:

  def __init__(self):
    with open(f"{path}/styles.html", "r") as f:
      styles = f.read()
      display(HTML(value = styles))

  @staticmethod
  def heading_template(name) -> str:
    return f"""
              <h2 class="heading"> <b> {name} </b> </h2>
            """

  def item_template(self, name: str, tags: List[str], url: str):
    tag_template = lambda tag_name: f"<li class='tag'>{tag_name}</li>"
    tags = " ".join([tag_template(tag) for tag in tags])

    tags_html = f"""
                <ul class="tags">
                  <li style="border: none; color: {Colors.light_blue};" class="tag"><b>Tags: </b></li>
                  {tags}
                </ul>
                """

    value = f"""
              <div class="container">
                {self.heading_template(name)}
                <div class="main">
                  <img src="data:image/jpeg;base64,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"
                    alt="" class="info">
                  {tags_html}
                </div>
                  <a href={url} class="url" target="_blank">MAL</a>
              </div>
            """

    return HTML(value=value, layout=Layout(flex="3 1 100%"))

  def info_template(self, name: str, texts: List[str]):
    text_template = lambda text: f"<li class='text-box'><p>{text}</p></li>"
    texts_html = " ".join([text_template(text) for text in texts])

    value = f"""
              <p style="background: {Colors.black};">
                {self.heading_template(name)}
                <ol>
                  {texts_html}
                </ol>
              </p>
              <hr style="border: 1px solid {Colors.light_purple};">
              <br>
            """
    return HTML(value=value,layout=Layout(flex="0 1 85%"))

  @property
  def loading_widget(self):
    value = f"""
              <h3 style="color: {Colors.peach2};">searching . . .</h3>
              <div class="loader"></div>
            """
    return HTML(value=value,layout=Layout(flex="0 1 auto",align_self="center"))

  @property
  def logo(self):
    value = """ <h1 class="logo">Yuno</h1> """
    return HTML(value=value,layout=Layout(flex="0 1 auto",align_self="center"))


  @property
  def search_btn(self):
    btn = Button(description="search", icon="search", layout=Layout(flex="1 1 15%"))
    btn.add_class("main-btn")
    return btn

  @property
  def info_btn(self):
    btn = Button(description="More Info", layout=Layout(flex="0 1 12%", align_self="center", margin="0 0 0 1%"))
    btn.add_class("main-btn")
    return btn

  @property
  def back_btn(self):
    btn = Button(description="Back")
    btn.add_class("back-btn")
    return btn

  @property
  def search_bar(self):
    search_bar = Text(placeholder="search ...", layout=Layout(flex="3 1 85%"))
    search_bar.add_class("searchTerm")
    return search_bar
