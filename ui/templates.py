from ipywidgets import Button,HTML,Layout,Box,Text, IntSlider
from typing import List
from pathlib import Path

path = Path(__file__).parent


class Colors:
  black = "#1F1D36"
  navy = "#3f3351"
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

  def item_template(self, name: str, tags: List[str], url: str, img_url: str) -> HTML:
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
                  <img src={img_url} alt="" class="info">
                  {tags_html}
                </div>
                  <a href={url} class="url" target="_blank">MAL</a>
              </div>
            """

    return HTML(value=value, layout=Layout(flex="3 1 100%"))

  def info_template(self, name: str, texts: List[str]) -> HTML:
    text_template = lambda text: f"<li class='text-box'><p>{text}</p></li>"
    texts_html = " ".join([text_template(text) for text in texts])

    value = f"""
            <div class="texts">
              <p style="background: {Colors.black};">
                {self.heading_template(name)}
                <ol>
                  {texts_html}
                </ol>
              </p>
              <hr style="border: 1px solid {Colors.light_purple};">
            </div>
              <br>
            """
    return HTML(value=value,layout=Layout(flex="0 1 85%"))

  @property
  def loading_widget(self) -> HTML:
    value = f"""
              <h3 style="color: {Colors.peach2};">searching . . .</h3>
              <div class="loader"></div>
            """
    return HTML(value=value,layout=Layout(flex="0 1 auto",align_self="center"))

  @property
  def logo(self) -> HTML:
    value = """ <h1 class="logo">Yuno</h1> """
    return HTML(value=value,layout=Layout(flex="0 1 auto",align_self="center"))


  @property
  def search_btn(self) -> Button:
    btn = Button(description="search", icon="search", layout=Layout(flex="1 1 15%"))
    btn.add_class("main-btn")
    return btn

  @property
  def info_btn(self) -> Button:
    btn = Button(description="More Info", layout=Layout(flex="0 1 12%", align_self="center", margin="0 0 0 1%"))
    btn.add_class("main-btn")
    return btn

  @property
  def back_btn(self) -> Button:
    btn = Button(description="Back")
    btn.add_class("back-btn")
    return btn

  @property
  def search_bar(self) -> Text:
    search_bar = Text(placeholder="search ...", layout=Layout(flex="3 1 85%"))
    search_bar.add_class("searchTerm")
    return search_bar

  def curiosity_widget(self) -> IntSlider:

    x = IntSlider(value=128,min=32,max=284,
        step=8,description="Curiosity ",layout=Layout(flex="0 1 50%", align_self="center"))
    x.add_class("slider")
    x.style.handle_color = Colors.red
    return x
