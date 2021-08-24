from typing import Any

class Toto:
  def __init__(self, name:str) -> None:
    self.name = name
    for attr in ['age', 'height']:
      setattr(self, attr, 0)
  def __setattr__(self, name:str, value:Any):
    super().__setattr__(name, value)



