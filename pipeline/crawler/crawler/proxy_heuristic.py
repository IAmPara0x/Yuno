from enum import Enum
from dataclasses import dataclass, field
from functools import total_ordering
import time
import random

class ProxyType(Enum):
  working = 0
  cooldown = 5
  dead = 10

@total_ordering
class Proxy:
  def __init__(self, url):
    self.url: str = url
    self.status: ProxyType = ProxyType.working
    self.time: float = 0

  def __eq__(self, other: "Proxy") -> bool:
    if other.__class__ == self.__class__:
      return (self.status == other.status) and (self.time == other.time)
    else:
      raise NotImplemented

  def __lt__(self, other):
    if other.status == self.status:
      return self.time < other.time
    else:
      return self.status.value < other.status.value

  def __repr__(self):
    return f"Proxy: {self.url}, STATUS: {self.status}"

class ProxyPool:
  def __init__(self, file_path: str, sample_size=5) -> None:

    with open(file_path, "r") as f:
      self.pool = f.readlines()
      self.pool = [Proxy(i[:-1]) for i in self.pool]

    self.curr_idx = 0
    self.sample_size = sample_size

  def change_status(self, new_status: ProxyType) -> None:
    self.pool[self.curr_idx].status = new_status
    self.pool = sorted(self.pool)

  def change_time(self) -> None:
    self.pool[self.curr_idx].time = time.time()
    self.pool = sorted(self.pool)

  def get_proxy(self) -> Proxy:
    self.curr_idx = random.sample(range(self.sample_size),1)[0]
    return self.pool[self.curr_idx]
