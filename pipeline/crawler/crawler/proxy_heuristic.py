from typing import List
from enum import Enum, auto
from functools import total_ordering
import time
import random
import math
import pickle
import pathlib
from cytoolz.curried import (filter,
                             first,
                             compose,
                             )

FILEPATH = pathlib.Path(__file__).parent.parent.resolve()


class ProxyType(Enum):
  working = auto()
  cooldown = auto()
  dead = auto()


@total_ordering
class Proxy:
  def __init__(self, addr):
    self.status: ProxyType = ProxyType.working
    self.time: float = math.inf
    self.__addr: str = addr

  def __repr__(self):
    return f"Proxy: {self.__addr}, Status: {self.status}, Request Time: {self.__time}"

  @property
  def addr(self):
    self.__time = time.time()
    return self.__addr

  def __eq__(self, other):
    if other.__class__ is self.__class__:
      return (self.status == other.status) and (self.time == other.time)
    else:
      return NotImplemented

  def __ne__(self, other):
    if other.__class__ is self.__class__:
      return not self.__eq__(other)
    else:
      return NotImplemented

  def __lt__(self, other):
    if other.status == self.status:
      return (self.time) > (other.time)
    elif other.status == ProxyType.working:
      return True
    elif other.status == ProxyType.cooldown and self.status == ProxyType.dead:
      return True
    else:
      return False


class ProxyPool():
  def __init__(self, use_cached_proxy:str , proxy_file_path: str, prob_try_dead_proxy: float):

    if use_cached_proxy == "True":
      with open(f"{FILEPATH}/proxy-list-state.pkl", "rb") as f:
        self.pool: List[Proxy]= pickle.load(f)
    else:
      with open(proxy_file_path, "r") as f:
        proxies = [line[:-1] for line in f.readlines()]
        self.pool = [Proxy(proxy) for proxy in proxies]

    counter = lambda type: compose(len,
                                   list,
                                   filter
                                   )(lambda x: x.status == type, self.pool)

    self.total_proxies = len(self.pool)

    self.pool_status = {"working": counter(ProxyType.working),
                        "cooldown": counter(ProxyType.cooldown),
                        "dead": counter(ProxyType.dead)
                        }
    self.prob_try_dead_proxy = prob_try_dead_proxy

  def get_proxy(self) -> Proxy:
    self.pool = sorted(self.pool, reverse=True)

    sample_size =self.pool_status["working"]

    if sample_size == 0:
      proxy = random.choice(self.pool)
    else:
      if self.pool[0] != math.inf:
        if random.random() < self.prob_try_dead_proxy and (self.total_proxies != sample_size):
          proxy = random.choice(self.pool[sample_size:])
        else:
          proxy = random.choice(self.pool[:sample_size])
      else:
        proxy = self.pool[0]

    return proxy

  def set_status(self, addr: str, status: ProxyType) -> None:
    proxy = compose(first, filter(lambda proxy: proxy.addr == addr))(self.pool)
    self.pool_status[proxy.status.name] -= 1
    proxy.status = status
    self.pool_status[proxy.status.name] += 1
    self.pool = sorted(self.pool, reverse=True)

  def dump_proxy(self) -> None:
    with open("proxy-list-state.pkl", "wb") as f:
      pickle.dump(self.pool, f)
