import logging
from scrapy import signals
from twisted.internet import task

from .proxy_heuristic import ProxyPool, ProxyType


logger = logging.getLogger(__name__)


class RotatingProxies(object):

  def __init__(self, proxies: ProxyPool,
               max_proxies_try: int,
               logstats_interval: float
               ):

    self.proxies: ProxyPool = proxies
    self.max_proxies_try:int = max_proxies_try
    self.logstats_interval: float = logstats_interval

  @classmethod
  def from_crawler(cls, crawler) -> "RotatingProxies":
    s = crawler.settings
    proxies = ProxyPool(s.get("use_cached_proxy"),
                        s.get("proxy_file_path"),
                        s.getfloat("PROB_TRY_DEAD_PROXY", 0.15)
                        )

    rot_proxies = cls(proxies,
                      s.getfloat("ROTATING_PROXY_STATS_INTERVAL", 30),
                      s.getint("MAX_PROXIES_TRY", 6)
                      )

    crawler.signals.connect(rot_proxies.engine_started,
                            signal=signals.engine_started
                            )

    crawler.signals.connect(proxies.dump_proxy,
                            signal=signals.spider_closed
                            )

    return rot_proxies

  def engine_started(self):
    self.log_task = task.LoopingCall(self.log_stats)
    self.log_task.start(self.logstats_interval, now=True)

  def process_request(self, request, spider):

    proxy = self.proxies.get_proxy()
    request.meta["download_timeout"] = 5
    if "dont_use_proxy" not in request.meta:
      request.meta["proxy"] = proxy.addr
    else:
      print(f"{'='*25}\nusing default IP\n{'='*25}")

  def process_response(self, request, response, spider):

    if "dont_use_proxy" not in request.meta:

      print(f"{'='*25}\nproxy : {request.meta['proxy']} \
                        status : {response.status} \
                        url : {response.url} \
            \n{'='*25}")

      addr: str = request.meta["proxy"]

      if response.status == 200:
        self.proxies.set_status(addr, ProxyType.working)
        return response

      elif response.status == 403:
        self.proxies.set_status(addr, ProxyType.cooldown)
        return self._retry(request, spider)

      else:
        self.proxies.set_status(addr, ProxyType.dead)
        return self._retry(request, spider)

    else:
      return response

  def process_exception(self, request, exception, spider):

    print(f"{'='*25}\nEXCEPTION : {exception} url : {request.url} \n{'='*25}")

    if "dont_use_proxy" not in request.meta:
      addr = request.meta["proxy"]
      self.proxies.set_status(addr,ProxyType.dead)
      return self._retry(request, spider)

  def _retry(self, request, _):
    print(f"{'='*25}\nRETRYING {request.url}\n{'='*25}")

    retries = request.meta.get('proxy_retry_times', 1)
    retryreq = request.copy()
    retryreq.dont_filter = True

    if retries <= self.max_proxies_try:
      retryreq.meta['proxy_retry_times'] = retries + 1
      return retryreq

    else:
      del retryreq.meta["proxy"]
      retryreq.meta["dont_use_proxy"] = True
      return retryreq

  def log_stats(self):

    print(f"{'='*25}\n PROXY STATS: TOTAL({self.proxies.total_proxies}), \
            WORKING({self.proxies.pool_status['working']}), \
            COOLDOWN({self.proxies.pool_status['cooldown']}), \
            DEAD({self.proxies.pool_status['dead']}) \
           \n{'='*25}")

