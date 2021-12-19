# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals

# useful for handling different item types with a single interface
import logging
from itemadapter import is_item, ItemAdapter
from cytoolz.curried import compose, filter, map
from twisted.internet import task

from .proxy_heuristic import ProxyPool, ProxyType

logger = logging.getLogger(__name__)

class CrawlerSpiderMiddleware:
  # Not all methods need to be defined. If a method is not defined,
  # scrapy acts as if the spider middleware does not modify the
  # passed objects.

  @classmethod
  def from_crawler(cls, crawler):
    # This method is used by Scrapy to create your spiders.
    s = cls()
    crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
    return s

  def process_spider_input(self, response, spider):
    # Called for each response that goes through the spider
    # middleware and into the spider.

    # Should return None or raise an exception.
    return None

  def process_spider_output(self, response, result, spider):
    # Called with the results returned from the Spider, after
    # it has processed the response.

    # Must return an iterable of Request, or item objects.
    for i in result:
      yield i

  def process_spider_exception(self, response, exception, spider):
    # Called when a spider or process_spider_input() method
    # (from other spider middleware) raises an exception.

    # Should return either None or an iterable of Request or item objects.
    pass

  def process_start_requests(self, start_requests, spider):
    # Called with the start requests of the spider, and works
    # similarly to the process_spider_output() method, except
    # that it doesn’t have a response associated.

    # Must return only requests (not items).
    for r in start_requests:
      yield r

  def spider_opened(self, spider):
    spider.logger.info('Spider opened: %s' % spider.name)


class RotatingProxies:
  # Not all methods need to be defined. If a method is not defined,
  # scrapy acts as if the downloader middleware does not modify the
  # passed objects.

  def __init__(self, proxy_file_path=""):
    self.proxies = ProxyPool("proxy-list.txt")
    self.max_proxies_try = len(self.proxies.pool)//4
    self.logstats_interval = 30
    self.total_proxies = len(self.proxies.pool)

  @classmethod
  def from_crawler(cls, crawler):
    # This method is used by Scrapy to create your spiders.
    s = cls()
    crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)

    settings = crawler.settings

    rotating_proxies = cls(proxy_file_path=settings.get("proxy_file_path"))
    crawler.signals.connect(rotating_proxies.engine_started,
                            signal=signals.engine_started
                            )
    return rotating_proxies

  def engine_started(self):
    self.log_task = task.LoopingCall(self.log_stats)
    self.log_task.start(self.logstats_interval, now=True)

  def process_request(self, request, spider):
    # Called for each request that goes through the downloader
    # middleware.

    # Must either:
    # - return None: continue processing this request
    # - or return a Response object
    # - or return a Request object
    # - or raise IgnoreRequest: process_exception() methods of
    #   installed downloader middleware will be called
    request.meta["download_timeout"] = 5
    if "dont_use_proxy" not in request.meta:
      proxy = self.proxies.get_proxy()
      logger.debug(f"{proxy}\n{'='*25}")
      request.meta["proxy"] = proxy.url

    return None

  def process_response(self, request, response, spider):

    print(f"{'='*25}\nRESPONSE: {response.status}, URL: {response.url}\n{'='*25}")

    if response.status == 200:
      self.proxies.change_status(ProxyType.working)
      return response
    elif response.status == 403:
      logger.debug(f"RESPONSE: {response.status}, URL: {response.url}")
      self.proxies.change_status(ProxyType.cooldown)
      return self._retry(request, spider)
    else:
      logger.debug(f"RESPONSE: {response.status}, URL: {response.url}")
      self.proxies.change_status(ProxyType.dead)
      return self._retry(request, spider)

  def _retry(self, request, spider):
    print(f"{'='*25}\nRETRYING {request.url}\n{'='*25}")

    if "dont_use_proxy" not in request.meta:

      retries = request.meta.get('proxy_retry_times', 0) + 1
      max_proxies_try = request.meta.get('max_proxies_try',
                                            self.max_proxies_try)
      retryreq = request.copy()
      retryreq.meta['proxy_retry_times'] = retries
      retryreq.dont_filter = True

      if retries <= max_proxies_try:
        return retryreq
      else:
        del retryreq.meta["proxy"]
        retryreq.meta["dont_use_proxy"] = True
        print(f"{'='*25}\nUSING DEFAULT IP\n{'='*25}")
        return retryreq

  def process_exception(self, request, exception, spider):
    logger.warn(f"EXCEPTION: {exception}\n{'='*25}")

    if "dont_use_proxy" not in request.meta:
      self.proxies.change_status(ProxyType.dead)
      return self._retry(request, spider)

  def spider_opened(self, spider):
    spider.logger.info('Spider opened: %s' % spider.name)

  def log_stats(self):
    working = compose(len, list, filter)(lambda x: x.status == ProxyType.working, self.proxies.pool)
    cooldown = compose(len, list, filter)(lambda x: x.status == ProxyType.cooldown, self.proxies.pool)
    dead = self.total_proxies - (working + cooldown)

    logger.warn(f"TOTAL PROXIES: {self.total_proxies},\
        WORKING: {working}, COOLDOWN: {cooldown}, DEAD: {dead}")
