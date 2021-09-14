import operator
from toolz.curried import compose, reduce, curry
from typing import Union, Tuple
from .config import *
from .base_classes import *


class Search:
  def __init__(self,search_base:SearchBase, config:Config):

    for name,val in zip(search_base._fields,search_base.__iter__()):
      setattr(self,name,val)

    config_name = f"{self.name()}_config"
    search_config = getattr(config,config_name)

    for name,val in zip(search_config._fields,search_config.__iter__()):
      setattr(self,name,val)

  def __call__(self, text: str) -> SearchResult:
    return self.knn_search(text)

  @normalize(t_min=1,t_max=2)
  @sort_search
  def knn_search(self, text:str) -> SearchResult:
    q_embedding = self.MODEL(text)
    q_embedding = np.expand_dims(q_embedding,0)
    distances,n_id = self.INDEX.search(q_embedding,self.top_k)
    distances = 1/distances.squeeze()
    n_id = n_id.squeeze()

    n_embeddings, n_anime_uids = self.EMBEDDINGS[n_id], self.LABELS[n_id]
    n_anime = compose(list,map)(lambda uid: self.ALL_ANIME[uid], n_anime_uids)
    query = Query(text,q_embedding.squeeze())

    return SearchResult(query,n_embeddings,n_id,distances,list(n_anime))

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()


class TagReIndexer(ReIndexerBase):

  @normalize(t_min=1,t_max=4,inverse=False)
  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    query_mat = self.tags_mat(search_result.query)

    if self.tag_indexing_method == TagIndexing.per_category:
      similarity_scores = compose(list,map)(self.per_category_indexing(query_mat),search_result.anime_infos)
    elif self.tag_indexing_method == TagIndexing.all:
      query_mat = query_mat.reshape(-1)
      similarity_scores = compose(list,map)(self.all_category_indexing(query_mat),search_result.anime_infos)
    else:
      raise Exception(f"{self.tag_indexing_method} is not a corret type.")

    similarity_scores *= anime_infos.scores
    return SearchResult.new_search_result(search_result,scores=similarity_scores)

  @staticmethod
  def cosine_similarity(v1: np.ndarray,v2: np.ndarray) -> int:
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

  def _get_tag_pos(self,tag_uid:int) -> Tuple[int,int]:
    i = self.ALL_TAGS[tag_uid].category_uid
    j = self.ALL_TAGS_CATEGORY[i].tag_pos(tag_uid)
    return (i,j)

  def tags_mat(self, x:Union[Anime,Query]) -> np.ndarray:
    len_tags_category = len(self.ALL_TAGS_CATEGORY.keys())
    max_tags_uids = compose(max,map)(lambda val: len(val.tags_uid), self.ALL_TAGS_CATEGORY.values())
    tags_mat = np.zeros((len_tags_category, max_tags_uids))

    def assign_score(uid,score):
      pos = self._get_tag_pos(uid)
      tags_mat[pos] = score

    if isinstance(x,Anime):
      for (uid,score) in zip(x.tags_uid,x.tags_score):
        assign_score(uid,score)
    elif isinstance(x,Query):
      for uid in self.ALL_TAGS.keys():
        score = self.cosine_similarity(self.ALL_TAGS[uid].embedding, x.embedding)
        assign_score(uid,score)
    else:
      raise Exception(f"Only supported types are Anime and Query but {type(x)} is None of them")
    return tags_mat

  @curry
  def per_category_indexing(self, query_mat: np.ndarray, anime_info: Anime) -> int:
    anime_mat = self.tags_mat(anime_info)
    x = compose(np.diag,np.dot)(anime_mat,query_mat.T)
    y = compose(np.diag,np.dot)(anime_mat,anime_mat.T)
    return np.dot(x,y)

  @curry
  def all_category_indexing(self, query_mat: np.ndarray, anime_info: Anime) -> int:
    anime_mat = self.tags_mat(anime_info)
    anime_mat = anime_mat.reshape(-1)
    return self.cosine_similarity(anime_mat,query_mat)


class AccReIndexer(ReIndexerBase):

  @normalize(sigmoid=True)
  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    if self.acc_indexing_metric == AccIndexingMetric.add:
      return self.acc_result(search_result,operator.add,initial_val=0)
    elif self.acc_indexing_metric == AccIndexingMetric.mul:
      return self.acc_result(search_result,operator.mul,initial_val=1)
    else:
      raise Exception("not correct type.")

  def acc_result(self, search_result: SearchResult,
      metric: Callable[[float,float],float], initial_val: float) -> SearchResult:

    where = lambda anime: [idx for idx, x in enumerate(search_result.anime_infos) if x == anime] #NOTE: can improve this?

    def acc(anime):
      idxs = where(anime)
      result_index = search_result.result_indexs[idxs[0]]
      result_embedding = search_result.result_embeddings[idxs[0]]
      score = reduce(metric,search_result.scores[idxs],initial_val)
      return result_index,result_embedding,score

    result = map(lambda anime: (anime,*acc(anime)), set(search_result.anime_infos))

    anime_infos,result_indexs,result_embeddings,scores = zip(*result)

    return SearchResult.new_search_result(search_result,anime_infos=list(anime_infos),result_indexs=np.array(result_indexs),
                                          result_embeddings=np.array(result_embeddings),scores=np.array(scores))

