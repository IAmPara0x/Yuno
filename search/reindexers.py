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

  @normalize
  @sort_search
  def knn_search(self, text:str) -> SearchResult:
    q_embedding = self.MODEL(text)
    q_embedding = np.expand_dims(q_embedding,0)
    distances,n_id = self.INDEX.search(q_embedding,self.top_k)
    distances = 1/distances.squeeze()
    n_id = n_id.squeeze()

    n_embeddings = self.EMBEDDINGS[n_id]

    n_anime_uids = self.LABELS[n_id]
    n_anime = map(lambda uid: self.ALL_ANIME[uid], n_anime_uids)

    query = Query(text,q_embedding)

    return SearchResult(query,n_embeddings,n_id,distances,list(n_anime))

  @classmethod
  def name(cls) -> str:
    return cls.__name__.lower()


class TagReIndexer(ReIndexerBase):

  @normalize
  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    query_mat = self.tags_mat(search_result.query)

    if self.tag_indexing_method == TagIndexing.per_category:
      similarity_scores = list(map(lambda anime_info: self.per_category_indexing(anime_info, query_mat),
                                search_result.anime_infos))
    elif self.tag_indexing_method == TagIndexing.all:
      query_mat = query_mat.reshape(-1)
      similarity_scores = list(map(lambda anime_info: self.all_category_indexing(anime_info, query_mat),
                                search_result.anime_infos))
    else:
      raise Exception(f"{self.tag_indexing_method} is not a corret type.")

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
    max_tags_uids = max(map(lambda val: len(val.tags_uid), self.ALL_TAGS_CATEGORY.values()))
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

  def per_category_indexing(self, anime_info: Anime, query_mat: np.ndarray) -> int:
    anime_mat = self.tags_mat(anime_info)
    x = np.diag(np.dot(anime_mat,query_mat.T))
    y = np.diag(np.dot(anime_mat,anime_mat.T))
    return np.dot(x,y)

  def all_category_indexing(self, anime_info: Anime, query_mat: np.ndarray) -> int:
    anime_mat = self.tags_mat(anime_info)
    anime_mat = anime_mat.reshape(-1)
    return self.cosine_similarity(anime_mat,query_mat)


class AccReIndexer(ReIndexerBase):
  def __call__(self, search_result: SearchResult) -> SearchResult:
    return self.acc_score(search_result)

  @normalize
  @sort_search
  def acc_score(self,search_result: SearchResult) -> SearchResult:
    #NOTE: implement this function in a better way

    result_embeddings = []
    result_indexs = []
    scores = []
    anime_infos = []

    def helper(idx,anime_info: Anime):
      if anime_info not in anime_infos:
        anime_infos.append(anime_info)
        embedding,index,score,_ = search_result.get_result(idx)
        result_embeddings.append(embedding)
        result_indexs.append(index)
        idxs = np.where(search_result.anime_infos==anime_info)[0]
        scores.append(sum(search_result.scores[idxs]))

    for idx, anime_info in enumerate(search_result.anime_infos):
      helper(idx,anime_info)
    return SearchResult.new_search_result(search_result,scores=scores,result_embeddings=result_embeddings,result_indexs=result_indexs,anime_infos=anime_infos)

