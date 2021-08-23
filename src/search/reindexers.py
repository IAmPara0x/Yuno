from .config import Config
from .base_classes import SearchResult,SearchBase,ReIndexerBase,ReIndexingPipeline

class Search:
  def __init__(self,search_base:SearchBase,config:Config):

    for name,val in zip(search_base._fields,search_base.__iter__()):
      setattr(self,name,val)

    config_name = f"{Search.__name__.lower()}_config" #FIXME: find a way to get the class name dynamically
    search_config = getattr(config,config_name)

    for name,val in zip(search_config._fields,search_config.__iter__()):
      setattr(self,name,val)

  def knn_search(self, text:str) -> SearchResult:
    q_embedding = self.model(text)
    distances,n_id = self.INDEX.search(q_embedding,self.TOP_K)
    distances = distances.squeeze()
    n_id = n_id.squeeze()

    n_embeddings = self.EMBEDDINGS[n_id]

    n_anime_uids = self.LABELS[n_id]
    n_anime = map(lambda uid: self.ALL_ANIME[uid], n_anime_uids)

    return SearchResult(q_embedding,n_embeddings,n_id,distances,list(n_anime))
