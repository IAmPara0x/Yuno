#NOTE: This indexer doesn't score very well
@dataclass(init=True, frozen=True)
class TagIdxr(IndexerBase):
  indexing_method: TagIdxingMethod
  indexing_metric: TagIdxingMetric

  @staticmethod
  def new(search_base: SearchBase, config: TagIdxrCfg) -> "TagIdxr":
    return TagIdxr(search_base, config.indexing_method, config.indexing_metric)

  @process_result(rescale_scores(t_min=1, t_max=8, inverse=False))
  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    query_mat = self.tags_mat(search_result.query)

    if self.indexing_method == TagIdxingMethod.per_category:
      similarity_scores = compose(list,
                                  map)(self.per_category_indexing(query_mat),
                                       self.get_animes(search_result))
    elif self.indexing_method == TagIdxingMethod.all:
      query_mat = query_mat.reshape(-1)
      similarity_scores = compose(list,
                                  map)(self.all_category_indexing(query_mat),
                                       self.get_animes(search_result))
    else:
      raise Exception(f"{self.indexing_method} is not a corret type.")

    similarity_scores = rescale_scores(t_min=1, t_max=3, inverse=False)(
        np.array(similarity_scores, dtype=np.float32))
    similarity_scores *= search_result.scores
    return SearchResult.new(search_result, scores=similarity_scores)

  def tags_mat(self, x: Union[Anime, Query]) -> np.ndarray:
    tag_cats = self.get_tagcats(AllData())
    rows, cols = len(tag_cats), compose(max,
                                        map)(lambda cat: len(cat.tag_uids),
                                             tag_cats)
    tags_mat = np.zeros((rows, cols))

    def tag_pos(tag: Tag) -> Tuple[int, int]:
      i = [idx for idx, cat in enumerate(tag_cats)
           if cat.uid == tag.cat_uid][0]
      j = [
          idx for idx, tag_uid in enumerate(tag_cats[i].tag_uids)
          if tag_uid == tag.uid
      ][0]
      return (i, j)

    if isinstance(x, Anime):
      anime_tags = self.get_tags(x)
      i_s, j_s = zip(*map(tag_pos, anime_tags))
      tags_mat[(i_s, j_s)] = x.tag_scores
    elif isinstance(x, Query):
      all_tags = self.get_tags(AllData())
      i_s, j_s = zip(*map(tag_pos, all_tags))
      scores = [cos_sim(x.embedding, tag.embedding).item() for tag in all_tags]
      tags_mat[(i_s, j_s)] = scores
    else:
      raise Exception(
          f"Only supported types are Anime and Query but {type(x)} is None of them"
      )
    return tags_mat

  @curry
  def per_category_indexing(self, query_mat: np.ndarray,
                            anime_info: Anime) -> int:
    anime_mat = self.tags_mat(anime_info)
    x = compose(np.diag, np.dot)(anime_mat, query_mat.T)
    y = compose(np.diag, np.dot)(anime_mat, anime_mat.T)
    return np.dot(x, y).item()

  @curry
  def all_category_indexing(self, query_mat: np.ndarray,
                            anime_info: Anime) -> int:
    anime_mat = self.tags_mat(anime_info)
    anime_mat = anime_mat.reshape(-1)
    return cos_sim(anime_mat, query_mat).item()
