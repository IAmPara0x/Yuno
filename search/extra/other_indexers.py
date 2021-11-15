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

#NOTE: This indexer takes too much time coz of brute force search.
@dataclass(init=True, frozen=True)
class ContextIdxr(IndexerBase):
  """
    ContextIdxr class inherits from IndexerBase.
    uses brute force search with some pruning to search
    combinations of different sentences that best
    linearly approximate query.

    ...

    Attributes:
    ----------
      cfg: ContextIdxrCfg
        contains all the default parameters that is need for __call__.

    Methods:
    ----------
      new(search_base: SearchBase, cfg: ContextIdxrCfg) -> ContextIdxr
        returns new instance of ContextIdxr class.

      __call__(search_result: SearchResult) -> SearchResult
        calls context_idxr with given search_result and default cfg or given cfg.

      context_idxr(search_result: SearchResult, cfg: ContextIdxrCfg) -> SearchResult
        scores the data according to the score returned by method context_score.

      context_score(q: Tensor, mat: Tensor, combinations: int,
          cutoff_sim: float, device: str) -> Tuple[float, List[int]]

        uses brute force search with pruning that searches for best combination of
        vectors that best linearly approximate the q vector.
  """

  cfg: ContextIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: ContextIdxrCfg) -> "ContextIdxr":
    """
      creates new instance of ContextIdxr class.

      Parameters
      ----------
        search_base: SearchBase
        cfg: ContextIdxrCfg

      Returns
      ----------
        ContextIdxr
    """

    return ContextIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    """
      calls method context_idxr with cfg argument if provided in search_result or else
      uses default cfg attribute. The datas returned by context_idxr is
      then sorted according to according to their scores.

      Parameters
      ----------
        search_result: SearchResult

      Returns
      ----------
        SearchResult
    """

    cfg = get_config(search_result.config, self.cfg, "nodeindexer_cfg")
    return self.context_idxr(search_result, cfg)

  def context_idxr(self, search_result: SearchResult,
                   cfg: ContextIdxrCfg) -> SearchResult:
    """
      scores the data in search_result according to the score returned
      by method context_score which is then linearly scaled between min=0.5,max=3.

      Parameters
      ----------
        search_result: SearchResult
        cfg: ContextIdxrCfg

      Returns
      ----------
        SearchResult
    """

    def contextscore_acc(acc_data, data):
      a_uid = data.anime_uid
      a_datas = datas_filter(lambda data: data.type == DataType.recs,
                             self.get_datas(a_uid))

      if len(a_datas):
        embds = from_vstack([data.embedding
                             for data in a_datas]).to(cfg.device)
        idxs = compose(sorted, map(snd), sorted,
                       filter(lambda x: x[0] >= cfg.sim_thres), zip)(
                           torch.cosine_similarity(q, embds),
                           range(embds.shape[0]),
                       )[-cfg.topk:]

        if len(idxs):
          new_score, max_idxs = self.context_score(q, embds[idxs], 2,
                                                   cfg.cutoff_sim, cfg.device)
          new_datas = [
              Data.new(a_datas[idxs[midx]],
                       anime_uid=a_uid,
                       type=DataType.short) for midx in max_idxs
          ]
        else:
          new_datas = [data]
          new_score = 0.5

      else:
        new_datas = [data]
        new_score = cfg.sim_thres
      acc_data.append((new_datas, new_score))
      return acc_data

    q = torch.from_numpy(search_result.query.embedding).unsqueeze(0).to(
        cfg.device)
    new_datas, new_scores = zip(
        *reduce(contextscore_acc, search_result.datas, []))
    new_scores = (compose(rescale_scores(0.5, 3), np.array)(new_scores) *
                  search_result.scores)
    new_scores = compose(list, concat)(
        [[score] * len(data) for data, score, in zip(new_datas, new_scores)])

    return SearchResult.new(search_result,
                            datas=compose(list, concat)(new_datas),
                            scores=np.array(new_scores).squeeze())

  def context_score(self, q: Tensor, mat: Tensor, combinations: int,
                    cutoff_sim: float, device: str) -> Tuple[float, List[int]]:
    """
      uses brute force search with pruning to find a combination
      of vectors that best linearly approximate the target vector q.

      Parameters
      ----------
        q: Tensor
          the target vector which is being approximated.
        mat: Tensor
          set of vectors that will be used to find approximation of the target vector q.
        combinations: int
          number of vectors to combine to approximate the target vector.
        cutoff_sim: int
          pruning some combinations of vector from set of vectors mat that are too similar to
          each other.
        device: str
          "cpu" or "cuda".

      Returns
      ----------
        Tuple[float,List[int]]
          float: context score between the best combination of the vectors and
                 target vector
          List[int]: indexs of vector that formed the best combination.
    """

    if combinations > mat.shape[0]:
      sims = torch.cosine_similarity(q, mat)
      return torch.mean(sims).item(), [int(torch.argmax(sims))]

    else:
      _adjmat = pair_sim(mat, mat)
      comb_idxs = compose(
          list,
          filter(lambda idx: _adjmat[tuple(idx)] <= cutoff_sim),
          torch.combinations,
      )(torch.arange(mat.shape[0]), combinations)

      if len(comb_idxs):
        q_mat = torch.vstack([q for _ in range(len(comb_idxs))
                               ]).unsqueeze(-1).to(device)
        mat = torch.vstack([mat[c_idx].T.unsqueeze(0) for c_idx in comb_idxs])

        mat_t = torch.movedim(mat.T, -1, 0).to(device)
        r = l2_approx(q_mat, mat, mat_t)
        max_idx = torch.argmax(torch.sum(r, 1))
        return (
            torch.cosine_similarity(
                torch.movedim(mat[max_idx] @ r[max_idx], 1, 0), q).item(),
            comb_idxs[max_idx],
        )
      else:
        r = l2_approx(q.squeeze(), mat.T, mat)
        return torch.cosine_similarity((mat.T @ r).unsqueeze(0),
                                       q).item(), [int(torch.argmax(r))]

"""
NOTE: working context indexer but not scores very well because of SVD
      and l2 approximation somehow doesn't work very well together.
"""

@dataclass(init=True, frozen=True)
class ContextIdxrV2(IndexerBase):
  cfg: ContextIdxrCfg

  @staticmethod
  def new(search_base: SearchBase, cfg: ContextIdxrCfg) -> "ContextIdxr":
    return ContextIdxr(search_base, cfg)

  @sort_search
  def __call__(self, search_result: SearchResult) -> SearchResult:
    cfg = get_config(search_result.config, self.cfg, "contextindexer_cfg")
    return self.context_idxr(search_result, cfg)

  def context_idxr(self, search_result: SearchResult, cfg: ContextIdxrCfg) -> SearchResult:


    def context_map(data: Data) -> Tuple[float,List[Data]]:
      a_uid = data.anime_uid
      a_datas = datas_filter(lambda data: data.type == DataType.recs,
                             self.get_datas(a_uid))

      if a_datas:
        embds = from_vstack([data.embedding for data in a_datas]).to(cfg.device)
        context_score,recs_idxs = self.context_search(q,embds,cfg.sim_threshold,cfg.stride)
        rec_datas = compose(list,map)(lambda x,*xs:
                                        a_datas[x].text.extend([data.text for data in itemgetter(*xs)(a_data)]),
                                      recs_idxs)
      else:
        context_score,rec_datas = cfg.sim_threshold,[data]

      return (context_score,
              [Data.new(rec_data,anime_uid=a_uid,type=DataType.final) for rec_data in rec_datas])


    q = torch.from_numpy(search_result.query.embedding).unsqueeze(0).to(cfg.device)
    context_scores,context_datas = zip(*map(context_map,search_result.datas))
    new_scores = compose(rescale_scores(0.25,3),np.array,list)(context_scores) * search_result.scores

    new_scores = compose(list, concat)(
        [[score] * len(context_data) for context_data, score, in zip(context_datas, new_scores)])

    return SearchResult.new(search_result,
                            datas=compose(list, concat)(context_datas),
                            scores=np.array(new_scores).squeeze())

  def context_search(self, q: Tensor, mat: Tensor,
                     sim_threshold: float, stride: int) -> Tuple[float,List[List[Int]]]:
    qsim_vec = torch.cosine_similarity(q,mat)
    sim_idxs = torch.where(qsim_vec >= sim_threshold)[0]
    qsim_vec = qsim_vec[sim_idxs]

    if len(sim_idxs):
      axis_sim_idxs = torch.argmax(
        torch.cosine_similarity(q.reshape(1,(1280//stride),stride),
                                mat.reshape(-1,(1280//stride),stride), dim=-1)
                      dim=0)
      axis_sim_idxs = torch.unique(axis_sim_idxs).cpu().tolist()
      axis_sim_mat = pair_sim(mat[axis_sim_idxs],mat[sim_idxs])

      context_scores = []
      sel_idxs = []

      for i,axis_idx in enumerate(axis_sim_idxs):
        diff_idxs = sim_idxs[torch.where(qsim_vec > axis_sim_mat[i])[0]]

        if len(diff_idxs):
          A = torch.vstack([mat[diff_idxs],mat[axis_idx]])
          x = q.squeeze()
          rank = torch.matrix_rank(A)

          if rank != A.shape[0]:
            u,s,v = torch.svd(A.T)
            A = (torch.diag(s[:rank]) @ v[:,:rank].T).T
            x = torch.diag(s[:rank]) @ (torch.inverse(torch.diag(s[:rank])) @ u[:,:rank].T @ x)

          contrib = l2_approx(x,A.T,A)[:,0]
          context_score.append(torch.cosine_similarity((A.T@contrib).unsqueeze(0),q))

          contrib = contrib.masked_fill(contrib < 0, 0)
          sel_idxs.append([axis_idx,*top_subset_sum(contrib[:-1],torch.sum(contrib[:-1])*0.2)])

        else:
          sel_idxs.append([axis_idx])
          context_scores.append(torch.cosine_similarity(q,mat[axis_idx].unsqueeze(0)).item())

      score = sum(context_scores) / len(context_scores)

    else:
      score = qsim_vec.max().item() * 0.25
      sel_idxs = [[qsim_vec.argmax().item()]]

    return score,sel_idxs
