#Imports
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import multiprocessing


plt.rcParams["figure.figsize"] = [18,12]

#Params
DIMS: int = 2
PERPLEXITIES: List[int] = [5,30,50,75]
STEPS:int = 5_000
TSNE_RESULTS = {}
CPUS: int = multiprocessing.cpu_count()
DATA_SIZE: int = 164_000
USE_PCA: bool = True
PCA_OUT_DIMS: int = 64

class ReduceDim:
  def __init__(self, out_dim, perplexity, steps, jobs, **kwargs):
    self.out_dim = out_dim
    self.perplexity = perplexity
    self.steps = steps
    self.jobs = jobs
    self.tsne = TSNE(n_components=self.out_dim, verbose=1,
                     perplexity=self.perplexity,
                     n_iter=self.steps, n_jobs=self.jobs)

    if "use_pca" in kwargs.keys():
      if kwargs["use_pca"]:
        self.pca_out_dims = kwargs["pca_out_dims"]
        self.pca = PCA(n_components=self.pca_out_dims)
        self.use_pca = True
      else:
        self.use_pca = False
    else:
        self.use_pca = False

  def fit_transform(self, data):
    if self.use_pca:
      print("calculating principal components")
      pca_results = self.pca.fit_transform(data)
      print(f"Done. The total variance captured with {self.pca_out_dims} dims\
              is {np.sum(self.pca.explained_variance_ratio_)}")

    print(f"Reducing dim using T-sne with steps: {self.steps} perplexity: {self.perplexity}. Training Now.")
    tsne_result = self.tsne.fit_transform(pca_results)
    return tsne_result

def main():
  with open("/kaggle/input/anime-search-visualization/data.pkl", "rb") as f:
    DATA = pickle.load(f)

  with open("data.pkl", "wb") as f:
    pickle.dump(DATA,f)

  embeddings,labels,tokenized_sents = shuffle(DATA["embeddings"],DATA["labels"],DATA["tokenized_sents"])
  embeddings = embeddings[:DATA_SIZE]
  labels = np.array(labels[:DATA_SIZE])
  tokenized_sents = tokenized_sents[:DATA_SIZE]

  with open("tsne_data.pkl", "wb") as f:
    tsne_data = {}
    tsne_data["embeddings"] = embeddings
    tsne_data["labels"] = labels
    tsne_data["tokenized_sents"] = tokenized_sents
    pickle.dump(tsne_data,f)

  for perplexity in PERPLEXITIES:
    model = reducedim(out_dim=dims,perplexity=perplexity,
              steps=steps,jobs=cpus,use_pca=use_pca,
              pca_out_dims=pca_out_dims)
    tsne_results[f"tsne_result_{perplexity}"] = model.fit_transform(embeddings)

  with open("tsne_results.pkl","wb") as f:
    pickle.dump(TSNE_RESULTS,f)
