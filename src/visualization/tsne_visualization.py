
# imports
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets # for jupyter notebooks
from IPython.display import display # for jupyter notebooks
from mpl_toolkits.mplot3d import Axes3D # for jupyter notebooks

plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = [18,12]

def update_annot(ind):
  with out:
    pos = data_points[ind][0]
    anime_uid = labels[ind][0]
    print(f"uid: {anime_uid}, Text: {data_text[ind]}")
    annot.xy = pos
    text = f"{anime_uid}"
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(sc.get_facecolor()[0])
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
  vis = annot.get_visible()
  cont,ind = sc.contains(event)
  if cont:
    update_annot(ind["ind"])
    annot.set_visible(True)
    fig.canvas.draw_idle()
  else:
    if vis:
      annot.set_visible(False)
      fig.canvas.draw_idle()

def plot_anime(uid, key, ax):
  anime_labels_idx = np.where(labels == uid)[0]
  points = tsne_data[key][anime_labels_idx]
  ax.scatter(points[:,0], points[:,1], s=5, cmap=cmap, s=5, norm=norm)
  return ax

plot_points = None
plot_labels = None
plot_text = None

for uid in plot_uid:
  if plot_points is None:
    plot_points,plot_labels,plot_text = plot_anime(uid, "tsne_result_75")
  else:
    d_points d_labels, d_text = plot_anime(uid, "tsne_result_75")
    plot_points,plot_labels,plot_text = (np.concatenate((plot_points,d_points)),
                                         np.concatenate((plot_labels,d_labels)),
                                         np.concatenate((plot_text,d_text)))

