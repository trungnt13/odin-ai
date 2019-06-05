from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np

from odin import fuel as F, visual as V
from odin import ml

from sklearn.decomposition import PCA

random_state = 1234
np.random.seed(random_state)
path = '/tmp/tmp.pdf'
# ===========================================================================
# Load dataset
# ===========================================================================
iris = F.IRIS.load()
X_iris = iris['X'][:]
y_iris = iris['y'][:]

mnist = F.MNIST.load()
X_mnist = mnist['X_test'][:].reshape(-1, 28 * 28)
y_mnist = mnist['y_test'][:]

fmnist = F.FMNIST_original.load()
X_fmnist = fmnist['X_test'][:].reshape(-1, 28 * 28)
y_fmnist = fmnist['y_test'][:]

# ====== only take 1000 samples, easier and faster ====== #
num_samples = X_mnist.shape[0]
assert num_samples == X_fmnist.shape[0]
ids = np.random.permutation(num_samples)

X_mnist = X_mnist[ids][:1000]
y_mnist = y_mnist[ids][:1000]

X_fmnist = X_fmnist[ids][:1000]
y_fmnist = y_fmnist[ids][:1000]

# ===========================================================================
# Main comparison
# ===========================================================================
def compare_methods(X, y, dim, title, n_iter='auto', verbose=0, plda=False):
  print(title, ':', dim)
  #
  pca = PCA(n_components=dim, random_state=random_state)
  pca.fit(X)
  X_pca = pca.transform(X)
  #
  if plda:
    plda = ml.PLDA(n_phi=dim, verbose=verbose)
    plda.fit(X=X_iris, y=y_iris)
    X_plda = plda.transform(X_iris)
    n_col = 5
  else:
    plda = None
    X_plda = None
    n_col = 4
  #
  ppca = ml.PPCA(n_components=dim, verbose=verbose,
                 n_iter=n_iter, random_state=random_state)
  ppca.fit(X)
  X_ppca = ppca.transform(X)
  #
  sppca1 = ml.SupervisedPPCA(n_components=dim, verbose=verbose, extractor='supervised',
                             n_iter=n_iter, random_state=random_state)
  sppca1.fit(X, y)
  X_sppca1 = sppca1.transform(X)
  #
  sppca2 = ml.SupervisedPPCA(n_components=dim, verbose=verbose, extractor='unsupervised',
                             n_iter=n_iter, random_state=random_state)
  sppca2.fit(X, y)
  X_sppca2 = sppca2.transform(X)
  # T-SNE if necessary
  if dim > 2:
    X_pca = ml.fast_tsne(X_pca, n_components=2)
    X_ppca = ml.fast_tsne(X_ppca, n_components=2)
    X_sppca1 = ml.fast_tsne(X_sppca1, n_components=2)
    X_sppca2 = ml.fast_tsne(X_sppca2, n_components=2)
    if X_plda is not None:
      X_plda = ml.fast_tsne(X_plda, n_components=2)
  # Plotting
  V.plot_figure(nrow=4, ncol=18)
  plt.subplot(1, n_col, 1)
  plt.scatter(x=X_pca[:, 0], y=X_pca[:, 1], c=y, marker='o',
              alpha=0.5, s=1)
  plt.xticks([], []); plt.yticks([], [])
  plt.title("PCA")

  plt.subplot(1, n_col, 2)
  plt.scatter(x=X_ppca[:, 0], y=X_ppca[:, 1], c=y, marker='o',
              alpha=0.5, s=1)
  plt.xticks([], []); plt.yticks([], [])
  plt.title("PPCA")

  plt.subplot(1, n_col, 3)
  plt.scatter(x=X_sppca1[:, 0], y=X_sppca1[:, 1], c=y, marker='o',
              alpha=0.5, s=1)
  plt.xticks([], []); plt.yticks([], [])
  plt.title("S-PPCA (supervised extractor)")

  plt.subplot(1, n_col, 4)
  plt.scatter(x=X_sppca2[:, 0], y=X_sppca2[:, 1], c=y, marker='o',
              alpha=0.5, s=1)
  plt.xticks([], []); plt.yticks([], [])
  plt.title("S-PPCA (unsupervised extractor")

  if plda is not None:
    plt.subplot(1, n_col, 5)
    plt.scatter(x=X_plda[:, 0], y=X_plda[:, 1], c=y, marker='o',
                alpha=0.5, s=1)
    plt.xticks([], []); plt.yticks([], [])
    plt.title("PLDA")

  plt.suptitle('[%d]%s' % (dim, title))

compare_methods(X=X_iris, y=y_iris, dim=2, title='IRIS', plda=True)
compare_methods(X=X_mnist, y=y_mnist, dim=2, title='MNIST')
compare_methods(X=X_fmnist, y=y_fmnist, dim=2, title='Fashion-MNIST')

compare_methods(X=X_iris, y=y_iris, dim=3, title='IRIS', plda=True)
compare_methods(X=X_mnist, y=y_mnist, dim=3, title='MNIST')
compare_methods(X=X_fmnist, y=y_fmnist, dim=3, title='Fashion-MNIST')

compare_methods(X=X_mnist, y=y_mnist, dim=256, title='MNIST', verbose=1)
compare_methods(X=X_fmnist, y=y_fmnist, dim=256, title='Fashion-MNIST', verbose=1)

V.plot_save(path)
