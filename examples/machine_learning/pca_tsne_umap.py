from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from odin import ml
from odin import visual as vs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_umap = ml.fast_umap(X_train, X_test)
X_tsne = ml.fast_tsne(X_train, X_test)
X_pca = ml.fast_pca(X_train, X_test, n_components=2)

styles = dict(size=12, alpha=0.6, centroids=True)

vs.plot_figure(6, 12)
vs.plot_scatter(x=X_pca[0], color=y_train, ax=(1, 2, 1), **styles)
vs.plot_scatter(x=X_pca[1], color=y_test, ax=(1, 2, 2), **styles)

vs.plot_figure(6, 12)
vs.plot_scatter(x=X_tsne[0], color=y_train, ax=(1, 2, 1), **styles)
vs.plot_scatter(x=X_tsne[1], color=y_test, ax=(1, 2, 2), **styles)

vs.plot_figure(6, 12)
vs.plot_scatter(x=X_umap[0], color=y_train, ax=(1, 2, 1), **styles)
vs.plot_scatter(x=X_umap[1], color=y_test, ax=(1, 2, 2), **styles)

vs.plot_save()
