from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'float32,gpu,seed=1234'

import numpy as np
import tensorflow as tf

from odin.utils import ctext
from odin import backend as K, nnet as N, fuel as F
from odin.visual import plot_scatter, plot_save, plot_figure
from odin.ml import PLDA

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import BernoulliRBM
# ===========================================================================
# Const
# ===========================================================================
TRAINING_PERCENT = 0.5
POINT_SIZE = 8
NUM_DIM = 3
colors = ['r', 'b', 'g']
markers = ["o", "^", "s"]
SEED = K.get_rng().randint(0, 10e8)
# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.IRIS.load()
print(ds)

nb_samples = ds['X'].shape[0]
ids = K.get_rng().permutation(nb_samples)
X = ds['X'][ids]
y = ds['y'][ids]
labels = ds['name']
print("Labels:", ctext(labels))
assert len(colors) == len(labels) and len(markers) == len(labels)

X_train = X[:int(TRAINING_PERCENT * nb_samples)]
y_train = y[:int(TRAINING_PERCENT * nb_samples)]
y_train_color = [colors[i] for i in y_train]
y_train_marker = [markers[i] for i in y_train]

X_score = X[int(TRAINING_PERCENT * nb_samples):]
y_score = y[int(TRAINING_PERCENT * nb_samples):]
y_score_color = [colors[i] for i in y_score]
y_score_marker = [markers[i] for i in y_score]

print("Train:", X_train.shape, y_train.shape)
print("Score:", X_score.shape, y_score.shape)

legends = {(c, m): labels[i]
           for i, (c, m) in enumerate(zip(colors, markers))}
# ===========================================================================
# Classic
# ===========================================================================
# ====== PCA ====== #
pca = PCA(n_components=NUM_DIM, whiten=False, random_state=SEED)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_score_pca = pca.transform(X_score)
# ====== tsne + pca ====== #
tsne_pca = TSNE(n_components=NUM_DIM, perplexity=30.0, learning_rate=200.0, n_iter=1000,
                random_state=SEED)
X_train_tsne_pca = tsne_pca.fit_transform(X_train_pca)
X_score_tsne_pca = tsne_pca.fit_transform(X_score_pca)
# ====== tsne ====== #
tsne = TSNE(n_components=NUM_DIM, perplexity=30.0, learning_rate=200.0, n_iter=1000,
            random_state=SEED)
X_train_tsne = tsne.fit_transform(X_train)
X_score_tsne = tsne.fit_transform(X_score)
# ====== lda ====== #
lda = LinearDiscriminantAnalysis(n_components=NUM_DIM)
lda.fit(X_train, y_train)
X_train_lda = lda.transform(X_train)
X_score_lda = lda.transform(X_score)
# ====== plda ====== #
plda = PLDA(n_phi=NUM_DIM, random_state=SEED)
plda.fit(X_train, y_train)
X_train_plda = plda.predict_log_proba(X_train)
X_score_plda = plda.predict_log_proba(X_score)
# ====== gmm ====== #
gmm = GaussianMixture(n_components=NUM_DIM, max_iter=100, covariance_type='full',
                      random_state=SEED)
gmm.fit(X_train)
X_train_gmm = gmm._estimate_weighted_log_prob(X_train)
X_score_gmm = gmm._estimate_weighted_log_prob(X_score)
# ====== rbm ====== #
rbm = BernoulliRBM(n_components=NUM_DIM, batch_size=8, learning_rate=0.0008,
                   n_iter=8, verbose=2, random_state=SEED)
rbm.fit(X_train)
X_train_rbm = rbm.transform(X_train)
X_score_rbm = rbm.transform(X_score)
# ===========================================================================
# Deep Learning
# ===========================================================================

# ===========================================================================
# Visualize
# ===========================================================================
def plot(train, score, title, applying_pca=False):
  if applying_pca:
    pca = PCA(n_components=NUM_DIM)
    pca.fit(train)
    train = pca.transform(train)
    score = pca.transform(score)
  plot_figure(nrow=6, ncol=12)
  plot_scatter(x=train[:, 0], y=train[:, 1],
               z=None if NUM_DIM < 3 or train.shape[1] < 3 else train[:, 2],
               size=POINT_SIZE, color=y_train_color, marker=y_train_marker,
               fontsize=12, legend=legends,
               title='[train]' + str(title),
               ax=(1, 2, 1))
  plot_scatter(x=score[:, 0], y=score[:, 1],
               z=None if NUM_DIM < 3 or score.shape[1] < 3 else score[:, 2],
               size=POINT_SIZE, color=y_score_color, marker=y_score_marker,
               fontsize=12, legend=legends,
               title='[score]' + str(title),
               ax=(1, 2, 2))

plot(train=X_train_pca, score=X_score_pca, title='PCA')
plot(train=X_train_tsne, score=X_score_tsne, title='T-SNE')
plot(train=X_train_tsne_pca, score=X_score_tsne_pca, title='T-SNE + PCA')
plot(train=X_train_lda, score=X_score_lda, title='LDA')
plot(train=X_train_plda, score=X_score_plda, title='PLDA')
plot(train=X_train_plda, score=X_score_plda, title='PLDA + PCA', applying_pca=True)
plot(train=X_train_gmm, score=X_score_gmm, title='GMM')
plot(train=X_train_rbm, score=X_score_rbm, title='RBM')
plot_save('/tmp/tmp.pdf')
