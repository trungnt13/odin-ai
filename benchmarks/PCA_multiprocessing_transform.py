# ===========================================================================
# Single process:
# 0.0003s
# Multiprocessing:
# ncpu = 1: ~0.16s
# ncpu = 2: ~0.07s
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from odin import fuel as F, visual
from odin.ml import MiniBatchPCA
from sklearn.manifold import TSNE
from odin.utils import UnitTimer, TemporaryDirectory

iris = F.load_iris()
print(iris)
pca = MiniBatchPCA()

X = iris['X'][:]

i = 0
while i < X.shape[0]:
    x = X[i:i + 20]
    i += 20
    pca.partial_fit(x)
    print("Fitting PCA ...")

with UnitTimer():
    for i in range(8):
        x = pca.transform(X)

with UnitTimer():
    for i in range(8):
        x = pca.transform_mpi(X, keep_order=True, ncpu=1, n_components=2)
print("Output shape:", x.shape)

colors = ['r' if i == 0 else ('b' if i == 1 else 'g')
          for i in iris['y'][:]]
visual.plot_scatter(x[:, 0], x[:, 1], color=colors, size=8)
visual.plot_save('/tmp/tmp.pdf')
# bananab
