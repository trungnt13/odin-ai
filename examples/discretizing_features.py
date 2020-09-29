from __future__ import absolute_import, division, print_function

import os

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from odin import visual as vs
from odin.bay.vi import discretizing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)
sns.set()

shape = (1024, 1)
total_figures = 1 + 4 * 2
ncol = nrow = int(np.ceil(np.sqrt(total_figures)))
hist_bins = 120
for dist, fn in [('uniform', np.random.rand), ('normal', np.random.randn)]:
  x = fn(*shape)
  vs.plot_figure(nrow=12, ncol=12, dpi=120)
  ax = vs.subplot(nrow, ncol, 1)
  ax, _, _ = vs.plot_histogram(x, bins=hist_bins, title=dist, ax=ax)
  idx = 2
  for strategy in ('gmm', 'uniform', 'quantile', 'kmeans'):
    for n_bins in (5, 10):
      y = discretizing(x, n_bins=n_bins, strategy=strategy)
      title = '%s-%d' % (strategy, n_bins)
      ax = vs.subplot(nrow, ncol, idx)
      vs.plot_histogram(y, bins=hist_bins, ax=ax, title=title)
      idx += 1
  plt.tight_layout()

# ====== special case: GMM discretizing ====== #
vs.plot_figure()
y, gmm = discretizing(x, n_bins=2, strategy='gmm', return_model=True)
gmm = gmm[0]
vs.plot_gaussian_mixture(x,
                         gmm,
                         show_probability=True,
                         show_pdf=True,
                         show_components=True)
# ====== save everything ====== #
vs.plot_save()
