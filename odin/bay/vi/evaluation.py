from __future__ import absolute_import, division, print_function

from numbers import Number

import numpy as np
import tensorflow as tf

from odin import visual as vs
from odin.bay.vi import metrics, utils
from odin.bay.vi._evaluation import _Criticizer
from odin.ml import clustering, dimension_reduce

__all__ = ['Criticizer']


class Criticizer(_Criticizer, vs.Visualizer):

  ############## Ploting
  def _check_factors(self, factors):
    if factors is None:
      factors = list(range(self.n_factors))
    else:
      factors = [
          int(i) if isinstance(i, Number) else self.index(i)
          for i in tf.nest.flatten(factors)
      ]
    return factors

  def plot_histogram(self, factors=None, bins=120, original_factors=True):
    r"""
    orginal_factors : optional original factors before discretized by
      `Criticizer`
    """
    self._assert_sampled()
    from matplotlib import pyplot as plt
    # prepare the data
    factors = self._check_factors(factors)
    Z = np.concatenate(self.representations_mean, axis=0)
    F = np.concatenate(
        self.original_factors if original_factors else self.factors,
        axis=0)[:, factors]
    X = [i for i in F.T] + [i for i in Z.T]
    labels = self.factors_name[factors].tolist() + \
      self.code_name.tolist()
    # create the figure
    ncol = int(np.ceil(np.sqrt(len(X)))) + 1
    nrow = int(np.ceil(len(X) / ncol))
    fig = vs.plot_figure(nrow=18, ncol=25, dpi=80)
    for i, (x, lab) in enumerate(zip(X, labels)):
      vs.plot_histogram(x,
                        ax=(nrow, ncol, i + 1),
                        bins=int(bins),
                        title=lab,
                        alpha=0.8,
                        color='blue',
                        fontsize=16)
    plt.tight_layout()
    self.add_figure(
        "histogram_%s" % ("original" if original_factors else "discretized"),
        fig)
    return self

  def plot_code_factor_matrix(self, factors=None, original_factors=True):
    r"""
    factors : list of Integer or String. The index or name of factors taken
      into account for analyzing.
    """
    from matplotlib import pyplot as plt
    factors = self._check_factors(factors)
    Z = np.concatenate(self.representations_mean, axis=0)
    F = np.concatenate(
        self.original_factors if original_factors else self.factors,
        axis=0)[:, factors]
    labels = self.factors_name[factors]
    code_name = self.code_name
    # create the figure
    nrow = Z.shape[1]  # number representations
    ncol = F.shape[1]  # number factors
    fig, subplots = plt.subplots(nrows=nrow,
                                 ncols=ncol,
                                 sharex=True,
                                 sharey=True,
                                 squeeze=True,
                                 gridspec_kw=dict(wspace=0.01, hspace=0.01),
                                 figsize=(ncol * 3, nrow * 3),
                                 dpi=80)
    for code_idx, row in enumerate(subplots):
      for fact_idx, ax in enumerate(row):
        z = Z[:, code_idx]
        zname = code_name[code_idx]
        f = F[:, fact_idx]
        fname = labels[fact_idx]
        mean = np.mean(f)
        ax.scatter(f,
                   z,
                   s=6,
                   alpha=0.5,
                   c=['r' if i > mean else 'b' for i in f],
                   linewidths=0.)
        ax.grid(False)
        if fact_idx == 0:
          ax.set_ylabel(zname, fontsize=16)
        if code_idx == 0:
          ax.set_title(fname, fontsize=16)
    self.add_figure(
        "code_factor_%s" % ("original" if original_factors else "discretized"),
        fig)
    return self

  def plot_uncertainty_statistics(self, factors=None):
    r"""
    factors : list of Integer or String. The index or name of factors taken
      into account for analyzing.
    """
    factors = self._check_factors(factors)
    zmean = np.concatenate(self.representations_mean, axis=0)
    zstd = np.sqrt(np.concatenate(self.representations_variance, axis=0))
    labels = self.factors_name[factors]
    factors = np.concatenate(self.original_factors, axis=0)[:, factors]
    X = np.arange(zmean.shape[0])
    # create the figure
    nrow = self.n_representations
    ncol = len(labels)
    fig = vs.plot_figure(nrow=nrow * 4, ncol=ncol * 4, dpi=80)
    plot = 1
    for row, (code, mean, std) in enumerate(zip(self.code_name, zmean.T,
                                                zstd.T)):
      # prepare the code
      ids = np.argsort(mean)
      mean, std = mean[ids], std[ids]
      # show the factors
      for col, (name, y) in enumerate(zip(labels, factors.T)):
        axes = []
        # the variance
        ax = vs.plot_subplot(nrow, ncol, plot)
        ax.plot(mean, color='g', linestyle='--')
        ax.fill_between(X, mean - 2 * std, mean + 2 * std, alpha=0.2, color='b')
        if col == 0:
          ax.set_ylabel(code)
        if row == 0:
          ax.set_title(name)
        axes.append(ax)
        # factor
        y = y[ids]
        ax = ax.twinx()
        vs.plot_scatter_heatmap(x=X,
                                y=y,
                                val=y,
                                size=12,
                                colormap='bwr',
                                alpha=0.5)
        axes.append(ax)
        # update plot index
        for ax in axes:
          ax.tick_params(axis='both',
                         which='both',
                         top=False,
                         bottom=False,
                         left=False,
                         right=False,
                         labeltop=False,
                         labelleft=False,
                         labelright=False,
                         labelbottom=False)
        plot += 1
    fig.tight_layout()
    self.add_figure("uncertainty_stats", fig)
    return self

  def plot_uncertainty_scatter(self, factors=None, n_samples=2, algo='pca'):
    r"""
    factors : list of Integer or String. The index or name of factors taken
      into account for analyzing.
    """
    factors = self._check_factors(factors)
    # this all include tarin and test data separatedly
    z_mean = np.concatenate(self.representations_mean)
    z_var = np.concatenate(
        [np.mean(var, axis=1) for var in self.representations_variance])
    z_samples = [
        z for z in np.concatenate(self.representations_sample(int(n_samples)),
                                  axis=1)
    ]
    F = np.concatenate(self.original_factors, axis=0)[:, factors]
    labels = self.factors_name[factors]
    # preprocessing
    inputs = tuple([z_mean] + z_samples)
    Z = dimension_reduce(*inputs,
                         algo=algo,
                         n_components=2,
                         return_model=False,
                         random_state=self.randint)
    V = utils.discretizing(z_var[:, np.newaxis], n_bins=10).ravel()
    # the figure
    nrow = 3
    ncol = int(np.ceil(len(labels) / nrow))
    fig = vs.plot_figure(nrow=nrow * 4, ncol=ncol * 4, dpi=80)
    for idx, (name, y) in enumerate(zip(labels, F.T)):
      ax = vs.plot_subplot(nrow, ncol, idx + 1)
      for i, x in enumerate(Z):
        kw = dict(val=y,
                  color="coolwarm",
                  ax=ax,
                  x=x,
                  grid=False,
                  legend_enable=False,
                  centroids=True,
                  fontsize=12)
        if i == 0:  # the mean value
          vs.plot_scatter(size=V,
                          size_range=(8, 80),
                          alpha=0.4,
                          linewidths=0,
                          cbar=True,
                          cbar_horizontal=True,
                          title=name,
                          **kw)
        else:  # the samples
          # vs.plot_scatter(size=8, marker='x', alpha=0.3, **kw)
          vs.plot_scatter_text(size=8, marker='x', alpha=0.3, **kw)
    # fig.tight_layout()
    self.add_figure("uncertainty_scatter_%s" % algo, fig)
    return self
