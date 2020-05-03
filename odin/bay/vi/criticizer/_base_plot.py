import inspect
from numbers import Number

import numpy as np
import tensorflow as tf

from odin import visual as vs
from odin.bay.vi import metrics, utils
from odin.bay.vi.criticizer._base import CriticizerBase
from odin.ml import clustering, dimension_reduce


class CriticizerPlot(CriticizerBase, vs.Visualizer):

  def _check_factors(self, factors):
    if factors is None:
      factors = list(range(self.n_factors))
    else:
      try:
        factors = [
            int(i) if isinstance(i, Number) else self.index(i)
            for i in tf.nest.flatten(factors)
        ]
      except ValueError:
        raise ValueError("Cannot find factors: %s, from list of factors: %s" %
                         (str(factors), self.factors_name))
    return factors

  def plot_test(self):
    self.assert_sampled()
    return self

  def plot_histogram(self, histogram_bins=120, original_factors=True):
    r"""
    orginal_factors : optional original factors before discretized by
      `Criticizer`
    """
    self.assert_sampled()
    from matplotlib import pyplot as plt
    ## prepare the data
    Z = np.concatenate(self.representations_mean, axis=0)
    F = np.concatenate(
        self.original_factors if original_factors else self.factors, axis=0)
    X = [i for i in F.T] + [i for i in Z.T]
    labels = self.factors_name.tolist() + self.codes_name.tolist()
    # create the figure
    ncol = int(np.ceil(np.sqrt(len(X)))) + 1
    nrow = int(np.ceil(len(X) / ncol))
    fig = vs.plot_figure(nrow=18, ncol=25, dpi=80)
    for i, (x, lab) in enumerate(zip(X, labels)):
      vs.plot_histogram(x,
                        ax=(nrow, ncol, i + 1),
                        bins=int(histogram_bins),
                        title=lab,
                        alpha=0.8,
                        color='blue',
                        fontsize=16)
    plt.tight_layout()
    self.add_figure(
        "histogram_%s" % ("original" if original_factors else "discretized"),
        fig)
    return self

  def plot_histogram_heatmap(self,
                             factors=None,
                             factor_bins=15,
                             histogram_bins=80,
                             n_codes_per_factor=6,
                             corr_method='average',
                             original_factors=True):
    r""" The histogram bars are colored by the value of factors

    Arguments:
      factors : which factors will be used
      factor_bins : factor is discretized into bins, then a LogisticRegression
        model will predict the bin (with color) given the code as input.
      orginal_factors : optional original factors before discretized by
        `Criticizer`
    """
    self.assert_sampled()
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    if n_codes_per_factor is None:
      n_codes_per_factor = self.n_codes
    else:
      n_codes_per_factor = int(n_codes_per_factor)
    styles = dict(fontsize=12,
                  val_bins=int(factor_bins),
                  color='bwr',
                  bins=int(histogram_bins),
                  alpha=0.8)
    ## correlation
    train_corr, test_corr = self.cal_correlation_matrix(mean=True,
                                                        method=corr_method,
                                                        decode=False)
    corr = (train_corr + test_corr) / 2
    ## prepare the data
    factors = self._check_factors(factors)
    Z = np.concatenate(self.representations_mean, axis=0)
    F = np.concatenate(
        self.original_factors if original_factors else self.factors,
        axis=0)[:, factors]
    # annotations
    factors_name = self.factors_name[factors]
    codes_name = self.codes_name
    # create the figure
    nrow = F.shape[1]
    ncol = int(1 + n_codes_per_factor)
    fig = vs.plot_figure(nrow=nrow * 3, ncol=ncol * 3, dpi=80)
    plot_count = 1
    for fidx, (f, fname) in enumerate(zip(F.T, factors_name)):
      c = corr[:, fidx]
      vs.plot_histogram(f,
                        val=f,
                        ax=(nrow, ncol, plot_count),
                        cbar=True,
                        cbar_horizontal=False,
                        title=fname,
                        **styles)
      plot_count += 1
      # all codes are visualized
      if n_codes_per_factor == self.n_codes:
        all_codes = range(self.n_codes)
      # lower to higher correlation
      else:
        zids = np.argsort(c)
        bottom = zids[:n_codes_per_factor // 2]
        top = zids[-(n_codes_per_factor - n_codes_per_factor // 2):]
        all_codes = (top.tolist()[::-1] + bottom.tolist()[::-1])
      for i in all_codes:
        z = Z[:, i]
        zname = codes_name[i]
        vs.plot_histogram(z,
                          val=f,
                          ax=(nrow, ncol, plot_count),
                          title='[%.2g]%s' % (c[i], zname),
                          **styles)
        plot_count += 1
    fig.tight_layout()
    self.add_figure(
        "histogram_%s" % ("original" if original_factors else "discretized"),
        fig)
    return self

  def plot_code_factor_matrix(self, factors=None, original_factors=True):
    r""" Scatter plot with x-axis is groundtruth factor and y-axis is
    latent code

    Arguments:
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
    codes_name = self.codes_name
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
        zname = codes_name[code_idx]
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
    for row, (code, mean,
              std) in enumerate(zip(self.codes_name, zmean.T, zstd.T)):
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
        vs.plot_scatter(x=X,
                        y=y,
                        val=y,
                        size=12,
                        color='bwr',
                        alpha=0.5,
                        grid=False)
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

  def plot_uncertainty_scatter(self, factors=None, n_samples=2, algo='tsne'):
    r""" Plotting the scatter points of the mean and sampled latent codes,
    colored by the factors.

    Arguments:
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
                         combined=True,
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
                          alpha=0.3,
                          linewidths=0,
                          cbar=True,
                          cbar_horizontal=True,
                          title=name,
                          **kw)
        else:  # the samples
          vs.plot_scatter_text(size=8,
                               marker='x',
                               alpha=0.8,
                               weight='light',
                               **kw)
    # fig.tight_layout()
    self.add_figure("uncertainty_scatter_%s" % algo, fig)
    return self
