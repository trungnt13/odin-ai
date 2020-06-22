import inspect
from numbers import Number

import numpy as np
import tensorflow as tf
from six import string_types

from odin import visual as vs
from odin.bay.vi import metrics, utils
from odin.bay.vi.criticizer._criticizer_metrics import CriticizerMetrics
from odin.ml import clustering, dimension_reduce
from odin.search import diagonal_linear_assignment


class CriticizerPlot(CriticizerMetrics, vs.Visualizer):

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
                         (str(factors), self.factor_names))
    return factors

  def plot(self, n_bins):
    pass

  def plot_histogram(self,
                     histogram_bins=120,
                     original_factors=True,
                     return_figure=False):
    r"""
    original_factors : optional original factors before discretized by
      `Criticizer`
    """
    self.assert_sampled()
    from matplotlib import pyplot as plt
    ## prepare the data
    Z = np.concatenate(self.representations_mean, axis=0)
    F = np.concatenate(
        self.original_factors if original_factors else self.factors, axis=0)
    X = [i for i in F.T] + [i for i in Z.T]
    labels = self.factor_names.tolist() + self.code_names.tolist()
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
    fig.tight_layout()
    if return_figure:
      return fig
    return self.add_figure(
        f"histogram_{'original' if original_factors else 'discretized'}", fig)

  def plot_disentanglement(self,
                           factor_names=None,
                           n_bins_factors=15,
                           n_bins_codes=80,
                           corr_type='average',
                           original_factors=True,
                           show_all_codes=False,
                           title='',
                           return_figure=False):
    r""" To illustrate the disentanglement of the codes, the codes' histogram
    bars are colored by the value of factors.

    Arguments:
      factor_names : list of String or Integer.
        Name or index of which factors will be used for visualization.
      factor_bins : factor is discretized into bins, then a LogisticRegression
        model will predict the bin (with color) given the code as input.
      corr_type : {'spearman', 'pearson', 'lasso', 'average', 'mi', None, matrix}
        Type of correlation, with special case 'mi' for mutual information.
          - If None, no sorting by correlation provided.
          - If an array, the array must have shape `[n_codes, n_factors]`
      show_all_codes : a Boolean.
        if False, only show most correlated codes-factors, otherwise,
        all codes are shown for each factor.
        This option only in effect when `corr_type` is not `None`.
      original_factors : optional original factors before discretized by
        `Criticizer`
    """
    self.assert_sampled()
    ### prepare styled plot
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    styles = dict(fontsize=12,
                  cbar_horizontal=False,
                  bins_color=int(n_bins_factors),
                  bins=int(n_bins_codes),
                  color='bwr',
                  alpha=0.8)
    # get all relevant factors
    factor_ids = self._check_factors(factor_names)
    ### correlation
    if isinstance(corr_type, string_types):
      if corr_type == 'mi':
        train_corr, test_corr = self.create_mutualinfo_matrix(mean=True)
        score_type = 'mutual-info'
      else:
        train_corr, test_corr = self.create_correlation_matrix(mean=True,
                                                               method=corr_type)
        score_type = corr_type
      # [n_factors, n_codes]
      corr = ((train_corr + test_corr) / 2.).T
      corr = corr[factor_ids]
      code_ids = diagonal_linear_assignment(np.abs(corr), nan_policy=0)
      if not show_all_codes:
        code_ids = code_ids[:len(factor_ids)]
    # directly give the correlation matrix
    elif isinstance(corr_type, np.ndarray):
      corr = corr_type
      if self.n_codes != self.n_factors and corr.shape[0] == self.n_codes:
        corr = corr.T
      assert corr.shape == (self.n_factors, self.n_codes), \
        (f"Correlation matrix expect shape (n_factors={self.n_factors}, "
         f"n_codes={self.n_codes}) but given shape: {corr.shape}")
      score_type = 'score'
      corr = corr[factor_ids]
      code_ids = diagonal_linear_assignment(np.abs(corr), nan_policy=0)
      if not show_all_codes:
        code_ids = code_ids[:len(factor_ids)]
    # no correlation provided
    elif corr_type is None:
      train_corr, test_corr = self.create_correlation_matrix(mean=True,
                                                             method='spearman')
      score_type = 'spearman'
      # [n_factors, n_codes]
      corr = ((train_corr + test_corr) / 2.).T
      code_ids = np.arange(self.n_codes, dtype=np.int32)
    # exception
    else:
      raise ValueError(
          f"corr_type could be string, None or a matrix but given: {type(corr_type)}"
      )
    # applying the indexing
    corr = corr[:, code_ids]
    ### prepare the data
    # factors
    F = np.concatenate(
        self.original_factors if original_factors else self.factors,
        axis=0,
    )[:, factor_ids]
    factor_names = self.factor_names[factor_ids]
    # codes
    Z = np.concatenate(self.representations_mean, axis=0)[:, code_ids]
    code_names = self.code_names[code_ids]
    ### create the figure
    nrow = F.shape[1]
    ncol = Z.shape[1] + 1
    fig = vs.plot_figure(nrow=nrow * 3, ncol=ncol * 2.8, dpi=80)
    count = 1
    for fidx, (f, fname) in enumerate(zip(F.T, factor_names)):
      # the first plot show how the factor clustered
      ax = vs.plot_histogram(x=f,
                             color_val=f,
                             ax=(nrow, ncol, count),
                             cbar=False,
                             title=f"{fname}",
                             **styles)
      plt.gca().tick_params(axis='y', labelleft=False)
      count += 1
      # the rest of the row show how the codes align with the factor
      for zidx, (score, z, zname) in enumerate(zip(corr[fidx], Z.T,
                                                   code_names)):
        text = "*" if fidx == zidx else ""
        ax = vs.plot_histogram(x=z,
                               color_val=f,
                               ax=(nrow, ncol, count),
                               cbar=False,
                               title=f"{text}{fname}-{zname} (${score:.2f}$)",
                               bold_title=True if fidx == zidx else False,
                               **styles)
        plt.gca().tick_params(axis='y', labelleft=False)
        count += 1
    ### fine tune the plot
    fig.suptitle(f"[{score_type}]{title}", fontsize=12)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    if return_figure:
      return fig
    return self.add_figure(
        f"disentanglement_{'original' if original_factors else 'discretized'}",
        fig)

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
    labels = self.factor_names[factors]
    code_names = self.code_names
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
        zname = code_names[code_idx]
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
    labels = self.factor_names[factors]
    factors = np.concatenate(self.original_factors, axis=0)[:, factors]
    X = np.arange(zmean.shape[0])
    # create the figure
    nrow = self.n_representations
    ncol = len(labels)
    fig = vs.plot_figure(nrow=nrow * 4, ncol=ncol * 4, dpi=80)
    plot = 1
    for row, (code, mean,
              std) in enumerate(zip(self.code_names, zmean.T, zstd.T)):
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
    labels = self.factor_names[factors]
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
