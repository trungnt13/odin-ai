from __future__ import absolute_import, division, print_function

import os
import pickle
import warnings
from collections import defaultdict

import matplotlib.mlab as mlab
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm

from odin.stats import describe
from odin.utils import (ArgController, Progbar, UnitTimer, auto_logging,
                        batching, catch_warnings_ignore, get_script_path,
                        is_number, mpi, unique_labels)
from odin.visual import (Visualizer, generate_random_colors, merge_text_graph,
                         plot_confusion_matrix, plot_figure, plot_histogram,
                         plot_histogram_layers, plot_save, print_dist)

os.environ['ODIN'] = 'gpu,float32,seed=5218'

with warnings.catch_warnings():
  warnings.filterwarnings('ignore', category=ImportWarning)
  from sklearn.mixture import GaussianMixture
  from sklearn.neighbors import KernelDensity
  from sklearn.base import DensityMixin, BaseEstimator


# ===========================================================================
# Helpers
# ===========================================================================
def _clipping_quartile(x, alpha=1.5, test_mode=False):
  # result wider range threshold, keep more data points,
  # lower LLK
  x = x.astype('float32')
  Q1 = np.percentile(x, q=25)
  Q3 = np.percentile(x, q=75)
  IQR = Q3 - Q1

  high = Q3 + alpha * IQR
  low = Q1 - alpha * IQR

  if test_mode:
    x[x < low] = low
    x[x > high] = high
  else:
    x = x[np.logical_and(low <= x, x <= high)]
  return x


def _log_norm(x, scale_factor=10000):
  x = x.astype('float32')
  x_sum = np.sum(x)
  return np.log1p(x / (x_sum + np.finfo(x.dtype).eps) * scale_factor)


def _draw_hist(x, ax, title, n_bins, show_yticks=True):
  count, bins = plot_histogram(x=x,
                               bins=n_bins,
                               ax=ax,
                               normalize=False,
                               kde=False,
                               range_0_1=False,
                               covariance_factor=0.25,
                               centerlize=False,
                               fontsize=8,
                               title=title)
  plt.xlim((np.min(x), np.max(x)))
  plt.xticks(np.linspace(start=np.min(x),
                         stop=np.max(x),
                         num=5,
                         dtype='float32'),
             fontsize=6)
  if show_yticks:
    plt.yticks(np.linspace(start=np.min(count),
                           stop=np.max(count),
                           num=5,
                           dtype='int32'),
               fontsize=5)
  else:
    plt.yticks([], [])
  return count, bins


# ===========================================================================
# LabelThresholding class
# ===========================================================================
class ProbabilisticEmbedding(BaseEstimator, DensityMixin, Visualizer):
  r""" Probabilistic embedding of real values vectors using
    Gaussian Mixture Model

  Arguments:
    n_components_per_class : int
      number of GMM components for thresholding (default: 2)
    positive_component : int
      in case, 3 or more components are used, this is the index
      of the component from where it is positive during thresholding
      (the components are sorted by increasing mean order)
    log_norm : bool
    clip_quartile : float
    ci_threshold : float
    random_state: int
    verbose: bool

  """

  def __init__(self,
               n_components_per_class=2,
               positive_component=1,
               log_norm=True,
               clip_quartile=0.,
               remove_zeros=True,
               ci_threshold=-0.68,
               random_state=1,
               verbose=False):
    super(ProbabilisticEmbedding, self).__init__()
    self.n_components_per_class = int(n_components_per_class)
    self.positive_component = int(positive_component)
    assert self.positive_component > 0

    self.remove_zeros = bool(remove_zeros)
    self.log_norm = bool(log_norm)
    self.clip_quartile = float(clip_quartile)

    ci_threshold = float(ci_threshold)
    assert 0 <= np.abs(ci_threshold) <= 1
    self.ci_threshold = ci_threshold

    self.verbose = bool(verbose)
    self.random_state = random_state
    self._models = []

  # ******************** properties ******************** #
  @property
  def n_classes(self):
    return len(self._models)

  @property
  def means(self):
    """ Components' means for all classes

    Returns
    -------
    array
        means array (n_components, n_classes)
    """
    return np.hstack([
        gmm.means_.ravel()[order][:, np.newaxis] for order, gmm in self._models
    ])

  @property
  def precisions(self):
    """ Components' precision for all classes

    Returns
    -------
    array
        precisions array (n_components, n_classes)
    """
    return np.hstack([
        gmm.precisions_.ravel()[order][:, np.newaxis]
        for order, gmm in self._models
    ])

  # ******************** main ******************** #
  def normalize(self, x, test_mode=False):
    if x.ndim > 1:
      x = x.ravel()
    assert np.all(x >= 0), "Only support non-negative values"
    if self.remove_zeros and not test_mode:
      x = x[x > 0]
    if self.clip_quartile > 0:
      x = _clipping_quartile(x, alpha=self.clip_quartile, test_mode=test_mode)
    if self.log_norm:
      x = _log_norm(x)
    return x

  def fit(self, X):
    assert X.ndim == 2, "Only support input matrix but given: %s" % str(X.shape)

    n_classes = X.shape[1]
    it = tqdm(list(range(n_classes))) if self.verbose else range(n_classes)
    for i in it:
      # ====== normalizing ====== #
      x_train = self.normalize(X[:, i], test_mode=False)
      # ====== GMM ====== #
      gmm = GaussianMixture(n_components=self.n_components_per_class,
                            covariance_type='diag',
                            init_params='kmeans',
                            n_init=8,
                            max_iter=120,
                            random_state=self.random_state)
      gmm.fit(x_train[:, np.newaxis])
      # ====== save GMM ====== #
      means_ = gmm.means_.ravel()
      order = np.argsort(means_)
      self._models.append((order, gmm))

  def fit_transform(self, X, return_probabilities=True):
    self.fit(X)
    return self.predict_proba(X) if return_probabilities else self.predict(X)

  def _predict(self, X, threshold):
    assert X.shape[1] == self.n_classes, "Number of classes mis-match"
    y = []
    for i, (order, gmm) in enumerate(self._models):
      x_test = self.normalize(X[:, i], test_mode=True)

      # binary thresholding
      if threshold is not None:
        ci = stats.norm.interval(
            np.abs(threshold),
            loc=gmm.means_[order[self.positive_component]],
            scale=np.sqrt(1 / gmm.precisions_[order[self.positive_component]]))
        x_out = (x_test >=
                 (ci[0] if threshold < 0 else ci[1])).astype('float32')
        x_out = x_out[:, np.newaxis]
      # probabilizing
      else:
        probas = gmm.predict_proba(
            x_test[:, np.newaxis]).T[order][self.positive_component:]
        probas = np.mean(probas, axis=0)
        x_out = probas[:, np.newaxis]
      y.append(x_out)
    return np.concatenate(y, axis=1)

  def predict(self, X):
    return self._predict(X, threshold=self.ci_threshold)

  def predict_proba(self, X):
    return self._predict(X, threshold=None)

  def score_samples(self, X):
    r"""Compute the weighted log probabilities for each sample.

    Arguments:
      X : array-like, shape (n_samples, n_features)
          List of n_features-dimensional data points. Each row
          corresponds to a single data point.

    Returns:
      log_prob : array, shape (n_samples,)
          Log probabilities of each data point in X.
    """
    scores = []
    for x, (order, gmm) in zip(X.T, self._models):
      x = self.normalize(x, test_mode=True)
      s = gmm.score_samples(x[:, np.newaxis])[:, np.newaxis]
      scores.append(s)
    return np.mean(np.hstack(scores), axis=1)

  def score(self, X, y=None):
    r"""Compute the per-sample average log-likelihood of the given data X.

    Arguments:
      X : array-like, shape (n_samples, n_dimensions)
          List of n_features-dimensional data points. Each row
          corresponds to a single data point.

    Returns:
      log_likelihood : float
          Log likelihood of the Gaussian mixture given X.
    """
    return self.score_samples(X).mean()

  # ******************** methods for diagnosing ******************** #
  def _check_input(self, X, labels):
    assert X.ndim == 2, \
    "Only support input matrix but given: %s" % str(X.shape)

    n_classes = X.shape[1]
    assert n_classes == self.n_classes, \
    "Fitted with %d classes but give %d classes" % (self.n_classes, n_classes)

    if labels is None:
      labels = ['#%d' % i for i in range(n_classes)]
    assert len(labels) == n_classes, \
    "Number of classes and labels mismatch"
    return X, labels, n_classes

  def plot_diagnosis(self, X, labels=None, n_bins=200):
    X, labels, n_classes = self._check_input(X, labels)

    nrow = n_classes
    ncol = 1
    fig = plot_figure(nrow=nrow * 2, ncol=8)
    # add 1 for threshold color
    # add 1 for PDF color
    colors = sns.color_palette(n_colors=self.n_components_per_class + 2)

    for i, (name, (order, gmm)) in enumerate(zip(labels, self._models)):
      start = ncol * i

      means_ = gmm.means_.ravel()[order]
      precision_ = gmm.precisions_.ravel()[order]
      x = self.normalize(X[:, i], test_mode=False)

      # ====== scores ====== #
      # score
      score_llk = gmm.score(x[:, np.newaxis])
      score_bic = gmm.bic(x[:, np.newaxis])
      score_aic = gmm.aic(x[:, np.newaxis])

      # ====== the histogram ====== #
      ax = plt.subplot(nrow, ncol, start + 1)
      count, bins = _draw_hist(x,
                               ax=ax,
                               title="[%s] LLK:%.2f BIC:%.2f AIC:%.2f" %
                               (name, score_llk, score_bic, score_aic),
                               n_bins=n_bins,
                               show_yticks=True)

      # ====== draw GMM PDF ====== #
      y_ = np.exp(gmm.score_samples(bins[:, np.newaxis]))
      y_ = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_)) * np.max(count)
      ax.plot(bins, y_, color='red', linestyle='-', linewidth=1.5, alpha=0.6)

      # ====== draw the threshold ====== #
      ci = stats.norm.interval(
          np.abs(self.ci_threshold),
          loc=gmm.means_[order[self.positive_component]],
          scale=np.sqrt(1 / gmm.precisions_[order[self.positive_component]]))
      threshold = ci[0] if self.ci_threshold < 0 else ci[1]
      ids = np.where(bins >= threshold, True, False)
      ax.fill_between(bins[ids],
                      y1=0,
                      y2=np.max(count),
                      facecolor=colors[-2],
                      alpha=0.3)
      ax.text(np.min(bins[ids]), np.min(count), "%.2f" % threshold)

      # ====== plot GMM probability ====== #
      x_ = np.linspace(np.min(bins), np.max(bins), 1200)
      y_ = gmm.predict_proba(x_[:, np.newaxis]) * np.max(count)
      for c, j in zip(colors, y_.T):
        plt.plot(x_, j, color=c, linestyle='--', linewidth=1.8, alpha=0.6)

      # ====== draw the each Gaussian bell ====== #
      ax = ax.twinx()
      _x = np.linspace(start=np.min(x), stop=np.max(x), num=800)
      for c, m, p in zip(colors, means_, precision_):
        with catch_warnings_ignore(Warning):
          j = mlab.normpdf(_x, m, np.sqrt(1 / p))
        ax.plot(_x, j, color=c, linestyle='-', linewidth=1)
        ax.scatter(_x[np.argmax(j)],
                   np.max(j),
                   s=66,
                   alpha=0.8,
                   linewidth=0,
                   color=c)
      ax.yaxis.set_ticklabels([])

    fig.tight_layout()
    self.add_figure('diagnosis', fig)
    return self

  def plot_distribution(self, X, labels=None):
    X, labels, n_classes = self._check_input(X, labels)

    X_bin = self.predict(X)
    X_prob = self.predict_proba(X)

    normalize_to_01 = lambda x: x / np.sum(x)
    dist_raw = normalize_to_01(np.sum(X, axis=0))
    dist_bin = normalize_to_01(np.sum(X_bin, axis=0))
    dist_prob = normalize_to_01(np.sum(X_prob, axis=0))
    x = np.arange(n_classes)

    fig = plot_figure(nrow=3, ncol=int(n_classes * 1.2))
    ax = plt.gca()

    colors = sns.color_palette(n_colors=3)
    bar1 = ax.bar(x, dist_raw, width=0.2, color=colors[0], alpha=0.8)
    bar2 = ax.bar(x + 0.2, dist_bin, width=0.2, color=colors[1], alpha=0.8)
    bar3 = ax.bar(x + 0.4, dist_prob, width=0.2, color=colors[2], alpha=0.8)

    ax.set_xticks(x + 0.2)
    ax.set_xticklabels(labels, rotation=-10)
    ax.legend([bar1, bar2, bar3], ['Original', 'Binarized', 'Probabilized'])

    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    self.add_figure('distribution', fig)
    return self

  def boxplot(self, X, labels=None):
    X, labels, n_classes = self._check_input(X, labels)

    nrow = n_classes
    ncol = 3
    fig = plot_figure(nrow=3 * nrow, ncol=int(1.5 * ncol))

    for i, (x, name) in enumerate(zip(X.T, labels)):
      start = i * ncol

      ax = plt.subplot(nrow, ncol, start + 1)
      ax.boxplot(x,
                 whis=1.5,
                 labels=['Original'],
                 flierprops={
                     'marker': '.',
                     'markersize': 8
                 },
                 showmeans=True,
                 meanline=True)
      ax.set_ylabel(name)

      ax = plt.subplot(nrow, ncol, start + 2)
      ax.boxplot(x[x > 0],
                 whis=1.5,
                 labels=['NonZeros'],
                 flierprops={
                     'marker': '.',
                     'markersize': 8
                 },
                 showmeans=True,
                 meanline=True)

      ax = plt.subplot(nrow, ncol, start + 3)
      ax.boxplot(self.normalize(x, test_mode=False),
                 whis=1.5,
                 labels=['Normalized'],
                 flierprops={
                     'marker': '.',
                     'markersize': 8
                 },
                 showmeans=True,
                 meanline=True)

    plt.tight_layout()
    self.add_figure('boxplot', fig)
    return self
