from __future__ import absolute_import, division, print_function

import random
from collections import defaultdict
from numbers import Number

import numpy as np
from odin.visual.plot_utils import *
from scipy import stats


def _fit(x, y, n_bins):
  from odin.utils import catch_warnings_ignore
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import GridSearchCV
  from sklearn.preprocessing import KBinsDiscretizer
  x = x[:, np.newaxis]
  y = KBinsDiscretizer(n_bins=int(n_bins), encode='ordinal').fit_transform(
      y[:, np.newaxis]).ravel().astype(np.int64)
  with catch_warnings_ignore(UserWarning):
    lr = GridSearchCV(estimator=LogisticRegression(max_iter=500,
                                                   solver='liblinear',
                                                   random_state=1234),
                      cv=2,
                      param_grid=dict(C=np.linspace(0.5, 5, num=5)))
    lr.fit(x, y)
  return lr


def plot_histogram(x,
                   color_val=None,
                   bins_color=10,
                   bins=80,
                   ax=None,
                   normalize=False,
                   range_0_1=False,
                   kde=False,
                   covariance_factor=None,
                   color='blue',
                   color_kde='red',
                   grid=False,
                   cbar=False,
                   cbar_horizontal=False,
                   alpha=0.6,
                   centerlize=False,
                   linewidth=1.2,
                   fontsize=12,
                   bold_title=False,
                   title=None):
  r"""
  Arguments:
    x: array, data for ploting histogram
    color_val : array (optional), heatmap color value for each histogram bars
    bins_color : int, number of bins for the colors
    bins : int, number of histogram bins
    covariance_factor : None or float,
      smaller value lead to more detail for KDE plot
  """
  import matplotlib as mpl
  from matplotlib import pyplot as plt
  bins_color = int(bins_color)
  bins = int(bins)
  # ====== prepare ====== #
  # only 1-D
  if isinstance(x, (tuple, list)):
    x = np.array(x)
  x = x.ravel()
  ax = to_axis(ax, is_3D=False)
  ax.grid(grid)
  # ====== get the bins ====== #
  if range_0_1:
    x = (x - np.min(x, axis=0, keepdims=True)) /\
        (np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))
  hist, hist_bins = np.histogram(x, bins=bins, density=normalize)
  width = (hist_bins[1] - hist_bins[0]) / 1.36
  # colormap
  if color_val is not None:
    assert len(x) == len(color_val), \
      f"Given {len(x)} data points but {len(color_val)} color values"
    if color == 'blue':  # change the default color
      color = 'Blues'
    # create mapping x -> val_bins
    lr = _fit(x=x, y=color_val, n_bins=bins_color)
    # all colors
    vmin, vmax = np.min(color_val), np.max(color_val)
    cmap = plt.cm.get_cmap(color)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    all_colors = cmap(normalizer(np.linspace(vmin, vmax, num=bins_color)))
    # map the x bin to val bin then to color
    feat = np.array([(s + e) / 2 for s, e in zip(hist_bins[:-1], hist_bins[1:])
                    ])[:, np.newaxis]
    color = [all_colors[i] for i in lr.predict(feat).ravel()]
  # histogram bar
  ax.bar((hist_bins[:-1] + hist_bins[1:]) / 2 - width / 2,
         hist,
         width=width,
         color=color,
         alpha=alpha,
         linewidth=0.,
         edgecolor=None)
  # colorbar
  if color_val is not None and cbar:
    mappable = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    mappable.set_clim(vmin, vmax)
    cba = plt.colorbar(
        mappable,
        ax=ax,
        shrink=0.99,
        pad=0.01,
        orientation='horizontal' if cbar_horizontal else 'vertical')
    cba.ax.tick_params(labelsize=fontsize - 2)
  # ====== centerlize the data ====== #
  min_val = np.min(hist_bins)
  max_val = np.max(hist_bins)
  if centerlize:
    ax.set_xlim((min_val - np.abs(max_val) / 2, max_val + np.abs(max_val) / 2))
  # ====== kde ====== #
  if kde:
    if not normalize:
      raise ValueError(
          "KDE plot only applicable for normalized-to-1 histogram.")
    density = stats.gaussian_kde(x)
    if isinstance(covariance_factor, Number):
      density.covariance_factor = lambda: float(covariance_factor)
      density._compute_covariance()
    if centerlize:
      xx = np.linspace(
          np.min(x) - np.abs(max_val) / 2,
          np.max(x) + np.abs(max_val) / 2, 100)
    else:
      xx = np.linspace(np.min(x), np.max(x), 100)
    yy = density(xx)
    ax.plot(xx,
            yy,
            color=color_kde,
            alpha=min(1., alpha + 0.2),
            linewidth=linewidth,
            linestyle='-.')
  # ====== post processing ====== #
  ax.tick_params(axis='both', labelsize=int(0.8 * fontsize))
  if title is not None:
    ax.set_title(str(title),
                 fontsize=fontsize,
                 fontweight='bold' if bold_title else 'regular')
  return ax, hist, hist_bins


def plot_histogram_layers(Xs,
                          bins=50,
                          ax=None,
                          normalize=False,
                          range_0_1=False,
                          kde=False,
                          covariance_factor=None,
                          layer_name=None,
                          layer_color=None,
                          legend_loc='upper center',
                          legend_ncol=5,
                          legend_colspace=0.4,
                          grid=True,
                          fontsize=12,
                          title=None):
  r"""
  normalize : bool (default: False)
    pass
  range_0_1 : bool (default: False)
    if True, normalize each array in `Xs` so it min value is 0,
    and max value is 1
  covariance_factor : None or float
    if float is given, smaller mean more detail
  """
  if isinstance(Xs, np.ndarray):
    assert Xs.ndim == 2
    Xs = [Xs[:, i] for i in range(Xs.shape[1])]
  num_classes = len(Xs)
  ax = to_axis(ax, is_3D=True)
  # ====== validate input argument ====== #
  layer_name = check_arg_length(layer_name,
                                n=num_classes,
                                dtype=string_types,
                                default='',
                                converter=lambda x: str(x))
  layer_color = check_arg_length(layer_color,
                                 n=num_classes,
                                 dtype=string_types,
                                 default='blue',
                                 converter=lambda x: str(x))
  legends = []
  for name, a, c, z, x in zip(layer_name,
                              np.linspace(0.6, 0.9,
                                          num_classes)[::-1], layer_color,
                              np.linspace(0, 100, num_classes), Xs):
    if range_0_1:
      x = (x - np.min(x, axis=0, keepdims=True)) /\
          (np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))
    hist, hist_bins = np.histogram(x, bins=bins, density=normalize)
    width = (hist_bins[1] - hist_bins[0]) / 1.36
    _ = ax.bar(left=(hist_bins[:-1] + hist_bins[1:]) / 2 - width / 2,
               height=hist,
               width=width,
               zs=z,
               zdir='y',
               color=c,
               ec=c,
               alpha=a)
    if kde:
      if not normalize:
        raise ValueError(
            "KDE plot only applicable for normalized-to-1 histogram.")
      density = stats.gaussian_kde(x)
      if isinstance(covariance_factor, Number):
        density.covariance_factor = lambda: float(covariance_factor)
        density._compute_covariance()
      xx = np.linspace(np.min(x), np.max(x), 1000)
      yy = density(xx)
      zz = np.full_like(xx, fill_value=z)
      ax.plot(xs=xx,
              ys=zz,
              zs=yy,
              color=c,
              alpha=a,
              linewidth=1.2,
              linestyle='-.')
    # legend
    if len(name) > 0:
      legends.append((name, _))
  # ====== legend ====== #
  if len(legends) > 0:
    legends = ax.legend([i[1] for i in legends], [i[0] for i in legends],
                        markerscale=1.5,
                        scatterpoints=1,
                        scatteryoffsets=[0.375, 0.5, 0.3125],
                        loc=legend_loc,
                        bbox_to_anchor=(0.5, -0.01),
                        ncol=int(legend_ncol),
                        columnspacing=float(legend_colspace),
                        labelspacing=0.,
                        fontsize=fontsize,
                        handletextpad=0.1)
    for i, c in enumerate(layer_color):
      legends.legendHandles[i].set_color(c)
  # ====== config ====== #
  ax.set_xlabel('Value')
  ax.set_zlabel('Frequency', rotation=-90)
  ax.set_yticklabels([])
  ax.grid(grid)
  if title is not None:
    ax.set_title(str(title))
  return ax
