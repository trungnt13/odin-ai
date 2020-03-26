# -*- coding: utf-8 -*-
# ===========================================================================
# The waveform and spectrogram plot adapted from:
# [librosa](https://github.com/bmcfee/librosa)
# Copyright (c) 2016, librosa development team.
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import absolute_import, division, print_function

import colorsys
import copy
import itertools
import os
import sys
import warnings
from collections import Mapping, OrderedDict, defaultdict
from contextlib import contextmanager
from numbers import Number

import numpy as np
from scipy import stats
from six import string_types
from six.moves import range, zip

from odin.visual.heatmap_plot import *
from odin.visual.histogram_plot import *
from odin.visual.plot_utils import *
from odin.visual.scatter_plot import *
from odin.visual.stats_plot import *

# try:
#     import seaborn # import seaborn for pretty plot
# except:
#     pass


# ===========================================================================
# Helper for spectrogram
# ===========================================================================
def time_ticks(locs, *args, **kwargs):  # pylint: disable=star-args
  r'''Plot time-formatted axis ticks.
  Parameters
  ----------
  locations : list or np.ndarray
      Time-stamps for tick marks
  n_ticks : int > 0 or None
      Show this number of ticks (evenly spaced).
      If none, all ticks are displayed.
      Default: 5
  axis : 'x' or 'y'
      Which axis should the ticks be plotted on?
      Default: 'x'
  time_fmt : None or {'ms', 's', 'm', 'h'}
      - 'ms': milliseconds   (eg, 241ms)
      - 's': seconds         (eg, 1.43s)
      - 'm': minutes         (eg, 1:02)
      - 'h': hours           (eg, 1:02:03)
      If none, formatted is automatically selected by the
      range of the times data.
      Default: None
  fmt : str
      .. warning:: This parameter name was in librosa 0.4.2
          Use the `time_fmt` parameter instead.
          The `fmt` parameter will be removed in librosa 0.5.0.
  kwargs : additional keyword arguments.
      See `matplotlib.pyplot.xticks` or `yticks` for details.
  Returns
  -------
  locs
  labels
      Locations and labels of tick marks
  See Also
  --------
  matplotlib.pyplot.xticks
  matplotlib.pyplot.yticks
  Examples
  --------
  >>> # Tick at pre-computed beat times
  >>> librosa.display.specshow(S)
  >>> librosa.display.time_ticks(beat_times)
  >>> # Set the locations of the time stamps
  >>> librosa.display.time_ticks(locations, timestamps)
  >>> # Format in seconds
  >>> librosa.display.time_ticks(beat_times, time_fmt='s')
  >>> # Tick along the y axis
  >>> librosa.display.time_ticks(beat_times, axis='y')
  '''
  from matplotlib import pyplot as plt

  n_ticks = kwargs.pop('n_ticks', 5)
  axis = kwargs.pop('axis', 'x')
  time_fmt = kwargs.pop('time_fmt', None)

  if axis == 'x':
    ticker = plt.xticks
  elif axis == 'y':
    ticker = plt.yticks
  else:
    raise ValueError("axis must be either 'x' or 'y'.")

  if len(args) > 0:
    times = args[0]
  else:
    times = locs
    locs = np.arange(len(times))

  if n_ticks is not None:
    # Slice the locations and labels evenly between 0 and the last point
    positions = np.linspace(0, len(locs) - 1, n_ticks,
                            endpoint=True).astype(int)
    locs = locs[positions]
    times = times[positions]

  # Format the labels by time
  formats = {
      'ms':
          lambda t: '{:d}ms'.format(int(1e3 * t)),
      's':
          '{:0.2f}s'.format,
      'm':
          lambda t: '{:d}:{:02d}'.format(int(t / 6e1), int(np.mod(t, 6e1))),
      'h':
          lambda t: '{:d}:{:02d}:{:02d}'.format(int(
              t / 3.6e3), int(np.mod(t / 6e1, 6e1)), int(np.mod(t, 6e1)))
  }

  if time_fmt is None:
    if max(times) > 3.6e3:
      time_fmt = 'h'
    elif max(times) > 6e1:
      time_fmt = 'm'
    elif max(times) > 1.0:
      time_fmt = 's'
    else:
      time_fmt = 'ms'

  elif time_fmt not in formats:
    raise ValueError('Invalid format: {:s}'.format(time_fmt))

  times = [formats[time_fmt](t) for t in times]

  return ticker(locs, times, **kwargs)


def _cmap(data):
  r'''Get a default colormap from the given data.

  If the data is boolean, use a black and white colormap.

  If the data has both positive and negative values,
  use a diverging colormap ('coolwarm').

  Otherwise, use a sequential map: either cubehelix or 'OrRd'.

  Parameters
  ----------
  data : np.ndarray
      Input data


  Returns
  -------
  cmap : matplotlib.colors.Colormap
      - If `data` has dtype=boolean, `cmap` is 'gray_r'
      - If `data` has only positive or only negative values,
        `cmap` is 'OrRd' (`use_sns==False`) or cubehelix
      - If `data` has both positive and negatives, `cmap` is 'coolwarm'

  See Also
  --------
  matplotlib.pyplot.colormaps
  seaborn.cubehelix_palette
  '''
  import matplotlib as mpl
  from matplotlib import pyplot as plt

  _HAS_SEABORN = False
  try:
    _matplotlibrc = copy.deepcopy(mpl.rcParams)
    import seaborn as sns
    _HAS_SEABORN = True
    mpl.rcParams.update(**_matplotlibrc)
  except ImportError:
    pass

  data = np.atleast_1d(data)

  if data.dtype == 'bool':
    return plt.get_cmap('gray_r')

  data = data[np.isfinite(data)]

  robust = True
  if robust:
    min_p, max_p = 2, 98
  else:
    min_p, max_p = 0, 100

  max_val = np.percentile(data, max_p)
  min_val = np.percentile(data, min_p)

  if min_val >= 0 or max_val <= 0:
    if _HAS_SEABORN:
      return sns.cubehelix_palette(light=1.0, as_cmap=True)
    else:
      return plt.get_cmap('OrRd')

  return plt.get_cmap('coolwarm')


# ===========================================================================
# Plotting methods
# ===========================================================================
@contextmanager
def figure(nrow=8, ncol=8, dpi=180, show=False, tight_layout=True, title=''):
  from matplotlib import pyplot as plt
  inches_for_box = 2.4
  if nrow != ncol:
    nrow = inches_for_box * ncol
    ncol = inches_for_box * nrow
  else:
    nrow = inches_for_box * nrow
    ncol = inches_for_box * ncol
  nrow += 1.2  # for the title
  fig = plt.figure(figsize=(ncol, nrow), dpi=dpi)
  yield fig
  plt.suptitle(title)
  if show:
    plot_show(block=True, tight_layout=tight_layout)


def merge_figures(nrow, ncol):
  pass


def fig2data(fig):
  """w, h, 4"""
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
  buf.shape = (w, h, 4)
  buf = np.roll(buf, 3, axis=2)
  return buf


def data2fig(data):
  from matplotlib import pyplot as plt
  fig = plt.figure()
  plt.imshow(data)
  return fig


def plot_figure(nrow=8, ncol=8, dpi=180):
  from matplotlib import pyplot as plt
  fig = plt.figure(figsize=(ncol, nrow), dpi=dpi)
  return fig


def plot_title(title, fontsize=12):
  from matplotlib import pyplot as plt
  plt.suptitle(str(title), fontsize=fontsize)


def subplot(*arg, **kwargs):
  from matplotlib import pyplot as plt
  subplot = plt.subplot(*arg)
  if 'title' in kwargs:
    subplot.set_title(kwargs['title'])
  return subplot


def plot_frame(ax=None, left=None, right=None, top=None, bottom=None):
  """ Turn on, off the frame (i.e. the bounding box of an axis) """
  ax = to_axis(ax)
  if top is not None:
    ax.spines['top'].set_visible(bool(top))
  if right is not None:
    ax.spines['right'].set_visible(bool(right))
  if bottom is not None:
    ax.spines['bottom'].set_visible(bool(bottom))
  if left is not None:
    ax.spines['left'].set_visible(bool(left))
  return ax


def plot_aspect(aspect=None, adjustable=None, ax=None):
  r"""
  aspect : {'auto', 'equal'} or num
    'auto'  automatic; fill the position rectangle with data
    'equal' same scaling from data to plot units for x and y
    num a circle will be stretched such that the height is num times
    the width. aspect=1 is the same as aspect='equal'.

  adjustable : None or {'box', 'datalim'}, optional
    If not None, this defines which parameter will be adjusted to
    meet the required aspect. See set_adjustable for further details.
  """
  ax = to_axis(ax)
  if aspect is not None and adjustable is None:
    ax.axis(aspect)
  else:
    ax.set_aspect(aspect, adjustable)
  return ax


@contextmanager
def plot_gridSpec(nrow, ncol, wspace=None, hspace=None):
  """
  Example
  -------
  grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
  plt.subplot(grid[0, 0])
  plt.subplot(grid[0, 1:])
  plt.subplot(grid[1, :2])
  plt.subplot(grid[1, 2])
  """
  from matplotlib import pyplot as plt
  grid = plt.GridSpec(nrows=nrow, ncols=ncol, wspace=wspace, hspace=hspace)
  yield grid


def plot_gridSubplot(shape, loc, colspan=1, rowspan=1):
  """
  Example
  -------
  ax1 = plt.subplot2grid((3, 3), (0, 0))
  ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
  ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
  ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
  """
  from matplotlib import pyplot as plt
  return plt.subplot2grid(shape=shape,
                          loc=loc,
                          colspan=colspan,
                          rowspan=rowspan)


def plot_subplot(*args):
  from matplotlib import pyplot as plt
  return plt.subplot(*args)


def set_labels(ax, title=None, xlabel=None, ylabel=None):
  if title is not None:
    ax.set_title(title)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  if ylabel is not None:
    ax.set_ylabel(ylabel)


def plot_vline(x, ymin=0., ymax=1., color='r', ax=None):
  from matplotlib import pyplot as plt
  ax = ax if ax is not None else plt.gca()
  ax.axvline(x=x, ymin=ymin, ymax=ymax, color=color, linewidth=1, alpha=0.6)
  return ax


def plot_comparison_track(Xs,
                          legends,
                          tick_labels,
                          line_colors=None,
                          line_styles=None,
                          linewidth=1.,
                          marker_size=33,
                          marker_styles=None,
                          fontsize=10,
                          draw_label=True,
                          title=None):
  """ Plot multiple series for comparison
  Parameters
  ----------
  Xs : list (tuple) of series
    the list that contain list of data points
  legends : list of string
    name for each series
  tick_labels : list of string
    name for each data points
  draw_label : bool
    if True, drawing text of actual value of each point on top of it
  """
  if len(Xs) != len(legends):
    raise ValueError("Number of series (len(Xs)) is: %d different from "
                     "number of legends: %d" % (len(Xs), len(legends)))
  nb_series = len(Xs)
  if len(Xs[0]) != len(tick_labels):
    raise ValueError("Number of points for each series is: %d different from "
                     "number of xticks' labels: %d" %
                     (len(Xs[0], len(tick_labels))))
  nb_points = len(Xs[0])
  from matplotlib import pyplot as plt
  # ====== some default styles ====== #
  default_marker_styles = ['o', '^', 's', '*', '+', 'X', '|', 'D', 'H', '8']
  if marker_styles is None and nb_series <= len(default_marker_styles):
    marker_styles = default_marker_styles[:nb_series]
  # ====== init ====== #
  point_colors = []
  inited = False
  handlers = []
  # ====== plotting ====== #
  for idx, X in enumerate(Xs):
    kwargs = {}
    if line_colors is not None:
      kwargs['color'] = line_colors[idx]
    if line_styles is not None:
      kwargs['linestyle'] = line_styles[idx]
    else:
      kwargs['linestyle'] = '--'
    # lines
    handlers.append(plt.plot(X, linewidth=linewidth, **kwargs)[0])
    # points
    ax = plt.gca()
    for i, j in enumerate(X):
      style = 'o' if marker_styles is None else marker_styles[idx]
      if not inited:
        p = plt.scatter(i, j, s=marker_size, marker=style)
        point_colors.append(p.get_facecolor()[0])
      else:
        p = plt.scatter(i,
                        j,
                        s=marker_size,
                        marker=style,
                        color=point_colors[i])
      if draw_label:
        ax.text(i,
                1.01 * j,
                s=str(j),
                ha='center',
                va='bottom',
                fontsize=fontsize)
    inited = True
  # ====== legends and tick labels ====== #
  plt.gca().set_xticks(np.arange(len(tick_labels)))
  plt.gca().set_xticklabels(tick_labels, rotation=-60, fontsize=fontsize)
  plt.legend(handlers,
             legends,
             bbox_to_anchor=(1.05, 1),
             loc=2,
             borderaxespad=0.,
             fontsize=fontsize)
  if title is not None:
    plt.suptitle(title)


def plot_gaussian_mixture(x,
                          gmm,
                          bins=80,
                          fontsize=12,
                          linewidth=2,
                          show_pdf=False,
                          show_probability=False,
                          show_components=True,
                          legend=True,
                          ax=None,
                          title=None):
  from sklearn.mixture import GaussianMixture
  from odin.utils import as_tuple, catch_warnings_ignore
  import seaborn as sns
  from scipy import stats
  ax = to_axis(ax, is_3D=False)
  n_points = int(bins * 12)
  assert gmm.means_.shape[1] == 1, "Only support plotting 1-D series GMM"
  x = x.ravel()
  order = np.argsort(gmm.means_.ravel())
  means_ = gmm.means_.ravel()[order]
  precision_ = gmm.precisions_.ravel()[order]
  colors = sns.color_palette(n_colors=gmm.n_components + 2)
  # ====== Histogram ====== #
  count, bins = plot_histogram(x=x,
                               bins=int(bins),
                               ax=ax,
                               normalize=False,
                               kde=False,
                               range_0_1=False,
                               covariance_factor=0.25,
                               centerlize=False,
                               fontsize=fontsize,
                               alpha=0.25,
                               title=title)
  ax.set_ylabel("Histogram Count", fontsize=fontsize)
  ax.set_xlim((np.min(x), np.max(x)))
  ax.set_xticks(
      np.linspace(start=np.min(x), stop=np.max(x), num=5, dtype='float32'))
  ax.set_yticks(
      np.linspace(start=np.min(count), stop=np.max(count), num=5,
                  dtype='int32'))
  # ====== GMM PDF ====== #
  x_ = np.linspace(np.min(bins), np.max(bins), n_points)
  y_ = np.exp(gmm.score_samples(x_[:, np.newaxis]))
  y_ = (y_ - np.min(y_)) / (np.max(y_) - np.min(y_)) * np.max(count)
  if show_pdf:
    ax.plot(x_,
            y_,
            color='red',
            linestyle='-',
            linewidth=linewidth * 1.2,
            alpha=0.6,
            label="GMM log-likelihood")
  # ====== GMM probability ====== #
  twinx = None
  ymax = 0.0
  if show_probability:
    if twinx is None:
      twinx = ax.twinx()
    y_ = gmm.predict_proba(x_[:, np.newaxis])
    for idx, (c, j) in enumerate(zip(colors, y_.T)):
      twinx.plot(x_,
                 j,
                 color=c,
                 linestyle='--',
                 linewidth=linewidth,
                 alpha=0.8,
                 label=r"$p_{\#%d}(x)$" % idx)
    ymax = max(ymax, np.max(y_))
  # ====== draw the each Gaussian bell ====== #
  if show_components:
    if twinx is None:
      twinx = ax.twinx()
    for idx, (c, m, p) in enumerate(zip(colors, means_, precision_)):
      with catch_warnings_ignore(Warning):
        j = stats.norm.pdf(x_, m, np.sqrt(1 / p))
      twinx.plot(x_,
                 j,
                 color=c,
                 linestyle='-',
                 linewidth=linewidth,
                 label=r"$PDF_{\#%d}$" % idx)
      # mean, top of the bell
      twinx.scatter(x_[np.argmax(j)],
                    np.max(j),
                    s=88,
                    alpha=0.8,
                    linewidth=0,
                    color=c)
      ymax = max(ymax, np.max(j))
    twinx.set_ylabel("Probability Density", fontsize=fontsize)
    twinx.grid(False)
  # set the limit for twinx
  if twinx is not None:
    twinx.set_ylim(0.0, ymax * 1.05)
  # ====== show legend ====== #
  if twinx is not None:
    twinx.yaxis.label.set_color(colors[0])
    twinx.tick_params(axis='y', colors=colors[0])
  if legend:
    ax.legend(fontsize=fontsize)
    if twinx is not None:
      twinx.legend(fontsize=fontsize)
  return ax


def plot(x, y=None, ax=None, color='b', lw=1, **kwargs):
  '''Plot the amplitude envelope of a waveform.
  '''
  from matplotlib import pyplot as plt

  ax = ax if ax is not None else plt.gca()
  if y is None:
    ax.plot(x, c=color, lw=lw, **kwargs)
  else:
    ax.plot(x, y, c=color, lw=lw, **kwargs)
  return ax


def plot_ellipses(mean, sigma, color, alpha=0.75, ax=None):
  """ Plot an ellipse in 2-D
  If the data is more than 2-D, you can use PCA before
  fitting the GMM.
  """
  import matplotlib as mpl
  from matplotlib import pyplot as plt
  # ====== prepare ====== #
  mean = mean.ravel()
  assert len(mean) == 2, "mean must be vector of size 2"
  assert sigma.shape == (2, 2), "sigma must be matrix of shape (2, 2)"
  if ax is None:
    ax = plt.gca()
  covariances = sigma**2
  # ====== create the ellipses ====== #
  v, w = np.linalg.eigh(covariances)
  u = w[0] / np.linalg.norm(w[0])
  angle = np.arctan2(u[1], u[0])
  angle = 180 * angle / np.pi  # convert to degrees
  v = 2. * np.sqrt(2.) * np.sqrt(v)
  ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
  ell.set_clip_box(ax.bbox)
  ell.set_alpha(alpha)
  ax.add_artist(ell)


def plot_indices(idx, x=None, ax=None, alpha=0.3, ymin=0., ymax=1.):
  from matplotlib import pyplot as plt

  ax = ax if ax is not None else plt.gca()

  x = range(idx.shape[0]) if x is None else x
  for i, j in zip(idx, x):
    if i:
      ax.axvline(x=j, ymin=ymin, ymax=ymax, color='r', linewidth=1, alpha=alpha)
  return ax


def plot_multiple_features(features,
                           order=None,
                           title=None,
                           fig_width=4,
                           sharex=False):
  """ Plot a series of 1D and 2D in the same scale for comparison

  Parameters
  ----------
  features: Mapping
      pytho Mapping from name (string type) to feature matrix (`numpy.ndarray`)
  order: None or list of string
      if None, the order is keys of `features` sorted in alphabet order,
      else, plot the features or subset of features based on the name
      specified in `order`
  title: None or string
      title for the figure

  Note
  ----
  delta or delta delta features should have suffix: '_d1' and '_d2'
  """
  known_order = [
      # For audio processing
      'raw',
      'stft_energy',
      'stft_energy_d1',
      'stft_energy_d2',
      'frames_energy',
      'frames_energy_d1',
      'frames_energy_d2',
      'energy',
      'energy_d1',
      'energy_d2',
      'vad',
      'sad',
      'sap',
      'sap_d1',
      'sap_d2',
      'pitch',
      'pitch_d1',
      'pitch_d2',
      'loudness',
      'loudness_d1',
      'loudness_d2',
      'f0',
      'f0_d1',
      'f0_d2',
      'spec',
      'spec_d1',
      'spec_d2',
      'mspec',
      'mspec_d1',
      'mspec_d2',
      'mfcc',
      'mfcc_d1',
      'mfcc_d2',
      'sdc',
      'qspec',
      'qspec_d1',
      'qspec_d2',
      'qmspec',
      'qmspec_d1',
      'qmspec_d2',
      'qmfcc',
      'qmfcc_d1',
      'qmfcc_d2',
      'bnf',
      'bnf_d1',
      'bnf_d2',
      'ivec',
      'ivec_d1',
      'ivec_d2',
      # For image processing
      # For video processing
  ]

  from matplotlib import pyplot as plt
  if isinstance(features, (tuple, list)):
    features = OrderedDict(features)
  if not isinstance(features, Mapping):
    raise ValueError("`features` must be mapping from name -> feature_matrix.")
  # ====== check order or create default order ====== #
  if order is not None:
    order = [str(o) for o in order]
  else:
    if isinstance(features, OrderedDict):
      order = features.keys()
    else:
      keys = sorted(features.keys(
      ) if isinstance(features, Mapping) else [k for k, v in features])
      order = []
      for name in known_order:
        if name in keys:
          order.append(name)
      # add the remain keys
      for name in keys:
        if name not in order:
          order.append(name)
  # ====== get all numpy array ====== #
  features = [(name, features[name])
              for name in order
              if name in features and isinstance(features[name], np.ndarray) and
              features[name].ndim <= 4]
  plt.figure(figsize=(int(fig_width), len(features)))
  for i, (name, X) in enumerate(features):
    X = X.astype('float32')
    plt.subplot(len(features), 1, i + 1)
    # flatten 2D features with one dim equal to 1
    if X.ndim == 2 and any(s == 1 for s in X.shape):
      X = X.ravel()
    # check valid dimension and select appropriate plot
    if X.ndim == 1:
      plt.plot(X)
      plt.xlim(0, len(X))
      plt.ylabel(name, fontsize=6)
    elif X.ndim == 2:  # transpose to frequency x time
      plot_spectrogram(X.T, title=name)
    elif X.ndim == 3:
      plt.imshow(X)
      plt.xticks(())
      plt.yticks(())
      plt.ylabel(name, fontsize=6)
    else:
      raise RuntimeError("No support for >= 3D features.")
    # auto, equal
    plt.gca().set_aspect(aspect='auto')
    # plt.axis('off')
    plt.xticks(())
    # plt.yticks(())
    plt.tick_params(axis='y', size=6, labelsize=4, color='r', pad=0, length=2)
    # add title to the first subplot
    if i == 0 and title is not None:
      plt.title(str(title), fontsize=8)
    if sharex:
      plt.subplots_adjust(hspace=0)


def plot_spectrogram(x,
                     vad=None,
                     ax=None,
                     colorbar=False,
                     linewidth=0.5,
                     vmin='auto',
                     vmax='auto',
                     title=None):
  ''' Plotting spectrogram

  Parameters
  ----------
  x : np.ndarray
      2D array
  vad : np.ndarray, list
      1D array, a red line will be draw at vad=1.
  ax : matplotlib.Axis
      create by fig.add_subplot, or plt.subplots
  colorbar : bool, 'all'
      whether adding colorbar to plot, if colorbar='all', call this
      methods after you add all subplots will create big colorbar
      for all your plots
  path : str
      if path is specified, save png image to given path

  Notes
  -----
  Make sure nrow and ncol in add_subplot is int or this error will show up
   - ValueError: The truth value of an array with more than one element is
      ambiguous. Use a.any() or a.all()

  '''
  from matplotlib import pyplot as plt
  if vmin == 'auto':
    vmin = np.min(x)
  if vmax == 'auto':
    vmax = np.max(x)

  # colormap = _cmap(x)
  # colormap = 'spectral'
  colormap = 'nipy_spectral'

  if x.ndim > 2:
    raise ValueError('No support for > 2D')
  elif x.ndim == 1:
    x = x[:, None]

  if vad is not None:
    vad = np.asarray(vad).ravel()
    if len(vad) != x.shape[1]:
      raise ValueError('Length of VAD must equal to signal length, but '
                       'length[vad]={} != length[signal]={}'.format(
                           len(vad), x.shape[1]))
    # normalize vad
    vad = np.cast[np.bool](vad)

  ax = to_axis(ax, is_3D=False)
  ax.set_aspect('equal', 'box')
  # ax.tick_params(axis='both', which='major', labelsize=6)
  ax.set_xticks([])
  ax.set_yticks([])
  # ax.axis('off')
  if title is not None:
    ax.set_ylabel(str(title) + '-' + str(x.shape), fontsize=6)
  img = ax.imshow(x,
                  cmap=colormap,
                  interpolation='kaiser',
                  alpha=0.9,
                  vmin=vmin,
                  vmax=vmax,
                  origin='lower')
  # img = ax.pcolorfast(x, cmap=colormap, alpha=0.9)
  # ====== draw vad vertical line ====== #
  if vad is not None:
    for i, j in enumerate(vad):
      if j:
        ax.axvline(x=i,
                   ymin=0,
                   ymax=1,
                   color='r',
                   linewidth=linewidth,
                   alpha=0.3)
  # plt.grid(True)
  if colorbar == 'all':
    fig = ax.get_figure()
    axes = fig.get_axes()
    fig.colorbar(img, ax=axes)
  elif colorbar:
    plt.colorbar(img, ax=ax)
  return ax


def plot_images(X, tile_shape=None, tile_spacing=None, fig=None, title=None):
  r"""
  Parameters
  ----------
  x : 2D-gray or 3D-color images, or list of (2D, 3D images)
      for color image the color channel is the first dimension
  tile_shape : tuple
      resized shape of images
  tile_spacing : tuple
      space betwen rows and columns of images
  """
  from matplotlib import pyplot as plt
  if not isinstance(X, (tuple, list)):
    X = [X]
  X = [np.asarray(x) for x in X]
  if not isinstance(title, (tuple, list)):
    title = [title]

  n = int(np.ceil(np.sqrt(len(X))))
  for i, (x, t) in enumerate(zip(X, title)):
    if x.ndim == 3 or x.ndim == 2:
      cmap = plt.cm.Greys_r
    elif x.ndim == 4:
      cmap = None
    else:
      raise ValueError('NO support for %d dimensions image!' % x.ndim)

    x = tile_raster_images(x, tile_shape, tile_spacing)
    if fig is None:
      fig = plt.figure()
    subplot = fig.add_subplot(n, n, i + 1)
    subplot.imshow(x, cmap=cmap)
    if t is not None:
      subplot.set_title(str(t), fontsize=12, fontweight='bold')
    subplot.axis('off')

  fig.tight_layout()
  return fig


def plot_images_old(x, fig=None, titles=None, show=False):
  '''
  x : 2D-gray or 3D-color images
      for color image the color channel is second dimension
  '''
  from matplotlib import pyplot as plt
  if x.ndim == 3 or x.ndim == 2:
    cmap = plt.cm.Greys_r
  elif x.ndim == 4:
    cmap = None
    shape = x.shape[2:] + (x.shape[1],)
    x = np.vstack([i.T.reshape((-1,) + shape) for i in x])
  else:
    raise ValueError('NO support for %d dimensions image!' % x.ndim)

  if x.ndim == 2:
    ncols = 1
    nrows = 1
  else:
    ncols = int(np.ceil(np.sqrt(x.shape[0])))
    nrows = int(ncols)

  if fig is None:
    fig = plt.figure()
  if titles is not None:
    if not isinstance(titles, (tuple, list)):
      titles = [titles]
    if len(titles) != x.shape[0]:
      raise ValueError('Titles must have the same length with'
                       'the number of images!')

  for i in range(ncols):
    for j in range(nrows):
      idx = i * ncols + j
      if idx < x.shape[0]:
        subplot = fig.add_subplot(nrows, ncols, idx + 1)
        subplot.imshow(x[idx], cmap=cmap)
        if titles is not None:
          subplot.set_title(titles[idx])
        subplot.axis('off')

  if show:
    # plt.tight_layout()
    plt.show(block=True)
    input('<Enter> to close the figure ...')
  else:
    return fig


def plot_hinton(matrix, max_weight=None, ax=None):
  '''
  Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
  a weight matrix):
      Positive: white
      Negative: black
  squares, and the size of each square represents the magnitude of each value.
  * Note: performance significant decrease as array size > 50*50
  Example:
      W = np.random.rand(10,10)
      hinton_plot(W)
  '''
  from matplotlib import pyplot as plt
  """Draw Hinton diagram for visualizing a weight matrix."""
  ax = ax if ax is not None else plt.gca()

  if not max_weight:
    max_weight = 2**np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

  ax.patch.set_facecolor('gray')
  ax.set_aspect('equal', 'box')
  ax.xaxis.set_major_locator(plt.NullLocator())
  ax.yaxis.set_major_locator(plt.NullLocator())

  for (x, y), w in np.ndenumerate(matrix):
    color = 'white' if w > 0 else 'black'
    size = np.sqrt(np.abs(w))
    rect = plt.Rectangle([x - size / 2, y - size / 2],
                         size,
                         size,
                         facecolor=color,
                         edgecolor=color)
    ax.add_patch(rect)

  ax.autoscale_view()
  ax.invert_yaxis()
  return ax


# ===========================================================================
# Helper methods
# ===========================================================================
def plot_show(block=True, tight_layout=False):
  from matplotlib import pyplot as plt
  if tight_layout:
    plt.tight_layout()
  plt.show(block=block)
  if not block:  # manually block
    input('<enter> to close all plots')
  plt.close('all')


# ===========================================================================
# Detection plot
# ===========================================================================
def _ppndf(cum_prob):
  """ @Original code from NIST
  The input to this function is a cumulative probability.
  The output from this function is the Normal deviate
  that corresponds to that probability.
  """
  SPLIT = 0.42
  A0 = 2.5066282388
  A1 = -18.6150006252
  A2 = 41.3911977353
  A3 = -25.4410604963
  B1 = -8.4735109309
  B2 = 23.0833674374
  B3 = -21.0622410182
  B4 = 3.1308290983
  C0 = -2.7871893113
  C1 = -2.2979647913
  C2 = 4.8501412713
  C3 = 2.3212127685
  D1 = 3.5438892476
  D2 = 1.6370678189
  # ====== preprocess ====== #
  cum_prob = np.array(cum_prob)
  eps = np.finfo(cum_prob.dtype).eps
  cum_prob = np.clip(cum_prob, eps, 1 - eps)
  adj_prob = cum_prob - 0.5
  # ====== init ====== #
  R = np.empty_like(cum_prob)
  norm_dev = np.empty_like(cum_prob)
  # ====== transform ====== #
  centerindexes = np.argwhere(np.abs(adj_prob) <= SPLIT).ravel()
  tailindexes = np.argwhere(np.abs(adj_prob) > SPLIT).ravel()
  # do centerstuff first
  R[centerindexes] = adj_prob[centerindexes] * adj_prob[centerindexes]
  norm_dev[centerindexes] = adj_prob[centerindexes] * \
      (((A3 * R[centerindexes] + A2) * R[centerindexes] + A1) * R[centerindexes] + A0)
  norm_dev[centerindexes] = norm_dev[centerindexes] /\
      ((((B4 * R[centerindexes] + B3) * R[centerindexes] + B2) * R[centerindexes] + B1) * R[centerindexes] + 1.0)
  #find left and right tails
  right = np.argwhere(cum_prob[tailindexes] > 0.5).ravel()
  left = np.argwhere(cum_prob[tailindexes] < 0.5).ravel()
  # do tail stuff
  R[tailindexes] = cum_prob[tailindexes]
  R[tailindexes[right]] = 1 - cum_prob[tailindexes[right]]
  R[tailindexes] = np.sqrt((-1.0) * np.log(R[tailindexes]))
  norm_dev[tailindexes] = ((
      (C3 * R[tailindexes] + C2) * R[tailindexes] + C1) * R[tailindexes] + C0)
  norm_dev[tailindexes] = norm_dev[tailindexes] / (
      (D2 * R[tailindexes] + D1) * R[tailindexes] + 1.0)
  # swap sign on left tail
  norm_dev[tailindexes[left]] = norm_dev[tailindexes[left]] * -1.0
  return norm_dev


def plot_detection_curve(x,
                         y,
                         curve,
                         xlims=None,
                         ylims=None,
                         ax=None,
                         labels=None,
                         legend=True,
                         title=None,
                         linewidth=1.2,
                         pointsize=8.0):
  """
  Parameters
  ----------
  x: array, or list|tuple of array
      if list or tuple of array is given, plot multiple curves at once
  y: array, or list|tuple of array
      if list or tuple of array is given, plot multiple curves at once
  curve: {'det', 'roc', 'prc'}
      det: detection error trade-off
      roc: receiver operating curve
      prc: precision-recall curve
  xlims : (xmin, xmax) in float
      for DET, `xlims` should be in [0, 1]
  ylims : (ymin, ymax) in float
      for DET, `ylims` should be in [0, 1]
  labels: {list of str}
      labels in case ploting multiple curves

  Note
  ----
  for 'det': xaxis is FPR - Pfa, and yxais is FNR - Pmiss
  for 'roc': xaxis is FPR - Pfa, and yaxis is TPR
  for 'prc': xaxis is, yaxis is
  """
  from matplotlib import pyplot as plt
  from odin import backend as K
  from odin.utils import as_tuple
  # ====== preprocessing ====== #
  if not isinstance(x, (tuple, list)):
    x = (x,)
  if not isinstance(y, (tuple, list)):
    y = (y,)
  if len(x) != len(y):
    raise ValueError(
        "Given %d series for `x`, but only get %d series for `y`." %
        (len(x), len(y)))
  if not isinstance(labels, (tuple, list)):
    labels = (labels,)
  labels = as_tuple(labels, N=len(x))
  linewidth = float(linewidth)
  # ====== const ====== #
  eps = np.finfo(x[0].dtype).eps
  xticks, xticklabels = None, None
  yticks, yticklabels = None, None
  xlabel, ylabel = None, None
  lines = []
  points = []
  # ====== check input arguments ====== #
  curve = curve.lower()
  if curve not in ('det', 'roc', 'prc'):
    raise ValueError("`curve` can only be: 'det', 'roc', or 'prc'")
  if ax is None:
    ax = plt.gca()
  # ====== select DET curve style ====== #
  if curve == 'det':
    xlabel = "False Alarm probability (in %)"
    ylabel = "Miss probability (in %)"
    # 0.00001, 0.00002,
    # , 0.99995, 0.99998, 0.99999
    xticks = np.array([
        0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
        0.9995, 0.9998, 0.9999
    ])
    xticklabels = [
        str(i)[:-2] if '.0' == str(i)[-2:] else
        (str(i) if i > 99.99 else str(i)) for i in xticks * 100
    ]
    if xlims is None:
      xlims = (max(min(np.min(i) for i in x),
                   xticks[0]), min(max(np.max(i) for i in x), xticks[-1]))
    xlims = (
        [val for i, val in enumerate(xticks) if val <= xlims[0] or i == 0][-1] +
        eps, [
            val for i, val in enumerate(xticks)
            if val >= xlims[1] or i == len(xticks) - 1
        ][0] - eps)
    if ylims is None:
      ylims = (max(min(np.min(i) for i in y),
                   xticks[0]), min(max(np.max(i) for i in y), xticks[-1]))
    ylims = (
        [val for i, val in enumerate(xticks) if val <= ylims[0] or i == 0][-1] +
        eps, [
            val for i, val in enumerate(xticks)
            if val >= ylims[1] or i == len(xticks) - 1
        ][0] - eps)
    # convert to log scale
    xticks = _ppndf(xticks)
    yticks, yticklabels = xticks, xticklabels
    xlims, ylims = _ppndf(xlims), _ppndf(ylims)
    # main line
    # TODO: add EER value later
    name_fmt = lambda name, dcf, eer: ('EER=%.2f;minDCF=%.2f' % (eer * 100, dcf * 100)) \
        if name is None else \
        ('%s (EER=%.2f;minDCF=%.2f)' % (name, eer * 100, dcf * 100))
    labels_new = []
    for count, (Pfa, Pmiss, name) in enumerate(zip(x, y, labels)):
      eer = K.metrics.compute_EER(Pfa=Pfa, Pmiss=Pmiss)
      # DCF point
      dcf, Pfa_opt, Pmiss_opt = K.metrics.compute_minDCF(Pfa=Pfa, Pmiss=Pmiss)
      Pfa_opt = _ppndf((Pfa_opt,))
      Pmiss_opt = _ppndf((Pmiss_opt,))
      points.append(((Pfa_opt, Pmiss_opt), {'s': pointsize}))
      # det curve
      Pfa = _ppndf(Pfa)
      Pmiss = _ppndf(Pmiss)
      name = name_fmt(name, eer, dcf)
      lines.append(((Pfa, Pmiss), {
          'lw': linewidth,
          'label': name,
          'linestyle': '-' if count % 2 == 0 else '-.'
      }))
      labels_new.append(name)
    labels = labels_new
  # ====== select ROC curve style ====== #
  elif curve == 'roc':
    xlabel = "False Positive probability"
    ylabel = "True Positive probability"
    xlims = (0, 1)
    ylims = (0, 1)
    # roc
    name_fmt = lambda name, auc: ('AUC=%.2f' % auc) if name is None else \
        ('%s (AUC=%.2f)' % (name, auc))
    labels_new = []
    for count, (i, j, name) in enumerate(zip(x, y, labels)):
      auc = K.metrics.compute_AUC(i, j)
      name = name_fmt(name, auc)
      lines.append([(i, j), {
          'lw': linewidth,
          'label': name,
          'linestyle': '-' if count % 2 == 0 else '-.'
      }])
      labels_new.append(name)
    labels = labels_new
    # diagonal
    lines.append([(xlims, ylims), {
        'lw': 0.8,
        'linestyle': '-.',
        'color': 'black'
    }])
  # ====== select ROC curve style ====== #
  elif curve == 'prc':
    raise NotImplementedError
  # ====== ploting ====== #
  fontsize = 9
  if xticks is not None:
    ax.set_xticks(xticks)
  if xticklabels is not None:
    ax.set_xticklabels(xticklabels, rotation=-60, fontsize=fontsize)
  if yticks is not None:
    ax.set_yticks(yticks)
  if yticklabels is not None:
    ax.set_yticklabels(yticklabels, fontsize=fontsize)
  # axes labels
  ax.set_xlabel(xlabel, fontsize=12)
  ax.set_ylabel(ylabel, fontsize=12)
  # plot all lines
  for args, kwargs in lines:
    ax.plot(*args, **kwargs)
  # plot all points
  for arg, kwargs in points:
    ax.scatter(*arg, **kwargs)
  if xlims is not None:
    ax.set_xlim(xlims)
  if ylims is not None:
    ax.set_ylim(ylims)
  ax.grid(color='black', linestyle='--', linewidth=0.4)
  if title is not None:
    ax.set_title(title, fontsize=fontsize + 2)
  # legend
  if legend and any(i is not None for i in labels):
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ===========================================================================
# Micro-control
# ===========================================================================
def plot_colorbar(colormap,
                  vmin=0,
                  vmax=1,
                  ax=None,
                  orientation='vertical',
                  tick_location=None,
                  tick_labels=None,
                  label=None):
  """

  Parameters
  ----------
  colormap : string, ColorMap type
  vmin : float
  vmax : float
  ax : {None, matplotlib.figure.Figure or matplotlib.axes.Axes}
    if `Figure` is given, show the color bar in the right hand side or
    top side of the figure based on the `orientation`
  orientation : {'vertical', 'horizontal'}
  ticks : None
  label : text label
  fig : figure instance matplotlib
  """
  import matplotlib as mpl
  from matplotlib import pyplot as plt

  if isinstance(colormap, string_types):
    cmap = mpl.cm.get_cmap(name=colormap)
  else:
    cmap = colormap
  norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

  # ====== add colorbar for the whole figure ====== #
  if ax is None or isinstance(ax, mpl.figure.Figure):
    fig = plt.gcf() if ax is None else ax
    if orientation == 'vertical':
      cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    else:
      cbar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.02])
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation=orientation)
  # ====== add colorbar for only 1 Axes ====== #
  elif isinstance(ax, mpl.axes.Axes):
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])  # no idea why we need this
    cb1 = plt.colorbar(mappable,
                       ax=ax,
                       pad=0.03 if orientation == 'vertical' else 0.1,
                       shrink=0.7,
                       aspect=25)
  # ====== no idea ====== #
  else:
    raise ValueError("No support for `ax` type: %s" % str(type(ax)))

  # ====== final configuration ====== #
  if tick_location is not None:
    cb1.set_ticks(tick_location)
  if tick_labels is not None:
    cb1.set_ticklabels(tick_labels)
  if label is not None:
    cb1.set_label(str(label))

  return cb1


# ===========================================================================
# Shortcut
# ===========================================================================
def plot_close():
  from matplotlib import pyplot as plt
  plt.close('all')


def plot_save(path='/tmp/tmp.pdf',
              figs=None,
              dpi=180,
              tight_plot=False,
              clear_all=True,
              log=False,
              transparent=False):
  """
  Parameters
  ----------
  clear_all: bool
      if True, remove all saved figures from current figure list
      in matplotlib
  """
  import matplotlib.pyplot as plt
  if tight_plot:
    plt.tight_layout()
  if os.path.exists(path) and os.path.isfile(path):
    os.remove(path)
  if figs is None:
    figs = [plt.figure(n) for n in plt.get_fignums()]
  # ====== saving PDF file ====== #
  if '.pdf' in path.lower():
    saved_path = [path]
    try:
      from matplotlib.backends.backend_pdf import PdfPages
      pp = PdfPages(path)
      for fig in figs:
        fig.savefig(pp,
                    dpi=dpi,
                    transparent=transparent,
                    format='pdf',
                    bbox_inches="tight")
      pp.close()
    except Exception as e:
      sys.stderr.write('Cannot save figures to pdf, error:%s \n' % str(e))
  # ====== saving PNG file ====== #
  else:
    saved_path = []
    path = os.path.splitext(path)
    ext = path[-1][1:].lower()
    path = path[0]
    kwargs = dict(dpi=dpi, bbox_inches="tight")
    for idx, fig in enumerate(figs):
      if len(figs) > 1:
        out_path = path + ('.%d.' % idx) + ext
      else:
        out_path = path + '.' + ext
      fig.savefig(out_path, transparent=transparent, **kwargs)
      saved_path.append(out_path)
  # ====== clean ====== #
  if log:
    sys.stderr.write('Saved figures to:%s \n' % ', '.join(saved_path))
  if clear_all:
    plt.close('all')


def plot_save_show(path,
                   figs=None,
                   dpi=180,
                   tight_plot=False,
                   clear_all=True,
                   log=True):
  plot_save(path, figs, dpi, tight_plot, clear_all, log)
  os.system('open -a /Applications/Preview.app %s' % path)
