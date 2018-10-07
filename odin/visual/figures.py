# -*- coding: utf-8 -*-
# ===========================================================================
# The waveform and spectrogram plot adapted from:
# [librosa](https://github.com/bmcfee/librosa)
# Copyright (c) 2016, librosa development team.
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, absolute_import, division

import os
import sys
import copy
import warnings
import colorsys
import itertools
from numbers import Number
from six import string_types
from six.moves import zip, range
from contextlib import contextmanager
from collections import Mapping, OrderedDict

import numpy as np
from scipy import stats

# try:
#     import seaborn # import seaborn for pretty plot
# except:
#     pass

# import matplotlib
# for name, hex in matplotlib.colors.cnames.items():
#     print(name, hex)
line_styles = ['-', '--', '-.', ':']

marker_styles = [
    "o",
    "^",
    "s",
    "p",
    "|",
    "d",
    # ",",
    # ".",
    "v",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "_",
]


def generate_random_colors(n, seed=5218, return_hex=True):
  if seed is not None:
    np.random.seed(seed)
  colors = []
  for i in range(n):
    hue = 0.05 + i / n # we want maximizing hue
    lightness = 0.4 + np.random.rand(1)[0] / 3  # lightness
    saturation = 0.5 + np.random.rand(1)[0] / 10 # saturation
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    hex = "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255),
                                       int(rgb[1] * 255),
                                       int(rgb[2] * 255))
    colors.append(rgb if not return_hex else hex)
  return colors

def generate_random_marker(n, seed=5218):
  if n > len(marker_styles):
    raise ValueError("There are %d different marker styles, but need %d" %
      (len(marker_styles), n))
  return marker_styles[:n]
  # return np.random.choice(marker_styles, size=n, replace=False)

def to_axis(ax, is_3D=False):
  """ Convert: int, tuple, None, Axes object
  to proper matplotlib Axes (2D and 3D)
  """
  from matplotlib import pyplot as plt
  # 3D plot
  if is_3D:
    from mpl_toolkits.mplot3d import Axes3D
    if ax is not None:
      assert isinstance(ax, (Axes3D, Number, tuple, list)), \
      'Axes3D must be used for 3D plot (z is given)'
      if isinstance(ax, Number):
        ax = plt.gcf().add_subplot(ax, projection='3d')
      elif isinstance(ax, (tuple, list)):
        ax = plt.gcf().add_subplot(*ax, projection='3d')
    else:
      ax = Axes3D(fig=plt.gcf())
  # 2D plot
  else:
    if isinstance(ax, Number):
      ax = plt.gcf().add_subplot(ax)
    elif isinstance(ax, (tuple, list)):
      ax = plt.gcf().add_subplot(*ax)
    elif ax is None:
      ax = plt.gca()
  return ax

def _check_arg_length(dat, n, dtype, default, converter):
  """ Shortcut for validating sequence of uniform data type """
  if dat is None:
    dat = [default] * n
  elif isinstance(dat, dtype):
    dat = [dat] * n
  else:
    assert len(dat) == n
  dat = [converter(d) for d in dat]
  return dat

# ===========================================================================
# Helper for spectrogram
# ===========================================================================
def time_ticks(locs, *args, **kwargs):  # pylint: disable=star-args
  '''Plot time-formatted axis ticks.
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
  formats = {'ms': lambda t: '{:d}ms'.format(int(1e3 * t)),
             's': '{:0.2f}s'.format,
             'm': lambda t: '{:d}:{:02d}'.format(int(t / 6e1),
                                                 int(np.mod(t, 6e1))),
             'h': lambda t: '{:d}:{:02d}:{:02d}'.format(int(t / 3.6e3),
                                                        int(np.mod(t / 6e1,
                                                                   6e1)),
                                                        int(np.mod(t, 6e1)))}

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
  '''Get a default colormap from the given data.

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
# Helpers
# From DeepLearningTutorials: http://deeplearning.net
# ===========================================================================
def resize_images(x, shape):
  from scipy.misc import imresize

  reszie_func = lambda x, shape: imresize(x, shape, interp='bilinear')
  if x.ndim == 4:
    def reszie_func(x, shape):
      # x: 3D
      # The color channel is the first dimension
      tmp = []
      for i in x:
        tmp.append(imresize(i, shape).reshape((-1,) + shape))
      return np.swapaxes(np.vstack(tmp).T, 0, 1)

  imgs = []
  for i in x:
    imgs.append(reszie_func(i, shape))
  return imgs


def tile_raster_images(X, tile_shape=None, tile_spacing=(2, 2), spacing_value=0.):
  ''' This function create tile of images

  Parameters
  ----------
  X : 3D-gray or 4D-color images
      for color images, the color channel must be the second dimension
  tile_shape : tuple
      resized shape of images
  tile_spacing : tuple
      space betwen rows and columns of images
  spacing_value : int, float
      value used for spacing

  '''
  if X.ndim == 3:
    img_shape = X.shape[1:]
  elif X.ndim == 4:
    img_shape = X.shape[2:]
  else:
    raise ValueError('Unsupport %d dimension images' % X.ndim)
  if tile_shape is None:
    tile_shape = img_shape
  if tile_spacing is None:
    tile_spacing = (2, 2)

  if img_shape != tile_shape:
    X = resize_images(X, tile_shape)
  else:
    X = [np.swapaxes(x.T, 0, 1) for x in X]

  n = len(X)
  n = int(np.ceil(np.sqrt(n)))

  # create spacing
  rows_spacing = np.zeros_like(X[0])[:tile_spacing[0], :] + spacing_value
  nothing = np.vstack((np.zeros_like(X[0]), rows_spacing))
  cols_spacing = np.zeros_like(nothing)[:, :tile_spacing[1]] + spacing_value

  # ====== Append columns ====== #
  rows = []
  for i in range(n): # each rows
    r = []
    for j in range(n): # all columns
      idx = i * n + j
      if idx < len(X):
        r.append(np.vstack((X[i * n + j], rows_spacing)))
      else:
        r.append(nothing)
      if j != n - 1:   # cols spacing
        r.append(cols_spacing)
    rows.append(np.hstack(r))
  # ====== Append rows ====== #
  img = np.vstack(rows)[:-tile_spacing[0]]
  return img


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
  nrow += 1.2 # for the title
  fig = plt.figure(figsize=(ncol, nrow), dpi=dpi)
  yield fig
  plt.suptitle(title)
  if show:
    plot_show(block=True, tight_layout=tight_layout)

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

def plot_subplotGrid(shape, loc, colspan=1, rowspan=1):
  from matplotlib import pyplot as plt
  return plt.subplot2grid(shape=shape, loc=loc, colspan=colspan, rowspan=rowspan)

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

def plot_comparison_track(Xs, legends, tick_labels,
                          line_colors=None, line_styles=None, linewidth=1.,
                          marker_size=33, marker_styles=None,
                          fontsize=10, draw_label=True, title=None):
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
                     "number of xticks' labels: %d" % (len(Xs[0], len(tick_labels))))
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
    handlers.append(
        plt.plot(X, linewidth=linewidth, **kwargs)[0])
    # points
    ax = plt.gca()
    for i, j in enumerate(X):
      style = 'o' if marker_styles is None else marker_styles[idx]
      if not inited:
        p = plt.scatter(i, j, s=marker_size, marker=style)
        point_colors.append(p.get_facecolor()[0])
      else:
        p = plt.scatter(i, j, s=marker_size, marker=style, color=point_colors[i])
      if draw_label:
        ax.text(i, 1.01 * j, s=str(j), ha='center', va='bottom',
                fontsize=fontsize)
    inited = True
  # ====== legends and tick labels ====== #
  plt.gca().set_xticks(np.arange(len(tick_labels)))
  plt.gca().set_xticklabels(tick_labels, rotation=-60, fontsize=fontsize)
  plt.legend(handlers, legends,
             bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
             fontsize=fontsize)
  if title is not None:
    plt.suptitle(title)

def plot_histogram(x, bins=80, ax=None,
                   normalize=False, range_0_1=False, kde=False, covariance_factor=None,
                   color='blue', color_kde='red', alpha=0.6, centerlize=False,
                   linewidth=1.2, fontsize=12, title=None):
  """
  x: histogram
  covariance_factor : None or float
      if float is given, smaller mean more detail
  """
  # ====== prepare ====== #
  # only 1-D
  if isinstance(x, (tuple, list)):
    x = np.array(x)
  x = x.ravel()
  ax = to_axis(ax, is_3D=False)
  # ====== get the bins ====== #
  if range_0_1:
    x = (x - np.min(x, axis=0, keepdims=True)) /\
        (np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))
  hist, hist_bins = np.histogram(x, bins=bins, density=normalize)
  width = (hist_bins[1] - hist_bins[0]) / 1.36
  ax.bar((hist_bins[:-1] + hist_bins[1:]) / 2 - width / 2, hist,
         width=width, color=color, alpha=alpha)
  # ====== centerlize the data ====== #
  min_val = np.min(hist_bins)
  max_val = np.max(hist_bins)
  if centerlize:
    ax.set_xlim((min_val - np.abs(max_val) / 2,
                 max_val + np.abs(max_val) / 2))
  # ====== kde ====== #
  if kde:
    if not normalize:
      raise ValueError("KDE plot only applicable for normalized-to-1 histogram.")
    density = stats.gaussian_kde(x)
    if isinstance(covariance_factor, Number):
      density.covariance_factor = lambda: float(covariance_factor)
      density._compute_covariance()
    if centerlize:
      xx = np.linspace(np.min(x) - np.abs(max_val) / 2,
                       np.max(x) + np.abs(max_val) / 2, 100)
    else:
      xx = np.linspace(np.min(x), np.max(x), 100)
    yy = density(xx)
    ax.plot(xx, yy,
            color=color_kde, alpha=min(1., alpha + 0.2),
            linewidth=linewidth, linestyle='-.')
  # ====== post processing ====== #
  ax.tick_params(axis='both', labelsize=fontsize)
  if title is not None:
    ax.set_title(str(title), fontsize=fontsize)
  return hist, hist_bins

def plot_histogram_layers(Xs, bins=50, ax=None,
                          normalize=False, range_0_1=False, kde=False, covariance_factor=None,
                          layer_name=None, layer_color=None,
                          legend_loc='upper center', legend_ncol=5, legend_colspace=0.4,
                          grid=True, fontsize=12, title=None):
  """
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
  layer_name = _check_arg_length(layer_name, n=num_classes,
                                 dtype=string_types, default='',
                                 converter=lambda x:str(x))
  layer_color = _check_arg_length(layer_color, n=num_classes,
                                  dtype=string_types, default='blue',
                                  converter=lambda x:str(x))
  legends = []
  for name, a, c, z, x in zip(layer_name,
                              np.linspace(0.6, 0.9, num_classes)[::-1],
                              layer_color,
                              np.linspace(0, 100, num_classes),
                              Xs):
    if range_0_1:
      x = (x - np.min(x, axis=0, keepdims=True)) /\
          (np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))
    hist, hist_bins = np.histogram(x, bins=bins, density=normalize)
    width = (hist_bins[1] - hist_bins[0]) / 1.36
    _ = ax.bar(left=(hist_bins[:-1] + hist_bins[1:]) / 2 - width / 2,
               height=hist, width=width,
               zs=z, zdir='y', color=c, ec=c, alpha=a)
    if kde:
      if not normalize:
        raise ValueError("KDE plot only applicable for normalized-to-1 histogram.")
      density = stats.gaussian_kde(x)
      if isinstance(covariance_factor, Number):
        density.covariance_factor = lambda: float(covariance_factor)
        density._compute_covariance()
      xx = np.linspace(np.min(x), np.max(x), 1000)
      yy = density(xx)
      zz = np.full_like(xx, fill_value=z)
      ax.plot(xs=xx, ys=zz, zs=yy,
              color=c, alpha=a, linewidth=1.2, linestyle='-.')
    # legend
    if len(name) > 0:
      legends.append((name, _))
  # ====== legend ====== #
  if len(legends) > 0:
    legends = ax.legend([i[1] for i in legends], [i[0] for i in legends],
      markerscale=1.5, scatterpoints=1, scatteryoffsets=[0.375, 0.5, 0.3125],
      loc=legend_loc, bbox_to_anchor=(0.5, -0.01), ncol=int(legend_ncol),
      columnspacing=float(legend_colspace), labelspacing=0.,
      fontsize=fontsize, handletextpad=0.1)
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

# ===========================================================================
# Scatter plot
# ===========================================================================
def _validate_color_marker_legends(num_samples, color, marker, legend):
  default_color = 'b'
  default_marker = '.'
  # ====== color is given create legend and different marker ====== #
  if color is None:
    color = [default_color] * num_samples
  elif isinstance(color, string_types):
    color = [color] * num_samples
  if len(color) != num_samples:
    raise ValueError("There are %d colors, but %d data points" %
                     len(color), num_samples)
  # ====== check marker ====== #
  if marker is None:
    marker = [default_marker] * num_samples
  elif isinstance(marker, string_types):
    marker = [marker] * num_samples
  elif len(marker) != num_samples:
    raise ValueError("There are %d markers, but %d data points" %
                     len(marker), num_samples)
  # ====== check legend ====== #
  color_marker = set([(i, j) for i, j in zip(color, marker)])
  if legend is None:
    legend = {x: '' for x in color_marker}
  elif isinstance(legend, dict):
    assert all(isinstance(j, string_types) and
               isinstance(i, (tuple, list)) and len(i) == 2
               for i, j in legend.items())
    if isinstance(legend, OrderedDict):
      legend = OrderedDict([(i, j)
                            for i, j in legend.items()
                            if i in color_marker])
    else:
      legend = {x: legend[x] for x in color_marker}
  elif isinstance(legend, string_types):
    legend = {x: legend for x in color_marker}
  elif isinstance(legend, (tuple, list, np.ndarray)):
    assert len(legend) >= len(color_marker)
    legend = OrderedDict([(i, str(j)) for i, j in zip(color_marker, legend)])
  else:
    raise ValueError("No support for legend value: %s" % str(legend))
  return color, marker, legend

def plot_density():
  from matplotlib import pyplot as plt
  from scipy.stats import kde

  nbins = 300
  def fit_kde(X):
    x = X[:, 0]
    y = X[:, 1]
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi.reshape(xi.shape)
    return xi, yi, zi
  plt.pcolormesh(*fit_kde(X0), cmap=plt.cm.Reds)
  plt.pcolormesh(*fit_kde(X1), cmap=plt.cm.Greens)
  plt.pcolormesh(*fit_kde(X2), cmap=plt.cm.Blues)

def plot_scatter_layers(x_y_val, ax=None,
                        layer_name=None, layer_color=None, layer_marker=None,
                        size=4.0, z_ratio=4, elev=None, azim=88,
                        ticks_off=True, grid=True, surface=True,
                        wireframe=False, wireframe_resolution=10,
                        colorbar=False, colorbar_horizontal=False,
                        legend_loc='upper center', legend_ncol=3, legend_colspace=0.4,
                        fontsize=8, title=None):
  """
  Parameter
  ---------
  z_ratio: float (default: 4)
    the amount of compression that layer in z_axis will be closer
    to each others compared to (x, y) axes
  """
  from matplotlib import pyplot as plt
  assert len(x_y_val) > 1, "Use `plot_scatter_heatmap` to plot only 1 layer"
  max_z = -np.inf
  min_z = np.inf
  for x, y, val in x_y_val:
    assert len(x) == len(y) == len(val)
    max_z = max(max_z, np.max(x), np.max(y))
    min_z = min(min_z, np.min(x), np.min(y))
  ax = to_axis(ax, is_3D=True)
  num_classes = len(x_y_val)
  # ====== preparing ====== #
  # name
  layer_name = _check_arg_length(dat=layer_name, n=num_classes,
                                 dtype=string_types, default='',
                                 converter=lambda x: str(x))
  # colormap
  layer_color = _check_arg_length(dat=layer_color, n=num_classes,
                                  dtype=string_types, default='Blues',
                                  converter=lambda x: plt.get_cmap(str(x)))
  # class marker
  layer_marker = _check_arg_length(dat=layer_marker, n=num_classes,
                                   dtype=string_types, default='o',
                                   converter=lambda x: str(x))
  # size
  size = _check_arg_length(dat=size, n=num_classes,
                           dtype=Number, default=4.0,
                           converter=lambda x: float(x))
  # ====== plotting each class ====== #
  legends = []
  for idx, (alpha, z) in enumerate(zip(np.linspace(0.05, 0.4, num_classes),
                                     np.linspace(min_z / 4, max_z / 4, num_classes))):
    x, y, val = x_y_val[idx]
    num_samples = len(x)
    z = np.full(shape=(num_samples,), fill_value=z)
    _ = ax.scatter(x, y, z, c=val, s=size[idx], marker=layer_marker[idx],
                   cmap=layer_color[idx])
    # ploting surface and wireframe
    if surface or wireframe:
      x, y = np.meshgrid(np.linspace(min(x), max(x), wireframe_resolution),
                         np.linspace(min(y), max(y), wireframe_resolution))
      z = np.full_like(x, fill_value=z[0])
      if surface:
        ax.plot_surface(X=x, Y=y, Z=z,
                        color=layer_color[idx](0.5), edgecolor='none',
                        alpha=alpha)
      if wireframe:
        ax.plot_wireframe(X=x, Y=y, Z=z, linewidth=0.8,
                          color=layer_color[idx](0.8), alpha=alpha + 0.1)
    # legend
    name = layer_name[idx]
    if len(name) > 0:
      legends.append((name, _))
    # colorbar
    if colorbar:
      cba = plt.colorbar(_, shrink=0.5, pad=0.01,
        orientation='horizontal' if colorbar_horizontal else 'vertical')
      if len(name) > 0:
        cba.set_label(name, fontsize=fontsize)
  # ====== plot the legend ====== #
  if len(legends) > 0:
    legends = ax.legend([i[1] for i in legends], [i[0] for i in legends],
      markerscale=1.5, scatterpoints=1, scatteryoffsets=[0.375, 0.5, 0.3125],
      loc=legend_loc, bbox_to_anchor=(0.5, -0.01), ncol=int(legend_ncol),
      columnspacing=float(legend_colspace), labelspacing=0.,
      fontsize=fontsize, handletextpad=0.1)
    for i, c in enumerate(layer_color):
      legends.legendHandles[i].set_color(c(.8))
  # ====== some configuration ====== #
  if ticks_off:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
  ax.grid(grid)
  if title is not None:
    ax.set_title(str(title))
  if (elev is not None or azim is not None):
    ax.view_init(elev=ax.elev if elev is None else elev,
                 azim=ax.azim if azim is None else azim)
  return ax


def plot_scatter_heatmap(x, y, val, z=None, ax=None,
                         cls_indicator=None, cls_name=None,
                         cls_color=None, cls_marker=None,
                         size=4.0, elev=None, azim=None,
                         ticks_off=True, grid=True,
                         colorbar=False, colorbar_horizontal=False,
                         legend_loc='upper center', legend_ncol=3, legend_colspace=0.4,
                         fontsize=8, title=None):
  """
  Parameters
  ----------
  x : 1D-array (num_samples,)
  y : 1D-array (num_samples,)
  z : 1D-array or None (num_samples,)
    if provided, plot in 3D

  """
  from matplotlib import pyplot as plt
  assert len(x) == len(y) == len(val)
  if z is not None:
    assert len(y) == len(z)
  is_3D_mode = False if z is None else True
  ax = to_axis(ax, is_3D=is_3D_mode)
  num_samples = len(x)
  # ====== prepare classes information ====== #
  if cls_indicator is None:
    cls_indicator = [0] * num_samples
  else:
    assert len(cls_indicator) == num_samples
    assert all(isinstance(i, Number) for i in cls_indicator), "`cls_indicator` must be integer."
  cls_indicator = [int(i) for i in cls_indicator]
  num_classes = len(set(cls_indicator))
  # class name
  cls_name = _check_arg_length(dat=cls_name, n=num_samples,
                               dtype=string_types, default='',
                               converter=lambda x:str(x))
  # colormap
  cls_color = _check_arg_length(dat=cls_color, n=num_classes,
                                dtype=string_types, default='Blues',
                                converter=lambda x: plt.get_cmap(str(x)))
  # class marker
  cls_marker = _check_arg_length(dat=cls_marker, n=num_classes,
                                 dtype=string_types, default='o',
                                 converter=lambda x: str(x))
  # size
  size = _check_arg_length(dat=size, n=num_classes,
                           dtype=Number, default=4.0,
                           converter=lambda x: float(x))
  # ====== plotting each class ====== #
  legends = []
  for idx, clz in enumerate(set(cls_indicator)):
    x_, y_, z_, val_ = [], [], [], []
    for i, c in enumerate(cls_indicator):
      if c == clz:
        x_.append(x[i])
        y_.append(y[i])
        val_.append(val[i])
        if is_3D_mode:
          z_.append(z[i])
    kwargs = {'c':val_, 's':size[idx], 'marker':cls_marker[idx],
              'cmap': cls_color[idx]}
    # plot
    if is_3D_mode:
      _ = ax.scatter(x_, y_, z_, **kwargs)
    else:
      _ = ax.scatter(x_, y_, **kwargs)
    # legend
    name = cls_name[idx]
    if len(name) > 0:
      legends.append((name, _))
    # colorbar
    if colorbar:
      cba = plt.colorbar(_, shrink=0.5, pad=0.01,
        orientation='horizontal' if colorbar_horizontal else 'vertical')
      if len(name) > 0:
        cba.set_label(name, fontsize=fontsize)
  # ====== plot the legend ====== #
  if len(legends) > 0:
    legends = ax.legend([i[1] for i in legends], [i[0] for i in legends],
      markerscale=1.5, scatterpoints=1, scatteryoffsets=[0.375, 0.5, 0.3125],
      loc=legend_loc, bbox_to_anchor=(0.5, -0.01), ncol=int(legend_ncol),
      columnspacing=float(legend_colspace), labelspacing=0.,
      fontsize=fontsize, handletextpad=0.1)
    for i, c in enumerate(cls_color):
      legends.legendHandles[i].set_color(c(.8))
  # ====== some configuration ====== #
  if ticks_off:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if is_3D_mode:
      ax.set_zticklabels([])
  ax.grid(grid)
  if title is not None:
    ax.set_title(str(title))
  if is_3D_mode and (elev is not None or azim is not None):
    ax.view_init(elev=ax.elev if elev is None else elev,
                 azim=ax.azim if azim is None else azim)
  return ax

def plot_scatter(x, y, z=None,
                 ax=None, color=None, marker=None, size=4.0,
                 elev=None, azim=None,
                 ticks_off=True, grid=True,
                 legend=None, legend_loc='upper center',
                 legend_ncol=3, legend_colspace=0.4,
                 fontsize=8, title=None):
  ''' Plot the amplitude envelope of a waveform.

  Parameters
  ----------
  x : 1D-array (num_samples,)
  y : 1D-array (num_samples,)
  z : 1D-array or None (num_samples,)
    if provided, plot in 3D
  ax : {None, int, tuple of int, Axes object) (default: None)
    if int, `ax` is the location of the subplot (e.g. `111`)
    if tuple, `ax` is tuple of location (e.g. `(1, 1, 1)`)
    if Axes object, `ax` must be `mpl_toolkits.mplot3d.Axes3D` in case `z`
    is given
  color: array (nb_samples)
      list of colors for each class, check `generate_random_colors`,
      length of color must be equal to `x` and `y`
  marker: array (nb_samples,)
      different marker for each color, default marker is '.'
  legend : dict
      mapping {[color][marker] -> name, ...}
      for example: {'r.': 'reddot', 'b^': 'bluetriangle'}
      you can control the order of legend by using OrderDict
  legend_ncol : int (default: 3)
    number of columns for displaying legends
  legend_colspace : float (default: 0.4)
    space between columns in the legend
  legend_loc : {str, int}
    ‘best’  0
    ‘upper right’ 1
    ‘upper left’  2
    ‘lower left’  3
    ‘lower right’ 4
    ‘right’ 5
    ‘center left’ 6
    ‘center right’  7
    ‘lower center’  8
    ‘upper center’  9
    ‘center’  10
  elev : {None, Number} (default: None or 30 degree)
    stores the elevation angle in the z plane, with `elev=90` is
    looking from top down.
    This can be used to rotate the axes programatically.
  azim : {None, Number} (default: None or -60 degree)
    stores the azimuth angle in the x,y plane.
    This can be used to rotate the axes programatically.
  title : {None, string} (default: None)
    specific title for the subplot
  '''
  assert len(x) == len(y)
  if z is not None:
    assert len(y) == len(z)
  is_3D_mode = False if z is None else True
  # ====== prepare ====== #
  color, marker, legend = _validate_color_marker_legends(
      len(x), color, marker, legend)
  # 3D plot
  ax = to_axis(ax, is_3D_mode)
  # ====== plotting ====== #
  # group into color-marker then plot each set
  axes = []
  legend_ = []
  # prepare legend
  legend_iter = legend.items() if isinstance(legend, OrderedDict) else\
  sorted(legend.items(), key=lambda x: x[-1])
  # plotting
  for code, name in legend_iter:
    c, m = list(code)
    x_ = [i for i, j, k in zip(x, color, marker) if j == c and k == m]
    y_ = [i for i, j, k in zip(y, color, marker) if j == c and k == m]
    if is_3D_mode:
      z_ = [i for i, j, k in zip(z, color, marker) if j == c and k == m]
      _ = ax.scatter(x_, y_, z_, color=c, s=size, marker=m)
    else:
      _ = ax.scatter(x_, y_, color=c, s=size, marker=m)
    axes.append(_)
    if len(name) > 0:
      legend_.append(name)
  # ====== plot the legend ====== #
  if len(legend_) > 0:
    legend = ax.legend(axes, legend_, markerscale=1.5,
      scatterpoints=1, scatteryoffsets=[0.375, 0.5, 0.3125],
      loc=legend_loc, bbox_to_anchor=(0.5, -0.01), ncol=int(legend_ncol),
      columnspacing=float(legend_colspace), labelspacing=0.,
      fontsize=fontsize, handletextpad=0.1)
  # ====== some configuration ====== #
  if ticks_off:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if is_3D_mode:
      ax.set_zticklabels([])
  ax.grid(grid)
  if title is not None:
    ax.set_title(str(title))
  if is_3D_mode and (elev is not None or azim is not None):
    ax.view_init(elev=ax.elev if elev is None else elev,
                 azim=ax.azim if azim is None else azim)
  return ax

def plot_text_scatter(X, text, ax=None,
                      font_weight='bold', font_size=8, font_alpha=0.8,
                      elev=None, azim=None, title=None):
  """
  Parameters
  ----------
  X : numpy.ndarray
    2-D array
  text : {tuple, list, array}
    list of the text or character for plotting at each data point
  ax : {None, int, tuple of int, Axes object) (default: None)
    if int, `ax` is the location of the subplot (e.g. `111`)
    if tuple, `ax` is tuple of location (e.g. `(1, 1, 1)`)
    if Axes object, `ax` must be `mpl_toolkits.mplot3d.Axes3D` in case `z`
    is given
  elev : {None, Number} (default: None or 30 degree)
    stores the elevation angle in the z plane, with `elev=90` is
    looking from top down.
    This can be used to rotate the axes programatically.
  azim : {None, Number} (default: None or -60 degree)
    stores the azimuth angle in the x,y plane.
    This can be used to rotate the axes programatically.
  """
  assert X.ndim == 2, \
  "Only support `X` two dimension array, but given: %s" % str(X.shape)
  if X.shape[1] == 2:
    is_3D = False
  elif X.shape[1] == 3:
    is_3D = True
  else:
    raise ValueError("No support for `X` with shape: %s" % str(X.shape))
  ax = to_axis(ax, is_3D=is_3D)
  assert len(text) == len(X), \
  "`text` length: %d is different from `X` length: %d" % (len(text), len(X))
  from matplotlib import pyplot as plt
  # ====== normalize X ====== #
  x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
  X = (X - x_min) / (x_max - x_min)
  # ====== check y ====== #
  text = [str(i) for i in text]
  labels = sorted(set(text))
  # ====== start plotting ====== #
  font_dict = {'weight': font_weight,
               'size': font_size,
               'alpha':font_alpha}
  for x, t in zip(X, text):
    if is_3D:
      plt.gca().text(x[0], x[1], x[2], t,
                     color=plt.cm.tab20((labels.index(t) + 1) / float(len(labels))),
                     fontdict=font_dict)
    else:
      plt.text(x[0], x[1], t,
               color=plt.cm.tab20((labels.index(t) + 1) / float(len(labels))),
               fontdict=font_dict)
  # ====== minor adjustment ====== #
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  if is_3D:
    ax.set_zticklabels([])
  if title is not None:
    ax.set_title(title, fontsize=font_size + 2, weight='semibold')
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
  covariances = sigma ** 2
  # ====== create the ellipses ====== #
  v, w = np.linalg.eigh(covariances)
  u = w[0] / np.linalg.norm(w[0])
  angle = np.arctan2(u[1], u[0])
  angle = 180 * angle / np.pi  # convert to degrees
  v = 2. * np.sqrt(2.) * np.sqrt(v)
  ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle,
                            color=color)
  ell.set_clip_box(ax.bbox)
  ell.set_alpha(alpha)
  ax.add_artist(ell)

def plot_indices(idx, x=None, ax=None, alpha=0.3, ymin=0., ymax=1.):
  from matplotlib import pyplot as plt

  ax = ax if ax is not None else plt.gca()

  x = range(idx.shape[0]) if x is None else x
  for i, j in zip(idx, x):
    if i: ax.axvline(x=j, ymin=ymin, ymax=ymax,
                     color='r', linewidth=1, alpha=alpha)
  return ax


def plot_multiple_features(features, order=None, title=None, fig_width=4,
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
      'stft_energy', 'stft_energy_d1', 'stft_energy_d2',
      'frames_energy', 'frames_energy_d1', 'frames_energy_d2',
      'energy', 'energy_d1', 'energy_d2',
      'vad',
      'sad',
      'sap', 'sap_d1', 'sap_d2',
      'pitch', 'pitch_d1', 'pitch_d2',
      'loudness', 'loudness_d1', 'loudness_d2',
      'f0', 'f0_d1', 'f0_d2',
      'spec', 'spec_d1', 'spec_d2',
      'mspec', 'mspec_d1', 'mspec_d2',
      'mfcc', 'mfcc_d1', 'mfcc_d2',
      'sdc',
      'qspec', 'qspec_d1', 'qspec_d2',
      'qmspec', 'qmspec_d1', 'qmspec_d2',
      'qmfcc', 'qmfcc_d1', 'qmfcc_d2',
      'bnf', 'bnf_d1', 'bnf_d2',
      'ivec', 'ivec_d1', 'ivec_d2',
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
      keys = sorted(features.keys() if isinstance(features, Mapping) else
                    [k for k, v in features])
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
              if name in features and
              isinstance(features[name], np.ndarray) and
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
    elif X.ndim == 2: # transpose to frequency x time
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
    plt.tick_params(axis='y', size=6, labelsize=4, color='r', pad=0,
                    length=2)
    # add title to the first subplot
    if i == 0 and title is not None:
      plt.title(str(title), fontsize=8)
    if sharex:
      plt.subplots_adjust(hspace=0)

def plot_spectrogram(x, vad=None, ax=None, colorbar=False,
                     linewidth=0.5, vmin='auto', vmax='auto',
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
  img = ax.imshow(x, cmap=colormap, interpolation='kaiser', alpha=0.9,
                  vmin=vmin, vmax=vmax, origin='lower')
  # img = ax.pcolorfast(x, cmap=colormap, alpha=0.9)
  # ====== draw vad vertical line ====== #
  if vad is not None:
    for i, j in enumerate(vad):
      if j: ax.axvline(x=i, ymin=0, ymax=1, color='r', linewidth=linewidth,
                       alpha=0.3)
  # plt.grid(True)
  if colorbar == 'all':
    fig = ax.get_figure()
    axes = fig.get_axes()
    fig.colorbar(img, ax=axes)
  elif colorbar:
    plt.colorbar(img, ax=ax)
  return ax

def plot_images(X, tile_shape=None, tile_spacing=None,
                fig=None, title=None):
  '''
  Parameters
  ----------
  x : 2D-gray or 3D-color images, or list of (2D, 3D images)
      for color image the color channel is second dimension
  tile_shape : tuple
      resized shape of images
  tile_spacing : tuple
      space betwen rows and columns of images
  '''
  from matplotlib import pyplot as plt
  if not isinstance(X, (tuple, list)):
    X = [X]
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
      subplot.set_title(str(t), fontsize=12)
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


def plot_Cnorm(cnorm, labels, Ptrue=[0.1, 0.5], ax=None, title=None,
               fontsize=12):
  from matplotlib import pyplot as plt
  cmap = plt.cm.Blues
  cnorm = cnorm.astype('float32')
  if not isinstance(Ptrue, (tuple, list, np.ndarray)):
    Ptrue = (Ptrue,)
  Ptrue = [float(i) for i in Ptrue]
  if len(Ptrue) != cnorm.shape[0]:
    raise ValueError("`Cnorm` was calculated for %d Ptrue values, but given only "
                     "%d values for `Ptrue`: %s" %
                     (cnorm.shape[0], len(Ptrue), str(Ptrue)))
  ax = to_axis(ax, is_3D=False)
  ax.imshow(cnorm, interpolation='nearest', cmap=cmap)
  # axis.get_figure().colorbar(im)
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(Ptrue)))
  ax.set_xticklabels(labels, rotation=-57, fontsize=fontsize)
  ax.set_yticklabels([str(i) for i in Ptrue], fontsize=fontsize)
  ax.set_ylabel('Ptrue', fontsize=fontsize)
  ax.set_xlabel('Predicted label', fontsize=fontsize)
  # center text for value of each grid
  for i, j in itertools.product(range(len(Ptrue)),
                                range(len(labels))):
    color = 'red'
    weight = 'normal'
    fs = fontsize
    text = '%.2f' % cnorm[i, j]
    plt.text(j, i, text,
             weight=weight, color=color, fontsize=fs,
             verticalalignment="center",
             horizontalalignment="center")
  # Turns off grid on the left Axis.
  ax.grid(False)
  title = "Cnorm: %.6f" % np.mean(cnorm) if title is None else \
  "%s (Cnorm: %.6f)" % (str(title), np.mean(cnorm))
  ax.set_title(title, fontsize=fontsize + 2, weight='semibold')
  # axis.tight_layout()
  return ax

def plot_confusion_matrix(cm, labels, ax=None, fontsize=12, colorbar=False,
                          title=None):
  # TODO: new style for confusion matrix (using small and big dot)
  from matplotlib import pyplot as plt
  cmap = plt.cm.Blues
  ax = to_axis(ax, is_3D=False)
  # calculate F1
  N_row = np.sum(cm, axis=-1)
  N_col = np.sum(cm, axis=0)
  TP = np.diagonal(cm)
  FP = N_col - TP
  FN = N_row - TP
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  F1 = 2 / (1 / precision + 1 / recall)
  F1[np.isnan(F1)] = 0.
  F1_mean = np.mean(F1)
  # column normalize
  nb_classes = cm.shape[0]
  cm = cm.astype('float32') / np.sum(cm, axis=1, keepdims=True)
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  # axis.get_figure().colorbar(im)
  tick_marks = np.arange(len(labels))
  ax.set_xticks(tick_marks)
  ax.set_yticks(tick_marks)
  ax.set_xticklabels(labels, rotation=-57, fontsize=fontsize)
  ax.set_yticklabels(labels, fontsize=fontsize)
  ax.set_ylabel('True label', fontsize=fontsize)
  ax.set_xlabel('Predicted label', fontsize=fontsize)
  # center text for value of each grid
  worst_index = {i: np.argmax([val if j != i else -1
                               for j, val in enumerate(row)])
                 for i, row in enumerate(cm)}
  for i, j in itertools.product(range(nb_classes),
                                range(nb_classes)):
    color = 'black'
    weight = 'normal'
    fs = fontsize
    text = '%.2f' % cm[i, j]
    if i == j: # diagonal
      color = 'magenta'
      # color = "darkgreen" if cm[i, j] <= 0.8 else 'forestgreen'
      weight = 'bold'
      fs = fontsize
      text = '%.2f\nF1:%.2f' % (cm[i, j], F1[i])
    elif j == worst_index[i]: # worst mis-classified
      color = 'red'
      weight = 'semibold'
      fs = fontsize
    plt.text(j, i, text,
             weight=weight, color=color, fontsize=fs,
             verticalalignment="center",
             horizontalalignment="center")
  # Turns off grid on the left Axis.
  ax.grid(False)
  # ====== colorbar ====== #
  if colorbar == 'all':
    fig = ax.get_figure()
    axes = fig.get_axes()
    fig.colorbar(im, ax=axes)
  elif colorbar:
    plt.colorbar(im, ax=ax)
  # ====== set title ====== #
  if title is None:
    title = ''
  title += ' (F1: %.3f)' % F1_mean
  ax.set_title(title, fontsize=fontsize + 2, weight='semibold')
  # axis.tight_layout()
  return ax

def plot_weights(x, ax=None, colormap = "Greys", colorbar=False, keep_aspect=True):
  '''
  Parameters
  ----------
  x : np.ndarray
      2D array
  ax : matplotlib.Axis
      create by fig.add_subplot, or plt.subplots
  colormap : str
      colormap alias from plt.cm.Greys = 'Greys' ('spectral')
      plt.cm.gist_heat
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

  Example
  -------
  >>> x = np.random.rand(2000, 1000)
  >>> fig = plt.figure()
  >>> ax = fig.add_subplot(2, 2, 1)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 2)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 3)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 4)
  >>> dnntoolkit.visual.plot_weights(x, ax, path='/Users/trungnt13/tmp/shit.png')
  >>> plt.show()
  '''
  from matplotlib import pyplot as plt

  if colormap is None:
    colormap = plt.cm.Greys

  if x.ndim > 2:
    raise ValueError('No support for > 2D')
  elif x.ndim == 1:
    x = x[:, None]

  ax = ax if ax is not None else plt.gca()
  if keep_aspect:
    ax.set_aspect('equal', 'box')
  # ax.tick_params(axis='both', which='major', labelsize=6)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.axis('off')
  ax.set_title(str(x.shape), fontsize=6)
  img = ax.pcolorfast(x, cmap=colormap, alpha=0.8)
  plt.grid(True)

  if colorbar == 'all':
    fig = ax.get_figure()
    axes = fig.get_axes()
    fig.colorbar(img, ax=axes)
  elif colorbar:
    plt.colorbar(img, ax=ax)
  return ax

def plot_weights3D(x, colormap = "Greys"):
  '''
  Example
  -------
  >>> # 3D shape
  >>> x = np.random.rand(32, 28, 28)
  >>> dnntoolkit.visual.plot_conv_weights(x)
  '''
  from matplotlib import pyplot as plt

  if colormap is None:
    colormap = plt.cm.Greys

  shape = x.shape
  if len(shape) == 3:
    ncols = int(np.ceil(np.sqrt(shape[0])))
    nrows = int(ncols)
  else:
    raise ValueError('This function only support 3D weights matrices')

  fig = plt.figure()
  count = 0
  for i in range(nrows):
    for j in range(ncols):
      count += 1
      # skip
      if count > shape[0]:
        continue

      ax = fig.add_subplot(nrows, ncols, count)
      # ax.set_aspect('equal', 'box')
      ax.set_xticks([])
      ax.set_yticks([])
      if i == 0 and j == 0:
        ax.set_xlabel('Width:%d' % x.shape[-1], fontsize=6)
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Height:%d' % x.shape[-2], fontsize=6)
        ax.yaxis.set_label_position('left')
      else:
        ax.axis('off')
      # image data: no idea why pcolorfast flip image vertically
      img = ax.pcolorfast(x[count - 1][::-1, :], cmap=colormap, alpha=0.9)
      # plt.grid(True)

  plt.tight_layout()
  # colorbar
  axes = fig.get_axes()
  fig.colorbar(img, ax=axes)
  return fig

def plot_weights4D(x, colormap = "Greys"):
  '''
  Example
  -------
  >>> # 3D shape
  >>> x = np.random.rand(32, 28, 28)
  >>> dnntoolkit.visual.plot_conv_weights(x)
  '''
  from matplotlib import pyplot as plt

  if colormap is None:
    colormap = plt.cm.Greys

  shape = x.shape
  if len(shape) != 4:
    raise ValueError('This function only support 4D weights matrices')

  fig = plt.figure()
  imgs = []
  for i in range(shape[0]):
    imgs.append(tile_raster_images(x[i], tile_spacing=(3, 3)))

  ncols = int(np.ceil(np.sqrt(shape[0])))
  nrows = int(ncols)

  count = 0
  for i in range(nrows):
    for j in range(ncols):
      count += 1
      # skip
      if count > shape[0]:
        continue

      ax = fig.add_subplot(nrows, ncols, count)
      ax.set_aspect('equal', 'box')
      ax.set_xticks([])
      ax.set_yticks([])
      ax.axis('off')
      # image data: no idea why pcolorfast flip image vertically
      img = ax.pcolorfast(imgs[count - 1][::-1, :], cmap=colormap, alpha=0.9)

  plt.tight_layout()
  # colorbar
  axes = fig.get_axes()
  fig.colorbar(img, ax=axes)
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
    rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                         facecolor=color, edgecolor=color)
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
  if not block: # manually block
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
  norm_dev[tailindexes] = (((C3 * R[tailindexes] + C2) * R[tailindexes] + C1) * R[tailindexes] + C0)
  norm_dev[tailindexes] = norm_dev[tailindexes] / ((D2 * R[tailindexes] + D1) * R[tailindexes] + 1.0)
  # swap sign on left tail
  norm_dev[tailindexes[left]] = norm_dev[tailindexes[left]] * -1.0
  return norm_dev

def plot_detection_curve(x, y, curve, xlims=None, ylims=None,
                         ax=None, labels=None, legend=True,
                         title=None, linewidth=1.2, pointsize=8.0):
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
    raise ValueError("Given %d series for `x`, but only get %d series for `y`."
                     % (len(x), len(y)))
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
        0.00005, 0.0001, 0.0002, 0.0005,
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
        0.1, 0.2, 0.4, 0.6, 0.8, 0.9,
        0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
        0.9995, 0.9998, 0.9999])
    xticklabels = [str(i)[:-2] if '.0' == str(i)[-2:]
               else (str(i) if i > 99.99 else str(i))
               for i in xticks * 100]
    if xlims is None:
      xlims = (max(min(np.min(i) for i in x), xticks[0]),
               min(max(np.max(i) for i in x), xticks[-1]))
    xlims = ([val for i, val in enumerate(xticks) if val <= xlims[0] or i == 0][-1] + eps,
             [val for i, val in enumerate(xticks) if val >= xlims[1] or i == len(xticks) - 1][0] - eps)
    if ylims is None:
      ylims = (max(min(np.min(i) for i in y), xticks[0]),
               min(max(np.max(i) for i in y), xticks[-1]))
    ylims = ([val for i, val in enumerate(xticks) if val <= ylims[0] or i == 0][-1] + eps,
             [val for i, val in enumerate(xticks) if val >= ylims[1] or i == len(xticks) - 1][0] - eps)
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
      points.append(((Pfa_opt, Pmiss_opt),
                     {'s': pointsize}))
      # det curve
      Pfa = _ppndf(Pfa)
      Pmiss = _ppndf(Pmiss)
      name = name_fmt(name, eer, dcf)
      lines.append(((Pfa, Pmiss),
                    {'lw': linewidth, 'label': name,
                     'linestyle': '-' if count % 2 == 0 else '-.'}))
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
      lines.append([(i, j),
                    {'lw': linewidth, 'label': name,
                     'linestyle': '-' if count % 2 == 0 else '-.'}])
      labels_new.append(name)
    labels = labels_new
    # diagonal
    lines.append([(xlims, ylims),
                  {'lw': 0.8, 'linestyle': '-.', 'color': 'black'}])
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
# Header
# ===========================================================================
def plot_close():
  from matplotlib import pyplot as plt
  plt.close('all')

def plot_save(path='/tmp/tmp.pdf', figs=None, dpi=180,
              tight_plot=False, clear_all=True, log=True):
  """
  Parameters
  ----------
  clear_all: bool
      if True, remove all saved figures from current figure list
      in matplotlib
  """
  try:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    if tight_plot:
      plt.tight_layout()
    if os.path.exists(path) and os.path.isfile(path):
      os.remove(path)
    pp = PdfPages(path)
    if figs is None:
      figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
      fig.savefig(pp, format='pdf', bbox_inches="tight")
    pp.close()
    if log:
      sys.stderr.write('Saved pdf figures to:%s \n' % str(path))
    if clear_all:
      plt.close('all')
  except Exception as e:
    sys.stderr.write('Cannot save figures to pdf, error:%s \n' % str(e))

def plot_save_show(path, figs=None, dpi=180, tight_plot=False,
                   clear_all=True, log=True):
  plot_save(path, figs, dpi, tight_plot, clear_all, log)
  os.system('open -a /Applications/Preview.app %s' % path)
