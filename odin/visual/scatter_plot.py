from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from numbers import Number

import numpy as np
from six import string_types

from odin.utils import as_tuple
from odin.visual.plot_utils import (check_arg_length, generate_random_colormaps,
                                    generate_random_colors,
                                    generate_random_marker, to_axis)


# ===========================================================================
# Scatter plot
# ===========================================================================
def _parse_scatterXYZ(x, y, z):
  assert x is not None, "`x` cannot be None"
  # remove all `1` dimensions
  x = np.squeeze(x)
  if y is not None:
    y = np.squeeze(y)
    assert y.ndim == 1
  if z is not None:
    z = np.square(z)
    assert z.ndim == 1
  # infer y, z from x
  if x.ndim > 2:
    x = np.reshape(x, (-1, np.prod(x.shape[1:])))
  if x.ndim == 1:
    if y is None:
      y = x
      x = np.arange(len(y))
  elif x.ndim == 2:
    if x.shape[1] == 2:
      y = x[:, 1]
      x = x[:, 0]
    elif x.shape[1] > 2:
      z = x[:, 2]
      y = x[:, 1]
      x = x[:, 0]
  return x, y, z


def _validate_color_marker_size_legend(n_samples,
                                       color,
                                       marker,
                                       size,
                                       is_colormap=False,
                                       size_range=8,
                                       random_seed=1234):
  r""" Return: colors, markers, sizes, legends """
  from odin.backend import interpolation
  from matplotlib.colors import LinearSegmentedColormap
  # check size range
  if isinstance(size, Number):
    size_range = interpolation.const(vmax=size)
  if isinstance(size_range, Number):
    size_range = interpolation.const(vmax=size_range)
  elif isinstance(size_range, interpolation.Interpolation):
    pass
  else:
    vmin, vmax = as_tuple(size_range, N=2)
    size_range = interpolation.linear(vmin=float(vmin), vmax=float(vmax))
  # check others
  default_color = 'b'
  if isinstance(color, (string_types, LinearSegmentedColormap)):
    default_color = color
    color = None
  default_marker = '.'
  if isinstance(marker, string_types):
    default_marker = marker
    marker = None
  legend = [
      [None] * n_samples,  # color
      [None] * n_samples,  # marker
      [None] * n_samples,  # size
  ]
  create_label_map = lambda labs, default_val, fn_gen: \
      ({labs[0]: default_val}
       if len(labs) == 1 else
       {i: j for i, j in zip(labs, fn_gen(len(labs), seed=random_seed))})
  # ====== check arguments ====== #
  if color is None:
    color = [0] * n_samples
  else:
    legend[0] = color
  if marker is None:
    marker = [0] * n_samples
  else:
    legend[1] = marker
  #
  if isinstance(size, Number):
    size = [0] * n_samples
  elif size is None:
    size = [0] * n_samples
  else:  # given a list of labels
    legend[2] = size
  size_range.norm = np.max(size)
  # ====== validate the length ====== #
  for name, arr in [("color", color), ("marker", marker), ("size", size)]:
    assert len(arr) == n_samples, \
    "Given %d samples for `%s`, but require %d samples" % \
      (len(arr), name, n_samples)
  # ====== labels set ====== #
  color_labels = np.unique(color)
  color_map = create_label_map(
      color_labels, default_color,
      generate_random_colormaps if is_colormap else generate_random_colors)
  #
  marker_labels = np.unique(marker)
  marker_map = create_label_map(marker_labels, default_marker,
                                generate_random_marker)
  #
  size_labels = np.unique(size)
  size_map = create_label_map(size_labels, size_range.vmax,
                              lambda n, seed: size_range(np.arange(n)).numpy())
  # ====== prepare legend ====== #
  legend_name = []
  legend_style = []
  for c, m, s in zip(*legend):
    name = []
    style = []
    if c is None:
      name.append('')
      style.append(color_map[0])
    else:
      name.append(str(c))
      style.append(color_map[c])
    if m is None:
      name.append('')
      style.append(marker_map[0])
    else:
      name.append(str(m))
      style.append(marker_map[m])
    if s is None:
      name.append('')
      style.append(size_map[0])
    else:
      name.append(str(s))
      style.append(size_map[s])
    name = tuple(name)
    style = tuple(style)
    if name not in legend_name:
      legend_name.append(name)
      legend_style.append(style)
  legend = OrderedDict([(i, j) for i, j in zip(legend_style, legend_name)])
  # ====== return ====== #
  return ([color_map[i] for i in color], [marker_map[i] for i in marker],
          [size_map[i] for i in size], legend)


def _downsample_scatter_points(x, y, z, n_samples, *args):
  args = list(args)
  # downsample all data
  if n_samples is not None and n_samples < len(x):
    n_samples = int(n_samples)
    rand = np.random.RandomState(seed=1234)
    ids = rand.permutation(len(x))[:n_samples]
    x = np.array(x)[ids]
    y = np.array(y)[ids]
    if z is not None:
      z = np.array(z)[ids]
    args = [
        np.array(a)[ids] if isinstance(a, (tuple, list, np.ndarray)) else a
        for a in args
    ]
  return [len(x), x, y, z] + args


# ===========================================================================
# Main functions
# ===========================================================================
def plot_scatter_layers(x_y_val,
                        ax=None,
                        layer_name=None,
                        layer_color=None,
                        layer_marker=None,
                        size=4.0,
                        z_ratio=4,
                        elev=None,
                        azim=88,
                        ticks_off=True,
                        grid=True,
                        surface=True,
                        wireframe=False,
                        wireframe_resolution=10,
                        colorbar=False,
                        colorbar_horizontal=False,
                        legend_loc='upper center',
                        legend_ncol=3,
                        legend_colspace=0.4,
                        fontsize=8,
                        title=None):
  r"""
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
    assert len(x) == len(y) == len(val), "Number of samples mismatch"
    max_z = max(max_z, np.max(x), np.max(y))
    min_z = min(min_z, np.min(x), np.min(y))
  ax = to_axis(ax, is_3D=True)
  num_classes = len(x_y_val)
  # ====== preparing ====== #
  # name
  layer_name = check_arg_length(dat=layer_name,
                                n=num_classes,
                                dtype=string_types,
                                default='',
                                converter=lambda x: str(x))
  # colormap
  layer_color = check_arg_length(dat=layer_color,
                                 n=num_classes,
                                 dtype=string_types,
                                 default='Blues',
                                 converter=lambda x: plt.get_cmap(str(x)))
  # class marker
  layer_marker = check_arg_length(dat=layer_marker,
                                  n=num_classes,
                                  dtype=string_types,
                                  default='o',
                                  converter=lambda x: str(x))
  # size
  size = check_arg_length(dat=size,
                          n=num_classes,
                          dtype=Number,
                          default=4.0,
                          converter=lambda x: float(x))
  # ====== plotting each class ====== #
  legends = []
  for idx, (alpha, z) in enumerate(
      zip(np.linspace(0.05, 0.4, num_classes),
          np.linspace(min_z / 4, max_z / 4, num_classes))):
    x, y, val = x_y_val[idx]
    num_samples = len(x)
    z = np.full(shape=(num_samples,), fill_value=z)
    _ = ax.scatter(x,
                   y,
                   z,
                   c=val,
                   s=size[idx],
                   marker=layer_marker[idx],
                   cmap=layer_color[idx])
    # ploting surface and wireframe
    if surface or wireframe:
      x, y = np.meshgrid(np.linspace(min(x), max(x), wireframe_resolution),
                         np.linspace(min(y), max(y), wireframe_resolution))
      z = np.full_like(x, fill_value=z[0])
      if surface:
        ax.plot_surface(X=x,
                        Y=y,
                        Z=z,
                        color=layer_color[idx](0.5),
                        edgecolor='none',
                        alpha=alpha)
      if wireframe:
        ax.plot_wireframe(X=x,
                          Y=y,
                          Z=z,
                          linewidth=0.8,
                          color=layer_color[idx](0.8),
                          alpha=alpha + 0.1)
    # legend
    name = layer_name[idx]
    if len(name) > 0:
      legends.append((name, _))
    # colorbar
    if colorbar:
      cba = plt.colorbar(
          _,
          shrink=0.5,
          pad=0.01,
          orientation='horizontal' if colorbar_horizontal else 'vertical')
      if len(name) > 0:
        cba.set_label(name, fontsize=fontsize)
  # ====== plot the legend ====== #
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


def plot_scatter_heatmap(x,
                         val,
                         y=None,
                         z=None,
                         ax=None,
                         colormap='bwr',
                         marker='o',
                         size=4.0,
                         size_range=(8., 25.),
                         alpha=0.8,
                         linewidths=0.,
                         linestyle='-',
                         facecolors=None,
                         edgecolors=None,
                         elev=None,
                         azim=None,
                         ticks_off=True,
                         grid=True,
                         colorbar=False,
                         colorbar_horizontal=False,
                         colorbar_ticks=None,
                         legend_enable=True,
                         legend_loc='upper center',
                         legend_ncol=3,
                         legend_colspace=0.4,
                         n_samples=None,
                         fontsize=8,
                         title=None):
  r"""
  Parameters
  ----------
  x : {1D, or 2D array} [n_samples,]
  y : {None, 1D-array} [n_samples,]
  z : {None, 1D-array} [n_samples,]
    if provided, plot in 3D
  val : 1D-array (num_samples,)
    float value for the intensity of given class
  """
  from matplotlib import pyplot as plt
  from matplotlib.colors import LinearSegmentedColormap
  x, y, z = _parse_scatterXYZ(x, y, z)

  assert len(x) == len(y) == len(val), "Number of samples mismatch"
  if z is not None:
    assert len(y) == len(z)
  is_3D_mode = False if z is None else True
  ax = to_axis(ax, is_3D=is_3D_mode)
  min_val = np.min(val)
  max_val = np.max(val)
  assert isinstance(colormap, (string_types, LinearSegmentedColormap)), \
  "`colormap` can be string or instance of matplotlib Colormap, but given: %s" % type(colormap)
  # ====== downsampling points ====== #
  n_samples, x, y, z, val, marker, size = \
      _downsample_scatter_points(x, y, z, n_samples, val, marker, size)
  colormap, marker, size, legend = _validate_color_marker_size_legend(
      n_samples, colormap, marker, size, size_range=size_range)
  # ====== plotting each class ====== #
  axes = []
  legend_name = []
  for idx, (style, name) in enumerate(legend.items()):
    x_, y_, z_, val_ = [], [], [], []
    # get the right set of data points
    for i, (c, m, s) in enumerate(zip(colormap, marker, size)):
      if c == style[0] and m == style[1] and s == style[2]:
        x_.append(x[i])
        y_.append(y[i])
        val_.append(val[i])
        if is_3D_mode:
          z_.append(z[i])
    # plot
    kwargs = dict(
        c=val_,
        vmin=min_val,
        vmax=max_val,
        cmap=style[0],
        marker=style[1],
        linewidths=linewidths,
        linestyle=linestyle,
        edgecolors=edgecolors,
        s=style[2],
        alpha=alpha,
    )
    if is_3D_mode:
      _ = ax.scatter(x_, y_, z_, **kwargs)
    else:
      _ = ax.scatter(x_, y_, **kwargs)
    axes.append(_)
    # make the shortest name
    name = [i for i in name if len(i) > 0]
    short_name = []
    for i in name:
      if i not in short_name:
        short_name.append(i)
    name = ', '.join(short_name)
    if len(name) > 0:
      legend_name.append(name)
    # colorbar
    if colorbar and idx == 0:
      cba = plt.colorbar(
          _,
          shrink=0.99,
          pad=0.01,
          orientation='horizontal' if colorbar_horizontal else 'vertical')
      if colorbar_ticks is not None:
        cba.set_ticks(np.linspace(min_val, max_val, num=len(colorbar_ticks)))
        cba.set_ticklabels(colorbar_ticks)
      else:
        cba.set_ticks(np.linspace(min_val, max_val, num=8 - 1))
      cba.ax.tick_params(labelsize=fontsize)
      # if len(name) > 0:
      #   cba.set_label(name, fontsize=fontsize)
  # ====== plot the legend ====== #
  if len(legend_name) > 0 and bool(legend_enable):
    legend = ax.legend(axes,
                       legend_name,
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
  # if len(legend_name) > 0:
  #   legends = ax.legend([i[1] for i in legends], [i[0] for i in legends],
  #     markerscale=1.5, scatterpoints=1, scatteryoffsets=[0.375, 0.5, 0.3125],
  #     loc=legend_loc, bbox_to_anchor=(0.5, -0.01), ncol=int(legend_ncol),
  #     columnspacing=float(legend_colspace), labelspacing=0.,
  #     fontsize=fontsize, handletextpad=0.1)
  #   for i, c in enumerate(cls_color):
  #     legends.legendHandles[i].set_color(c(.8))
  # ====== some configuration ====== #
  if ticks_off:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if is_3D_mode:
      ax.set_zticklabels([])
  ax.grid(grid)
  if title is not None:
    ax.set_title(str(title), fontsize=fontsize, fontweight='regular')
  if is_3D_mode and (elev is not None or azim is not None):
    ax.view_init(elev=ax.elev if elev is None else elev,
                 azim=ax.azim if azim is None else azim)
  return ax


def plot_scatter(x,
                 y=None,
                 z=None,
                 color='b',
                 marker='.',
                 size=4.0,
                 size_range=(8., 25.),
                 alpha=1,
                 linewidths=0.,
                 linestyle='-',
                 facecolors=None,
                 edgecolors=None,
                 elev=None,
                 azim=None,
                 ticks_off=True,
                 grid=True,
                 legend_enable=True,
                 legend_loc='upper center',
                 legend_ncol=3,
                 legend_colspace=0.4,
                 centroids=False,
                 n_samples=None,
                 fontsize=8,
                 ax=None,
                 title=None):
  r""" Plot the amplitude envelope of a waveform.

  Parameters
  ----------
  x : {1D, or 2D array} [n_samples,]
  y : {None, 1D-array} [n_samples,]
  z : {None, 1D-array} [n_samples,]
    if provided, plot in 3D

  ax : {None, int, tuple of int, Axes object) (default: None)
    if int, `ax` is the location of the subplot (e.g. `111`)
    if tuple, `ax` is tuple of location (e.g. `(1, 1, 1)`)
    if Axes object, `ax` must be `mpl_toolkits.mplot3d.Axes3D` in case `z`
    is given

  color: array [n_samples,]
      list of colors for each class, check `generate_random_colors`,
      length of color must be equal to `x` and `y`

  marker: array [n_samples,]
      different marker for each color, default marker is '.'

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

  centroids : Boolean. If True, annotate the labels on centroid of each cluster.

  title : {None, string} (default: None)
    specific title for the subplot
  """
  x, y, z = _parse_scatterXYZ(x, y, z)
  assert len(x) == len(y), "Number of samples mismatch"
  if z is not None:
    assert len(y) == len(z)
  is_3D_mode = False if z is None else True
  ax = to_axis(ax, is_3D_mode)
  # ====== perform downsample ====== #
  n_samples, x, y, z, color, marker, size = _downsample_scatter_points(
      x, y, z, n_samples, color, marker, size)
  color, marker, size, legend = _validate_color_marker_size_legend(
      n_samples, color, marker, size, size_range=size_range)
  # ====== plotting ====== #
  # group into color-marker then plot each set
  axes = []
  legend_name = []
  text_styles = dict(horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=fontsize + 2,
                     weight="bold",
                     bbox=dict(boxstyle="circle",
                               facecolor="black",
                               alpha=0.48,
                               pad=0.,
                               edgecolor='none'))

  for style, name in legend.items():
    x_, y_, z_ = [], [], []
    # get the right set of data points
    for i, (c, m, s) in enumerate(zip(color, marker, size)):
      if c == style[0] and m == style[1] and s == style[2]:
        x_.append(x[i])
        y_.append(y[i])
        if is_3D_mode:
          z_.append(z[i])
    # plotting
    if is_3D_mode:
      _ = ax.scatter(x_,
                     y_,
                     z_,
                     color=style[0],
                     marker=style[1],
                     s=style[2],
                     alpha=alpha,
                     linewidths=linewidths,
                     edgecolors=edgecolors,
                     facecolors=facecolors,
                     linestyle=linestyle)
    else:
      _ = ax.scatter(x_,
                     y_,
                     color=style[0],
                     marker=style[1],
                     s=style[2],
                     alpha=alpha,
                     linewidths=linewidths,
                     edgecolors=edgecolors,
                     facecolors=facecolors,
                     linestyle=linestyle)
      if centroids:
        ax.text(np.mean(x_),
                np.mean(y_),
                s=name[0],
                color=style[0],
                **text_styles)
    axes.append(_)
    # make the shortest name
    name = [i for i in name if len(i) > 0]
    short_name = []
    for i in name:
      if i not in short_name:
        short_name.append(i)
    name = short_name
    if len(name) > 0:
      legend_name.append(', '.join(name))
  # ====== plot the legend ====== #
  if len(legend_name) > 0 and bool(legend_enable):
    legend = ax.legend(axes,
                       legend_name,
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
  # ====== some configuration ====== #
  if ticks_off:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if is_3D_mode:
      ax.set_zticklabels([])
  ax.grid(grid)
  if title is not None:
    ax.set_title(str(title), fontsize=fontsize, fontweight='regular')
  if is_3D_mode and (elev is not None or azim is not None):
    ax.view_init(elev=ax.elev if elev is None else elev,
                 azim=ax.azim if azim is None else azim)
  return ax


def plot_scatter_text(X,
                      text,
                      ax=None,
                      font_weight='bold',
                      font_size=8,
                      font_alpha=0.8,
                      elev=None,
                      azim=None,
                      title=None):
  r"""
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
  font_dict = {'weight': font_weight, 'size': font_size, 'alpha': font_alpha}
  for x, t in zip(X, text):
    if is_3D:
      plt.gca().text(x[0],
                     x[1],
                     x[2],
                     t,
                     elev=elev,
                     azim=azim,
                     color=plt.cm.tab20(
                         (labels.index(t) + 1) / float(len(labels))),
                     fontdict=font_dict)
    else:
      plt.text(x[0],
               x[1],
               t,
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
