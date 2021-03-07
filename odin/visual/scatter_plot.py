from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from numbers import Number

import numpy as np
from six import string_types

from odin.utils import as_tuple
from odin.visual.plot_utils import (check_arg_length, generate_palette_colors,
                                    generate_random_colormaps,
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


def _validate_color_marker_size_legend(max_n_points,
                                       color,
                                       marker,
                                       size,
                                       text_marker=False,
                                       is_colormap=False,
                                       size_range=8,
                                       random_seed=1):
  """Return: colors, markers, sizes, legends"""
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
  # marker
  default_marker = '.'
  if isinstance(marker, string_types):
    default_marker = marker
    marker = None
  legend = [
      [None] * max_n_points,  # color
      [None] * max_n_points,  # marker
      [None] * max_n_points,  # size
  ]
  #
  create_label_map = lambda labs, default_val, fn_gen: \
      ({labs[0]: default_val}
       if len(labs) == 1 else
       {i: j for i, j in zip(labs, fn_gen(len(labs), seed=random_seed))})
  # ====== check arguments ====== #
  if color is None:
    color = [0] * max_n_points
  else:
    legend[0] = color
  #
  if marker is None:
    marker = [0] * max_n_points
  else:
    legend[1] = marker
  #
  if isinstance(size, Number):
    size = [0] * max_n_points
  elif size is None:
    size = [0] * max_n_points
  else:  # given a list of labels
    legend[2] = size
  size_range.norm = np.max(size)
  # ====== validate the length ====== #
  for name, arr in [("color", color), ("marker", marker), ("size", size)]:
    assert len(arr) == max_n_points, \
    "Given %d samples for `%s`, but require %d samples" % \
      (len(arr), name, max_n_points)
  # ====== labels set ====== #
  color_labels = np.unique(color)
  color_map = create_label_map(
      color_labels, default_color,
      generate_random_colormaps if is_colormap else generate_palette_colors)
  # generate_random_colors
  marker_labels = np.unique(marker)
  if text_marker:
    fn = lambda mrk, seed: marker_labels
  else:
    fn = generate_random_marker
  marker_map = create_label_map(marker_labels, default_marker, fn)
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
    if c is None:  # color
      name.append('')
      style.append(color_map[0])
    else:
      name.append(str(c))
      style.append(color_map[c])
    if m is None:  # marker style
      name.append('')
      style.append(marker_map[0])
    else:
      name.append(str(m))
      style.append(marker_map[m])
    if s is None:  # size
      name.append('')
      style.append(size_map[0])
    else:
      name.append(str(s))
      style.append(size_map[s])
    # name
    name = tuple(name)
    style = tuple(style)
    if name not in legend_name:
      legend_name.append(name)
      legend_style.append(style)
  #
  legend = OrderedDict([(i, j) for i, j in zip(legend_style, legend_name)])
  # ====== return ====== #
  return ([color_map[i] for i in color], [marker_map[i] for i in marker],
          [size_map[i] for i in size], legend)


def _downsample_scatter_points(x, y, z, max_n_points, *args):
  args = list(args)
  # downsample all data
  if max_n_points is not None and max_n_points < len(x):
    max_n_points = int(max_n_points)
    rand = np.random.RandomState(seed=1)
    ids = rand.permutation(len(x))[:max_n_points]
    x = np.array(x)[ids]
    y = np.array(y)[ids]
    if z is not None:
      z = np.array(z)[ids]
    args = [
        np.array(a)[ids] if isinstance(a, (tuple, list, np.ndarray)) else a
        for a in args
    ]
  return [len(x), x, y, z] + args


def _plot_scatter_points(*, x, y, z, val, color, marker, size, size_range,
                         alpha, max_n_points, cbar, cbar_horizontal,
                         cbar_nticks, cbar_ticks_rotation, cbar_title,
                         cbar_fontsize, legend_enable, legend_loc, legend_ncol,
                         legend_colspace, elev, azim, ticks_off, grid, fontsize,
                         centroids, xlabel, ylabel, title, ax, **kwargs):
  from matplotlib import pyplot as plt
  import matplotlib as mpl
  # keep the marker as its original text
  text_marker = kwargs.get('text_marker', False)
  x, y, z = _parse_scatterXYZ(x, y, z)
  assert len(x) == len(y), "Number of samples mismatch"
  if z is not None:
    assert len(y) == len(z)
  is_3D_mode = False if z is None else True
  ax = to_axis(ax, is_3D_mode)
  ### check the colormap
  if val is None:
    vmin, vmax, color_normalizer = None, None, None
    is_colormap = False
  else:
    from matplotlib.colors import LinearSegmentedColormap
    vmin = np.min(val)
    vmax = np.max(val)
    color_normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    is_colormap = True
    if is_colormap:
      assert isinstance(color, (string_types, LinearSegmentedColormap)), \
      "`colormap` can be string or instance of matplotlib Colormap, " + \
        "but given: %s" % type(color)
  if not is_colormap and isinstance(color, string_types) and color == 'bwr':
    color = 'b'
  ### perform downsample and select the styles
  max_n_points, x, y, z, color, marker, size = _downsample_scatter_points(
      x, y, z, max_n_points, color, marker, size)
  color, marker, size, legend = _validate_color_marker_size_legend(
      max_n_points,
      color,
      marker,
      size,
      text_marker=text_marker,
      is_colormap=is_colormap,
      size_range=size_range)
  ### centroid style
  centroid_style = dict(horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=fontsize + 2,
                        weight="bold",
                        bbox=dict(boxstyle="circle",
                                  facecolor="black",
                                  alpha=0.48,
                                  pad=0.,
                                  edgecolor='none'))
  ### plotting
  artist = []
  legend_name = []
  for plot_idx, (style, name) in enumerate(legend.items()):
    style = list(style)
    x_, y_, z_, val_ = [], [], [], []
    # get the right set of data points
    for i, (c, m, s) in enumerate(zip(color, marker, size)):
      if c == style[0] and m == style[1] and s == style[2]:
        x_.append(x[i])
        y_.append(y[i])
        if is_colormap:
          val_.append(val[i])
        if is_3D_mode:
          z_.append(z[i])
    # 2D or 3D plot
    if not is_3D_mode:
      z_ = None
    # colormap or normal color
    if not is_colormap:
      val_ = None
    else:
      cm = plt.cm.get_cmap(style[0])
      val_ = color_normalizer(val_)
      style[0] = cm(val_)
    # yield for plotting
    n_art = len(artist)
    yield ax, artist, x_, y_, z_, style
    # check new axis added
    assert len(artist) > n_art, \
      "Forgot adding new art object created by plotting"
    # check if ploting centroid
    if centroids:
      if is_3D_mode:
        ax.text(np.mean(x_),
                np.mean(y_),
                np.mean(z_),
                s=name[0],
                color=style[0],
                **centroid_style)
      else:
        ax.text(np.mean(x_),
                np.mean(y_),
                s=name[0],
                color=style[0],
                **centroid_style)
    # make the shortest name
    name = [i for i in name if len(i) > 0]
    short_name = []
    for i in name:
      if i not in short_name:
        short_name.append(i)
    name = ', '.join(short_name)
    if len(name) > 0:
      legend_name.append(name)
  ### at the end of the iteration, axis configuration
  if len(artist) == len(legend):
    ## colorbar (only enable when colormap is provided)
    if is_colormap and cbar:
      mappable = plt.cm.ScalarMappable(norm=color_normalizer, cmap=cm)
      mappable.set_clim(vmin, vmax)
      cba = plt.colorbar(
          mappable,
          ax=ax,
          shrink=0.99,
          pad=0.01,
          orientation='horizontal' if cbar_horizontal else 'vertical')
      if isinstance(cbar_nticks, Number):
        cbar_range = np.linspace(vmin, vmax, num=int(cbar_nticks))
        cbar_nticks = [f'{i:.2g}' for i in cbar_range]
      elif isinstance(cbar_nticks, (tuple, list, np.ndarray)):
        cbar_range = np.linspace(vmin, vmax, num=len(cbar_nticks))
        cbar_nticks = [str(i) for i in cbar_nticks]
      else:
        raise ValueError(f"No support for cbar_nticks='{cbar_nticks}'")
      cba.set_ticks(cbar_range)
      cba.set_ticklabels(cbar_nticks)
      if cbar_title is not None:
        if cbar_horizontal:  # horizontal colorbar
          cba.ax.set_xlabel(str(cbar_title), fontsize=cbar_fontsize)
        else:  # vertical colorbar
          cba.ax.set_ylabel(str(cbar_title), fontsize=cbar_fontsize)
      cba.ax.tick_params(labelsize=cbar_fontsize,
                         labelrotation=cbar_ticks_rotation)
    ## plot the legend
    if len(legend_name) > 0 and bool(legend_enable):
      markerscale = 1.5
      if isinstance(artist[0], mpl.text.Text):  # text plot special case
        for i, art in enumerate(list(artist)):
          pos = [art._x, art._y]
          if is_3D_mode:
            pos.append(art._z)
          if is_colormap:
            c = art._color
          else:
            c = art._color
          artist[i] = ax.scatter(*pos, c=c, s=0.1)
          markerscale = 25
      # sort the legends
      legend_name, artist = zip(
          *sorted(zip(legend_name, artist), key=lambda t: t[0]))
      legend = ax.legend(artist,
                         legend_name,
                         markerscale=markerscale,
                         scatterpoints=1,
                         scatteryoffsets=[0.375, 0.5, 0.3125],
                         loc=legend_loc,
                         bbox_to_anchor=(0.5, -0.01),
                         ncol=int(legend_ncol),
                         columnspacing=float(legend_colspace),
                         labelspacing=0.,
                         fontsize=fontsize,
                         handletextpad=0.1)
    ## tick configuration
    if ticks_off:
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      if is_3D_mode:
        ax.set_zticklabels([])
    ax.grid(grid)
    if xlabel is not None:
      ax.set_xlabel(str(xlabel), fontsize=fontsize - 1)
    if ylabel is not None:
      ax.set_ylabel(str(ylabel), fontsize=fontsize - 1)
    if title is not None:
      ax.set_title(str(title), fontsize=fontsize, fontweight='regular')
    if is_3D_mode and (elev is not None or azim is not None):
      ax.view_init(elev=ax.elev if elev is None else elev,
                   azim=ax.azim if azim is None else azim)


# ===========================================================================
# Main functions
# ===========================================================================
def plot_scatter(x,
                 y=None,
                 z=None,
                 val=None,
                 ax=None,
                 color='bwr',
                 marker='o',
                 size=4.0,
                 size_range=(8., 25.),
                 alpha=0.8,
                 linewidths=0.,
                 linestyle='-',
                 edgecolors=None,
                 elev=None,
                 azim=None,
                 ticks_off=True,
                 grid=True,
                 cbar=False,
                 cbar_horizontal=False,
                 cbar_nticks=10,
                 cbar_ticks_rotation=-30,
                 cbar_fontsize=10,
                 cbar_title=None,
                 legend_enable=True,
                 legend_loc='upper center',
                 legend_ncol=3,
                 legend_colspace=0.4,
                 centroids=False,
                 max_n_points=None,
                 fontsize=10,
                 xlabel=None,
                 ylabel=None,
                 title=None):
  """Generalized function for plotting scatter points colored or heatmap.

  Parameters
  ----------
  x : {1D, or 2D array} [n_samples,]
  y : {None, 1D-array} [n_samples,]
  z : {None, 1D-array} [n_samples,]
    if provided, plot in 3D
  val : 1D-array (num_samples,)
    float value for the intensity of given class
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
  centroids : Boolean. If True, annotate the labels on centroid of
    each cluster.
  xlabel, ylabel: str (optional)
    label for x-axis and y-axis
  title : {None, string} (default: None)
    specific title for the subplot
  """
  from matplotlib import pyplot as plt
  for ax, artist, x, y, z, \
    (color, marker, size) in _plot_scatter_points(**locals()):
    kwargs = dict(
        c=color,
        marker=marker,
        s=size,
        linewidths=linewidths,
        linestyle=linestyle,
        edgecolors=edgecolors,
        alpha=alpha,
    )
    if z is not None:  # 3D plot
      art = ax.scatter(x, y, z, **kwargs)
    else:  # 2D plot
      art = ax.scatter(x, y, **kwargs)
    artist.append(art)
  return ax


def plot_scatter_text(x,
                      y=None,
                      z=None,
                      val=None,
                      ax=None,
                      color='bwr',
                      marker='o',
                      weight='normal',
                      size=4.0,
                      size_range=(8., 25.),
                      alpha=0.8,
                      linewidths=0.,
                      linestyle='-',
                      edgecolors=None,
                      elev=None,
                      azim=None,
                      ticks_off=True,
                      grid=True,
                      cbar=False,
                      cbar_horizontal=False,
                      cbar_nticks=10,
                      cbar_ticks_rotation=-30,
                      cbar_title=None,
                      legend_enable=True,
                      legend_loc='upper center',
                      legend_ncol=3,
                      legend_colspace=0.4,
                      centroids=False,
                      max_n_points=None,
                      fontsize=10,
                      title=None):
  r"""
  Arguments:
    x : {1D, or 2D array} [n_samples,]
    y : {None, 1D-array} [n_samples,]
    z : {None, 1D-array} [n_samples,]
      if provided, plot in 3D
    marker : {tuple, list, array}
      list of the text or character for plotting at each data point
    weight : {'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'}.
      Font weight
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
  xlim = (np.inf, -np.inf)
  ylim = (np.inf, -np.inf)
  zlim = (np.inf, -np.inf)
  for ax, artist, x, y, z, \
    (color, marker, size) in _plot_scatter_points(text_marker=True, **locals()):
    if len(color) != len(x):
      color = [color] * len(x)
    # axes limits
    xlim = (min(xlim[0], min(x)), max(xlim[1], max(x)))
    ylim = (min(ylim[0], min(y)), max(ylim[1], max(y)))
    if z is not None:
      zlim = (min(zlim[0], min(z)), max(zlim[1], max(z)))
    # font style
    fontdict = dict(size=size, weight=weight)
    # alignment
    kwargs = dict(horizontalalignment='center', verticalalignment='center')
    if z is not None:  # 3D plot
      for a, b, c, col in zip(x, y, z, color):
        fontdict.update(color=col)
        art = ax.text(a,
                      b,
                      c,
                      s=str(marker),
                      elev=elev,
                      azim=azim,
                      fontdict=fontdict,
                      **kwargs)
    else:  # 2D plot
      for a, b, col in zip(x, y, color):
        fontdict.update(color=col)
        art = ax.text(a, b, s=str(marker), fontdict=fontdict, **kwargs)
    # store the art
    artist.append(art)
  # set the axes limits
  adjust = lambda mi, ma: (mi - 0.1 * np.abs(mi), ma + 0.1 * ma)
  ax.set_xlim(adjust(*xlim))
  ax.set_ylim(adjust(*ylim))
  if z is not None:
    ax.set_zlim(adjust(*zlim))
  return ax


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
