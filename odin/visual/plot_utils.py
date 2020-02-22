from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from numbers import Number

import numpy as np

line_styles = ['-', '--', '-.', ':']

# this is shuffled by hand to make sure everything ordered
# in the most intuitive way
marker_styles = [
    ".", "_", "|", "2", "s", "P", "+", "x", "^", "*", "h", "p", "d", "v", "H",
    "<", "8", ">", "X", "1", "3", "4", "D", "o"
]


def get_all_named_colors(to_hsv=False):
  from matplotlib import colors as mcolors
  colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
  # Sort colors by hue, saturation, value and name.
  if to_hsv:
    by_hsv = sorted(
        (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
        for name, color in colors.items())
    colors = OrderedDict([(name, color) for color, name in by_hsv])
  return colors


def generate_random_colors(n,
                           seed=1234,
                           lightness_value=None,
                           return_hsl=False,
                           return_hex=True):
  if seed is not None:
    rand = np.random.RandomState(seed)
  n = int(n)
  colors = []
  # we want maximizing the differences in hue
  all_hue = np.linspace(0., 0.88, num=n)
  for i, hue in enumerate(all_hue):
    saturation = 0.6 + rand.rand(1)[0] / 2.5  # saturation
    if lightness_value is None:
      lightness = 0.25 + rand.rand(1)[0] / 1.4  # lightness
    else:
      lightness = float(lightness_value)
    # select color scheme to return
    if return_hsl:
      colors.append((hue, saturation, lightness))
    else:
      rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
      colors.append(rgb if not return_hex else "#{:02x}{:02x}{:02x}".
                    format(int(rgb[0] * 255), int(rgb[1] *
                                                  255), int(rgb[2] * 255)))
  return colors


def generate_random_colormaps(n, seed=1234, bicolors=False):
  from matplotlib.colors import LinearSegmentedColormap
  color_maps = []
  interpolate_hsl = lambda h, s, l: \
      [(h, l + 0.49, s),
       (h, l, s),
       (h, l - 0.1, min(s + 0.1, 1.))]
  if bicolors:
    base_colors = generate_random_colors(n * 2,
                                         lightness_value=0.5,
                                         seed=seed,
                                         return_hsl=True)
    base_colors = list(zip(base_colors[:n], base_colors[n:]))
  else:
    base_colors = generate_random_colors(n,
                                         lightness_value=0.5,
                                         seed=seed,
                                         return_hsl=True)
  for i, c in enumerate(base_colors):
    if bicolors:
      cA, cB = c
      colors = [
          colorsys.hls_to_rgb(*i)
          for i in interpolate_hsl(*cB)[::-1] + interpolate_hsl(*cA)
      ]
    else:
      hue, saturation, lightness = c
      colors = [colorsys.hls_to_rgb(*i) for i in interpolate_hsl(*c)]
    color_maps.append(
        LinearSegmentedColormap.from_list(name='Colormap%d' % i,
                                          colors=colors,
                                          N=256,
                                          gamma=1))
  return color_maps


def generate_random_marker(n, seed=1234):
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


def check_arg_length(dat, n, dtype, default, converter):
  r""" Shortcut for validating sequence of uniform data type """
  if dat is None:
    dat = [default] * n
  elif isinstance(dat, dtype):
    dat = [dat] * n
  else:
    assert len(dat) == n
  dat = [converter(d) for d in dat]
  return dat


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


def tile_raster_images(X,
                       tile_shape=None,
                       tile_spacing=(2, 2),
                       spacing_value=0.):
  r''' This function create tile of images

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
  for i in range(n):  # each rows
    r = []
    for j in range(n):  # all columns
      idx = i * n + j
      if idx < len(X):
        r.append(np.vstack((X[i * n + j], rows_spacing)))
      else:
        r.append(nothing)
      if j != n - 1:  # cols spacing
        r.append(cols_spacing)
    rows.append(np.hstack(r))
  # ====== Append rows ====== #
  img = np.vstack(rows)[:-tile_spacing[0]]
  return img
