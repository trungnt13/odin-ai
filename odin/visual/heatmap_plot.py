from __future__ import absolute_import, division, print_function

import itertools
from numbers import Number

import numpy as np
import scipy as sp

from odin.visual.plot_utils import tile_raster_images, to_axis


def plot_heatmap(data,
                 cmap="Blues",
                 ax=None,
                 xticklabels=None,
                 yticklabels=None,
                 xlabel=None,
                 ylabel=None,
                 cbar_title=None,
                 cbar=False,
                 fontsize=12,
                 gridline=0,
                 hide_spines=True,
                 annotation=None,
                 text_colors=dict(diag="black",
                                  minrow=None,
                                  mincol=None,
                                  maxrow=None,
                                  maxcol=None,
                                  other="black"),
                 title=None):
  r""" Showing heatmap matrix """
  from matplotlib import pyplot as plt
  ax = to_axis(ax, is_3D=False)
  ax.grid(False)
  fig = ax.get_figure()
  # figsize = fig.get_size_inches()
  # prepare labels
  if xticklabels is None and yticklabels is not None:
    xticklabels = ["X#%d" % i for i in range(data.shape[1])]
  if yticklabels is None and xticklabels is not None:
    yticklabels = ["Y#%d" % i for i in range(data.shape[0])]
  # Plot the heatmap
  im = ax.imshow(data,
                 interpolation='nearest',
                 cmap=cmap,
                 aspect='equal',
                 origin='upper')
  # Create colorbar
  if cbar:
    cb = plt.colorbar(im, fraction=0.02, pad=0.02)
    if cbar_title is not None:
      cb.ax.set_ylabel(cbar_title, rotation=-90, va="bottom", fontsize=fontsize)
  ## major ticks
  if xticklabels is not None and yticklabels is not None:
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(list(yticklabels), fontsize=fontsize)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=-30,
             ha="right",
             rotation_mode="anchor")
  else:  # turn-off all ticks
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
  ## axis label
  if ylabel is not None:
    ax.set_ylabel(ylabel, fontsize=fontsize + 1)
  if xlabel is not None:
    ax.set_xlabel(xlabel, fontsize=fontsize + 1)
  ## Turn spines off
  if hide_spines:
    for edge, spine in ax.spines.items():
      spine.set_visible(False)
  ## minor ticks and create white grid.
  # (if no minor ticks, the image will be cut-off)
  ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
  ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
  if gridline > 0:
    ax.grid(which="minor", color="w", linestyle='-', linewidth=gridline)
    ax.tick_params(which="minor", bottom=False, left=False)
  # set the title
  if title is not None:
    ax.set_title(str(title), fontsize=fontsize + 2, weight='semibold')
  # prepare the annotation
  if annotation is not None and annotation is not False:
    if annotation is True:
      annotation = np.array([['%.2g' % x for x in row] for row in data])
    assert annotation.shape == data.shape
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=fontsize)
    # np.log(max(2, np.mean(data.shape) - np.mean(figsize)))
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    minrow = text_colors.get('minrow', None)
    maxrow = text_colors.get('maxrow', None)
    mincol = text_colors.get('mincol', None)
    maxcol = text_colors.get('maxcol', None)
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        # basics text config
        if i == j:
          kw['weight'] = 'bold'
          color = text_colors.get('diag', 'black')
        else:
          kw['weight'] = 'normal'
          color = text_colors.get('other', 'black')
        # min, max of row
        if data[i, j] == min(data[i]) and minrow is not None:
          color = minrow
        elif data[i, j] == max(data[i]) and maxrow is not None:
          color = maxrow
        # min, max of column
        if data[i, j] == min(data[:, j]) and mincol is not None:
          color = mincol
        elif data[i, j] == max(data[:, j]) and maxcol is not None:
          color = maxcol
        # show text
        text = im.axes.text(j, i, annotation[i, j], color=color, **kw)
        texts.append(text)
  return ax


def plot_confusion_matrix(cm=None,
                          labels=None,
                          cmap="Blues",
                          ax=None,
                          fontsize=12,
                          cbar=False,
                          title=None,
                          y_true=None,
                          y_pred=None,
                          **kwargs):
  r"""
  cm : a square matrix of raw count
  kwargs : arguments for `odin.visual.plot_heatmap`
  """
  # TODO: new style for confusion matrix (using small and big dot)
  if cm is None:
    assert y_true is not None and y_pred is not None, \
      "Provide either cm explicitly or y_true and y_pred together"
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
  assert cm.shape[0] == cm.shape[1], \
    "Plot confusion matrix only applied for squared matrix"
  if labels is None:
    labels = ['#%d' % i for i in range(max(cm.shape))]
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
  # generate annotation
  annotation = np.empty(shape=(nb_classes, nb_classes), dtype=object)
  for i, j in itertools.product(range(nb_classes), range(nb_classes)):
    if i == j:  # diagonal
      text = '%.2f\nF1:%.2f' % (cm[i, j], F1[i])
    else:
      text = '%.2f' % cm[i, j]
    annotation[i, j] = text
  # plotting
  return plot_heatmap(\
      data=cm,
      xticklabels=labels,
      yticklabels=labels,
      xlabel="Prediction",
      ylabel="True",
      cmap=cmap,
      ax=ax,
      fontsize=fontsize,
      cbar=cbar,
      cbar_title="Accuracy",
      annotation=annotation,
      text_colors=dict(diag='magenta', other='black', minrow='red'),
      title='%s(F1: %.3f)' % ('' if title is None else str(title), F1_mean),
      **kwargs)


def plot_Cnorm(cnorm,
               labels,
               Ptrue=[0.1, 0.5],
               ax=None,
               title=None,
               fontsize=12):
  from matplotlib import pyplot as plt
  cmap = plt.cm.Blues
  cnorm = cnorm.astype('float32')
  if not isinstance(Ptrue, (tuple, list, np.ndarray)):
    Ptrue = (Ptrue,)
  Ptrue = [float(i) for i in Ptrue]
  if len(Ptrue) != cnorm.shape[0]:
    raise ValueError(
        "`Cnorm` was calculated for %d Ptrue values, but given only "
        "%d values for `Ptrue`: %s" % (cnorm.shape[0], len(Ptrue), str(Ptrue)))
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
  for i, j in itertools.product(range(len(Ptrue)), range(len(labels))):
    color = 'red'
    weight = 'normal'
    fs = fontsize
    text = '%.2f' % cnorm[i, j]
    plt.text(j,
             i,
             text,
             weight=weight,
             color=color,
             fontsize=fs,
             verticalalignment="center",
             horizontalalignment="center")
  # Turns off grid on the left Axis.
  ax.grid(False)
  title = "Cnorm: %.6f" % np.mean(cnorm) if title is None else \
  "%s (Cnorm: %.6f)" % (str(title), np.mean(cnorm))
  ax.set_title(title, fontsize=fontsize + 2, weight='semibold')
  # axis.tight_layout()
  return ax


def plot_weights(x, ax=None, colormap="Greys", cbar=False, keep_aspect=True):
  r'''
  Parameters
  ----------
  x : np.ndarray
      2D array
  ax : matplotlib.Axis
      create by fig.add_subplot, or plt.subplots
  colormap : str
      colormap alias from plt.cm.Greys = 'Greys' ('spectral')
      plt.cm.gist_heat
  cbar : bool, 'all'
      whether adding cbar to plot, if cbar='all', call this
      methods after you add all subplots will create big cbar
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

  if cbar == 'all':
    fig = ax.get_figure()
    axes = fig.get_axes()
    fig.colorbar(img, ax=axes)
  elif cbar:
    plt.colorbar(img, ax=ax)
  return ax


def plot_weights3D(x, colormap="Greys"):
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
  # cbar
  axes = fig.get_axes()
  fig.colorbar(img, ax=axes)
  return fig


def plot_weights4D(x, colormap="Greys"):
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


def plot_distance_heatmap(X,
                          labels,
                          lognorm=True,
                          colormap='hot',
                          ax=None,
                          legend_enable=True,
                          legend_loc='upper center',
                          legend_ncol=3,
                          legend_colspace=0.2,
                          fontsize=10,
                          cbar=True,
                          title=None):
  r"""

  Arguments:
    X : (n_samples, n_features). Coordination for scatter points
    labels : (n_samples,). List of classes index or name
  """
  from matplotlib import pyplot as plt
  from matplotlib.lines import Line2D
  from odin import backend as K
  import seaborn as sns
  # prepare data
  X = K.length_norm(X, axis=-1, epsilon=np.finfo(X.dtype).eps)
  ax = to_axis(ax)
  n_samples, n_dim = X.shape
  # processing labels
  labels = np.array(labels).ravel()
  assert labels.shape[0] == n_samples, "labels must be 1-D array."
  is_continuous = isinstance(labels[0], Number) and int(labels[0]) != labels[0]
  # float values label (normalize -1 to 1) or binary classification
  if is_continuous:
    min_val = np.min(labels)
    max_val = np.max(labels)
    labels = 2 * (labels - min_val) / (max_val - min_val) - 1
    n_labels = 2
    labels_name = {'-1': 0, '+1': 1}
  else:
    labels_name = {name: i for i, name in enumerate(np.unique(labels))}
    n_labels = len(labels_name)
    labels = np.array([labels_name[name] for name in labels])
  # ====== sorting label and X ====== #
  order_X = np.vstack(
      [x for _, x in sorted(zip(labels, X), key=lambda pair: pair[0])])
  order_label = np.vstack(
      [y for y, x in sorted(zip(labels, X), key=lambda pair: pair[0])])
  distance = sp.spatial.distance_matrix(order_X, order_X)
  if bool(lognorm):
    distance = np.log1p(distance)
  min_non_zero = np.min(distance[np.nonzero(distance)])
  distance = np.clip(distance, a_min=min_non_zero, a_max=np.max(distance))
  # ====== convert data to image ====== #
  cm = plt.get_cmap(colormap)
  distance_img = cm(distance)
  # diagonal black line (i.e. zero distance)
  # for i in range(n_samples):
  #   distance_img[i, i] = (0, 0, 0, 1)
  # labels colormap
  width = max(int(0.032 * n_samples), 8)
  if n_labels == 2:
    cm = plt.get_cmap('bwr')
    horz_bar = np.repeat(cm(order_label.T), repeats=width, axis=0)
    vert_bar = np.repeat(cm(order_label), repeats=width, axis=1)
    all_colors = np.array((cm(np.min(labels)), cm(np.max(labels))))
  else:  # use seaborn color palette here is better
    cm = [i + (1.,) for i in sns.color_palette(n_colors=n_labels)]
    c = np.stack([cm[i] for i in order_label.ravel()])
    horz_bar = np.repeat(np.expand_dims(c, 0), repeats=width, axis=0)
    vert_bar = np.repeat(np.expand_dims(c, 1), repeats=width, axis=1)
    all_colors = cm
  # image
  final_img = np.zeros(shape=(n_samples + width, n_samples + width,
                              distance_img.shape[2]),
                       dtype=distance_img.dtype)
  final_img[width:, width:] = distance_img
  final_img[:width, width:] = horz_bar
  final_img[width:, :width] = vert_bar
  assert np.sum(final_img[:width, :width]) == 0, \
  "Something wrong with my spacial coordination when writing this code!"
  # ====== plotting ====== #
  ax.imshow(final_img)
  ax.axis('off')
  # ====== legend ====== #
  if bool(legend_enable):
    legend_elements = [
        Line2D([0], [0],
               marker='o',
               color=color,
               label=name,
               linewidth=0,
               linestyle=None,
               lw=0,
               markerfacecolor=color,
               markersize=fontsize // 2)
        for color, name in zip(all_colors, labels_name.keys())
    ]
    ax.legend(handles=legend_elements,
              markerscale=1.,
              scatterpoints=1,
              scatteryoffsets=[0.375, 0.5, 0.3125],
              loc=legend_loc,
              bbox_to_anchor=(0.5, -0.01),
              ncol=int(legend_ncol),
              columnspacing=float(legend_colspace),
              labelspacing=0.,
              fontsize=fontsize - 1,
              handletextpad=0.1)
  # ====== final configurations ====== #
  if title is not None:
    ax.set_title(str(title), fontsize=fontsize)
  if cbar:
    from odin.visual import plot_colorbar
    plot_colorbar(colormap,
                  vmin=np.min(distance),
                  vmax=np.max(distance),
                  ax=ax,
                  orientation='vertical')
  return ax
