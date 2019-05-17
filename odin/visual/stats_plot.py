# Collection of helpers methods for plotting series or
# image and its statistics
# ax_1 = ax.twinx()
from __future__ import print_function, division, absolute_import
from numbers import Number

import numpy as np

# ===========================================================================
# Helpers
# ===========================================================================
def to_axis2D(ax):
  from matplotlib import pyplot as plt
  if isinstance(ax, Number):
    ax = plt.gcf().add_subplot(ax)
  elif isinstance(ax, (tuple, list)):
    ax = plt.gcf().add_subplot(*ax)
  elif ax is None:
    ax = plt.gca()
  return ax

# ===========================================================================
# Main plotting
# ===========================================================================
def _preprocess_series(observed, expected, total_stdev, explained_stdev):
  if observed is None and expected is None:
    raise ValueError("Either `observed` or `expected` must be not None")
  n = len(observed) if observed is not None else len(expected)
  if observed is not None:
    assert len(observed) == n
  if expected is not None:
    assert len(expected) == n
  if total_stdev is not None:
    if np.isscalar(total_stdev):
      total_stdev = np.array([total_stdev] * n)
    assert len(total_stdev) == n
  if explained_stdev is not None:
    if np.isscalar(explained_stdev):
      explained_stdev = np.array([explained_stdev] * n)
    assert len(explained_stdev) == n
  return observed, expected, total_stdev, explained_stdev

def _get_sort_indices(observed, expected,
                      sort_by, sort_ascending):
  if sort_by is not None:
    if 'observed' in str(sort_by).lower():
      sort_indices = np.argsort(observed)
    elif 'expected' in str(sort_by).lower():
      sort_indices = np.argsort(expected)
    else:
      raise ValueError("No support for `sort_by` value: %s" % str(sort_by))
    if not sort_ascending:
      sort_indices = sort_indices[::-1]
  else:
    sort_indices = slice(None)
  return sort_indices

def plot_series_statistics(observed=None, expected=None,
                           total_stdev=None, explained_stdev=None,
                           color_set='Set2',
                           xscale="linear", yscale="linear",
                           xlabel="feature", ylabel="value", y_cutoff=None,
                           sort_by='expected', sort_ascending=True, despine=True,
                           legend_enable=True, legend_title=None, legend_loc='best',
                           alpha=None, markersize=1.0, linewdith=1.2,
                           fontsize=8, ax=None, title=None,
                           return_handles=False, return_indices=False):
  """ This function can plot 2 comparable series, and the
  scale are represented in 2 y-axes (major axis - left) and
  the right one


  Parameters
  ----------
  xcale, yscale : {"linear", "log", "symlog", "logit", ...}
      text or instance in `matplotlib.scale`

  despine : bool (default: True)
      if True, remove the top and right spines from plot,
      otherwise, only remove the top spine

  Example
  -------
  >>> import numpy as np
  >>> from matplotlib import pyplot as plt
  >>> np.random.seed(5218)
  >>> x = np.random.randn(8000)
  >>> y = np.random.randn(8000)
  ...
  >>> z = np.random.rand(8000) + 3
  >>> w = np.random.rand(8000) + 3
  ...
  >>> ax, handles1 = V.plot_series_statistics(observed=x, expected=y,
  ...                                        explained_stdev=np.std(x),
  ...                                        total_stdev=np.std(y),
  ...                                        color_set='Set1',
  ...                                        legend_enable=False, legend_title="Series_1",
  ...                                        return_handles=True)
  >>> _, handles2 = V.plot_series_statistics(observed=z, expected=w,
  ...                                        explained_stdev=np.std(z),
  ...                                        total_stdev=np.std(w),
  ...                                        color_set='Set2',
  ...                                        legend_enable=False, legend_title="Series_2",
  ...                                        return_handles=True,
  ...                                        ax=ax.twinx(), alpha=0.2)
  >>> plt.legend(handles=handles1 + handles2, loc='best', fontsize=8)
  """
  import seaborn
  import matplotlib

  ax = to_axis2D(ax)
  observed, expected, total_stdev, explained_stdev = _preprocess_series(
      observed, expected, total_stdev, explained_stdev)
  # ====== color palette ====== #
  if isinstance(color_set, (tuple, list)):
    observed_color, expected_color, \
    expected_total_standard_deviations_color, \
    expected_explained_standard_deviations_color = color_set
  else:
    standard_palette = seaborn.color_palette(color_set, 8)
    observed_color = standard_palette[0]
    expected_palette = seaborn.light_palette(standard_palette[1], 5)
    expected_color = expected_palette[-1]
    expected_total_standard_deviations_color = expected_palette[1]
    expected_explained_standard_deviations_color = expected_palette[3]
  # ====== prepare ====== #
  sort_indices = _get_sort_indices(observed, expected,
                                   sort_by, sort_ascending)
  # ====== plotting expected and observed ====== #
  indices = np.arange(len(observed)
                      if observed is not None else
                      len(expected)) + 1
  handles = []
  # ====== series title ====== #
  if legend_title is not None:
    _, = ax.plot([], marker='None', linestyle='None',
                 label="$%s$" % legend_title)
    handles.append(_)
  # ====== plotting expected and observed ====== #
  if observed is not None:
    _, = ax.plot(indices, observed[sort_indices],
            label="Observations",
            color=observed_color,
            linestyle="", marker="o", zorder=2,
            markersize=markersize)
    handles.append(_)
  if expected is not None:
    _, = ax.plot(indices, expected[sort_indices],
            label="Expectation",
            color=expected_color,
            linestyle="-", marker="", zorder=3,
            linewidth=linewdith)
    handles.append(_)
  # ====== plotting stdev ====== #
  if total_stdev is not None:
    lower = expected - total_stdev
    upper = expected + total_stdev
    ax.fill_between(
        indices, lower[sort_indices], upper[sort_indices],
        color=expected_total_standard_deviations_color,
        zorder=0,
        alpha=alpha,
    )
    _ = matplotlib.patches.Patch(
        label="Stdev(Total)",
        color=expected_total_standard_deviations_color
    )
    handles.append(_)
  if explained_stdev is not None:
    lower = expected - explained_stdev
    upper = expected + explained_stdev
    ax.fill_between(
        indices, lower[sort_indices], upper[sort_indices],
        color=expected_explained_standard_deviations_color,
        zorder=1,
        alpha=alpha,
    )
    _ = matplotlib.patches.Patch(
        label="Stdev(Explained)",
        color=expected_explained_standard_deviations_color
    )
    handles.append(_)
  # ====== legend ====== #
  if legend_enable:
    ax.legend(handles=handles, loc=legend_loc, fontsize=fontsize)
  # ====== adjusting ====== #
  if bool(despine):
    seaborn.despine(top=True, right=True)
  else:
    seaborn.despine(top=True, right=False)
  ax.set_yscale(yscale, nonposy="clip")
  ax.set_ylabel('[%s]%s' % (yscale, ylabel), fontsize=fontsize)
  ax.set_xscale(xscale)
  ax.set_xlabel('[%s]%s%s' %
    (xscale, xlabel,
     ' (sorted by "%s")' % str(sort_by).lower() if sort_by is not None else ''),
      fontsize=fontsize)
  # ====== set y-cutoff ====== #
  y_min, y_max = ax.get_ylim()
  if y_cutoff is not None:
    if yscale == "linear":
      y_max = y_cutoff
    elif yscale == "log":
      y_min = y_cutoff
  ax.set_ylim(y_min, y_max)
  ax.tick_params(axis='both', labelsize=fontsize)
  # ====== title ====== #
  if title is not None:
    ax.set_title(title, fontsize=fontsize, fontweight='bold')
  ret = [ax]
  if return_handles:
    ret.append(handles)
  if return_indices:
    ret.append(sort_indices)
  return ax if len(ret) == 1 else tuple(ret)

# ===========================================================================
# Others
# ===========================================================================
def plot_relative_series(X, row_name=None, col_name=None,
                         linestyle='--', linewidth=1,
                         markerstyle='o', markersize=32,
                         grid=True, fontsize=12, text_rotation=0,
                         ax=None):
  """ First row in X will be used as baseline

  """
  import seaborn
  X = np.asarray(X)
  assert X.ndim == 2, \
  "Matrix must be provided for X, for example, each system in row, " + \
  "and different score type in column"
  n_row, n_col = X.shape
  if row_name is None:
    row_name = ["Row#%d" % i for i in range(n_row)]
  if col_name is None:
    col_name = ["Col#%d" % i for i in range(n_col)]
  assert n_row == len(row_name)
  assert n_col == len(col_name)

  colors = seaborn.color_palette(n_colors=n_col)

  # ====== normalize X to relative different to the first row ====== #
  min_ = np.min(X)
  max_ = np.max(X)

  baseline = X[:1, :]
  X = X - baseline
  y_min = 0
  y_max = np.max(X)

  ids = np.arange(n_row)

  ax = to_axis2D(ax)
  for col_idx in range(n_col):
    ax.plot(X[:, col_idx], color=colors[col_idx],
            linestyle='--', linewidth=1, alpha=0.5,
            label=col_name[col_idx])
    ax.scatter(ids, X[:, col_idx],
               color=colors[col_idx], s=markersize, marker=markerstyle,
               alpha=0.8)

  ax.set_xticks(ids)
  ax.set_xticklabels(row_name, fontsize=fontsize, rotation=text_rotation)

  ax.set_yticks(np.linspace(y_min, y_max, 5))
  ax.set_yticklabels(['%.2f' % i for i in np.linspace(min_, max_, 5)],
                     fontsize=fontsize)

  if bool(grid):
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.5, alpha=0.5)

  lg = ax.legend(fontsize=fontsize + 2)
  for line in lg.get_lines():
    line.set_linewidth(3)
    line.set_alpha(0.8)

  return ax
