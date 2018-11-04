# Collection of helpers methods for plotting series or
# image and its statistics
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
def plot_series_statistics(observed, expected,
                           total_stdev=None, explained_stdev=None,
                           xscale="linear", yscale="linear",
                           xlabel="feature", ylabel="value",
                           y_cutoff=None,
                           sort_by='expected', sort_ascending=True,
                           ax=None, markersize=1, linewdith=1,
                           fontsize=8, title=None):
  """
  Parameters
  ----------

  xcale, yscale : {"linear", "log", "symlog", "logit", ...}
      text or instance in `matplotlib.scale`

  """
  import seaborn
  import matplotlib

  ax = to_axis2D(ax)
  n = len(observed)
  assert len(expected) == n
  if total_stdev is not None:
    assert len(total_stdev) == n
  if explained_stdev is not None:
    assert len(explained_stdev) == n
  # ====== color palette ====== #
  standard_palette = seaborn.color_palette('Set2', 8)
  observed_color = standard_palette[0]
  expected_palette = seaborn.light_palette(standard_palette[1], 5)
  expected_color = expected_palette[-1]
  expected_total_standard_deviations_color = expected_palette[1]
  expected_explained_standard_deviations_color = expected_palette[3]
  # ====== prepare ====== #
  indices = np.arange(n) + 1
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
  handles = []
  # ====== plotting expected and observed ====== #
  _, = ax.plot(indices, observed[sort_indices],
          label="Observations",
          color=observed_color,
          linestyle="", marker="o", zorder=2,
          markersize=markersize)
  handles.append(_)
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
        zorder=0
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
        zorder=1
    )
    _ = matplotlib.patches.Patch(
        label="Stdev(Explained)",
        color=expected_explained_standard_deviations_color
    )
    handles.append(_)
  # ====== legend ====== #
  ax.legend(handles=handles, loc="best", fontsize=fontsize)
  seaborn.despine()
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
  return ax
