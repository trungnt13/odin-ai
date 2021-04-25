from __future__ import absolute_import, division, print_function

import os
import sys
from collections import defaultdict
from typing import Dict, Text, Optional

from matplotlib import pyplot as plt

_FIGURE_LIST = defaultdict(dict)
_FIGURE_COUNT = defaultdict(lambda: defaultdict(int))


class Visualizer(object):
  r""" Visualizer """

  def assert_figure(self, fig):
    assert isinstance(fig, plt.Figure), \
      'fig must be instance of matplotlib.Figure, but given: %s' % str(
        type(fig))
    return fig

  def assert_axis(self, ax):
    from matplotlib import pyplot as plt
    from odin.visual.figures import to_axis
    ax = to_axis(ax)
    assert isinstance(ax, plt.Axes), \
      'ax must be instance of matplotlib.Axes, but given: %s' % str(type(ax))
    return ax

  @property
  def figures(self) -> Dict[Text, plt.Figure]:
    return _FIGURE_LIST[id(self)]

  def add_figure(self, name: str, fig: plt.Figure) -> 'Visualizer':
    self.assert_figure(fig)
    figures = _FIGURE_LIST[id(self)]
    count = _FIGURE_COUNT[id(self)]
    count[name] += 1
    if count[name] > 1:
      name = f"{name}_{count[name] - 1}"
    figures[name] = fig
    return self

  def save_figures(self,
                   path: str = '/tmp/tmp.pdf',
                   dpi: Optional[int] = None,
                   verbose: bool = False) -> 'Visualizer':
    """ Saving all stored figures to path

    Parameters
    ----------
    path : a String.
        path to a pdf or image file, or a directory in case saving the figures
        to separated image files.
    dpi : int, optional
        dot-per-inch
    verbose : bool
        print out the log
    """
    if dpi is None and hasattr(self, 'dpi'):
      dpi = self.dpi
    # checking arguments
    figures = _FIGURE_LIST[id(self)]
    if len(figures) == 0:
      return self
    # ====== saving PDF file ====== #
    if '.pdf' == path[-4:].lower():
      try:
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(path)
        for fig in figures.values():
          fig: plt.Figure
          fig.savefig(pp,
                      dpi=dpi,
                      transparent=False,
                      format='pdf',
                      bbox_inches="tight")
          plt.close(fig)
        pp.close()
        if verbose:
          sys.stdout.write(f"Saved figures to:{path}\n")
      except Exception as e:
        sys.stderr.write(f'Cannot save figures to pdf, error:{str(e)}\n')
    # ====== saving PNG file ====== #
    else:
      if not os.path.exists(path):
        os.makedirs(path)
      assert os.path.isdir(path), f'Invalid directory path: {path}'
      kwargs = dict(dpi=dpi, bbox_inches="tight")
      for name, fig in figures.items():
        fig: plt.Figure
        img_path = os.path.join(path, f'{name}.png')
        fig.savefig(img_path, transparent=False, **kwargs)
        plt.close(fig)
        if verbose:
          sys.stdout.write(f"Saved figures to:{img_path}\n")
    # clean
    figures.clear()
    return self
