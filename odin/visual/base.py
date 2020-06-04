from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict
from typing import Dict, Text

_FIGURE_LIST = defaultdict(dict)


class Visualizer(object):
  r""" Visualizer """

  def assert_figure(self, fig):
    from matplotlib import pyplot as plt
    assert isinstance(fig, plt.Figure), \
    'fig must be instance of matplotlib.Figure, but given: %s' % str(type(fig))
    return fig

  def assert_axis(self, ax):
    from matplotlib import pyplot as plt
    from odin.visual.figures import to_axis
    ax = to_axis(ax)
    assert isinstance(ax, plt.Axes), \
    'ax must be instance of matplotlib.Axes, but given: %s' % str(type(ax))
    return ax

  @property
  def figures(self) -> Dict[Text, 'Figure']:
    return _FIGURE_LIST[id(self)]

  def add_figure(self, name, fig):
    from matplotlib import pyplot as plt
    self.assert_figure(fig)
    figures = _FIGURE_LIST[id(self)]
    if name in figures:
      i = name.split('_')[-1]
      try:
        i = min(9, int(i) + 1)
      except:
        i = ''
      name = f"{name}_{i}"
    figures[name] = fig
    return self

  def save_figures(self,
                   path='/tmp/tmp.pdf',
                   dpi=100,
                   separate_files=True,
                   clear_figures=True,
                   verbose=False):
    r""" Saving all stored figures to path

    Arguments:
      path : a String.
        path to a pdf or image file, or a directory in case saving the figures
        to separated image files.
      dpi : dot-per-inch
      separate_files : save each figure in separated file
      clear_figures : remove and close all stored figures
      verbose : print out the log
    """
    from odin.utils import ctext
    from matplotlib import pyplot as plt
    # checking arguments
    if os.path.isfile(path) or '.pdf' == path[-4:].lower():
      separate_files = False
      assert '.pdf' == path[-4:].lower(), \
      "If a file is given, it must be PDF file"
    figures = _FIGURE_LIST[id(self)]
    n_figures = len(figures)
    if n_figures == 0:
      return self
    # ====== saving PDF file ====== #
    if verbose:
      print(f"Saving {n_figures} figures to: {path}")
    if not separate_files:
      if dpi is None:
        dpi = 48
      if '.pdf' not in path:
        path = path + '.pdf'
      from matplotlib.backends.backend_pdf import PdfPages
      pp = PdfPages(path)
      for key, fig in figures.items():
        try:
          fig.savefig(pp, dpi=dpi, format='pdf', bbox_inches="tight")
          if verbose:
            print(" - Saved '%s' to pdf file" % ctext(key, 'cyan'))
        except Exception as e:
          if verbose:
            print(" - Error '%s'" % ctext(key, 'cyan'))
            print("  ", e)
      pp.close()
    # ====== saving PNG file ====== #
    else:
      if dpi is None:
        dpi = 160
      if not os.path.exists(path):
        os.mkdir(path)
      assert os.path.isdir(path), "'%s' must be path to a folder" % path
      kwargs = dict(dpi=dpi, bbox_inches="tight")
      for key, fig in figures.items():
        out_path = os.path.join(path, key + '.png')
        try:
          fig.savefig(out_path, **kwargs)
          if verbose:
            print(" - Saved '%s' to %s" %
                  (ctext(key, 'cyan'), ctext(out_path, 'yellow')))
        except Exception as e:
          if verbose:
            print(" - Error '%s'" % ctext(key, 'cyan'))
            print("  ", e)
    # ====== clear figures ====== #
    if clear_figures:
      for fig in figures.values():
        plt.close(fig)
      figures.clear()
    return self
