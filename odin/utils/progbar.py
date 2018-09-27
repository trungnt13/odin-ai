# -*- coding: utf-8 -*-
##################################################################
# Example of usage:
##################################################################
from __future__ import print_function, division, absolute_import

import sys
import time
from numbers import Number
from datetime import datetime
from contextlib import contextmanager
from collections import OrderedDict, defaultdict

import numpy as np

from odin.visual.bashplot import print_bar, print_confusion
from .decorators import functionable

try:
  from tqdm import __version__ as tqdm_version
  tqdm_version = int(tqdm_version.split(".")[0])
  if tqdm_version < 4:
    raise ImportError
  from tqdm import tqdm as _tqdm
  from tqdm._utils import _environ_cols_wrapper
except ImportError:
  sys.stderr.write("[ERROR] Cannot import `tqdm` version >= 4.\n")
  exit()

try:
  import colorama
  colorama.init()
  from colorama import Fore as _Fore
  _RED = _Fore.RED
  _YELLOW = _Fore.YELLOW
  _CYAN = _Fore.CYAN
  _MAGENTA = _Fore.MAGENTA
  _RESET = _Fore.RESET
except ImportError:
  _RED, _YELLOW, _CYAN, _MAGENTA, _RESET = '', '', '', '', ''

_NUMBERS_CH = {
    ord('0'): 0,
    ord('1'): 1,
    ord('2'): 2,
    ord('3'): 3,
    ord('4'): 4,
    ord('5'): 5,
    ord('6'): 6,
    ord('7'): 7,
    ord('8'): 8,
    ord('9'): 9,
}

# ===========================================================================
# Helper
# ===========================================================================
_LAST_UPDATED_PROG = [None]

def add_notification(msg):
  msg = _CYAN + "[%s]Notification:" % \
      datetime.now().strftime('%d/%b-%H:%M:%S') + _RESET + msg + ''
  _tqdm.write(msg)

# ===========================================================================
# Progressbar
# ===========================================================================
class Progbar(object):

  """ Original progress bar is just failed illusion of time and
  estimation.

  Parameters
  ----------
  target: int
      total number of steps expected
  interval: float
      Minimum progress display update interval, in seconds.
  keep: bool
      whether to keep the progress bar when the epoch finished
  print_report: bool
      print updated report along with the progress bar for each update
  print_summary: bool
      print epoch summary after each epoch
  count_func: call-able
      a function takes the returned batch and return an integer for upating
      progress.
  report_func: call-able
      a function takes the returned batch and a collection of pair
      (key, value) for constructing the report.
  name: str or None
      specific name for the progress bar

  Examples
  --------
  >>> import numpy as np
  >>> from odin.utils import Progbar
  >>> x = list(range(10))
  >>> for i in Progbar(target=x):
  ...     pass

  Note
  ----
  Some special case:
      * any report key contain "confusionmatrix" will be printed out using
      `print_confusion`
      * any report key
  """
  FP = sys.stderr

  def __init__(self, target, interval=0.08, keep=False,
               print_report=True, print_summary=False,
               count_func=None, report_func=None, progress_func=None,
               name=None):
    self.__pb = None # tqdm object
    if isinstance(target, Number):
      self.target = int(target)
      self.__iter_obj = None
    elif hasattr(target, '__len__'):
      self.target = len(target)
      self.__iter_obj = target
    else:
      raise ValueError("Unsupport for `target` type: %s" %
                       str(target.__class__))

    self._seen_so_far = defaultdict(int) # mapping: epoch_idx -> seen_so_far

    n = len(str(self.target))
    self._counter_fmt = '(%%%dd/%%%dd)' % (n, n)

    if name is None:
      name = "Progress-%s" % datetime.utcnow()
    self._name = name
    # ====== flags ====== #
    self.__interval = float(interval)
    self.__keep = keep
    self.print_progress = True
    self.print_report = print_report
    self.print_summary = print_summary
    # ====== for history ====== #
    self._report = OrderedDict()
    self._last_report = None
    self._last_print_time = None
    self._epoch_summarizer = {}
    # ====== recording history ====== #
    # dictonary: {epoch_id: {key: [value1, value2, ...]}}
    self._epoch_hist = defaultdict(lambda: defaultdict(list))
    self._epoch_summary = defaultdict(dict)
    self._epoch_idx = 0
    self._epoch_start_time = None
    # ====== iter information ====== #
    if self.__iter_obj is None and \
    (count_func is not None or report_func is not None):
      raise RuntimeError("`count_func` and `report_func` can only be used "
                         "when `target` is an iterator with specific length.")
    #
    if count_func is not None:
      if not hasattr(count_func, '__call__'):
        raise ValueError("`count_func` must be call-able or None.")
      self.__count_func = count_func
    else:
      self.__count_func = lambda x: len(x)
    #
    if report_func is not None:
      if not hasattr(report_func, '__call__'):
        raise ValueError("`report_func` must be call-able or None.")
      self.__report_func = report_func
    else:
      self.__report_func = lambda x: None
    # ====== check progress function ====== #
    if progress_func is not None:
      if not hasattr(progress_func, '__call__'):
        raise ValueError("`progress_func` must be call-able or None.")
    self._progress_func = progress_func
    # ====== other ====== #
    self._labels = None # labels for printing the confusion matrix

  # ==================== History management ==================== #
  def __getitem__(self, key):
    return self._report.__getitem__(key)

  def __setitem__(self, key, val):
    self._epoch_hist[self.epoch_idx][key].append(val)
    return self._report.__setitem__(key, val)

  def __delitem__(self, key):
    return self._report.__delitem__(key)

  def __iter__(self):
    if self.__iter_obj is None:
      raise RuntimeError("This Progbar cannot be iterated, "
                         "the set `target` must be iterable.")
    for X in self.__iter_obj:
      count = self.__count_func(X)
      report = self.__report_func(X)
      if report is not None:
        for key, val in report:
          self[key] = val
      self.add(int(count))
      yield X
    del self.__iter_obj
    del self.__count_func
    del self.__report_func

  # ==================== screen control ==================== #
  @property
  def epoch_idx(self):
    return self._epoch_idx

  @property
  def nb_epoch(self):
    return self._epoch_idx + 1

  @property
  def name(self):
    return self._name

  @property
  def labels(self):
    """ Special labels for printing the confusion matrix. """
    return self._labels

  @property
  def history(self):
    """ Return history recording all add item (timestamp, key, value)
    to this progress

    Return
    ------
    dictonary: {epoch_id: {key: [value1, value2, ...]}}
    """
    return self._epoch_hist

  def get_report(self, epoch=-1, key=None):
    if epoch < 0:
      epoch = self.nb_epoch + epoch - 1
    return self._epoch_hist[epoch] if key is None else \
    self._epoch_hist[epoch][key]

  def set_summarizer(self, key, fn):
    """ Epoch summarizer is a function, searching in the
    report for given key, and summarize all the stored values
    of each epoch into a readable format

    i.e. the input arguments is a list of stored epoch report,
    the output is a string.
    """
    if not hasattr(fn, '__call__'):
      raise ValueError('`fn` must be call-able.')
    key = str(key)
    self._epoch_summarizer[key] = functionable(func=fn)
    return self

  def set_name(self, name):
    self._name = str(name)
    return self

  def set_labels(self, labels):
    if labels is not None:
      self._labels = tuple([str(l) for l in labels])
    return self

  def _formatted_report(self, report_dict, margin='', inc_name=True):
    """ Convert a dictionary of key -> value to well formatted string."""
    if inc_name:
      text = _MAGENTA + "\t%s" % self.name + _RESET + '\n'
    else:
      text = ""
    report_dict = sorted(report_dict.items(), key=lambda x: str(x[0]))
    for i, (key, value) in enumerate(report_dict):
      # ====== check value of key and value ====== #
      key = margin + str(key).replace('\n', ' ')
      # ====== special cases ====== #
      if "confusionmatrix" in key.lower() or \
      "confusion_matrix" in key.lower() or \
      "confusion-matrix" in key.lower() or \
      "confusion matrix" in key.lower():
        value = print_confusion(value, labels=self.labels,
                                inc_stats=True)
      # just print out string representation
      else:
        value = str(value)
      # ====== multiple lines or not ====== #
      if '\n' in value:
        text += _YELLOW + key + _RESET + ":\n"
        for line in value.split('\n'):
          text += margin + ' ' + line + '\n'
      else:
        text += _YELLOW + key + _RESET + ": " + value + "\n"
    return text[:-1]

  @property
  def progress_bar(self):
    if self.__pb is None:
      it = range(self.target)
      self.__pb = _tqdm(iterable=it,
                    desc="Epoch%s" % str(self.epoch_idx),
                    leave=self.__keep, total=self.target,
                    file=Progbar.FP, unit='obj',
                    mininterval=self.__interval, maxinterval=10,
                    miniters=0, position=0)
      self.__pb.clear()
      self._epoch_start_time = time.time()
    return self.__pb

  @property
  def seen_so_far(self):
    return self._seen_so_far[self.epoch_idx]

  def _generate_epoch_summary(self, epoch, inc_name=False, inc_counter=True):
    seen_so_far = self._seen_so_far[epoch]
    if seen_so_far == 0:
      return ''
    # ====== include name ====== #
    if inc_name:
      s = _MAGENTA + "%s" % self.name + _RESET
    else:
      s = ""
    # ====== create epoch summary ====== #
    if seen_so_far == self.target: # epoch already finished
      speed = (1. / self._epoch_summary[epoch]['__avg_time__'])
      elapsed = self._epoch_summary[epoch]['__total_time__']
    else: # epoch hasn't finished
      avg_time = (time.time() - self._epoch_start_time) / self.seen_so_far \
      if self.progress_bar.avg_time is None else \
      self.progress_bar.avg_time
      speed = 1. / avg_time
      elapsed = time.time() - self._epoch_start_time
    # ====== counter ====== #
    if inc_counter:
      frac = seen_so_far / self.target
      counter_epoch = self._counter_fmt % (seen_so_far, self.target)
      percentage = "%6.2f%%%s " % (frac * 100, counter_epoch)
    else:
      percentage = ''
    s += _RED + " Epoch %d " % epoch + _RESET + "%.4f(s) %s%.4f(obj/s)" % \
        (elapsed, percentage, speed)
    # epoch summary
    summary = dict(self._epoch_summary[epoch])
    if len(summary) > 2:
      summary.pop('__total_time__', None)
      summary.pop('__avg_time__', None)
      s += '\n' + self._formatted_report(summary, margin='   ', inc_name=False)
    return s

  @property
  def summary(self):
    s = _MAGENTA + "Report \"%s\"    TotalEpoch: %d\n" % \
    (self.name, self.nb_epoch) + _RESET
    # ====== create summary for each epoch ====== #
    s += '\n'.join([self._generate_epoch_summary(i)
                    for i in range(self.nb_epoch)])
    return s[:-1]

  # ==================== same actions ==================== #
  def add_notification(self, msg):
    msg = _CYAN + "[%s][%s]Notification:" % \
        (datetime.now().strftime('%d/%b-%H:%M:%S'),
            _MAGENTA + self.name + _CYAN) + _RESET + msg
    _tqdm.write(msg)
    return self

  def _new_epoch(self):
    if self.__pb is None:
      return
    # calculate number of offset lines from last report
    if self._last_report is None:
      nlines = 0
    else:
      nlines = len(self._last_report.split('\n'))
    # ====== reset progress bar (tqdm) ====== #
    if self.__keep: # keep the last progress on screen
      self.__pb.moveto(nlines)
    else: # clear everything
      for i in range(nlines):
        Progbar.FP.write('\r')
        Progbar.FP.write(' ' * _environ_cols_wrapper()(Progbar.FP))
        Progbar.FP.write('\r')  # place cursor back at the beginning of line
        self.__pb.moveto(1)
      self.__pb.moveto(-(nlines * 2))
    self.__pb.close()
    # ====== create epoch summary ====== #
    for key, values in self._epoch_hist[self._epoch_idx].items():
      values = [v for v in values]
      if key in self._epoch_summarizer: # provided summarizer
        self._epoch_summary[self._epoch_idx][key] = self._epoch_summarizer[key](values)
      elif isinstance(values[0], Number):
        self._epoch_summary[self._epoch_idx][key] = np.mean(values)
      elif isinstance(values[0], np.ndarray):
        self._epoch_summary[self._epoch_idx][key] = sum(v for v in values)
    # total epoch time
    total_time = time.time() - self._epoch_start_time
    self._epoch_summary[self._epoch_idx]['__total_time__'] = total_time
    # average time for 1 object
    avg_time = self.__pb.avg_time
    if avg_time is None:
      avg_time = total_time / self.target
    self._epoch_summary[self._epoch_idx]['__avg_time__'] = avg_time
    # reset all flags
    self.__pb = None
    self._last_report = None
    self._last_print_time = None
    self._epoch_start_time = None
    self._epoch_idx += 1
    return self

  @contextmanager
  def safe_progress(self):
    """ This context manager will automatically call `pause` if the
    progress unfinished, hence, it doesn't mesh up the screen. """
    yield None
    if 0 < self.seen_so_far < self.target:
      self.pause()

  def pause(self):
    """ Call `pause` if progress is running, hasn't finish, and
    you want to print something else on the scree.
    """
    # ====== clear the report ====== #
    if self._last_report is not None:
      nlines = len(self._last_report.split("\n"))
      self.__pb.moveto(-nlines)
      for i in range(nlines):
        Progbar.FP.write('\r')
        Progbar.FP.write(' ' * _environ_cols_wrapper()(Progbar.FP))
        Progbar.FP.write('\r')  # place cursor back at the beginning of line
        self.__pb.moveto(1)
    else:
      nlines = 0
    # ====== clear the bar ====== #
    if self.__pb is not None:
      self.__pb.clear()
      self.__pb.moveto(-nlines)
    # ====== reset the last report ====== #
    # because we already clean everything,
    # set _last_report=None prevent
    # further moveto(-nlines) in add()
    self._last_report = None
    return self

  def add(self, n=1):
    """ You need to call pause if """
    if not isinstance(n, Number):
      if self._progress_func is None:
        raise RuntimeError(
            "`n` is an object, but no given `progress_func` for preprocessing")
      n = self._progress_func(n)
    if n <= 0:
      return self
    fp = Progbar.FP
    # ====== update information ====== #
    seen_so_far = min(self._seen_so_far[self.epoch_idx] + n, self.target)
    self._seen_so_far[self.epoch_idx] = seen_so_far
    # ====== check last updated progress, for automatically pause ====== #
    if _LAST_UPDATED_PROG[0] is None:
      _LAST_UPDATED_PROG[0] = self
    elif _LAST_UPDATED_PROG[0] != self:
      _LAST_UPDATED_PROG[0].pause()
      _LAST_UPDATED_PROG[0] = self
    # ====== show report ====== #
    if self.print_report:
      curr_time = time.time()
      # update the report
      if self._last_print_time is None or \
      time.time() - self._last_print_time > self.__interval or\
      seen_so_far >= self.target:
        self._last_print_time = curr_time
        # move the cursor to last point
        if self._last_report is not None:
          nlines = len(self._last_report.split('\n'))
          self.progress_bar.moveto(-nlines)
        report = self._formatted_report(self._report)
        # clear old report
        if self._last_report is not None:
          for i, l in enumerate(self._last_report.split('\n')):
            fp.write('\r')
            fp.write(' ' * len(l))
            fp.write('\r')  # place cursor back at the beginning of line
            self.progress_bar.moveto(1)
          self.progress_bar.clear()
          self.progress_bar.moveto(-i - 1)
        fp.write(report)
        fp.flush()
        self._last_report = report
        self.progress_bar.moveto(1)
    # ====== show progress ====== #
    if self.print_progress:
      self.progress_bar.update(n=n)
    else:
      self.progress_bar
    # ====== end of epoch ====== #
    if seen_so_far >= self.target:
      self._new_epoch()
      # print summary of epoch
      if self.print_summary:
        _tqdm.write(self._generate_epoch_summary(self.epoch_idx - 1,
                                                 inc_name=True,
                                                 inc_counter=False))
    return self
