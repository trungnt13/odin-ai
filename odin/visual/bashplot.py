# -*- coding: utf-8 -*-
# ===========================================================================
# This module is adpated from: https://github.com/glamp/bashplotlib
# Original work Copyright (c) 2013 Greg Lamp
# Modified work Copyright 2016-2017 TrungNT
# NOTE: the module is intentionally made for self-efficient, hence,
#       no need for external libraries
# ===========================================================================

from __future__ import print_function, absolute_import, division

import re
import math
import warnings
from collections import Mapping

import numpy as np

__all__ = [
    'remove_text_color',
    'escape_text_color',
    'merge_text_graph',
    'print_dist',
    'print_confusion',
    'print_hist',
    'print_bar',
    'print_scatter',
    'print_hinton'
]
# ===========================================================================
# Helper
# ===========================================================================
_chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

isiterable = lambda x: hasattr(x, '__iter__') or hasattr(x, '__getitem__')

bcolours = {
    "white": '\033[97m',
    "aqua": '\033[96m',
    "pink": '\033[95m',
    "blue": '\033[94m',
    "yellow": '\033[93m',
    "green": '\033[92m',
    "red": '\033[91m',
    "grey": '\033[90m',
    "black": '\033[30m',
    "default": '\033[39m',
    "ENDC": '\033[39m',
}

colour_help = ', '.join([colour for colour in bcolours if colour != "ENDC"])


def get_colour(colour):
  """
  Get the escape code sequence for a colour
  """
  return bcolours.get(colour, bcolours['ENDC'])

def print_return_str(text, end='\n'):
  # if not return_str:
        # print(text, end=end)
  return text + end

def printcolour(text, sameline=False, colour=get_colour("ENDC")):
  """
  Print color text using escape codes
  """
  if sameline:
    sep = ''
  else:
    sep = '\n'
  if colour == 'default' or colour == 'ENDC' or colour is None:
    return print_return_str(text, sep)
  return print_return_str(get_colour(colour) + text + bcolours["ENDC"], sep)

def ctext(s, color='red'):
  try:
    from colorama import Fore
    color = color.upper()
    color = getattr(Fore, color, '')
    return color + s + Fore.RESET
  except ImportError:
    pass
  return s

def remove_text_color(s):
  s = re.sub(pattern=r"\\033\[\d\dm", repl='', string=s)
  return s

def escape_text_color(s):
  return remove_text_color(s)

def merge_text_graph(*graphs, padding=' '):
  """ To merge multiple graph together, this function
  will remove all the color coded text to properly align
  all the graph text.
  """
  padding = str(padding)
  assert len(graphs) >= 1
  if len(graphs) == 1:
    return graphs[0]

  def normalizing_lines(text):
    lines = [remove_text_color(l) for l in text.split('\n')]
    maxlen = max([len(l) for l in lines])
    lines = [l + ' ' * (maxlen - len(l))
             if len(l) < maxlen else l
             for l in lines]
    return lines

  graphs = [normalizing_lines(g) for g in graphs]
  maxlen = max(len(i) for i in graphs)
  final_text = ''
  for l in range(maxlen):
    for i, g in enumerate(graphs):
      if l < len(g):
        final_text += (padding if i > 0 else '') + g[l]
    final_text += '\n'
  return final_text

def drange(start, stop, step=1.0, include_stop=False):
  """
  Generate between 2 numbers w/ optional step, optionally include upper bound
  """
  if step == 0:
    step = 0.01
  r = start

  if include_stop:
    while r <= stop:
      yield r
      r += step
      r = round(r, 10)
  else:
    while r < stop:
      yield r
      r += step
      r = round(r, 10)


def box_text(text, width, offset=0):
  """
  Return text inside an ascii textbox
  """
  box = " " * offset + "-" * (width + 2) + "\n"
  box += " " * offset + "|" + text.center(width) + "|" + "\n"
  box += " " * offset + "-" * (width + 2)
  return box


def get_scale(series, is_y=False, steps=20):
  min_val = min(series)
  max_val = max(series)
  scaled_series = []
  for x in drange(min_val, max_val, (max_val - min_val) / steps,
                  include_stop=True):
    if x > 0 and scaled_series and max(scaled_series) < 0:
      scaled_series.append(0.0)
    scaled_series.append(x)

  if is_y:
    scaled_series.reverse()
  return scaled_series

def calc_bins(n, min_val, max_val, h=None, binwidth=None):
  """
  Calculate number of bins for the histogram
  """
  if not h:
    h = max(10, math.log(n + 1, 2))
  if binwidth == 0:
    binwidth = 0.1
  if binwidth is None:
    binwidth = (max_val - min_val) / h
  for b in drange(min_val, max_val, step=binwidth, include_stop=True):
    if b.is_integer():
      yield int(b)
    else:
      yield b

def read_numbers(numbers):
  """
  Read the input data in the most optimal way
  """
  if isiterable(numbers):
    for number in numbers:
      yield float(str(number).strip())
  else:
    for number in open(numbers):
      yield float(number.strip())

# ===========================================================================
# Main
# ===========================================================================
def print_dist(d, height=12, pch="o", show_number=False,
               title=None):
  """ Printing a figure of given distribution

  Parameters
  ----------
  d: dict, list
      a dictionary or a list, contains pairs of: "key" -> "count_value"
  height: int
      number of maximum lines for the graph
  pch : str
      shape of the bars in the plot, e.g 'o'

  Return
  ------
  str

  """
  LABEL_COLOR = ['cyan', 'yellow', 'blue', 'magenta', 'green']
  MAXIMUM_YLABEL = 4
  try:
    if isinstance(d, Mapping):
      d = d.items()
    orig_d = [(str(name), int(count))
              for name, count in d]
    d = [(str(name)[::-1].replace('-', '|').replace('_', '|'), count)
         for name, count in d]
    labels = [[c for c in name] for name, count in d]
    max_labels = max(len(name) for name, count in d)
    max_count = max(count for name, count in d)
    min_count = min(count for name, count in d)
  except Exception as e:
    raise ValueError('`d` must be distribution dictionary contains pair of: '
                     'label_name -> disitribution_count, error: "%s"' % str(e))
  # ====== create figure ====== #
  # draw height, 1 line for minimum bar, 1 line for padding the label,
  # then the labels
  nb_lines = int(height) + 1 + 1 + max_labels
  unit = (max_count - min_count) / height
  fig = ""
  # ====== add unit and total ====== #
  fig += ctext("Unit: ", 'red') + \
      '10^%d' % max(len(str(max_count)) - MAXIMUM_YLABEL, 0) + '  '
  fig += ctext("Total: ", 'red') + \
      str(sum(count for name, count in d)) + '\n'
  # ====== add the figure ====== #
  for line in range(nb_lines):
    value = max_count - unit * line
    # draw the y_label
    if line % 2 == 0 and line <= int(height): # value
      fig += ctext(
          ('%' + str(MAXIMUM_YLABEL) + 's') % str(int(value))[:MAXIMUM_YLABEL],
          color='red')
    else: # blank
      fig += ' ' * MAXIMUM_YLABEL
    fig += '|' if line <= int(height) else ' '
    # draw default line
    if line == int(height):
      fig += ''.join([ctext(pch + ' ',
                            color=LABEL_COLOR[i % len(LABEL_COLOR)])
                      for i in range(len(d))])
    # draw seperator for the label
    elif line == int(height) + 1:
      fig += '-' * (len(d) * 2)
    # draw the labels
    elif line > int(height) + 1:
      for i, lab in enumerate(labels):
        fig += ctext(' ' if len(lab) == 0 else lab.pop(),
                     LABEL_COLOR[i % len(LABEL_COLOR)]) + ' '
    # draw the histogram
    else:
      for i, (name, count) in enumerate(d):
        fig += ctext(pch if count - value >= 0 else ' ',
                     LABEL_COLOR[i % len(LABEL_COLOR)]) + ' '
    # new line
    fig += '\n'
  # ====== add actual number of necessary ====== #
  maximum_fig_length = MAXIMUM_YLABEL + 1 + len(orig_d) * 2
  if show_number:
    line_length = 0
    name_fmt = '%' + str(max_labels) + 's'
    for name, count in orig_d:
      n = len(name) + len(str(count)) + 4
      text = ctext(name_fmt % name, 'red') + ': %d ' % count
      if line_length + n >= maximum_fig_length:
        fig += '\n'
        line_length = n
      else:
        line_length += n
      fig += text
  # ====== add title ====== #
  if title is not None:
    title = ctext('"%s"' % str(title), 'red')
    padding = '  '
    n = (maximum_fig_length - len(title) // 2) // 2 - len(padding) * 3
    fig = '=' * n + padding + title + padding + '=' * n + '\n' + fig
  return fig[:-1]

def _float2str(x, fp=2):
  fmt = '%.' + '%d' % fp + 'f'
  return '1.' + '0' * (fp - 1) \
  if x > 0.99 else (fmt % x)[1:]

def print_confusion(arr, labels=None, side_bar=True, inc_stats=True,
                    float_precision=2):
  """
  Parameters
  ----------
  side_bar : bool (default: True)
    if True, include the side bar of precision, recall, F1,
    false alarm and number of samples

  inc_stats : bool (default: True)
    if True, include Precision, Recall, F1 ,False Alarm

  float_precision : int (.. > 0)
    number of floating point precision for printing the entries
  """
  side_bar = bool(side_bar)
  inc_stats = bool(inc_stats)
  float_precision = int(float_precision)
  assert float_precision > 0
  # ====== preprocessing ====== #
  LABEL_COLOR = 'magenta'
  if arr.ndim != 2:
    raise ValueError("Can only process 2-D confusion matrixf")
  if arr.shape[0] != arr.shape[1]:
    raise ValueError("`arr` must be square matrix.")
  nb_classes = arr.shape[0]
  if labels is None:
    labels = ['%d' % i for i in range(nb_classes)]
  else:
    labels = [str(i) for i in labels]
  max_label_length = max(len(i) for i in labels)
  lab_fmt = '%-' + str(max_label_length) + 's '
  # ====== calculate precision, recall, F1 ====== #
  arr_sum_row = arr.sum(-1).astype('float64')
  total_samples = np.sum(arr_sum_row)
  arr_sum_col = arr.sum(0).astype('float64')
  info = {}
  nb_info = 5 # Precision, Recall, F1, FA, Sum
  if inc_stats or side_bar:
    for i in range(nb_classes):
      TP = arr[i, i] # True positive
      FN = arr_sum_row[i] - arr[i, i] # False negative
      FP = arr_sum_col[i] - arr[i, i] # False positive
      precision = 0. if (TP + FP) == 0 else TP / (TP + FP)
      recall = 0. if (TP + FN) == 0 else TP / (TP + FN)
      f1 = 0. if precision == 0. or recall == 0. else \
          2 / (1 / precision + 1 / recall)
      fa = 0. if (total_samples - arr_sum_row[i]) == 0 else \
          FP / (total_samples - arr_sum_row[i]) # False alarm
      info[i] = (precision, recall, f1, fa, arr_sum_row[i])
  # normalize the confusion
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    arr = np.nan_to_num(arr.astype('float64') / arr_sum_row[:, None])
  # print title of the side bar
  fig = " " * ((3 + 1) * nb_classes + max_label_length + 2)
  if side_bar:
    fig += ctext('|'.join(['Pre', 'Rec', ' F1', ' FA', 'Sum']), 'red') + '\n'
    longest_line = len(fig) - 1
  else:
    longest_line = len(fig) - 1
    fig = ""
  # confusion matrix
  for row_id, row in enumerate(arr):
    row_text = ctext(lab_fmt % labels[row_id], LABEL_COLOR)
    # get the worst performed point
    most_misclassified = np.argsort(row, kind='quicksort').tolist()
    most_misclassified.remove(row_id)
    most_misclassified = most_misclassified[-1]
    if row[most_misclassified] == 0.:
      most_misclassified = None
    # iterate over each column value
    for col_id, col in enumerate(row):
      col = _float2str(col, fp=float_precision)
      if col_id == row_id:
        row_text += ctext(col,
                          color='blue' if col == '1.0' else 'cyan') + ' '
      elif most_misclassified is not None and \
      col_id == most_misclassified:
        row_text += ctext(col, color='red') + ' '
      else:
        row_text += col + ' '
    # new line
    if side_bar:
      # print float, except the last one is int
      info_str = [_float2str(val) if i < (nb_info - 1) else
                  ('%d' % np.round(val))
                  for i, val in enumerate(info[row_id])]
      fig += row_text + ' ' + '|'.join(info_str) + '\n'
    else:
      fig += row_text + '\n'
  # ====== draw labels at the bottom ====== #
  labels = [[c for c in i.replace('-', '|').replace('_', '|')[::-1]]
            for i in labels]
  for i in range(max_label_length):
    fig += ' ' * (max_label_length + 1)
    row = ''
    for l in labels:
      row += ' ' + (l.pop() if len(l) > 0 else ' ') + ' ' * float_precision
    fig += ctext(row, 'magenta') + '\n'
  # ====== Add the average values of stats ====== #
  if inc_stats:
    n = 0
    for i, name in enumerate(['Pre', 'Rec', ' F1', ' FA', '  Sum']):
      avr = np.mean([stats[i] for stats in info.values()])
      avr = '%.4g' % avr
      if avr[0] == '0':
        avr = avr[1:]
      avr = ctext('%s:' % name, 'red') + avr
      if n + len(avr) >= longest_line: # new line
        avr += '\n'
        n = 0
      else: # same line
        avr += '\t'
      n += len(avr) + len('\t')
      fig += avr
  return fig[:-1]

def print_hist(f, height=20.0, bincount=None, binwidth=None, pch="o",
    colour="default", title="", xlab=True, showSummary=False,
    regular=False):
  ''' Plot histogram.
   1801|       oo
   1681|       oo
   1561|      oooo
    961|      oooo
    841|      oooo
    721|     ooooo
    601|     oooooo
    241|     oooooo
    121|    oooooooo
      1| oooooooooooooo
        --------------
  Parameters
  ----------
  f : list(number), numpy.ndarray, str(filepath)
      input array
  height : float
      the height of the histogram in # of lines
  bincount : int
      number of bins in the histogram
  binwidth : int
      width of bins in the histogram
  pch : str
      shape of the bars in the plot, e.g 'o'
  colour : str
      white,aqua,pink,blue,yellow,green,red,grey,black,default,ENDC
  title : str
      title at the top of the plot, None = no title
  xlab : boolean
      whether or not to display x-axis labels
  showSummary : boolean
      whether or not to display a summary
  regular : boolean
      whether or not to start y-labels at 0
  return_str : boolean
      return string represent the plot or print it out, default: False
  '''
  if pch is None:
    pch = "o"
  splot = ''
  if isinstance(f, str):
    f = open(f).readlines()

  min_val, max_val = None, None
  n, mean, sd = 0.0, 0.0, 0.0

  for number in read_numbers(f):
    n += 1
    if min_val is None or number < min_val:
      min_val = number
    if max_val is None or number > max_val:
      max_val = number
    mean += number

  mean /= n

  for number in read_numbers(f):
    sd += (mean - number)**2

  sd /= (n - 1)
  sd **= 0.5

  bins = list(calc_bins(n, min_val, max_val, bincount, binwidth))
  hist = dict((i, 0) for i in range(len(bins)))

  for number in read_numbers(f):
    for i, b in enumerate(bins):
      if number <= b:
        hist[i] += 1
        break
    if number == max_val and max_val > bins[len(bins) - 1]:
      hist[len(hist) - 1] += 1

  min_y, max_y = min(hist.values()), max(hist.values())

  start = max(min_y, 1)
  stop = max_y + 1

  if regular:
    start = 1

  if height is None:
    height = stop - start
    if height > 20:
      height = 20

  ys = list(drange(start, stop, float(stop - start) / height))
  ys.reverse()

  nlen = max(len(str(min_y)), len(str(max_y))) + 1

  if title:
    splot += print_return_str(box_text(title, max(len(hist) * 2, len(title)), nlen))
  splot += print_return_str('')

  used_labs = set()
  for y in ys:
    ylab = str(int(y))
    if ylab in used_labs:
      ylab = ""
    else:
      used_labs.add(ylab)
    ylab = " " * (nlen - len(ylab)) + ylab + "|"

    splot += print_return_str(ylab, end=' ')

    for i in range(len(hist)):
      if int(y) <= hist[i]:
        splot += printcolour(pch, True, colour)
      else:
        splot += printcolour(" ", True, colour)
    splot += print_return_str('')
  xs = hist.keys()

  splot += print_return_str(" " * (nlen + 1) + "-" * len(xs))

  if xlab:
    xlen = len(str(float((max_y) / height) + max_y))
    for i in range(0, xlen):
      splot += printcolour(" " * (nlen + 1), True, colour)
      for x in range(0, len(hist)):
        num = str(bins[x])
        if x % 2 != 0:
          pass
        elif i < len(num):
          splot += print_return_str(num[i], end=' ')
        else:
          splot += print_return_str(" ", end=' ')
      splot += print_return_str('')

  center = max(map(len, map(str, [n, min_val, mean, max_val])))
  center += 15

  if showSummary:
    splot += print_return_str('')
    splot += print_return_str("-" * (2 + center))
    splot += print_return_str("|" + "Summary".center(center) + "|")
    splot += print_return_str("-" * (2 + center))
    summary = "|" + ("observations: %d" % n).center(center) + "|\n"
    summary += "|" + ("min value: %f" % np.min(f)).center(center) + "|\n"
    summary += "|" + ("mean : %f" % np.mean(f)).center(center) + "|\n"
    summary += "|" + (" 1-per: %f" % np.percentile(f, 1)).center(center) + "|\n"
    summary += "|" + (" 5-per: %f" % np.percentile(f, 5)).center(center) + "|\n"
    summary += "|" + ("25-per: %f" % np.percentile(f, 25)).center(center) + "|\n"
    summary += "|" + ("median: %f" % np.median(f)).center(center) + "|\n"
    summary += "|" + ("75-per: %f" % np.percentile(f, 75)).center(center) + "|\n"
    summary += "|" + ("95-per: %f" % np.percentile(f, 95)).center(center) + "|\n"
    summary += "|" + ("99-per: %f" % np.percentile(f, 99)).center(center) + "|\n"
    summary += "|" + ("sd : %f" % np.std(f)).center(center) + "|\n"
    summary += "|" + ("max value: %f" % np.max(f)).center(center) + "|\n"
    summary += "-" * (2 + center)
    splot += print_return_str(summary)
  return splot

def print_bar(f, height=20.0, bincount=None, binwidth=None, pch="o",
    colour="default", title="", xlab=None, showSummary=False,
    regular=False):
  ''' Plot bar.

  Parameters
  ----------
  f : list(number), numpy.ndarray, str(filepath)
      input array
  height : float
      the height of the histogram in # of lines
  bincount : int
      number of bins in the histogram
  binwidth : int
      width of bins in the histogram
  pch : str
      shape of the bars in the plot, e.g 'o'
  colour : str
      white,aqua,pink,blue,yellow,green,red,grey,black,default,ENDC
  title : str
      title at the top of the plot, None = no title
  xlab : boolean
      whether or not to display x-axis labels
  showSummary : boolean
      whether or not to display a summary
  regular : boolean
      whether or not to start y-labels at 0

  Example
  -------
  >>> y = np.random.rand(50)
  >>> bash_bar(y, bincount=50, colour='red')

  >>> 0.971|
  >>> 0.923|                o         o             o   o
  >>> 0.875|                o   o     o           o o  oo
  >>> 0.827|                o   o     o     o     o o ooo
  >>> 0.779|   o            o   o     o   o o     o o ooo
  >>> 0.731|   o    o       o   o     o   o o     o o ooo
  >>> 0.683|   oo   oo      o   o o o o   o o     o o ooo
  >>> 0.635|   oo   oo      o  oo o o o   o o    oo o ooo
  >>> 0.587|   oo   oo      o  oo o ooo   o o    oo o ooo
  >>> 0.539|   oo   oo      o  oooo ooo   o o    oo o ooo o
  >>> 0.491|   ooo ooo      o  oooo ooo   o o   ooo o ooo o
  >>> 0.443|   ooo ooo     oo ooooo oooo  o o   ooo o ooo o
  >>> 0.395|   ooo ooo    ooooooooo oooo  o o   ooo o ooo o
  >>> 0.347|  oooo ooo    ooooooooo ooooo o o   ooo o ooo o
  >>> 0.299|  oooo ooo    ooooooooo ooooo o o   ooo o ooo o
  >>> 0.251|  oooo ooo    ooooooooo ooooo o o   ooo o ooo o o
  >>> 0.203|  oooo ooo    ooooooooo ooooo ooo   ooo o ooo o o
  >>> 0.155|  oooo ooo oo ooooooooo ooooo ooo   ooo ooooooo oo
  >>> 0.107| ooooo ooo oo ooooooooo oooooooooo  ooooooooooo ooo
  >>> 0.059| ooooo oooooo ooooooooo ooooooooooooooooooooooo ooo
  >>> 0.011| oooooooooooooooooooooo ooooooooooooooooooooooooooo
  >>>       --------------------------------------------------
  '''
  if len(f) == 1:
    f = [min(0., np.min(f))] + [i for i in f]

  if pch is None:
    pch = "o"

  splot = ''
  if isinstance(f, str):
    f = open(f).readlines()

  # ====== Create data ====== #
  min_val, max_val = None, None
  n, mean, sd = 0.0, 0.0, 0.0

  # pick mode and get data
  numbers = [i for i in read_numbers(f)]
  int_mode = False
  if numbers[0].is_integer():
    int_mode = True

  # rescale big enough to show on bars
  min_orig = min(numbers) # original
  max_orig = max(numbers)
  numbers = [1000 * (i - min_orig) / (max_orig - min_orig + 1e-8) for i in numbers]

  # statistics
  n = len(numbers)
  min_val = min(numbers)
  max_val = max(numbers)
  mean = sum(numbers) / n
  sd = (sum([(mean - i)**2 for i in numbers]) / (n - 1)) ** 0.5

  # bins is index
  if bincount is not None:
    bincount = min(bincount, n)
  bins = list(calc_bins(n, 0., n + 0., bincount, binwidth))
  bins = [int(i) for i in bins]
  hist = dict((i, 0) for i in range(len(bins) - 1))

  # hist is the mean value of array with indices within bin
  for idx, (i, j) in enumerate(zip(bins, bins[1:])):
    if i == j: j += 1
    arr = numbers[i:j]
    hist[idx] = sum(arr) / len(arr) # calculate mean

  # ====== Start plot ====== #
  min_y, max_y = min(hist.values()), max(hist.values())

  start = max(min_y, 1)
  stop = max_y + 1

  if regular:
    start = 1

  if height is None:
    height = stop - start
    if height > 20:
      height = 20

  ys = list(drange(start, stop, float(stop - start) / height))
  ys.reverse()

  nlen = max(len(str(min_y)), len(str(max_y))) + 1

  if title:
    splot += print_return_str(
        box_text(title, max(len(hist) * 2, len(title)), nlen))
  splot += print_return_str('')

  used_labs = set()
  for y in ys:
    if int_mode:
      ylab = '%d' % int(y * (max_orig - min_orig + 1e-8) / 1000 + min_orig)
    else:
      ylab = '%.3f' % float(y * (max_orig - min_orig + 1e-8) / 1000 + min_orig)
    if ylab in used_labs:
      ylab = ""
    else:
      used_labs.add(ylab)
    ylab = " " * (nlen - len(ylab)) + ylab + "|"

    splot += print_return_str(ylab, end=' ')

    for i in range(len(hist)):
      if int(y) <= hist[i]:
        splot += printcolour(pch, True, colour)
      else:
        splot += printcolour(" ", True, colour)
    splot += print_return_str('')
  xs = hist.keys()

  splot += print_return_str(" " * (nlen + 1) + "-" * len(xs))

  if xlab:
    xlen = len(str(float((max_y) / height) + max_y))
    for i in range(0, xlen):
      splot += printcolour(" " * (nlen + 1), True, colour)
      for x in range(0, len(hist)):
        num = str(bins[x])
        if x % 2 != 0:
          pass
        elif i < len(num):
          splot += print_return_str(num[i], end=' ')
        else:
          splot += print_return_str(" ", end=' ')
      splot += print_return_str('')

  center = max(map(len, map(str, [n, min_val, mean, max_val])))
  center += 15

  if showSummary:
    splot += print_return_str('')
    splot += print_return_str("-" * (2 + center))
    splot += print_return_str("|" + "Summary".center(center) + "|")
    splot += print_return_str("-" * (2 + center))
    summary = "|" + ("observations: %d" % n).center(center) + "|\n"
    summary += "|" + ("min value: %f" % np.min(f)).center(center) + "|\n"
    summary += "|" + ("mean : %f" % np.mean(f)).center(center) + "|\n"
    summary += "|" + ("sd : %f" % np.std(f)).center(center) + "|\n"
    summary += "|" + ("max value: %f" % np.max(f)).center(center) + "|\n"
    summary += "-" * (2 + center)
    splot += print_return_str(summary)

  return splot


def print_scatter(xs, ys, size=None, pch='o',
                colour='red', title=None):
  ''' Scatter plot.
  ----------------------
  |                 *   |
  |               *     |
  |             *       |
  |           *         |
  |         *           |
  |        *            |
  |       *             |
  |      *              |
  -----------------------
  Parameters
  ----------
  xs : list, numpy.ndarray
      list of x series
  ys : list, numpy.ndarray
      list of y series
  size : int
      width of plot
  pch : str
      any character to represent a points
  colour : str, list(str)
      white,aqua,pink,blue,yellow,green,red,grey,black,default,ENDC
  title : str
      title for the plot, None = not show
  return_str : boolean
      return string represent the plot or print it out, default: False
  '''
  splot = ''
  plotted = set()
  cs = colour

  if size is None:
    size = 13

  if title:
    splot += print_return_str(box_text(title, 2 * len(get_scale(xs, False, size)) + 1))

  # ====== Top line ====== #
  splot += print_return_str(' ' + "-" * (len(get_scale(xs, False, size)) + 2))
  # ====== Main plot ====== #
  for y in get_scale(ys, True, size):
    splot += print_return_str("|", end=' ')
    for x in get_scale(xs, False, size):
      point = " "
      for (i, (xp, yp)) in enumerate(zip(xs, ys)):
        if xp <= x and yp >= y and (xp, yp) not in plotted:
          point = pch
          plotted.add((xp, yp))
          if isinstance(cs, list):
            colour = cs[i]
      splot += printcolour(point, True, colour)
    splot += print_return_str(" |")
  # ====== Bottom line ====== #
  splot += print_return_str(' ' + "-" * (len(get_scale(xs, False, size)) + 2))
  return splot


def print_hinton(arr, max_arr=None):
  ''' Print bar string, fast way to visual magnitude of value in terminal

  Example:
  -------
  >>> W = np.random.rand(10,10)
  >>> print_hinton(W)
  >>> ▁▃▄█▅█ ▅▃▅
  >>> ▅▂▆▄▄ ▅▅
  >>> ▄██▆▇▆▆█▆▅
  >>> ▄▄▅▂▂▆▅▁▅▆
  >>> ▂ ▁  ▁▄▆▅▁
  >>> ██▃█▃▃▆ ▆█
  >>>  ▁▂▁ ▁▃▃▆▂
  >>> ▅▂▂█ ▂ █▄▅
  >>> ▃▆▁▄▁▆▇▃▅▁
  >>> ▄▁▇ ██▅ ▂▃
  Returns
  -------
  return : str
      plot of array, for example: ▄▅▆▇
  '''
  arr = np.asarray(arr)
  if len(arr.shape) == 1:
    arr = arr[None, :]

  def visual_func(val, max_val):
    if abs(val) == max_val:
      step = len(_chars) - 1
    else:
      step = int(abs(float(val) / max_val) * len(_chars))
    colourstart = ""
    colourend = ""
    if val < 0:
      colourstart, colourend = '\033[90m', '\033[0m'
    return colourstart + _chars[step] + colourend

  if max_arr is None:
    max_arr = arr
  max_val = max(abs(np.max(max_arr)), abs(np.min(max_arr)))
  # print(np.array2string(arr,
  #                       formatter={'float_kind': lambda x: visual(x, max_val)},
  #                       max_line_width=5000)
  # )
  f = np.vectorize(visual_func)
  result = f(arr, max_val) # array of ▄▅▆▇
  rval = ''
  for r in result:
    rval += ''.join(r) + '\n'

  return rval[:-1]
