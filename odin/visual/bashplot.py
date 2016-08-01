# -*- coding: utf-8 -*-
# ===========================================================================
# This module is adpated from: https://github.com/glamp/bashplotlib
# Original work Copyright (c) 2013 Greg Lamp
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================

from __future__ import print_function, absolute_import, division

import math
import numpy as np

__all__ = [
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
def print_hist(f, height=20.0, bincount=None, binwidth=None, pch="o",
    colour="default", title="", xlab=None, showSummary=False,
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
