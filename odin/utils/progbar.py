# -*- coding: utf-8 -*-
##################################################################
# # Example of usage:
# with Progbar(target=50, name="Prog1").context() as p1:
#     for i in range(200):
#         time.sleep(0.2)
#         p1['value1'] = i
#         p1.add(1)
#         if i == 30:
#             with Progbar(target=20, name="Prog2").context() as p2:
#                 for j in range(60):
#                     time.sleep(0.2)
#                     p2['value2'] = j
#                     p2.add(1)
##################################################################
from __future__ import print_function, division, absolute_import

import sys
import time
from numbers import Number
from datetime import datetime
from contextlib import contextmanager
from collections import OrderedDict, defaultdict

import numpy as np

from odin.visual.bashplot import print_bar

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


@contextmanager
def _progbar(pb, print_report, print_summary):
    org_report = pb.print_report
    org_summary = pb.print_summary
    if print_report is not None:
        pb.print_report = print_report
    if print_summary is not None:
        pb.print_summary = print_summary
    yield pb
    pb.pause()
    pb.print_report = org_report
    pb.print_summary = org_summary


def add_notification(msg):
    msg = _CYAN + "\n[%s]Notification:" % \
        datetime.now().strftime('%d/%b-%H:%M:%S') + _RESET + msg + '\n'
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
    name: str or None
        specific name for the progress bar
    """
    FP = sys.stderr

    def __init__(self, target, interval=0.1, keep=False,
                 print_report=False, print_summary=False,
                 name=None):
        self.__pb = None
        self.target = int(target)
        self._seen_so_far = defaultdict(int) # mapping: epoch_idx -> seen_so_far

        n = len(str(self.target))
        self._counter_fmt = '(%%%dd/%%%dd)' % (n, n)

        if name is None:
            name = "Progress-%s" % datetime.utcnow()
        self._name = name

        self.__interval = float(interval)
        self.__keep = keep
        self.print_report = print_report
        self.print_summary = print_summary
        # ====== for history ====== #
        self._report = OrderedDict()
        self._last_report = None
        self._last_print_time = None
        # ====== recording history ====== #
        # dictonary: {epoch_id: {key: [value1, value2, ...]}}
        self._epoch_hist = defaultdict(lambda: defaultdict(list))
        self._epoch_summary = defaultdict(dict)
        self._epoch_idx = 0
        self._epoch_start_time = None

    # ==================== History management ==================== #
    def __getitem__(self, key):
        return self._report.__getitem__(key)

    def __setitem__(self, key, val):
        self._epoch_hist[self.epoch_idx][key].append(val)
        return self._report.__setitem__(key, val)

    def __delitem__(self, key):
        return self._report.__delitem__(key)

    def __iter__(self):
        return self._report.__iter__()

    # ==================== screen control ==================== #
    def interactive(self, print_report=None, print_summary=None):
        return _progbar(self, print_report, print_summary)

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
    def history(self):
        """ Return history recording all add item (timestamp, key, value)
        to this progress

        Return
        ------
        dictonary: {epoch_id: {key: [value1, value2, ...]}}
        """
        return self._epoch_hist

    def formatted_report(self, report_dict, margin='', inc_name=True):
        if inc_name:
            text = _MAGENTA + "\t%s" % self.name + _RESET + '\n'
        else:
            text = ""
        report_dict = sorted(report_dict.items(), key=lambda x: str(x[0]))
        for i, (key, value) in enumerate(report_dict):
            # ====== check value of key and value ====== #
            key = margin + str(key).replace('\n', ' ')
            value = '\n'.join([s if _ == 0 else
                               ' ' * (len(key) + 2) + s
                               for _, s in enumerate(str(value).split('\n'))])
            text += _YELLOW + key + _RESET + ": " + value + "\n"
        return text[:-1]

    @property
    def progress_bar(self):
        if self.__pb is None:
            it = range(self.target)
            self.__pb = _tqdm(iterable=it, desc="Epoch %d" % self.epoch_idx,
                              leave=self.__keep,
                              total=self.target, file=Progbar.FP, unit='obj',
                              mininterval=self.__interval, position=0)
            self.__pb.clear()
            self._epoch_start_time = time.time()
        return self.__pb

    @property
    def seen_so_far(self):
        return self._seen_so_far[self.epoch_idx]

    def _generate_epoch_summary(self, i, inc_name=False, inc_counter=True):
        seen_so_far = self._seen_so_far[i]
        if seen_so_far == 0:
            return ''
        # ====== include name ====== #
        if inc_name:
            s = _MAGENTA + "%s" % self.name + _RESET
        else:
            s = ""
        # ====== create epoch summary ====== #
        if seen_so_far == self.target:
            speed = (1. / self._epoch_summary[i]['__avg_time__'])
            elapsed = self._epoch_summary[i]['__total_time__']
        else:
            speed = 1. / self.progress_bar.avg_time
            elapsed = time.time() - self._epoch_start_time
        # ====== counter ====== #
        if inc_counter:
            frac = seen_so_far / self.target
            counter_epoch = self._counter_fmt % (seen_so_far, self.target)
            percentage = "%6.2f%%%s " % (frac * 100, counter_epoch)
        else:
            percentage = ''
        s += _RED + " Epoch %d " % i + _RESET + \
            "%.4f(s) %s%.4f(obj/s)" % \
            (elapsed, percentage, speed)
        # epoch summary
        summary = dict(self._epoch_summary[i])
        if len(summary) > 2:
            summary.pop('__total_time__', None)
            summary.pop('__avg_time__', None)
            s += '\n' + self.formatted_report(summary, margin='   ',
                                              inc_name=False)
        return s

    @property
    def summary(self):
        s = _MAGENTA + "Report \"%s\"    #Epoch: %d\n" % (self.name, self.nb_epoch) + _RESET
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
        if self.__pb is None or self._last_report is None:
            return
        avg_time = self.__pb.avg_time
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
        self.__pb = None
        # create epoch summary
        for key, values in self._epoch_hist[self._epoch_idx].iteritems():
            values = [v for v in values]
            if isinstance(values[0], Number):
                self._epoch_summary[self._epoch_idx][key] = np.mean(values)
            elif isinstance(values[0], np.ndarray):
                self._epoch_summary[self._epoch_idx][key] = sum(v for v in values)
        self._epoch_summary[self._epoch_idx]['__avg_time__'] = avg_time
        self._epoch_summary[self._epoch_idx]['__total_time__'] = \
            time.time() - self._epoch_start_time
        # reset all flags
        self._last_report = None
        self._last_print_time = None
        self._epoch_start_time = None
        self._epoch_idx += 1
        return self

    def pause(self):
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
        # because we already clean everythin, set _last_report=None prevent
        # further moveto(-nlines) in add()
        self._last_report = None
        return self

    def add(self, n=1):
        """ You need to call pause if

        """
        if n <= 0:
            return self
        fp = Progbar.FP
        # ====== update information ====== #
        seen_so_far = min(self._seen_so_far[self.epoch_idx] + n, self.target)
        self._seen_so_far[self.epoch_idx] = seen_so_far
        # ====== check last updated prog, for automatically pause ====== #
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
                report = self.formatted_report(self._report)
                # clear old report
                if self._last_report is not None:
                    for i, l in enumerate(self._last_report.split('\n')):
                        fp.write('\r')
                        fp.write(' ' * len(l))
                        fp.write('\r')  # place cursor back at the beginning of line
                        self.progress_bar.moveto(1)
                    self.progress_bar.clear(nomove=True)
                    self.progress_bar.moveto(-i - 1)
                fp.write(report)
                fp.flush()
                self.progress_bar.moveto(1)
                self._last_report = report
        # ====== show progress ====== #
        self.progress_bar.update(n=n)
        # ====== end of epoch ====== #
        if seen_so_far >= self.target:
            self._new_epoch()
            if self.print_summary: # print summary of epoch
                _tqdm.write(self._generate_epoch_summary(self.epoch_idx - 1,
                                                         inc_name=True,
                                                         inc_counter=False))
        return self
