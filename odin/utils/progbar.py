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
    import curses
except ImportError:
    print("[WARNING] Cannot import `curses` for manipulating terminal outputs, "
          "it is probably because of using Windows for Machine Learning.")
_NCOLS = 0
_NROWS = 0
_HEIGHT = 0
_Y = 0
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

_stdscr = None


def stdscr():
    global _stdscr, _NROWS, _NCOLS, _HEIGHT
    if _stdscr is None:
        win = curses.initscr()
        (_NROWS, _NCOLS) = win.getmaxyx()
        _HEIGHT = _NROWS
        _stdscr = curses.newpad(_HEIGHT, _NCOLS)
        _stdscr.nodelay(1)
        _stdscr.scrollok(True)
    return _stdscr


def addstr(y, x, text, style=curses.A_NORMAL):
    if y >= _HEIGHT:
        global _HEIGHT
        _HEIGHT = _HEIGHT * 2
        _stdscr.resize(_HEIGHT, _NCOLS)
    stdscr().addstr(y, x, text, style)


def scrollUP(amount):
    global _Y
    _Y = max(0, _Y - amount)


def scrollDOWN(amount):
    global _Y
    _Y = min(_HEIGHT - 1, _Y + amount)


def scrollRESET():
    global _Y
    _Y = 0


def refresh():
    stdscr().refresh(_Y, 0, 0, 0, _NROWS - 1, _NCOLS)

# ===========================================================================
# Helper
# ===========================================================================
_TUTORIAL_TEXT = \
"""******************************************************************************
*                  Welcome to `ProgressBar` mode in O.D.I.N                  *
******************************************************************************
* a) Press `S` for toggle (on/off) summary of current progress               *
* b) `i` for scroll UP and `k` for scoll DOWN, `r` for reset the scroll      *
* c) Number from `0`, `1`, ..., `9` to show summary of the last 10 progress  *
* d) If `confirm_exit` press <ENTER> to exit when Progress finished          *
******************************************************************************"""
_PROGBAR_STACK = []
_FINISHED_PROGBAR = []
_PROGBAR_HISTORY = [] # last 10 ran ProgBar
_CURRENT_PROG_SUMMARY = [None]


def _show_tutorial(n_line, n_col):
    for line in _TUTORIAL_TEXT.split('\n'):
        addstr(n_line, 0, line, curses.A_BOLD)
        n_line += 1
    return n_line, n_col


def _show_progbar_hist(n_line, n_col):
    intro_text = "The last 10 progress"
    addstr(n_line, 0, intro_text, curses.A_STANDOUT)
    curr_col = len(intro_text)
    for i, prog in enumerate(_PROGBAR_HISTORY):
        text = ' %d:"%s" ' % (i, prog.name)
        if len(text) + curr_col > _NCOLS:
            n_line += 1
            curr_col = len(intro_text)
        addstr(n_line, curr_col, text, curses.A_NORMAL)
        curr_col += len(text)
    return n_line + 1, n_col


# ===========================================================================
# Handling notification
# ===========================================================================
_NOTIFICATION_MSG = []


def add_notification(msg):
    if len(_PROGBAR_STACK) > 0:
        _NOTIFICATION_MSG.append((datetime.now().strftime('%d/%b-%H:%M:%S '),
                                  str(msg)))


def clear_notification():
    global _NOTIFICATION_MSG
    _NOTIFICATION_MSG = []


def _show_notification(n_line, n_col):
    for timestamp, msg in _NOTIFICATION_MSG:
        addstr(n_line, 0, timestamp, curses.A_BOLD)
        addstr(n_line, len(timestamp), msg, curses.A_NORMAL)
        n_line += 1
    return n_line, n_col


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
    """

    def __init__(self, target, name=None):
        self.target = int(target)
        self._seen_so_far = defaultdict(int)

        if name is None:
            name = "Progress: %s" % datetime.utcnow()
        self._name = name

        self._report = OrderedDict()
        # ====== time mark ====== #
        self._last_update = None
        # store for each epoch
        self._total_time = defaultdict(int)
        self._update_time_hist = defaultdict(list)
        # ====== recording history ====== #
        # dictonary: {epoch_id: {key: [value1, value2, ...]}}
        self._epoch_hist = defaultdict(lambda: defaultdict(list))
        self._epoch_summary = defaultdict(dict)
        self._epoch_idx = 0

    def __getitem__(self, key):
        return self._report.__getitem__(key)

    def __setitem__(self, key, val):
        # erase old screen if update new information
        if self in _PROGBAR_STACK:
            stdscr().erase()
        self._epoch_hist[self.epoch_idx][key].append(val)
        return self._report.__setitem__(key, val)

    def __delitem__(self, key):
        # erase old screen if update new information
        if self in _PROGBAR_STACK:
            stdscr().erase()
        return self._report.__delitem__(key)

    def __iter__(self):
        return self._report.__iter__()

    def context(self, print_progress=True, print_summary=False, confirm_exit=False):
        return _progbar(self, print_progress=print_progress,
                        print_summary=print_summary, confirm_exit=confirm_exit)

    @property
    def epoch_idx(self):
        return self._epoch_idx

    @property
    def nb_epoch(self):
        return len([i for i in self._update_time_hist.values()
                    if len(i) > 0])

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

    @property
    def formatted_report(self):
        text = []
        for i, (key, value) in enumerate(self._report.iteritems()):
            # ====== check value of key and value ====== #
            key = ' '.join(str(key).split('\n'))
            value = str(value).split('\n')
            # ====== created formatted text ====== #
            value = [(len(key) + 2, v.replace('\n', ''), curses.A_NORMAL)
                     for v in value]
            key = (0, str(key) + ":", curses.A_BOLD)
            text.append((key, value))
        return text

    @property
    def seen_so_far(self):
        return self._seen_so_far[self.epoch_idx]

    @property
    def progress_bar(self):
        # ====== get information ====== #
        seen_so_far = self._seen_so_far[self.epoch_idx]
        frac = seen_so_far / self.target
        total_time = max(self._total_time[self.epoch_idx], 10e-8)
        it_per_sec = seen_so_far / total_time
        n = len(str(self.target))
        # ====== progress info ====== #
        elapsed = ' Elapsed: %.2f(s)' % total_time
        if it_per_sec > 0:
            remain = " Estimate: %.2f(s)" % ((self.target - seen_so_far) / it_per_sec)
        else:
            remain = " Estimate: inf(s)"
        # calculate speed
        speed = ' [Min:%%-%d.2f  Avr:%%-%d.2f  Max:%%-%d.2f (obj/s)]' % (n, n, n)
        update_time_hist = self._update_time_hist[self.epoch_idx]
        min_it = (1. / max(update_time_hist)) if len(update_time_hist) > 0 else 0
        max_it = (1. / min(update_time_hist)) if len(update_time_hist) > 0 else 0
        avr_it = (1. / np.mean(update_time_hist)) if len(update_time_hist) > 0 else 0
        speed = speed % (min_it, avr_it, max_it)
        report = ' '.join([elapsed, remain, speed])
        # ====== bar ====== #
        percentage = '(Epoch %d) % 3.2f%%' % (self.epoch_idx, frac * 100)
        counter = '(%%%dd/%%%dd)' % (n, n)
        counter = counter % (seen_so_far, self.target)
        N_BARS = len(report) - len(percentage) - len(counter)
        bar_length = int(frac * N_BARS)
        bar = '|%s|' % ('#' * bar_length + '-' * (N_BARS - bar_length))
        progress = percentage + counter + bar
        # ====== final progress ====== #
        return [report, progress]

    @property
    def summary(self):
        s = "Name: \"%s\"    #Epoch: %d\n" % (self.name, self.nb_epoch)
        n = len(str(self.target))
        counter = '(%%%dd/%%%dd)' % (n, n)
        # ====== create summary for each epoch ====== #
        for i in range(self.epoch_idx + 1):
            update_time_hist = self._update_time_hist[i]
            if len(update_time_hist) == 0:
                continue
            seen_so_far = self._seen_so_far[i]
            frac = seen_so_far / self.target
            total_time = max(self._total_time[i], 10e-8)
            it_per_sec = seen_so_far / total_time
            s += " Epoch %d\n" % i
            s += '\tElapsed: %.2f(s)\n' % total_time
            counter_epoch = counter % (seen_so_far, self.target)
            s += '\tProgress: % 3.2f%% %s\n' % (frac * 100, counter_epoch)
            s += '\tSpeed: %.2f(obj/s)\n' % it_per_sec
            if len(update_time_hist) > 0:
                s += "\tSpeed chart (obj/s):\n"
                s += print_bar([1. / i for i in update_time_hist],
                               height=8.)
                s += '\n'
        return s

    # ==================== same actions ==================== #
    def add_notification(self, msg):
        add_notification(msg)
        return self

    def reset(self):
        # create epoch summary
        for key, values in self._epoch_hist[self._epoch_idx].iteritems():
            values = [v for v in values]
            if isinstance(values[0], Number):
                self._epoch_summary[self._epoch_idx][key] = np.mean(values)
            elif isinstance(values[0], np.ndarray):
                self._epoch_summary[self._epoch_idx][key] = sum(v for v in values)
        # reset all flags
        self._last_update = None
        self._epoch_idx += 1
        return self

    def pause(self):
        self._last_update = None
        return self

    def _flush_str(self, n_line, n_col):
        # ====== print ====== #
        name = self.name
        addstr(n_line, n_col + 0, ' ' + '-' * (len(name) + 2))
        n_line += 1
        addstr(n_line, n_col + 0, '| ')
        addstr(n_line, n_col + 2, name, curses.A_STANDOUT)
        addstr(n_line, n_col + 2 + len(name), ' |')
        n_line += 1
        addstr(n_line, n_col + 0, ' ' + '-' * (len(name) + 2))
        n_line += 1
        # report
        for key, value in self.formatted_report:
            addstr(n_line, n_col + key[0], key[1], key[2])
            # Draw multiple lines of values
            for v in value:
                addstr(n_line, n_col + v[0], v[1], v[2])
                n_line += 1
        # bar
        for text in self.progress_bar:
            addstr(n_line, n_col + 0, text)
            n_line += 1
        return n_line

    def _flush_summary_str(self, n_line, n_col):
        name = "SUMMARY: " + self.name
        addstr(n_line, min(_NCOLS - len(name), n_col + 12), name, curses.A_STANDOUT)
        n_line += 1
        for i in range(self.nb_epoch):
            summary = self._epoch_summary[i]
            addstr(n_line, n_col, "Epoch %d" % i, curses.A_STANDOUT)
            n_line += 1
            for key, value in sorted(summary.iteritems(), key=lambda x: x[0]):
                addstr(n_line, n_col, key, curses.A_BOLD)
                # print value in multiple line
                value = str(value).split('\n')
                for v in value:
                    addstr(n_line, len(key) + 2, v, curses.A_NORMAL)
                    n_line += 1

    def add(self, n):
        if n <= 0:
            return self
        # ====== update information ====== #
        seen_so_far = min(self._seen_so_far[self.epoch_idx] + n, self.target)
        current_timestamp = time.time()
        self._seen_so_far[self.epoch_idx] = seen_so_far
        if self._last_update is None:
            self._last_update = current_timestamp
        else:
            curr_time = current_timestamp
            duration = curr_time - self._last_update
            self._total_time[self.epoch_idx] += duration
            self._update_time_hist[self.epoch_idx].append(duration / n)
            self._last_update = curr_time
        # prepare print screen
        if self in _PROGBAR_STACK:
            # print tutorial
            start_line, start_col = _show_notification(
                *_show_progbar_hist(*_show_tutorial(0, 0)))
            # processing the keyboard input
            ch = stdscr().getch()
            if ch != curses.ERR:
                if ch == ord('s'): # handle toggle Summary
                    scrollRESET()
                    stdscr().erase()
                    if _CURRENT_PROG_SUMMARY[0] is not None:
                        _CURRENT_PROG_SUMMARY[0] = None
                    else:
                        _CURRENT_PROG_SUMMARY[0] = self
                elif ch == ord('i'): # scroll UP
                    scrollUP(3)
                elif ch == ord('k'): # scoll DOWN
                    scrollDOWN(3)
                elif ch == ord('r'): # scoll RESET
                    scrollRESET()
                # show the summary of 1 progress in the history
                elif ch in _NUMBERS_CH:
                    i = _NUMBERS_CH[ch]
                    if i < len(_PROGBAR_HISTORY):
                        scrollRESET()
                        stdscr().erase()
                        _CURRENT_PROG_SUMMARY[0] = _PROGBAR_HISTORY[i]
            # Printout information
            if _CURRENT_PROG_SUMMARY[0] is not None:
                # check summary timeout
                _CURRENT_PROG_SUMMARY[0]._flush_summary_str(start_line, start_col)
            else: # flush progress bar
                # update child progress
                for pb in _PROGBAR_STACK:
                    if pb is self:
                        break
                    start_line = pb._flush_str(start_line, start_col) + 1
                # update the self progress
                self._flush_str(start_line, start_col)
            refresh()
        # check if finish the process, reset new epoch
        if seen_so_far == self.target:
            self.reset()
        return self


@contextmanager
def _progbar(prog, print_progress, print_summary, confirm_exit):
    if not print_progress:
        yield prog
    else:
        _exception_happend = None
        try:
            # initialize window
            stdscr()
            if len(_PROGBAR_STACK) == 0:
                # Turn off echoing of keys, and enter cbreak mode,
                # where no buffering is performed on keyboard input
                curses.noecho()
                curses.cbreak()
            else:
                # pause all other progess
                for pb in _PROGBAR_STACK:
                    if pb is not prog:
                        pb.pause()
            # ====== run the progress ====== #
            if prog not in _PROGBAR_STACK:
                _PROGBAR_STACK.append(prog)
            # record history
            if prog in _PROGBAR_HISTORY:
                _PROGBAR_HISTORY.remove(prog)
            _PROGBAR_HISTORY.insert(0, prog)
            if len(_PROGBAR_HISTORY) > 10:
                _PROGBAR_HISTORY.pop()
            yield prog
            # ====== finish the ProgBar ====== #
            _PROGBAR_STACK.remove(prog)
            prog.pause()
            _FINISHED_PROGBAR.append((time.time(), prog))
            # erase old progress
            stdscr().erase()
            # ====== confirm exit ====== #
            if len(_PROGBAR_STACK) == 0 and confirm_exit:
                scrollRESET()
                stdscr().erase()
                prog._flush_summary_str(
                    *_show_notification(
                        *_show_progbar_hist(
                            *_show_tutorial(0, 0)))); refresh()
                while True:
                    ch = stdscr().getch()
                    if ch == curses.KEY_ENTER or ch == 10 or ch == 13: # Exit
                        break
                    elif ch == ord('i'): # scroll UP
                        scrollUP(3); refresh()
                    elif ch == ord('k'): # scoll DOWN
                        scrollDOWN(3); refresh()
                    elif ch == ord('r'): # scoll RESET
                        scrollRESET(); refresh()
                    elif ch in _NUMBERS_CH:
                        i = _NUMBERS_CH[ch]
                        if i < len(_PROGBAR_HISTORY):
                            scrollRESET()
                            stdscr().erase()
                            _PROGBAR_HISTORY[i]._flush_summary_str(
                                *_show_notification(
                                    *_show_progbar_hist(
                                        *_show_tutorial(0, 0)))); refresh()
        except Exception as e:
            _exception_happend = e
        finally:
            # Set everything back to normal
            if len(_PROGBAR_STACK) == 0 or _exception_happend is not None:
                curses.echo()
                curses.nocbreak()
                curses.endwin()
                # check if exception happened
                if _exception_happend:
                    raise _exception_happend
                # print summary
                if print_summary:
                    summary = [p[1].summary
                        for p in sorted(_FINISHED_PROGBAR, key=lambda x: x[0])][::-1]
                    for s in summary:
                        sys.stdout.write(s); sys.stdout.flush()
                # remove all old data
                for i in range(len(_FINISHED_PROGBAR)):
                    _FINISHED_PROGBAR.pop()
                # clear notification
                clear_notification()
