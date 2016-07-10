# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

import sys
import time
import timeit
import cPickle
from datetime import datetime
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy as np

from odin.nnet import NNOps
from odin.utils import Progbar
from odin.utils.decorators import functionable

__all__ = [
    'Callback',
    'CallbackList',
    'History',
    'EarlyStopGeneralizationLoss',
    'EarlyStopPatience',
    'ProgressMonitor',
    'Checkpoint',
    'Debug',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _parse_result(result):
    if isinstance(result, (tuple, list)) and len(str(result)) > 20:
        type_str = ''
        if len(result) > 0:
            type_str = type(result[0]).__name__
        return 'list;%d;%s' % (len(result), type_str)
    s = str(result)
    return s[:20]


TASK_TYPES = ['task', 'subtask', 'crosstask', 'othertask']


# ===========================================================================
# Callbacks
# ===========================================================================
class Callback(object):

    """Callback

    Properties
    ----------
    task: current task
    results: return results
    iter: current number of iteration
    epoch: current number of epoch
    mode: 'task', 'subtask', 'crosstask', 'save'
    mainloop: the running loop

    Callbacks
    --------
    task_start(self)
    task_end(self)
    epoch_start(self)
    epoch_end(self)
    batch_end(self)

    Note
    ----
    This object can be used for many different task, just call reset before
    switching to other task

    """

    def __init__(self):
        super(Callback, self).__init__()
        self._task = None
        self._results = None
        self._iter = defaultdict(int)
        self._epoch = defaultdict(int)

        self._mode = None # 'task', 'subtask', 'crosstask'
        self._mainloop = None

    def __setstate__(self, value):
        self._task = None
        self._results = None
        self._iter = value[0]
        self._epoch = value[1]
        self._mode = None # 'itask', 'subtask', 'crosstask'
        self._mainloop = None

    def __getstate__(self):
        return self._iter, self._epoch

    # ==================== helpers ==================== #
    @property
    def mainloop(self):
        return self._mainloop

    @mainloop.setter
    def mainloop(self, value):
        self._mainloop = value

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if hasattr(value, 'name'):
            self._task = value

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in TASK_TYPES:
            self._mode = value

    @property
    def iter(self):
        return self._iter[self.task]

    @iter.setter
    def iter(self, value):
        self._iter[self.task] = value

    @property
    def epoch(self):
        return self._epoch[self.task]

    @epoch.setter
    def epoch(self, value):
        self._epoch[self.task] = value

    def reset(self):
        self._task = None
        self._results = None

    # ==================== main callback methods ==================== #
    def task_start(self):
        pass

    def task_end(self):
        pass

    def epoch_start(self):
        pass

    def epoch_end(self):
        pass

    def batch_end(self):
        pass


class CallbackList(Callback):

    ''' Broadcast signal to all its children'''

    def __init__(self, *args):
        super(CallbackList, self).__init__()
        # ====== check duplicate callback types ====== #
        seen_callback_type = [i.__class__ for i in args]
        if len(set(seen_callback_type)) != len(seen_callback_type):
            raise Exception('You cannot have 2 or more callbacks of the '
                            'same type in CallbackList.')
        # ====== add callback to list ====== #
        self._callbacks = []
        for i in args:
            self.add_callback(i)

    def add_callback(self, callback):
        if isinstance(callback, Callback) and callback not in self._callbacks:
            self._callbacks.append(callback)

    def __getitem__(self, key):
        if isinstance(key, str):
            for i in self._callbacks:
                if key in i.__class__.__name__:
                    return i
        return self._callbacks[key]

    def __setstate__(self, value):
        super(CallbackList, self).__setstate__(value[0])
        self._callbacks = value[1]

    def __getstate__(self):
        return super(CallbackList, self).__getstate__(), self._callbacks

    def __str__(self):
        return 'CallbackList: ' + ', '.join([i.__class__.__name__ for i in self._callbacks])

    # ==================== helpers ==================== #
    @Callback.mainloop.setter
    def mainloop(self, value):
        self._mainloop = value
        for i in self._callbacks:
            i.mainloop = value

    @Callback.task.setter
    def task(self, value):
        self._task = value
        for i in self._callbacks:
            i.task = value

    @Callback.results.setter
    def results(self, value):
        self._results = value
        for i in self._callbacks:
            i.results = value

    @Callback.mode.setter
    def mode(self, value):
        self._mode = value
        for i in self._callbacks:
            i.mode = value

    @Callback.iter.setter
    def iter(self, value):
        self._iter[self.task] = value
        for i in self._callbacks:
            i.iter = value

    @Callback.epoch.setter
    def epoch(self, value):
        self._epoch[self.task] = value
        for i in self._callbacks:
            i.epoch = value

    def reset(self):
        self.task = None
        self.results = None
        for i in self._callbacks:
            i.reset()

    # ==================== main callback methods ==================== #
    def task_start(self):
        for i in self._callbacks:
            i.task_start()

    def task_end(self):
        for i in self._callbacks:
            i.task_end()

    def epoch_start(self):
        for i in self._callbacks:
            i.epoch_start()

    def epoch_end(self):
        for i in self._callbacks:
            i.epoch_end()

    def batch_end(self):
        for i in self._callbacks:
            i.batch_end()


# ===========================================================================
# Checkpoint utilities
# ===========================================================================
class Checkpoint(Callback):
    """ Checkpoint
    Note
    ----
    Checkpoint is created once whenever MainLoop.save() is called.
    This class (by defaults) pickles everything
    """

    def __init__(self, path):
        super(Checkpoint, self).__init__()
        self.path = path
        self._save_obj = None

    def task_start(self):
        if self.mode == 'othertask' and self.task.name == 'save':
            self._save()

    def set_obj(self, obj):
        self._save_obj = obj
        return self

    def _save(self):
        if self._save_obj is None:
            raise Exception('You must set_obj for Checkpoint first.')
        cPickle.dump(self._save_obj,
                     open(self.path, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)

    # ==================== Pickling ==================== #
    def __getstate__(self):
        return super(Checkpoint, self).__getstate__(), self.path

    def __setstate__(self, value):
        super(Checkpoint, self).__setstate__(value[0])
        self.path = value[1]
        self._save_obj = None


# ===========================================================================
# EarlyStop utilities
# ===========================================================================
@add_metaclass(ABCMeta)
class EarlyStop(Callback):
    """ Early Stopping algorithm based on Generalization Loss criterion,
    this is strict measure on validation

    Parameters
    ----------
    threshold : float
        for example, threshold = 5, if we loss 5% of performance on validation
        set, then stop
    task : string
        task name for checking this criterion
    get_value : function
        function to process the results of given task.

    Note
    ----
    The early stop checking will be performed at the end of an epoch.
    By default, the return value from epoch mean the loss value, i.e lower
    is better
    """

    def __init__(self, threshold, task, get_value=lambda x: np.mean(x)):
        super(EarlyStop, self).__init__()
        self.threshold = threshold
        self.task_name = task
        if get_value is not None and not hasattr(get_value, '__call__'):
            raise ValueError('get_value must callable')
        self.get_value = None if get_value is None else functionable(get_value)
        self._history = []

    # ==================== main callback methods ==================== #
    def task_start(self):
        pass

    def task_end(self):
        pass

    def epoch_start(self):
        pass

    def epoch_end(self):
        if self.task.name == self.task_name:
            value = self.results
            if self.get_value is not None:
                value = self.get_value(value)
            self._history.append(value)
            # ====== check early stop ====== #
            shouldSave, shouldStop = self.earlystop(self._history)
            if shouldSave > 0:
                self.mainloop.save()
            if shouldStop > 0:
                self.mainloop.stop()

    def batch_end(self):
        pass

    @abstractmethod
    def earlystop(self, history):
        """ Any algorithm return: shouldSave, shouldStop """
        pass

    # ==================== Pickling ==================== #
    def __getstate__(self):
        return (super(EarlyStop, self).__getstate__(),
        self._history, self.threshold, self.task_name, self.get_value)

    def __setstate__(self, value):
        super(EarlyStop, self).__setstate__(value[0])
        self._history = value[1]
        self.threshold = value[2]
        self.task_name = value[3]
        self.get_value = value[4]


class EarlyStopGeneralizationLoss(EarlyStop):

    def earlystop(self, history):
        gl_exit_threshold = self.threshold
        epsilon = 1e-5

        if len(history) == 0: # no save, no stop
            return 0, 0
        shouldStop = 0
        shouldSave = 0

        gl_t = 100 * (history[-1] / (min(history) + epsilon) - 1)
        if gl_t <= 0 and np.argmin(history) == (len(history) - 1):
            shouldSave = 1
            shouldStop = -1
        elif gl_t > gl_exit_threshold:
            shouldStop = 1
            shouldSave = -1

        # check stay the same performance for so long
        if len(history) > gl_exit_threshold:
            remain_detected = 0
            j = history[-int(gl_exit_threshold)]
            for i in history[-int(gl_exit_threshold):]:
                if abs(i - j) < epsilon:
                    remain_detected += 1
            if remain_detected >= gl_exit_threshold:
                shouldStop = 1
        return shouldSave, shouldStop


class EarlyStopPatience(EarlyStop):
    """ Adapted algorithm from keras:
    All contributions by François Chollet:
    Copyright (c) 2015, François Chollet.
    All rights reserved.

    All contributions by Google:
    Copyright (c) 2015, Google, Inc.
    All rights reserved.

    All other contributions:
    Copyright (c) 2015, the respective contributors.
    All rights reserved.

    LICENSE: https://github.com/fchollet/keras/blob/master/LICENSE

    Stop training when a monitored quantity has stopped improving.

    Parameters
    ----------
    patience: number of epochs with no improvement
        after which training will be stopped.
    """

    def earlystop(self, history):
        if not hasattr(self, 'wait'):
            self.wait = 0

        shouldSave, shouldStop = 0, 0
        if history[-1] <= np.min(history): # showed improvement
            self.wait = 0
            shouldSave = 1
        else:
            if self.wait >= self.threshold:
                shouldSave = -1
                shouldStop = 1
            self.wait += 1
        return shouldSave, shouldStop


# ===========================================================================
# Extension
# ===========================================================================
class ProgressMonitor(Callback):

    '''
    Parameters
    ----------
    title : str
        pattern to serialize return from function to string

    Example
    -------
    >>> t = training.Task(dataset=ds, batch_size=512)
    >>> t.set_callback(training.ProgressMonitor(title='Result: %.2f'))
    >>> t.run()
    # Result: 52751.29 98/98 [=======================================] - 0s
    '''

    def __init__(self, title=''):
        super(ProgressMonitor, self).__init__()
        self._format_results = False
        if len(list(title._formatter_parser())) > 0:
            self._format_results = True
        self._prog = Progbar(100, title='')
        self._title = title

    def __getstate__(self):
        return super(ProgressMonitor, self).__getstate__(), self._title, self._format_results

    def __setstate__(self, value):
        super(ProgressMonitor, self).__setstate__(value[0])
        self._title = value[1]
        self._format_results = value[2]
        self._prog = Progbar(100, title='')

    def batch_end(self):
        # do nothing for crosstask
        if self._mode == 'crosstask':
            return

        title = (self._title % self.results
                 if self._format_results else self._title)
        # title
        self._prog.title = 'Name:%-8s,Epoch:%2d,' % (self.task.name[:8], self.epoch) + title
        # progress
        iter_per_epoch = self.task.iter_per_epoch
        n = round(((self.iter % iter_per_epoch) / iter_per_epoch) * 100)
        self._prog.update(min(int(n), 99))

    def epoch_end(self):
        # get the last results
        title = (self._title % self.results[-1]
                 if self._format_results else self._title)
        # title
        self._prog.title = 'Name:%-8s,Epoch:%2d,' % (self.task.name[:8], self.epoch) + title
        # always 100% at the end of epoch
        self._prog.update(100)


class History(Callback):

    ''' Record the executing history in following format:
        |Datatime; event_type; task; result; iter; epoch|
    event_type : 6 events
        * task_start
        * task_end
        * batch_start
        * batch_end
        * epoch_start
        * epoch_end
    '''
    @staticmethod
    def time2date(timestamp):
        return datetime.fromtimestamp(timestamp).strftime('%y-%m-%d %H:%M:%S')

    @staticmethod
    def date2time(date):
        return time.mktime(datetime.datetime.strptime(date, '%y-%m-%d %H:%M:%S').timetuple())

    def __init__(self):
        super(History, self).__init__()
        self._history = []

    def task_start(self):
        t = timeit.default_timer()
        self._history.append((t, 'task_start', self.task.name,
                              self.results, self.iter, self.epoch))

    def task_end(self):
        t = timeit.default_timer()
        self._history.append((t, 'task_end', self.task.name,
                              self.results, self.iter, self.epoch))

    def epoch_start(self):
        t = timeit.default_timer()
        self._history.append((t, 'epoch_start', self.task.name,
                              self.results, self.iter, self.epoch))

    def epoch_end(self):
        t = timeit.default_timer()
        self._history.append((t, 'epoch_end', self.task.name,
                              self.results, self.iter, self.epoch))

    def batch_end(self):
        t = timeit.default_timer()
        self._history.append((t, 'batch_end', self.task.name,
                              self.results, self.iter, self.epoch))

    def get(self, task, event):
        """
        Parameters
        ----------
        task : str
            name of task
        event : str
            task_start, task_end, batch_end, epoch_start, epoch_end
            if 'task' or epoch event is queried, a list of results is
            returned
        """
        return [i[3] for i in self._history if i[1] == event and i[2] == task]

    def benchmark(self, task, event):
        '''
        Parameters
        ----------
        task : str
            name of given task want to benchmark
        event : 'batch', 'epoch', 'task'
            kind of event (e.g benchmark for each epoch, or batch)

        Return
        ------
        time : in second

        '''
        # ====== prepare ====== #
        if 'task' in event:
            event = 'task'
        elif 'epoch' in event:
            event = 'epoch'
        else:
            event = 'batch'
        history = [(i[0], i[1]) for i in self._history if task == i[2] and event in i[1]]
        # ====== benchmark ====== #
        if len(history) >= 2:
            if event == 'batch':
                history = [i[0] for i in history]
                return np.mean([j - i for i, j in zip(history, history[1:])])
            start = [i[0] for i in history if 'start' in i[1]]
            end = [i[0] for i in history if 'end' in i[1]]
            return np.mean([j - i for i, j in zip(start, end)])
        return None

    def __str__(self):
        format_str = "%s | %-12s | %-8s | %-20s | %-4s | %-4s"
        s = "=" * 24 + " N: %d " % len(self._history) + "=" * 24 + '\n'
        for i in self._history:
            i = (History.time2date(i[0]),) + i[1:]
            s += format_str % tuple([_parse_result(j) for j in i]) + '\n'
        return s

    # ==================== pickle interface ==================== #
    def __getstate__(self):
        return super(History, self).__getstate__(), self._history

    def __setstate__(self, value):
        super(History, self).__setstate__(value[0])
        self._history = value[1]


class Debug(Callback):

    def task_start(self):
        print()
        print('%-12s' % 'task_start', '%-12s' % self.task.name,
              None, '%4d' % self.iter, '%4d' % self.epoch)

    def task_end(self):
        print('%-12s' % 'task_end', '%-12s' % self.task.name,
              '%4d' % len(self.results), '%4d' % self.iter, '%4d' % self.epoch)
        print()

    def epoch_start(self):
        print('%-12s' % 'epoch_start', '%-12s' % self.task.name,
              None, '%4d' % self.iter, '%4d' % self.epoch)

    def epoch_end(self):
        print('%-12s' % 'epoch_end', '%-12s' % self.task.name,
              '%4d' % len(self.results), '%4d' % self.iter, '%4d' % self.epoch)

    def batch_end(self):
        # print('batch end', self.results)
        pass
