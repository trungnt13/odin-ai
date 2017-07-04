# -*- coding: utf-8 -*-
# ===========================================================================
# The following signal might returned by Callbacks:
# * SIG_TRAIN_STOP: stop the training task
# * SIG_TRAIN_SAVE: save the parameters during training
# * SIG_TRAIN_ROLLBACK: rollback the model to the best checkpoint
# ===========================================================================
from __future__ import division, absolute_import, print_function

import re
import sys
import time
import timeit
import warnings
from numbers import Number
from datetime import datetime
from collections import defaultdict
from abc import ABCMeta, abstractproperty, abstractmethod
from six import add_metaclass, string_types
from six.moves import cPickle

import numpy as np
import scipy as sp

from odin.utils import as_tuple, is_string, progbar
from odin.utils.decorators import functionable

__all__ = [
    'SIG_TRAIN_SAVE',
    'SIG_TRAIN_ROLLBACK',
    'SIG_TRAIN_STOP',
    'Callback',
    'Debug',
    'CallbackList',
    'NaNDetector',
    'EarlyStop',
    'EarlyStopGeneralizationLoss',
    'EarlyStopPatience',
]

# This SIGNAL can terminate running iterator (or generator),
# however, it cannot terminate them at the beginning,
# so you should not call .send(SIG_TERMINATE_ITERATOR)
# at beginning of iterator
SIG_TRAIN_SAVE = '__signal_train_save__'
SIG_TRAIN_ROLLBACK = '__signal_train_rollback__'
SIG_TRAIN_STOP = '__signal_train_stop__'

_ALLOW_MSG = {
    SIG_TRAIN_SAVE: 1,
    SIG_TRAIN_ROLLBACK: 1,
    SIG_TRAIN_STOP: 1,
}


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


def time2date(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%y-%m-%d %H:%M:%S')


def date2time(date):
    return time.mktime(datetime.datetime.strptime(date, '%y-%m-%d %H:%M:%S').timetuple())


# ===========================================================================
# Callbacks
# ===========================================================================
class Callback(object):

    """Callback

    Callbacks
    --------
    task_start(self)
    task_end(self)
    epoch_start(self)
    epoch_end(self)
    batch_start(self)
    batch_end(self)

    """

    def __init__(self):
        super(Callback, self).__init__()

    def batch_start(self, task, batch):
        pass

    def batch_end(self, task, batch_results):
        pass

    def epoch_start(self, task, data):
        pass

    def epoch_end(self, task, epoch_results):
        pass

    def task_start(self, task):
        pass

    def task_end(self, task, task_results):
        pass

    def send_notification(self, msg):
        progbar.add_notification('[%s]%s' % (self.__class__.__name__, msg))
        return self


class Debug(Callback):
    """docstring for Debug"""

    def __init__(self):
        super(Debug, self).__init__()

    def batch_start(self, task, batch):
        print("Batch Start:", task.name, task.curr_epoch, task.curr_samples,
              [(i.shape, i.dtype, type(i)) for i in batch])

    def batch_end(self, task, batch_results):
        print("Batch End:", task.name, task.curr_epoch, task.curr_samples,
              [(i.shape, i.dtype, type(i)) for i in as_tuple(batch_results)])

    def epoch_start(self, task, data):
        print("Epoch Start:", task.name, task.curr_epoch, task.curr_samples,
              [(i.shape, i.dtype, type(i)) for i in data])

    def epoch_end(self, task, epoch_results):
        print("Epoch End:", task.name, task.curr_epoch, task.curr_samples,
              [(i, len(j), type(j[0])) for i, j in epoch_results.iteritems()])

    def task_start(self, task):
        print("Task Start:", task.name, task.curr_epoch, task.curr_samples)

    def task_end(self, task, task_results):
        print("Task End:", task.name, task.curr_epoch, task.curr_samples,
            [(i, [(n, len(v), type(v[0])) for n, v in j.iteritems()])
             for i, j in task_results.iteritems()])


class CallbackList(Callback):

    ''' Broadcast signal to all its children'''

    def __init__(self, callbacks = None):
        super(CallbackList, self).__init__()
        self.set_callbacks(callbacks)

    def set_callbacks(self, callbacks):
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, CallbackList):
            callbacks = callbacks._callbacks
        else:
            callbacks = as_tuple(callbacks, t = Callback)
        self._callbacks = [i for i in set(callbacks)]
        return self

    def __str__(self):
        return '<CallbackList: ' + \
        ', '.join([i.__class__.__name__ for i in self._callbacks]) + '>'

    def batch_start(self, task, batch):
        msg = []
        for i in self._callbacks:
            m = i.batch_start(task, batch)
            msg += [j for j in as_tuple(m) if j in _ALLOW_MSG]
        return msg

    def batch_end(self, task, batch_results):
        msg = []
        for i in self._callbacks:
            m = i.batch_end(task, batch_results)
            msg += [j for j in as_tuple(m) if j in _ALLOW_MSG]
        return msg

    def epoch_start(self, task, data):
        msg = []
        for i in self._callbacks:
            m = i.epoch_start(task, data)
            msg += [j for j in as_tuple(m) if j in _ALLOW_MSG]
        return msg

    def epoch_end(self, task, epoch_results):
        msg = []
        for i in self._callbacks:
            m = i.epoch_end(task, epoch_results)
            msg += [j for j in as_tuple(m) if j in _ALLOW_MSG]
        return msg

    def task_start(self, task):
        msg = []
        for i in self._callbacks:
            m = i.task_start(task)
            msg += [j for j in as_tuple(m) if j in _ALLOW_MSG]
        return msg

    def task_end(self, task, task_results):
        msg = []
        for i in self._callbacks:
            m = i.task_end(task, task_results)
            msg += [j for j in as_tuple(m) if j in _ALLOW_MSG]
        return msg


# ===========================================================================
# NaN value detection
# ===========================================================================
class NaNDetector(Callback):
    """docstring for NaNDetector"""

    def __init__(self, task_name=None, patience=-1):
        super(NaNDetector, self).__init__()
        self._task_name = task_name
        self._patience = patience

    def batch_end(self, task, batch_results):
        if self._task_name is not None and task.name != self._task_name:
            return
        # found any NaN values
        if any(np.any(np.isnan(r)) for r in as_tuple(batch_results)):
            signal = SIG_TRAIN_ROLLBACK
            self._patience -= 1
            if self._patience <= 0: # but if out of patience, stop
                signal = SIG_TRAIN_STOP
            self.send_notification("Found NaN value, task:%s" % task.name)
            return signal


# ===========================================================================
# EarlyStop utilities
# ===========================================================================
@add_metaclass(ABCMeta)
class EarlyStop(Callback):
    """
    Early Stopping algorithm based on Generalization Loss criterion,
    this is strict measure on validation

    ``LOWER is better``

    Parameters
    ----------
    name : string
        task name for checking this criterion
    threshold : float
        for example, threshold = 5, if we loss 5% of performance on validation
        set, then stop
    patience: int
        how many cross the threshold that still can be rollbacked
    get_value : function
        function to process the results of whole epoch (i.e list of results
        returned from batch_end) to return comparable number.
        For example, lambda x: np.mean(x)
    stop_callback: function
        will be called when stop signal triggered
    save_callback: function
        will be called when save signal triggered

    Note
    ----
    * The early stop checking will be performed at the end of an epoch.
    * By default, the return value from epoch mean the loss value, i.e lower
    is better
    * If multiple value returned, you have to modify the get_value function
    """

    def __init__(self, task_name, output_name, threshold, patience=1,
                 get_value=lambda x: np.mean(x)):
        super(EarlyStop, self).__init__()
        self._task_name = str(task_name)
        self._output_name = output_name if is_string(output_name) \
            else output_name.name

        self._threshold = float(threshold)
        self._patience = int(patience)

        if get_value is None:
            get_value = lambda x: x
        elif not callable(get_value):
            raise ValueError('get_value must callable')
        self._get_value = functionable(get_value)
        # ====== history ====== #
        self._history = []

    # ==================== main callback methods ==================== #
    def epoch_end(self, task, epoch_results):
        if self._task_name != task.name:
            return
        self._history.append(self._get_value(epoch_results[self._output_name]))
        # ====== check early stop ====== #
        shouldSave, shouldStop = self.earlystop(self._history, self._threshold)
        msg = []
        if shouldSave > 0:
            msg = SIG_TRAIN_SAVE
        if shouldStop > 0:
            msg = SIG_TRAIN_ROLLBACK
            # check patience
            self._patience -= 1
            if self._patience <= 0:
                msg = SIG_TRAIN_STOP
        return msg

    @abstractmethod
    def earlystop(self, history, threshold):
        """ Any algorithm return: shouldSave, shouldStop """
        pass


class EarlyStopGeneralizationLoss(EarlyStop):
    """ Early Stopping algorithm based on Generalization Loss criterion,
    this is strict measure on validation

    ``LOWER is better``

    Parameters
    ----------
    name : string
        task name for checking this criterion
    threshold : float
        for example, threshold = 5, if we loss 5% of performance on validation
        set, then stop
    patience: int
        how many cross the threshold that still can be rollbacked
    get_value : function
        function to process the results of whole epoch (i.e list of results
        returned from batch_end) to return comparable number.

    Note
    ----
    The early stop checking will be performed at the end of an epoch.
    By default, the return value from epoch mean the loss value, i.e lower
    is better.
    By default, the `get_value` function will only take the first returned
    value for evaluation.
    """

    def earlystop(self, history, threshold):
        gl_exit_threshold = threshold
        longest_remain_performance = int(gl_exit_threshold + 1)
        epsilon = 1e-8

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
        if len(history) > longest_remain_performance:
            remain_detected = 0
            j = history[-longest_remain_performance]
            for i in history[-longest_remain_performance:]:
                if abs(i - j) < 1e-5:
                    remain_detected += 1
            if remain_detected >= longest_remain_performance:
                shouldStop = 1
        return shouldSave, shouldStop


class EarlyStopPatience(EarlyStop):
    """
    EarlyStopPatience(self, name, threshold, patience=1,
              get_value=lambda x: np.mean([i[0] for i in x]
                                          if isinstance(x[0], (tuple, list))
                                          else x),
              stop_callback=None,
              save_callback=None)


    Adapted algorithm from keras:
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
    name : string
        task name for checking this criterion
    threshold : float
        for example, threshold = 5, if we loss 5% of performance on validation
        set, then stop
    patience: int
        how many cross the threshold that still can be rollbacked
    get_value : function
        function to process the results of whole epoch (i.e list of results
        returned from batch_end) to return comparable number.
        for example, lambda x: np.mean(x)

    """

    def earlystop(self, history, threshold):
        if not hasattr(self, 'wait'): self.wait = 0
        shouldSave, shouldStop = 0, 0
        # showed improvement, should not equal to old best
        if len(history) <= 1 or history[-1] < np.min(history[:-1]):
            self.wait = 0
            shouldSave = 1
        else:
            if self.wait >= threshold:
                shouldSave = -1
                shouldStop = 1
            self.wait += 1
        return shouldSave, shouldStop
