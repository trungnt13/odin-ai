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
import cPickle
import warnings
from numbers import Number
from datetime import datetime
from collections import defaultdict
from abc import ABCMeta, abstractproperty, abstractmethod
from six import add_metaclass

import numpy as np
import scipy as sp

from odin import SIG_TRAIN_ROLLBACK, SIG_TRAIN_STOP, SIG_TRAIN_SAVE
from odin import backend as K
from odin.utils import Progbar, as_tuple
from odin.utils.decorators import functionable

__all__ = [
    'Callback',
    'CallbackList',
    'Checkpoint',
    'History',
    'NaNDetector',
    'EarlyStop',
    'EarlyStopGeneralizationLoss',
    'EarlyStopPatience',
    'ProgressMonitor',
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


def time2date(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%y-%m-%d %H:%M:%S')


def date2time(date):
    return time.mktime(datetime.datetime.strptime(date, '%y-%m-%d %H:%M:%S').timetuple())


# ===========================================================================
# Callbacks
# ===========================================================================
@add_metaclass(ABCMeta)
class Callback(object):

    """Callback

    Properties
    ----------
    task: current task
    results: return results
    iter: current number of iteration
    epoch: current number of epoch
    mode: 'task', 'subtask', 'crosstask', 'othertask'
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
        self.__event_name = None
        self.__event_type = None
        self.__results = None
        self._kwargs = None

        self.__nb_samples = defaultdict(int)
        self.__nb_iter = defaultdict(int)
        self.__nb_epoch = defaultdict(int)

    def __setstate__(self, value):
        self.__event_name = None
        self.__event_type = None
        self.__results = None
        self._kwargs = None

        self.__nb_iter = value[0]
        self.__nb_epoch = value[1]
        self.__nb_samples = value[2]

        # ====== load variables ====== #
        for i, j in value[-1].iteritems():
            setattr(self, i, j)

    def __getstate__(self):
        return (self.__nb_iter, self.__nb_epoch, self.__nb_samples,
                self._saveable_variables)

    # ==================== utilities methods ==================== #
    def record(self, event_name, event_type,
               nb_iter, nb_epoch, nb_samples,
               results, **kwargs):
        """
        Parameters
        ----------
        event_name: str
            identity of the event
        event_type: str
            batch_end, batch_start, epoch_end, epoch_start, ...
            the same function will be called
        nb_iter: int
            number of iteration
        nb_epoch: int
            number of epoch
        nb_samples: int
            number of samples has been trained on
        results: object
            any temporary results returned
        **kwargs: dict
            any additional information you want to record

        Return
        ------
        list of string signal after processed the event
        """
        # ====== updates ====== #
        self.__event_name = event_name
        self.__event_type = event_type
        self.__results = results
        self._kwargs = kwargs

        self.__nb_epoch[event_name] = max(nb_epoch, self.__nb_epoch[event_name])
        self.__nb_iter[event_name] = max(nb_iter, self.__nb_iter[event_name])
        self.__nb_samples[event_name] = max(nb_samples, self.__nb_samples[event_name])
        # ====== call appropriate function ====== #
        messages = []
        if hasattr(self, str(event_type)):
            handler = getattr(self, str(event_type))
            msg = handler()
            # handle both list of messages and only 1 message
            if isinstance(msg, (tuple, list)):
                messages += msg
            elif msg is not None:
                messages.append(msg)
        return messages

    @abstractproperty
    def _saveable_variables(self):
        """ Dictionary of attribute_name: attribute_value
        of callback that you will it be saved during pickling
        """
        return {}

    # ==================== helpers ==================== #
    @property
    def event_name(self):
        return self.__event_name

    @property
    def event_type(self):
        return self.__event_type

    @property
    def results(self):
        return self.__results

    @property
    def nb_iter(self):
        return self.__nb_iter[self.__event_name]

    @property
    def nb_samples(self):
        return self.__nb_samples[self.__event_name]

    @property
    def nb_epoch(self):
        return self.__nb_epoch[self.__event_name]

    def __getitem__(self, key):
        return self._kwargs[key]

    def __contains__(self, key):
        return key in self._kwargs


class CallbackList(Callback):

    ''' Broadcast signal to all its children'''

    def __init__(self, *args):
        super(CallbackList, self).__init__()
        # ====== check duplicate callback types ====== #
        args = list(set(args))
        # ====== add callback to list ====== #
        self._callbacks = args

    def __setstate__(self, value):
        super(CallbackList, self).__setstate__(value)

    def __getitem__(self, key):
        # get the callback
        if isinstance(key, str):
            for i in self._callbacks:
                if key in i.__class__.__name__:
                    return i
            raise Exception('Callback with type="%s" not found.' % key)
        # additional kwargs
        if key in self._kwargs:
            return self._kwargs[key]
        # otherwise, indexing by int or slice
        return self._callbacks[key]

    def __contains__(self, key):
        if isinstance(key, str):
            for i in self._callbacks:
                if key in i.__class__.__name__:
                    return True
            return False
        raise ValueError('CallbackList __contains__ only accept str input, but '
                         'given type is: %s' % type(key))

    def __str__(self):
        return 'CallbackList: ' + ', '.join(
            [i.__class__.__name__ for i in self._callbacks])

    # ==================== utilities methods ==================== #
    def record(self, event_name, event_type,
               nb_iter, nb_epoch, nb_samples,
               results, **kwargs):
        # ====== process the event ====== #
        super(CallbackList, self).record(event_name, event_type,
                                         nb_iter, nb_epoch, nb_samples,
                                         results, **kwargs)
        messages = []
        for cb in self._callbacks:
            msg = cb.record(event_name, event_type,
                            nb_iter, nb_epoch, nb_samples,
                            results, **kwargs)
            if msg is not None:
                messages += msg
        return messages

    @property
    def _saveable_variables(self):
        return {'__callbacks': self._callbacks}


# ===========================================================================
# History
# ===========================================================================
class Checkpoint(Callback):

    """ Checkpoint
    Return SIG_TRAIN_SAVE whenever the epoch_end is called with
    given event_name
    """

    def __init__(self, event_name):
        super(Checkpoint, self).__init__()
        self._name = event_name

    @property
    def _saveable_variables(self):
        return {'_name': self._name}

    def epoch_end(self):
        if self.event_name == self._name:
            return SIG_TRAIN_SAVE


class History(Callback):
    """ History
    [time, event_name, event_type, nb_samples, nb_iter, nb_epoch, results]
    """

    def __init__(self):
        super(History, self).__init__()
        self._history = []

    @property
    def history(self):
        return self._history

    @property
    def _saveable_variables(self):
        return {'_history': self._history}

    def record(self, event_name, event_type,
               nb_iter, nb_epoch, nb_samples,
               results, **kwargs):
        self._history.append([
            timeit.default_timer(), # time
            event_name, # name (Train, Valid ...)
            event_type, # batch_end, batch_start ...
            nb_samples, # nb_samples
            nb_iter, # nb_iterations
            nb_epoch, # nb_epoch
            results # results
        ])
        return None

    # ==================== helpers ==================== #
    @property
    def event_types(self):
        """ Return all exist event types in this history """
        return list(set([i[2] for i in self._history]))

    @property
    def event_names(self):
        """ Return all exist event name in this history """
        return list(set([i[1] for i in self._history]))

    def query(self, event_name=None, event_type=None):
        event_name = '' if event_name is None else str(event_name)
        event_type = '' if event_type is None else str(event_type)
        # return the results
        return [i[-1] for i in self._history
                if event_name in i[1] and event_type in i[2]]

    def benchmark(self, event_name, event_type):
        '''
        Return
        ------
        time : in second
            average time in second between the 2 appearances of given events


        '''
        event_name = '' if event_name is None else str(event_name)
        event_type = '' if event_type is None else str(event_type)
        # ====== prepare ====== #
        history = [i[0] for i in self._history
                   if event_name in i[1] and event_type in i[2]]
        # ====== benchmark ====== #
        if len(history) >= 2:
            return sp.stats.describe([j - i for i, j in zip(history, history[1:])])
        return None

    def print_info(self):
        """ Print information about all events appeared in the History """
        events = defaultdict(list)
        for t, name, typ, sa, it, ep, res in self._history:
            events[name].append(typ)
        # ====== print ====== #
        info = {}
        print('\n * [INFO] all event name and type of History:')
        for name, typ in events.iteritems():
            freq = sp.stats.itemfreq(typ)
            info[name] = freq
            print(name, ':', ', '.join(['%s-%d' % (i, int(j)) for i, j in freq]))
        return info

    def print_epoch(self, event_name):
        values = []
        for t, name, typ, sa, it, ep, res in self._history:
            if name == event_name:
                if typ == 'epoch_start': epoch = []
                elif typ == 'epoch_end': values.append(np.mean(epoch, 0))
                elif typ == 'batch_end':
                    if not isinstance(res, (tuple, list)):
                        res = (res,)
                    epoch.append([i for i in res if isinstance(i, Number) or
                                  (isinstance(i, np.ndarray) and i.ndim == 0)])
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        print("\n * [EPOCH] summarization for event: %s" % event_name)
        for i, v in enumerate(values.T):
            print('Results #%d:' % i)
            if len(v) <= 2:
                s = event_name + ':' + str(v)
            else:
                from odin.visual.bashplot import print_bar
                s = print_bar(v, height=20,
                              bincount=max(20, len(v)),
                              showSummary=True)
            print(s)

    def print_batch(self, event_name):
        values = []
        for t, name, typ, sa, it, ep, res in self._history:
            if name == event_name and typ == 'batch_end':
                if not isinstance(res, (tuple, list)):
                    res = (res,)
                values.append([i for i in res if isinstance(i, Number) or
                              (isinstance(i, np.ndarray) and i.ndim == 0)])
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        print("\n * [BATCH] summarization for event: %s" % event_name)
        for i, v in enumerate(values.T):
            print('Results #%d:' % i)
            if len(v) <= 2:
                s = event_name + ':' + str(v)
            else:
                from odin.visual.bashplot import print_bar
                s = print_bar(v, height=20,
                              bincount=min(20, len(v)),
                              showSummary=True)
            print(s)


# ===========================================================================
# NaN value detection
# ===========================================================================
class NaNDetector(Callback):
    """ NaNStop
    Simply stop training, or rollback the model when any batch of given
    with event name return NaN results

    This method search for "mainloop" in kwargs provided in `record`
    to call .stop() function

    Parameters
    ----------
    name: str, or list of str
        list of all event name that will be checked for NaN
    patience: int
        if patience > 0, send rollback signal until patience reduced to 0,
        else the training process will never stop because of NaN
    rollback: boolean
        if True, rollback the model at the best checkpoint whenever NaN
        value detected


    """

    def __init__(self, name, patience=-1, rollback=True):
        super(NaNDetector, self).__init__()
        self.name = as_tuple(name, t=(str, unicode))
        self.patience = patience
        self.rollback = rollback
        self._current_patience = patience

    @property
    def _saveable_variables(self):
        return {'name': self.name,
                'patience': self.patience,
                'rollback': self.rollback,
                '_current_patience': self.patience}

    def batch_end(self):
        if any(self.event_name == n for n in self.name):
            results = self.results if isinstance(self.results, (tuple, list)) \
                else (self.results,)
            # found any NaN values
            if any(np.any(np.isnan(r)) for r in results):
                signal = None
                if self.rollback: # maybe rollback
                    signal = SIG_TRAIN_ROLLBACK
                if self.patience > 0: # but if out of patience, stop
                    if self._current_patience > 0:
                        self._current_patience -= 1
                    else:
                        signal = SIG_TRAIN_STOP
                print('\nNaN value(s) was detected in task:"%s" results, '
                      '"%s" ...' % (self.event_name, signal))
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

    def __init__(self, name, threshold, patience=1,
                 get_value=lambda x: np.mean([i[0] for i in x]
                                             if isinstance(x[0], (tuple, list))
                                             else x),
                 stop_callback=None,
                 save_callback=None):
        super(EarlyStop, self).__init__()
        self.name = str(name)
        self.threshold = float(threshold)
        self.patience = int(patience)
        self._current_patience = self.patience

        if get_value is not None and not callable(get_value):
            raise ValueError('get_value must callable')
        self.get_value = functionable(get_value)
        # ====== history ====== #
        self._working_history = []
        self._history = []
        # ====== lr ====== #
        self.stop_callback = stop_callback
        self.save_callback = save_callback

    @property
    def _saveable_variables(self):
        return {'threshold': self.threshold,
                'name': self.name,
                'get_value': self.get_value,
                '_history': self._history,
                '_working_history': [],
                'patience': self.patience,
                '_current_patience': self._current_patience,
                'stop_callback': None,
                'save_callback': None}

    # ==================== main callback methods ==================== #
    def batch_end(self):
        if self.event_name != self.name:
            return
        self._working_history.append(self.results)

    def epoch_end(self):
        if self.event_name != self.name:
            return
        value = self._working_history
        self._working_history = [] # reset working history
        # ====== new value ====== #
        if self.get_value is not None:
            value = self.get_value(value)
        self._history.append(value)
        # ====== check early stop ====== #
        shouldSave, shouldStop = self.earlystop(self._history)
        messages = []
        if shouldSave > 0:
            messages.append(SIG_TRAIN_SAVE)
            # save callback
            if callable(self.save_callback):
                self.save_callback()
        if shouldStop > 0:
            messages.append(SIG_TRAIN_ROLLBACK)
            # call stop callback
            if callable(self.stop_callback):
                self.stop_callback()
            # check patience
            if self._current_patience > 0:
                self._current_patience -= 1
            else:
                messages.append(SIG_TRAIN_STOP)
        return messages

    @abstractmethod
    def earlystop(self, history):
        """ Any algorithm return: shouldSave, shouldStop """
        pass


class EarlyStopGeneralizationLoss(EarlyStop):
    """
    EarlyStopGeneralizationLoss(self, name, threshold, patience=1,
              get_value=lambda x: np.mean([i[0] for i in x]
                                          if isinstance(x[0], (tuple, list))
                                          else x),
              stop_callback=None,
              save_callback=None)


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
    stop_callback: function
        will be called when stop signal triggered, can be used to
        decrease learning rate.
        (for example, lambda x: learning_rate / 2)
    save_callback: function
        will be called when save signal triggered

    Note
    ----
    The early stop checking will be performed at the end of an epoch.
    By default, the return value from epoch mean the loss value, i.e lower
    is better
    """

    def earlystop(self, history):
        gl_exit_threshold = self.threshold
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
    stop_callback: function
        will be called when stop signal triggered
    save_callback: function
        will be called when save signal triggered

    """

    def task_start(self):
        if self.name == self.event_name:
            self.wait = 0

    def earlystop(self, history):
        shouldSave, shouldStop = 0, 0
        # showed improvement, should not equal to old best
        if len(history) <= 1 or history[-1] < np.min(history[:-1]):
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
_PLACEHOLDER = [re.compile('\{\d*\}'), re.compile('\{:.{0,3}\D\}')]


class ProgressMonitor(Callback):

    '''
    Parameters
    ----------
    title : str
        pattern to serialize return from function to string
    format: str
        format for the results, using the new python style
        (e.g. {0}, {1}, {:.4f}), ... and not %s, %d ...)
    tracking: list
        list of [(index, postprocessing_func)], tracking information at given
        index of the return value during batch_end, then, postprocess and print
        it in epoch_end.

    Example
    -------
    >>> t = training.Task(dataset=ds, batch_size=512)
    >>> t.set_callback(training.ProgressMonitor(title='Result: {:.2f}'))
    >>> t.run()
    # Result: 52751.29 98/98 [=======================================] - 0s

    Note
    ----
    This callback require specify `samples_size` in **kwargs of record
    '''

    def __init__(self, name, format='', tracking=[]):
        super(ProgressMonitor, self).__init__()
        self.name = name
        self._history = []
        self._prog = Progbar(100, title='')
        # ====== format ====== #
        self._format_results = 0
        for i in _PLACEHOLDER:
            self._format_results += len(i.findall(format))
        self._format = format
        # ====== one-time tracking at epoch_end ====== #
        if isinstance(tracking, dict):
            tracking = tracking.iteritems()
        self.tracking = [(int(i), j) for i, j in tracking if callable(j)]
        self._tracking_history = defaultdict(list)

    @property
    def _saveable_variables(self):
        return {'_format': self._format,
                '_format_results': self._format_results,
                '_history': [],
                'tracking': self.tracking,
                '_tracking_history': defaultdict(list),
                '_prog': Progbar(100, title=''),
                'name': self.name}

    def epoch_start(self):
        # reset tiem of ProgressBar
        if self.name == self.event_name:
            self._prog.start = time.time()

    def batch_end(self):
        # do nothing for not specified task
        if self.name != self.event_name or 'samples_size' not in self:
            return
        samples_size = self['samples_size']
        # ====== title ====== #
        r = self.results if isinstance(self.results, (tuple, list)) else (self.results,)
        r = [i.tolist() if isinstance(i, np.ndarray) and i.ndim == 0 else i
             for i in r]
        # ====== tracking ====== #
        for i, j in self.tracking:
            self._tracking_history[i].append(r[i])
        r = r[:self._format_results]
        self._history.append(r)
        title = (self._format.format(*r)
                 if self._format_results else self._format)
        # title
        self._prog.title = 'Name:%-8s,Epoch:%2d,' % \
        (self.name[:8], self.nb_epoch) + title
        # progress
        n = round(((self.nb_samples % samples_size) / samples_size) * 100)
        self._prog.update(min(int(n), 99))

    def epoch_end(self):
        # do nothing for not specified task
        if self.name != self.event_name:
            return
        # risky move: get the mean of all results
        if self._format_results:
            r = np.mean(self._history, axis=0).tolist()
            title = self._format.format(*r)
        else:
            title = self._format
        # reset
        self._history = []
        # title
        self._prog.title = 'Name:%-8s,Epoch:%2d,' % (
            self.event_name, self.nb_epoch) + title
        # always 100% at the end of epoch
        self._prog.target = 100; self._prog.update(100)
        # tracking
        for i, f in self.tracking:
            r = self._tracking_history[i]
            r = f(r)
            print('Tracking name-"%s" at location-%d:' % (self.name, i))
            print(r)
        self._tracking_history = defaultdict(list)


class CustomCallback(Callback):

    def __init__(self, name):
        pass


class Debug(Callback):

    FORMAT = "Name:%-8s Type:%-8s Samples:%d Iter:%d Epoch:%d Kwargs:%s"

    @property
    def _saveable_variables(self):
        return {'debug': True}

    def task_start(self):
        print(Debug.FORMAT % (self.event_name, self.event_type,
                              self.nb_samples, self.nb_iter, self.nb_epoch,
                              str(self._kwargs)))
        return 'event:task_start'

    def task_end(self):
        print(Debug.FORMAT % (self.event_name, self.event_type,
                              self.nb_samples, self.nb_iter, self.nb_epoch,
                              str(self._kwargs)))
        return 'event:task_end'

    def epoch_start(self):
        print(Debug.FORMAT % (self.event_name, self.event_type,
                              self.nb_samples, self.nb_iter, self.nb_epoch,
                              str(self._kwargs)))
        return 'event:epoch_start'

    def epoch_end(self):
        print(Debug.FORMAT % (self.event_name, self.event_type,
                              self.nb_samples, self.nb_iter, self.nb_epoch,
                              str(self._kwargs)))
        return 'event:epoch_end'

    def batch_start(self):
        print(Debug.FORMAT % (self.event_name, self.event_type,
                              self.nb_samples, self.nb_iter, self.nb_epoch,
                              str(self._kwargs)))
        return 'event:batch_start'

    def batch_end(self):
        print(Debug.FORMAT % (self.event_name, self.event_type,
                              self.nb_samples, self.nb_iter, self.nb_epoch,
                              str(self._kwargs)))
        return 'event:batch_end'
