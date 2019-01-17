# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

import os
import re
import sys
import time
import inspect
from enum import Enum
from numbers import Number
from datetime import datetime
from collections import defaultdict
from abc import ABCMeta, abstractproperty, abstractmethod
from six import add_metaclass, string_types
from six.moves import cPickle

import numpy as np
import scipy as sp

from odin.utils import (as_tuple, is_string, add_notification, ctext,
                        FuncDesc)

__all__ = [
    'TrainSignal',
    'TaskSignal',
    'Callback',
    'Debug',
    'LambdaCallback',
    'CallbackList',
    'NaNDetector',
    'CheckpointEpoch',
    'CheckpointGeneralization',
    'EarlyStop',
    'EarlyStopGeneralizationLoss',
    'EarlyStopPatience',
    'LRdecay',
    'EpochSummary'
]

# This SIGNAL can terminate running iterator (or generator),
# however, it cannot terminate them at the beginning,
# so you should not call .send(SIG_TERMINATE_ITERATOR)
# at beginning of iterator
class TrainSignal(Enum):
  """ TrainingSignal """
  NONE = 0 # nothing happened
  SAVE = 1 # save the parameters during training
  SAVE_BEST = 2 # saving the best model, signaled by EarlyStop
  ROLLBACK = 3 # rollback the model to the best checkpoint
  STOP = 4 # stop the training task

class TaskSignal(Enum):
  """ Signal represent current activities of a Training, Validating or
  Evaluating task """
  BatchStart = 0
  BatchEnd = 1
  EpochStart = 2
  EpochEnd = 3
  TaskStart = 4
  TaskEnd = 5

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
  Order of execution:
   - task_start(self, task)
   - epoch_start(self, task, data)
   - batch_start(self, task, data)
   - batch_end(self, task, results)
   - epoch_end(self, task, results)
   - task_end(self, task, results)

  For `batch_start` and `epoch_start`, `data` is the input data to training, validating
  or any Task

  For `batch_end`, `results` is the mapping `tensor_name` -> `returned_value`

  For `epoch_end`, `results` is the mapping `tensor_name` -> `list_of_returned_value`

  For `task_end`, results is the mapping `epoch_index` -> defaultdict(`epoch_results`),
  where `epoch_results` is the results of `epoch_end`.

  `task` is an instance of `odin.training.trainer.Task`, which
  indicate which Task is running (e.g. Training, Validating, ...)
  Some accessible properties from `odin.training.Task`:
   - curr_epoch: Total number of epoch finished since the beginning of the Task
   - curr_iter: Total number of iteration finished since the beginning of the Task
   - curr_samples: Total number of samples finished since the beginning of the Task
   - curr_epoch_iter: Number of iteration within current epoch
   - curr_epoch_samples: Number of samples within current epoch
  """

  def __init__(self, logging=True):
    super(Callback, self).__init__()
    self._log = bool(logging)

  def set_notification(self, is_enable):
    """ Turn notification on and off """
    self._log = bool(is_enable)
    return self

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

  def event(self, event_name):
    """ This function is directly called by MainLoop when
    special event triggered, `odin.training.callback.TrainSignal`
    """
    pass

  def send_notification(self, msg):
    if self._log:
      add_notification(
          '[%s] %s' % (ctext(self.__class__.__name__, 'magenta'), msg))
    return self

class CallbackList(Callback):

  ''' Broadcast signal to all its children'''

  def __init__(self, callbacks=None):
    super(CallbackList, self).__init__()
    self.set_callbacks(callbacks)

  def set_callbacks(self, callbacks):
    if callbacks is None:
      callbacks = []
    elif isinstance(callbacks, CallbackList):
      callbacks = callbacks._callbacks
    else:
      callbacks = as_tuple(callbacks,
        t=lambda x: isinstance(x, (Callback, type(None))))
      callbacks = [i for i in callbacks if i is not None]
    self._callbacks = [i for i in set(callbacks)]
    return self

  def set_notification(self, is_enable):
    """ Turn notification on and off """
    is_enable = bool(is_enable)
    self._log = is_enable
    for cb in self._callbacks:
      cb.set_notification(is_enable)
    return self

  def __str__(self):
    return '<CallbackList: ' + \
    ', '.join([i.__class__.__name__ for i in self._callbacks]) + '>'

  def batch_start(self, task, batch):
    msg = []
    for i in self._callbacks:
      m = i.batch_start(task, batch)
      msg += [j for j in as_tuple(m) if isinstance(j, TrainSignal)]
    return msg

  def batch_end(self, task, batch_results):
    msg = []
    for i in self._callbacks:
      m = i.batch_end(task, batch_results)
      msg += [j for j in as_tuple(m) if isinstance(j, TrainSignal)]
    return msg

  def epoch_start(self, task, data):
    msg = []
    for i in self._callbacks:
      m = i.epoch_start(task, data)
      msg += [j for j in as_tuple(m) if isinstance(j, TrainSignal)]
    return msg

  def epoch_end(self, task, epoch_results):
    msg = []
    for i in self._callbacks:
      m = i.epoch_end(task, epoch_results)
      msg += [j for j in as_tuple(m) if isinstance(j, TrainSignal)]
    return msg

  def task_start(self, task):
    msg = []
    for i in self._callbacks:
      m = i.task_start(task)
      msg += [j for j in as_tuple(m) if isinstance(j, TrainSignal)]
    return msg

  def task_end(self, task, task_results):
    msg = []
    for i in self._callbacks:
      m = i.task_end(task, task_results)
      msg += [j for j in as_tuple(m) if isinstance(j, TrainSignal)]
    return msg

  def event(self, event_type):
    assert isinstance(event_type, TrainSignal), \
    "event must be enum `TrainSignal` but given: %s" % str(type(event_type))
    for i in self._callbacks:
      i.event(event_type)

class Debug(Callback):
  """ Debug """

  def __init__(self, signal=None):
    super(Debug, self).__init__()
    if signal is None:
      pass

  def batch_start(self, task, batch):
    print("Batch Start:", task.name, task.curr_epoch, task.curr_samples,
          [(i.shape, i.dtype, type(i)) for i in batch])

  def batch_end(self, task, batch_results):
    print(ctext("Batch End:", 'cyan'),
          task.name, task.curr_epoch, task.curr_samples,
          [(i.shape, i.dtype, type(i)) for i in as_tuple(batch_results)])

  def epoch_start(self, task, data):
    print(ctext("Epoch Start:", 'cyan'),
          task.name, task.curr_epoch, task.curr_samples,
          [(i.shape, i.dtype, type(i)) for i in data])

  def epoch_end(self, task, epoch_results):
    print(ctext("Epoch End:", 'cyan'),
          task.name, task.curr_epoch, task.curr_samples,
          [(i, len(j), type(j[0])) for i, j in epoch_results.items()])

  def task_start(self, task):
    print(ctext("Task Start:", 'cyan'),
          task.name, task.curr_epoch, task.curr_samples)

  def task_end(self, task, task_results):
    print(ctext("Task End:", 'cyan'),
        task.name, task.curr_epoch, task.curr_samples,
        [(i, [(n, len(v), type(v[0])) for n, v in j.items()])
         for i, j in task_results.items()])

  def event(self, event_type):
    print(ctext("[Debug] Event:", 'cyan'), event_type)

# ===========================================================================
# Others
# ===========================================================================
class LambdaCallback(Callback):
  """ Some accessible properties from `odin.training.Task`:
   - curr_epoch: Total number of epoch finished since the beginning of the Task
   - curr_iter: Total number of iteration finished since the beginning of the Task
   - curr_samples: Total number of samples finished since the beginning of the Task
   - curr_epoch_iter: Number of iteration within current epoch
   - curr_epoch_samples: Number of samples within current epoch

  Parameters
  ----------
  fn : function
    a function, must follow one of the option:
      - 0 argument: only the function is called
      - 1 argument: only instance of `odin.training.Task`
      - 2 arguments: instance of `odin.training.Task`, and
                     input_data or returned_results
  name : string
    name of the Task when this callback will be activated
  signal : odin.training.callbacks.TaskSignal
    signal that trigger this callback

  """

  def __init__(self, fn, task_name,
               signal=TaskSignal.EpochEnd):
    super(LambdaCallback, self).__init__()
    assert hasattr(fn, '__call__')
    # ====== check function ====== #
    spec = inspect.signature(fn)
    assert len(spec.parameters) in (0, 1, 2),\
    "`fn` must accept 0, 1 or 2 arguments"
    self.n_args = len(spec.parameters)
    self.fn = FuncDesc(fn)
    # ====== others ====== #
    self.task_name = str(task_name)
    assert isinstance(signal, TaskSignal)
    self.signal = signal

  def _call_fn(self, task, data):
    if self.n_args == 0:
      return self.fn()
    elif self.n_args == 1:
      return self.fn(task)
    elif self.n_args == 2:
      return self.fn(task, data)
    else:
      raise RuntimeError()

  def batch_start(self, task, data):
    if self.signal == TaskSignal.BatchStart and \
    self.task_name == task.name:
      self._call_fn(task, data)

  def batch_end(self, task, batch_results):
    if self.signal == TaskSignal.BatchEnd and \
    self.task_name == task.name:
      self._call_fn(task, batch_results)

  def epoch_start(self, task, data):
    if self.signal == TaskSignal.EpochStart and \
    self.task_name == task.name:
      self._call_fn(task, data)

  def epoch_end(self, task, epoch_results):
    if self.signal == TaskSignal.EpochEnd and \
    self.task_name == task.name:
      self._call_fn(task, epoch_results)

  def task_start(self, task):
    if self.signal == TaskSignal.TaskStart and \
    self.task_name == task.name:
      self._call_fn(task, None)

  def task_end(self, task, task_results):
    if self.signal == TaskSignal.TaskEnd and \
    self.task_name == task.name:
      self._call_fn(task, task_results)

# ===========================================================================
# NaN value detection
# ===========================================================================
class NaNDetector(Callback):
  """ NaNDetector

  Parameters
  ----------
  task_name : {str, None}
    name of specific Task will be applied,
    if `None`, all tasks are considered

  patience : {int}
    the Task will be stopped if `patience` < 0, `patience` will
    subtracted by 1 every time NaN is detected

  logging : bool (default: True)
    show notification when NaN detected

  """

  def __init__(self, task_name=None, patience=-1,
               detect_inf=False, logging=True):
    super(NaNDetector, self).__init__(logging=logging)
    self._task_name = task_name
    self._patience = patience
    self._detect_inf = bool(detect_inf)

  def batch_end(self, task, batch_results):
    if self._task_name is not None and task.name != self._task_name:
      return
    # found any NaN values
    if self._detect_inf:
      fn = lambda x: np.logical_or(np.isinf(x), np.isnan(x))
    else:
      fn = lambda x: np.isnan(x)

    if any(np.any(fn(r)) for r in as_tuple(batch_results)):
      signal = TrainSignal.ROLLBACK
      self._patience -= 1
      if self._patience <= 0: # but if out of patience, stop
        signal = TrainSignal.STOP
      self.send_notification('Found NaN or Inf value, task:"%s"' % task.name)
      return signal

# ===========================================================================
# Checkpoint technique
# ===========================================================================
class CheckpointEpoch(Callback):
  """ CheckpointEpoch

  Periodically saving the model after fixed percents of epoch have
  passed (using the estimated number of iteration to calculate current
  epoch percents)

  """

  def __init__(self, task_name, epoch_percent=1., logging=True):
    super(CheckpointEpoch, self).__init__(logging=logging)
    self._task_name = task_name

    epoch_percent = float(epoch_percent)
    self._epoch_percent = epoch_percent

  def batch_end(self, task, batch_results):
    if task.name == self._task_name:
      if task.curr_epoch_iter % int(self._epoch_percent * task.iter_per_epoch) == 0:
        return TrainSignal.SAVE
    return None

class CheckpointGeneralization(Callback):
  """ Save the model when improvement on generalization score

  Parameters
  ----------
  task_name : string
      task name for checking this criterion

  output_name : {Tensor, string}
      name of the Tensor or the Tensor itself for monitoring

  improvement_margin : float (>= 0)
      improvement in percentage that will be accepted,
      default, 0% mean that any number is smaller will trigger
      the checkpoint.
  """

  def __init__(self, task_name, output_name,
               lower_better=True, improvement_margin=0, logging=True):
    super(CheckpointGeneralization, self).__init__(logging=logging)
    self._task_name = str(task_name)
    self._output_name = output_name if is_string(output_name) \
        else output_name.name
    self._lower_better = bool(lower_better)
    self._improvement_margin = float(improvement_margin)
    assert self._improvement_margin >= 0
    self._best_score = None

  def epoch_end(self, task, epoch_results):
    if task.name != self._task_name:
      return None
    if self._output_name not in epoch_results:
      return None
    # get the score
    score = np.mean(epoch_results[self._output_name])
    if not self._lower_better:
      score *= -1
    epsilon = 1e-8
    # first time, nothing to do
    if self._best_score is None:
      self._best_score = score
      return None

    gl_t = score / (self._best_score + epsilon)
    if gl_t < 1 - (self._improvement_margin / 100):
      self._best_score = score
      return TrainSignal.SAVE_BEST

# ===========================================================================
# Learning rate manipulation
# ===========================================================================
class LRdecay(Callback):
  """ LRdecay
  whenever TrainSignal.ROLLBACK is triggered, decrease the learning
  rate by `decay_rate`
  """

  def __init__(self, lr, decay_rate=0.5):
    super(LRdecay, self).__init__()
    from odin import backend as K
    self.lr = lr
    self.lr_value = K.get_value(lr)
    self.decay_rate = decay_rate

  def event(self, event_name):
    if event_name == TrainSignal.ROLLBACK:
      from odin import backend as K
      self.lr_value *= self.decay_rate
      K.set_value(self.lr, self.lr_value)

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
  task_name : string
      task name for checking this criterion

  output_name : {Tensor, string}
      name of the Tensor or the Tensor itself for monitoring

  threshold : float
      for example, threshold = 5, if we loss 5% of performance on validation
      set, then stop

  patience: int
      how many cross the threshold that still can be rollbacked

  get_value : function
      function to process the results of whole epoch (i.e list of results
      returned from batch_end) to return comparable number.
      For example, lambda x: np.mean(x)

  Note
  ----
  * The early stop checking will be performed at the end of an epoch.
  * By default, the return value from epoch mean the loss value, i.e lower
  is better
  * If multiple value returned, you have to modify the get_value function
  """

  def __init__(self, task_name, output_name, threshold, patience=1,
               get_value=lambda x: np.mean(x),
               logging=True):
    super(EarlyStop, self).__init__(logging=logging)
    self._task_name = str(task_name)
    self._output_name = output_name if is_string(output_name) \
        else output_name.name

    self._threshold = float(threshold)
    self._patience = int(patience)

    if get_value is None:
      get_value = lambda x: x
    elif not hasattr(get_value, '__call__'):
      raise ValueError('get_value must call-able')
    self._get_value = get_value
    # ====== history ====== #
    self._history = []

  # ==================== main callback methods ==================== #
  def epoch_end(self, task, epoch_results):
    if self._task_name != task.name:
      return
    self._history.append(self._get_value(epoch_results[self._output_name]))
    # ====== check early stop ====== #
    shouldSave, shouldStop = self.earlystop(self._history, self._threshold)
    msg = None
    if shouldSave > 0:
      msg = TrainSignal.SAVE_BEST
    if shouldStop > 0:
      msg = TrainSignal.ROLLBACK
      # check patience
      self._patience -= 1
      if self._patience < 0:
        msg = TrainSignal.STOP
    self.send_notification('Message "%s"' % str(msg))
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

  def __init__(self, task_name, output_name, threshold=5, patience=1,
               get_value=lambda x: np.mean(x), logging=True):
    super(EarlyStopGeneralizationLoss, self).__init__(
        task_name, output_name, threshold, patience,
        get_value, logging=logging)

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

  def __init__(self, task_name, output_name, threshold, patience=1,
               get_value=lambda x: np.mean(x), logging=True):
    super(EarlyStopPatience, self).__init__(
        task_name, output_name, threshold, patience,
        get_value, logging=logging)

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

# ===========================================================================
# Monitoring
# ===========================================================================
class EpochSummary(Callback):
  """
  Parameters
  ----------
  output_name : {list of string, list of Tensor}
    list of outputs for monitoring

  print_plot : bool (default: False)
    using bashplot to print out the figure directly in terminal

  save_path : {str, None} (default: None)
    if provided, save pdf figure to given path

  epoch_percent : float (0 < .. < 1)
    determine the periodic of the summary within single epoch

  repeat_freq : int (.. > 0)
    determine the repeating frequency (e.g. `repeat_freq=2`
    is for every 2 epoch)

  fn_reduce : bool (default:False)


  """

  def __init__(self, task_name, output_name,
               fn_reduce=lambda x: (np.mean(x)
                                    if isinstance(x[0], Number) else
                                    sum(i for i in x)),
               print_plot=False, save_path=None,
               repeat_freq=1, logging=True):
    super(EpochSummary, self).__init__(logging=logging)
    self._task_name = as_tuple(task_name, t=str)
    # ====== scheduling ====== #
    assert repeat_freq >= 1
    self._repeat_freq = int(repeat_freq)
    self._count = self._repeat_freq * len(self._task_name)
    self._epoch_results = defaultdict(dict)
    # ====== output identity ====== #
    if not isinstance(output_name, (tuple, list, set)):
      output_name = (output_name,)
    output_name = [i if is_string(i) else i.name
                   for i in output_name]
    self.output_name = tuple(output_name)
    self.fn_reduce = FuncDesc(func=fn_reduce)
    # ====== how to output ====== #
    self.print_plot = bool(print_plot)
    self.save_path = save_path

  def epoch_end(self, task, epoch_results):
    output_name = self.output_name
    if len(output_name) == 0: # nothing to do
      return
    task_name = self._task_name

    if task.name in task_name:
      self._count -= 1
      # ====== processing results ====== #
      assert all(name in epoch_results for name in output_name),\
      "Given outputs with name: %s; but task: '%s' results only contain name: %s" % \
      (', '.join(self.output_name), str(task), ', '.join(tuple(epoch_results.keys())))

      for name in output_name:
        batch_results = epoch_results[name]
        if name not in self._epoch_results[task.name]:
          self._epoch_results[task.name][name] = []
        self._epoch_results[task.name][name].append(self.fn_reduce(batch_results))
      # ====== start plotting ====== #
      if self._count == 0:
        self._count = self._repeat_freq * len(task_name)
        from odin import visual as V
        n_col = len(task_name)
        n_row = len(output_name)
        if self.save_path is not None:
          from matplotlib import pyplot as plt
        save_figures = False
        override = True

        for o_idx, o_name in enumerate(output_name):
          results = {task_name: r[o_name]
                     for task_name, r in self._epoch_results.items()}
          if all(len(i) >= 2 for i in results.values()):
            # ====== print text plot ====== #
            if self.print_plot:
              text = []
              for t_name in task_name:
                values = results[t_name]
                if isinstance(values[0], Number):
                  t = V.print_bar(f=values, height=8,
                                  title=t_name + "/" + o_name)
                elif isinstance(values[0], np.ndarray) and values[0].ndim == 2 and \
                values[0].shape[0] == values[0].shape[1]:
                  t = V.print_confusion(arr=values[-1],
                                       side_bar=False, inc_stats=True,
                                       float_precision=2)
                else:
                  t = ''
                if len(t) > 0:
                  text.append(t)
              if len(text) > 1:
                print(V.merge_text_graph(*text, padding='  '))
              else:
                print(text[0])
            # ====== matplotlib plot and save pdf ====== #
            if self.save_path is not None:
              for t_idx, t_name in enumerate(task_name):
                values = results[t_name]
                # plotting series
                if isinstance(values[0], Number):
                  if not save_figures:
                    V.plot_figure(nrow=int(n_row * 1.8), ncol=20)
                  save_figures = True

                  max_epoch = np.argmax(values)
                  max_val = values[max_epoch]
                  min_epoch = np.argmin(values)
                  min_val = values[min_epoch]

                  plt.subplot(n_row, n_col, o_idx * len(task_name) + t_idx + 1)
                  plt.plot(values)
                  plt.scatter(max_epoch, max_val, s=180, alpha=0.4, c='r')
                  plt.scatter(min_epoch, min_val, s=180, alpha=0.4, c='g')

                  plt.xlim((0, len(values) - 1))
                  if not np.any(np.isinf(values)):
                    eps = 0.1 * (max_val - min_val)
                    plt.ylim((min_val - eps, max_val + eps))
                  plt.xticks(np.linspace(0, len(values) - 1, num=12,
                                         dtype='int32'))

                  title_text = '[%s]' % o_name if t_idx == 0 else ''
                  title_text += t_name
                  plt.title('%s' % title_text,
                            fontsize=8, fontweight='bold')
        # save figure to pdf or image files
        if save_figures:
          if override:
            save_path = self.save_path
          else:
            path, ext = os.path.splitext(self.save_path)
            save_path = path + ('.%d' % (task.curr_epoch + 1)) + ext
          V.plot_save(save_path, tight_plot=True,
                      clear_all=True, log=False, dpi=180)
          self.send_notification("Saved summary at: %s" % save_path)
    return None
