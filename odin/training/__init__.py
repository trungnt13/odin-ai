# Address: {b6ba96161f8b621ce7d0424c6502c38e2f089fbc}
from __future__ import division, absolute_import, print_function

import os
from itertools import chain
from collections import defaultdict
from six.moves import range, zip, cPickle

import numpy as np

from odin.config import get_rng
from odin import fuel, backend as K, nnet as N
from odin.fuel import Dataset, as_data
from odin.utils import struct, as_tuple, is_number, Progbar, add_notification

from .callbacks import *


# ===========================================================================
# Helper
# ===========================================================================
def __format_string(nb_of_float):
  x = ["{:.4f}"] * int(nb_of_float)
  return ";".join(x)


def _plot_each_epoch(name, results, task_type):
  """ results list of each epoch results
  [(epoch1_r1, epoch1_r2, ...),
   (epoch2_r1, epoch2_r2, ...), ...]
  """
  from matplotlib import pyplot as plt
  nb_epoch = len(results)
  ncol = 3; nrow = int(np.ceil(nb_epoch / ncol))
  _ = list(chain(*results))
  max_ = np.max(_); min_ = np.min(_)
  # ====== plot an overall view of all epoch ====== #
  plt.figure()
  line = plt.plot(range(1, len(results) + 1),
           [np.mean(epoch) for epoch in results])[0]
  plt.setp(line, linewidth=2, color='r')
  plt.ylim([min_, max_])
  plt.xlabel("#Epoch")
  plt.ylabel(name)
  plt.suptitle(task_type + name, fontsize=20)
  # ====== plot each epoch ====== #
  plt.figure()
  for i, x in enumerate(results):
    ax = plt.subplot(nrow, ncol, i + 1)
    ax.plot(x); ax.set_ylim([min_, max_])
    ax.tick_params(labelsize=8)
    plt.xlabel("[Epoch%d]Iteration" % (i + 1), fontsize=8)
  plt.tight_layout()


# ===========================================================================
# Tasks
# ===========================================================================
class Task(object):
  """
  Parameters
  ----------
  func: call-able
      function will be executed for each iteration
  data: single or list of odin.fuel.Data, numpy.ndarray
      iterate over all these data and execute function on
      the data.
  epoch: int
      how many epoch will be repeated
  p: float (0.0 - 1.0)
      probability the `func` will be execute for each iteration
  batch_size: int (> 0)
      number of samples for each iteration
  seed: int
      random seed for shuffling the data
  shuffle_level: int (0, 1, 2)
      if 0, shuffle the file lists
      if 1, shuffle the buffer (i.e. list of processing files) and
          all the previous
      if 2, shuffle the returned batch and all the previous
  callbacks: None, or list of `odin.training.Callback`
      callback will be promoted during the execution of the task
  labels: None, or list of string
      labels for printing the confusion matrix in `odin.utils.Progbar`
  name: None or string
      unique name for Task identity.
  """

  def __init__(self, func, data, epoch=1, p=1.0,
               batch_size=128, seed=None, shuffle_level=2,
               callbacks=None, labels=None, name=None):
    super(Task, self).__init__()
    self.set_func(func, data)
    # this Progbar will record the history as well
    self._labels = [str(l) for l in labels] \
        if labels is not None else None
    self._progbar = Progbar(target=self.nb_samples, name=name,
                            print_report=True, print_summary=True)
    self._progbar.set_labels(self._labels)
    # ====== assign other arguments ====== #
    self._nb_epoch = epoch
    self._p = np.clip(p, 0., 1.)
    self._seed = seed
    self.set_batch(batch_size, seed, shuffle_level)
    self._name = name
    # ====== current info ====== #
    self._curr_epoch = 0
    self._curr_iter = 0
    self._curr_samples = 0
    self._curr_epoch_iter = 0
    self._curr_epoch_samples = 0
    self._callback_msg = []
    # ====== iter tracking ====== #
    self._created_iter = None
    self._stop = False
    self._callback = CallbackList(callbacks)

  def __getstate__(self):
    return (self._progbar, self._nb_epoch, self._p, self._name,
            self._batch_size, self._rng, self._seed, self._shuffle_level)

  def __setstate__(self, states):
    (self._progbar, self._nb_epoch, self._p, self._name,
     self._batch_size, self._rng, self._seed, self._shuffle_level) = states
    # ====== current info ====== #
    self._curr_epoch = 0
    self._curr_iter = 0
    self._curr_samples = 0
    self._curr_epoch_iter = 0
    self._curr_epoch_samples = 0
    self._callback_msg = []
    # ====== iter tracking ====== #
    self._created_iter = None
    self._stop = False

  def set_callbacks(self, callbacks):
    self._callback.set_callbacks(callbacks)
    return self

  def set_func(self, func, data):
    # ====== check function ====== #
    self._func = func
    if isinstance(func, K.Function):
      self._output_info = [(o.name, o.get_shape().as_list())
                           for o in self._func.outputs]
    elif hasattr(func, '__call__'):
      self._output_info = [] # No info (normal function)
    else:
      raise ValueError("No support for function type: %s" %
          func.__class__.__name__)
    # ====== check data ====== #
    if not isinstance(data, (tuple, list)):
      data = [data]
    self._data = [fuel.as_data(i) for i in data]
    self._nb_samples = min([len(d) for d in self._data])
    return self

  def set_batch(self, batch_size=None, seed=-1, shuffle_level=None):
    if batch_size is not None:
      self._batch_size = batch_size
    if seed is None or seed >= 0:
      if seed is not None:
        self._rng = np.random.RandomState(seed)
      else:
        self._rng = struct()
        self._rng.randint = lambda x: None
        self._rng.rand = get_rng().rand
    if shuffle_level is not None:
      self._shuffle_level = min(max(int(shuffle_level), 0), 2)
    return self

  # ==================== Properties ==================== #
  @property
  def name(self):
    return str(self._name)

  @property
  def labels(self):
    return self._labels

  @property
  def nb_epoch(self):
    return self._nb_epoch

  @property
  def nb_samples(self):
    ''' Estimated number of iteration for each epoch '''
    return self._nb_samples

  @property
  def probability(self):
    """Chance that the func will be execute during iteration"""
    return self._p

  @property
  def iter_per_epoch(self):
    ''' Estimated number of iteration for each epoch '''
    return int(np.ceil(self._nb_samples / self._batch_size))

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def curr_epoch(self):
    """Total number of epoch finished since the beginning of the Task"""
    return self._curr_epoch

  @property
  def curr_iter(self):
    """Total number of iteration finished since the beginning of the Task"""
    return self._curr_iter

  @property
  def curr_samples(self):
    """Total number of samples finished since the beginning of the Task"""
    return self._curr_samples

  @property
  def curr_epoch_iter(self):
    """Number of iteration within current epoch"""
    return self._curr_epoch_iter

  @property
  def curr_epoch_samples(self):
    """Number of samples within current epoch"""
    return self._curr_epoch_samples

  @property
  def callback_msg(self):
    return self._callback_msg

  # ==================== control function ==================== #
  def stop(self):
    """ Stop all iterations running for this Task"""
    if self._created_iter is not None:
      self._stop = True
      # just run to end of the iterators
      for i in self._created_iter:
        pass
      self._stop = False
      self._created_iter = None

  def copy(self):
    return Task(self._func, self._data,
                epoch=self.nb_epoch, p=self.probability,
                batch_size=self.batch_size, seed=self._seed,
                shuffle_level=self._shuffle_level,
                name=self._name)

  def __iter(self):
    '''
    Return
    ------
    One of the following:
    * 'task_start':
    * 'epoch_start' : beginning of epoch
    * 'epoch_end' : epoch ended
    * 'task_end' : task ended
    * (results, nb_iter, nb_samples,
       nb_total_samples, nb_epoch) : results of execute function on data

    Note
    ----
    'end_task' also end of final epoch
    '''
    yield None # just for initalize the iterator
    self._callback_msg = self._callback.task_start(self)
    yield 'task_start'
    if self._stop:
      yield 'task_end'
    else:
      # ====== start of training ====== #
      while self._curr_epoch < self._nb_epoch:
        self._callback_msg = self._callback.epoch_start(self, self._data)
        yield 'epoch_start'
        seed = self._rng.randint(10e8)
        # if only 1 Data, don't need zip or we will mess up
        if len(self._data) == 1:
          data_it = iter(self._data[0].set_batch(batch_size=self._batch_size,
                                                 seed=seed,
                                                 shuffle_level=self._shuffle_level))
          data = data_it
        else:
          data_it = [iter(d.set_batch(batch_size=self._batch_size,
                                      seed=seed,
                                      shuffle_level=self._shuffle_level))
                     for d in self._data]
          data = zip(*data_it)
        # ======  start the iteration ====== #
        self._curr_epoch_samples = 0
        self._curr_epoch_iter = 0
        for i, x in enumerate(data):
          # alread terminated, try to exhausted the iterator
          # if forced_to_terminate: continue
          # preprocessed the data
          if not isinstance(x, (tuple, list)):
            x = [x]
          # update some info
          shape0 = x[0].shape[0]
          self._curr_samples += shape0
          self._curr_iter += 1
          self._curr_epoch_samples += shape0
          self._curr_epoch_iter += 1
          self._callback_msg = self._callback.batch_start(self, x)
          # apply the function
          if self.probability >= 1. or self._rng.rand() < self.probability:
            results = self._func(*x)
            # add msg from batch_end event
            self._callback_msg += self._callback.batch_end(self, results)
            # return results
            yield results
            # update the progress bar
            for (name, shape), res in zip(self._output_info,
                                          as_tuple(results)):
              if len(shape) == 0: # return single value
                self._progbar[name] = res
              else: # return tensor
                self._progbar[name] = res
            self._progbar.add(shape0)
          # check TERMINATE signal
          if self._stop:
            # send signal to the data iterators also
            for i in data_it:
              if hasattr(i, 'stop'):
                i.stop()
              else: # just iterate all over
                for _ in i: pass
            # break the epoch loop
            break
        # Epoch end signaling
        self._curr_epoch += 1
        self._callback_msg = self._callback.epoch_end(
            self, self._progbar.history[self._curr_epoch - 1])
        yield 'epoch_end'
        # ====== check if we got the right number for epoch iter ====== #
        if self._curr_epoch_samples != self._nb_samples:
          # just for sure should not smaller than the real number
          self._nb_samples = self._curr_epoch_samples
        # ======  end_epoch or task ====== #
        if self._stop or self._curr_epoch >= self._nb_epoch:
          self._callback_msg = self._callback.task_end(
              self, self._progbar.history)
          yield 'task_end'
          self._progbar.add_notification('Task "%s" ended!' % str(self.name))
          break
    # ====== end of iteration ====== #
    self._created_iter = None

  def __iter__(self):
    if self._created_iter is None:
      # reset all information
      self._curr_epoch = 0
      self._curr_iter = 0
      self._curr_samples = 0
      self._curr_epoch_iter = 0
      self._curr_epoch_samples = 0
      self._callback_msg = []
      # create new iter
      self._created_iter = self.__iter()
      # initialize the iteration
      next(self._created_iter)
    return self._created_iter

  def __del__(self):
    self.stop()


# ===========================================================================
# MainLoop
# ===========================================================================
class Timer(object):
  """Timer to determine when a `Task` should be start within a `MainLoop`

  Note
  ----
  The `Task` will be commenced if any of the conditions is `True`
  """

  def __init__(self, epoch=None, iteration=None, samples=None,
               percentage=None):
    super(Timer, self).__init__()
    self._epoch = epoch
    self._iteration = iteration
    self._samples = samples
    self._percentage = percentage
    # ====== for the counter ====== #
    self._counter_max = 0 # store maximum amount of sample
    self._counter = 0 # store current amount of samples
    self._last_samples_checkpoint = -1

  def check(self, task):
    # task haven't started
    if task.curr_samples == 0:
      return False
    if self._epoch is not None and task.curr_epoch >= self._epoch:
      return True
    if self._iteration is not None and \
    task.curr_iter >= self._iteration:
      return True
    if self._samples is not None and \
    task.curr_samples >= self._samples:
      return True
    if self._percentage is not None and\
    task.curr_samples / (task.nb_epoch * task.nb_samples) >= self._percentage:
      return True
    return False

  def set_counter(self, task):
    nb_samples = task.nb_samples
    batch_size = task.batch_size
    if self._epoch is not None:
      self._counter_max = nb_samples * self._epoch
    elif self._iteration is not None:
      self._counter_max = batch_size * self._iteration
    elif self._samples is not None:
      self._counter_max = self._samples
    elif self._percentage is not None:
      self._counter_max = int(self._percentage * nb_samples)
    self._counter = self._counter_max
    return self

  def update_counter(self, task):
    if self._last_samples_checkpoint < 0:
      self._last_samples_checkpoint = task.curr_samples
    else:
      self._counter -= task.curr_samples - self._last_samples_checkpoint
      self._last_samples_checkpoint = task.curr_samples
      if self._counter <= 0:
        self._counter = self._counter_max
        return True
    return False


class MainLoop(object):

  """ MainLoop

  Parameters
  ----------
  batch_size: int
      size of each batch return when iterate this Data
  seed: None, int
      if None, no shuffling is performed while iterating,
      if < 0, do not change the current seed
      if >= 0, enable randomization with given seed
  shuffle_level: int
      0: only shuffle the order of each batch
      1: shuffle the order of batches and inside each batch as well.
      2: includes level 0 and 1, and custom shuffling (strongest form)
  rollback: bool
      if True, rollback to the best checkpoint whenever the validation
      performance is degraded.

  """

  def __init__(self, batch_size=256, seed=-1, shuffle_level=0, allow_rollback=True):
    super(MainLoop, self).__init__()
    self._main_task = None
    self._task = []
    self._subtask = []
    self._evaltask = []
    self._task_when = {} # mapping from `Task` to `Timer`
    self._task_freq = {} # mapping from `Task` to `Timer`
    self._allow_rollback = bool(allow_rollback)

    # create default RNG (no randomization)
    self._rng = struct()
    self._rng.randint = lambda *args, **kwargs: None
    # set batch
    self.set_batch(batch_size=batch_size, seed=seed,
                   shuffle_level=shuffle_level)

    self._callback = CallbackList()

    self._save_path = None
    self._save_hist = None
    self._save_obj = None
    self._save_variables = None

  # ==================== pickling ==================== #
  def __setstate__(self, value):
    self.set_batch(batch_size=value[0], shuffle_level=value[2])
    self._rng = value[1]

    self._callback = value[3]
    self._allow_rollback = value[4]

    self._task = []
    self._subtask = []
    self._main_task = None

  def __getstate__(self):
    return (self._batch_size, self._rng, self._shuffle_level,
            self._callback, self._allow_rollback)

  # ==================== Signal handling ==================== #
  def set_checkpoint(self, path, obj, variables=[]):
    """
    Parameters
    ----------
    path: str
        path to save the obj when the callback return save signal
    obj: object
        any pickle-able object you want to save
    variables : {list of tensorflow.Variable}
        external variables will be saved together with the
        model
    """
    self._save_path = path
    self._save_obj = obj
    self._save_variables = variables
    # save first checkpoint
    self._save()

  # ==================== properties ==================== #
  @property
  def batch_size(self):
    return self._batch_size

  def set_batch(self, batch_size=None, seed=-1, shuffle_level=None):
    """
    Parameters
    ----------
    batch_size: int
        size of each batch return when iterate this Data
    seed: None, int
        if None, no shuffling is performed while iterating,
        if < 0, do not change the current seed
        if >= 0, enable randomization with given seed
    start: int, float
        if int, start indicates the index of starting data points to
        iterate. If float, start is the percentage of data to start.
    end: int, float
        ending point of the interation
    shuffle_level: int
        0: only shuffle the order of each batch
        1: shuffle the order of batches and inside each batch as well.
        2: includes level 0 and 1, and custom shuffling (strongest form)
    """
    if batch_size is not None:
      self._batch_size = batch_size
    if seed >= 0 or seed is None:
      if seed is not None:
        self._rng = np.random.RandomState(seed)
      else:
        self._rng = struct()
        self._rng.randint = lambda *args, **kwargs: None
    if shuffle_level is not None:
      shuffle_level = min(max(int(shuffle_level), 0), 2)
      self._shuffle_level = shuffle_level
    # ====== set_batch for Tasks ====== #
    for i in self._task:
      i.set_batch(batch_size=batch_size, seed=seed,
                  shuffle_level=shuffle_level)
    for i in self._subtask:
      i.set_batch(batch_size=batch_size, seed=seed,
                  shuffle_level=shuffle_level)
    return self

  @property
  def callback(self):
    return self._callback

  def __str__(self):
    return 'Task'

  def set_callbacks(self, callbacks):
    self._callback.set_callbacks(callbacks)
    return self

  def __getitem__(self, key):
    """ Return callback from callback list"""
    return self._callback[key]

  # ==================== main ==================== #
  def set_train_task(self, func, data, epoch=1, p=1., when=None,
                     name="Train", labels=None):
    ''' The progress of the first task added as train_task will
    be used to determine Timer for all other task
    '''
    if len(self._task) == 0 and when is not None:
      raise ValueError("First train task will always be executed, you cannot"
          "specify `Timer` for this task.")
    if when is not None and not isinstance(when, Timer):
      raise ValueError("`when` must be instance of odin.training.Timer")
    t = Task(func, data, epoch=epoch, p=p, batch_size=self._batch_size,
             seed=self._rng.randint(10e8), shuffle_level=self._shuffle_level,
             labels=labels, name=name)
    self._task.append(t)
    self._task_when[t] = when
    self._task_freq[t] = Timer(samples=0)
    # assign main task
    if self._main_task is None:
      self._main_task = t
    return self

  def set_valid_task(self, func, data,
                     freq=Timer(epoch=1), when=Timer(samples=0),
                     name="Valid", labels=None):
    ''' A subtask is a repeative Task

    Parameters
    ----------
    when: float
        percentage of epoch of main task before this task is executed
        negative value => execute after final epoch of main task
    freq: float
        percentage of epoch of main task before this task is executed
    '''
    if not isinstance(when, Timer):
      raise ValueError("`when` must be instance of odin.training.Timer")
    t = Task(func, data, epoch=float('inf'), p=1., batch_size=self._batch_size,
             seed=None, shuffle_level=0, labels=labels, name=name)
    self._subtask.append(t)
    self._task_when[t] = when
    self._task_freq[t] = freq
    return self

  def set_eval_task(self, func, data, name="Eval", labels=None):
    t = Task(func, data, epoch=1, p=1., batch_size=self._batch_size,
             seed=None, shuffle_level=0, labels=labels, name=name)
    self._evaltask.append(t)
    self._task_when[t] = Timer(percentage=1.)
    self._task_freq[t] = Timer(samples=0)
    return self

  # ==================== logic ==================== #
  def _save(self):
    # trigger event for callbacks
    self._callback.event(SIG_TRAIN_SAVE)
    # default save procedure
    if self._save_path is not None and self._save_obj is not None:
      add_notification("Creating checkpoint at:" + self._save_path)
      if not os.path.exists(self._save_path):
        os.mkdir(self._save_path)
      elif os.path.isfile(self._save_path):
        raise ValueError("Save path for the model must be a folder.")
      N.serialize(nnops=self._save_obj, path=self._save_path,
                  save_variables=True, variables=self._save_variables,
                  override=True)

  def _rollback(self):
    if not self._allow_rollback:
      return
    # trigger event for callbacks
    self._callback.event(SIG_TRAIN_ROLLBACK)
    # default rollback procedure
    if self._save_path is not None and os.path.exists(self._save_path):
      add_notification("Rollback from:" + self._save_path)
      # restore previous checkpoint immediately
      N.deserialize(self._save_path, force_restore_vars=True)

  def _run(self):
    if self._main_task is None and len(self._evaltask) == 0:
      raise ValueError('You must call `set_task` and set the main task '
          'first, or you can specify evaluation task using `set_eval_task`.')
    for t in self._task + self._subtask + self._evaltask:
      t.set_callbacks(self._callback)
    # ====== prepare subtask ====== #
    if self._main_task is not None:
      finished_task = {i: False for i in self._task + self._subtask}
      task_iter = {i: iter(i) for i in self._task + self._subtask}
      for freq in self._task_freq.values():
        freq.set_counter(self._main_task)
    # ====== main logics ====== #
    while self._main_task is not None and \
    not finished_task[self._main_task]:
      for t in self._task:
        # ====== execute training task first ====== #
        if not finished_task[t]:
          res = next(task_iter[t])
          # task message
          if isinstance(res, str):
            if res == 'task_end':
              finished_task[t] = True
          else: # task result
            pass
          # process callback msg for tasks
          msg = t.callback_msg
          if SIG_TRAIN_SAVE in msg: self._save()
          if SIG_TRAIN_ROLLBACK in msg: self._rollback()
          if SIG_TRAIN_STOP in msg:
            self._callback.event(SIG_TRAIN_STOP)
            finished_task[self._main_task] = True
            break
        # ====== execute valid and eval task ====== #
        for st in self._subtask:
          if finished_task[st]: continue
          if self._task_when[st].check(self._main_task) and \
          self._task_freq[st].update_counter(self._main_task):
            # running 1 epoch of subtask
            for x in task_iter[st]:
              if isinstance(x, str): # signal
                if x == 'task_end': finished_task[st] = True
                if x == 'epoch_end': break
              else: # results
                pass
            # process callback msg for subtasks
            msg = st.callback_msg
            if SIG_TRAIN_SAVE in msg: self._save()
            if SIG_TRAIN_ROLLBACK in msg: self._rollback()
            if SIG_TRAIN_STOP in msg:
              self._callback.event(SIG_TRAIN_STOP)
              finished_task[self._main_task] = True
              break
    # ====== end main task ====== #
    for t in self._task + self._subtask:
      t.stop()
    # ====== Run eval task before finishing ====== #
    if self._allow_rollback:
      self._rollback()
    for et in self._evaltask:
      for x in iter(et):
        pass
    # everything finished
    self._evaltask = []

  def run(self):
    try:
      self._run()
    finally:
      try:
        import curses
        curses.echo()
        curses.nocbreak()
        curses.endwin()
      except Exception:
        pass
