from __future__ import print_function, division, absolute_import

import os
import sys
# Each process only run 1 threads
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import types
import pickle
import inspect
from six import add_metaclass
from decorator import decorator
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import (cpu_count, Process, Queue, Value, Lock,
                             current_process, Pipe)

import numpy as np

# ===========================================================================
# Threading
# ===========================================================================
_async_function_counter = defaultdict(int)
_NTHREADS = max(int(cpu_count()) // 2, 2)
_THREAD_POOL = None
_MAX_PIPE_BLOCK = 8 * 1024 * 1024 # 8MB
# this keep track all the running async-tasks that haven't
# finished
_RUNNING_ASYNC_TASKS = []

def set_max_threads(n):
  """ Set the maximum number of Threads for the
  ThreadPool that execute the async_task.
  """
  global _NTHREADS
  if _NTHREADS != n:
    _NTHREADS = n
    # re-init thread pool
    global _THREAD_POOL
    if _THREAD_POOL is not None:
      _THREAD_POOL.join()
      _THREAD_POOL.close()
    _THREAD_POOL = ThreadPool(processes=_NTHREADS)

def _full_func_name(f):
  return f.__module__ + "." + f.__class__.__qualname__ + '.' + f.__name__

class _async_task(object):
  """ A class converting blocking functions into
  asynchronous functions by using threads or processes.

  """

  def __init__(self, func, *args, **kw):
    self._is_threading = kw.pop('_is_threading')
    _async_function_counter[func] = _async_function_counter[func] + 1
    # specific name
    name = '<async_task:[%s]<%s|#%s>%s Finished:False>' % \
    ('Thread' if self._is_threading else 'Process',
        _full_func_name(func), _async_function_counter[func],
        '(%s)' % ';'.join(inspect.signature(func).parameters.keys()),
    )
    self._name = name
    self._func = func
    # check initialized pool
    # ====== using threading ====== #
    if self._is_threading:
      global _THREAD_POOL
      if _THREAD_POOL is None:
        _THREAD_POOL = ThreadPool(processes=_NTHREADS)
      self._async_task = _THREAD_POOL.apply_async(
          func=func, args=args, kwds=kw)
    # ====== using Multiprocessing ====== #
    else:
      def wrapped_func(conn, *args, **kwargs):
        results = self._func(*args, **kwargs)
        ret_data = pickle.dumps(results,
                                protocol=pickle.HIGHEST_PROTOCOL)
        ret_length = len(ret_data)
        conn.send(ret_length)
        sent_bytes = 0
        while sent_bytes < ret_length:
          data = ret_data[sent_bytes: sent_bytes + _MAX_PIPE_BLOCK]
          conn.send(data)
          sent_bytes += _MAX_PIPE_BLOCK

      parent_conn, child_conn = Pipe()
      self._async_task = Process(target=wrapped_func,
                                 args=(child_conn,) + tuple(args),
                                 kwargs=kw)
      self._async_task.start()
      self._conn = parent_conn

  @property
  def name(self):
    return self._name.replace(':False>', ':%s>' % bool(self.finished))

  def __str__(self):
    return self.name

  def __repr__(self):
    return str(self)

  @property
  def finished(self):
    if self._is_threading:
      return self._async_task.ready() and self._async_task.successful()
    else:
      return self._async_task.is_alive()

  @property
  def finish(self):
    return self.get()

  @property
  def result(self):
    return self.get()

  def get(self, timeout=None):
    """Return actual result of the function"""
    if not hasattr(self, '_result'):
      # ====== multi-threading ====== #
      if self._is_threading:
        self._result = self._async_task.get(timeout=timeout)
      # ====== multi-processing ====== #
      else:
        while not self._conn.poll(): # waiting for data
          pass
        ret_length = self._conn.recv()
        read_bytes = 0
        received = []
        while read_bytes < ret_length:
          data = self._conn.recv()
          received.append(data)
          read_bytes += len(data)
        received = b"".join(received)
        self._result = pickle.loads(received)
        # clean up
        self._conn.close()
        self._async_task.join(timeout=timeout)
    return self._result

def async(func=None):
  """ This create and asynchronized result using Threading instead of
  multiprocessing, you can take advantage from share memory in threading
  for async-IO using this decorator.

  Parameters
  ----------
  func: call-able
      main workload executed in this function and return the
      results.

  callback: call-able
      a callback function triggered when the task finished

  Example
  -------
  >>> from odin.utils import async
  ...
  >>> def test_fn(idx):
  ...   path = '/tmp/tmp%d.txt' % idx
  ...   with open(path, 'w') as f:
  ...     for i in range(100000):
  ...       f.write(str(i))
  ...     print("FINISH!", path)
  ...
  >>> f = async(test_fn)
  >>> x1 = f(1)
  >>> x2 = f(2)
  >>> x3 = f(3)
  ...
  >>> count = 0
  >>> while not x1.finished or not x2.finished or not x3.finished:
  ...   count += 1
  ...   print("Iteration:", count)
  ...   print(x1)
  ...   print(x2)
  ...   print(x3)

  """
  @decorator
  def _decorator_func_(func, *args, **kwargs):
    kwargs['_is_threading'] = True
    task = _async_task(func, *args, **kwargs)
    return task
  # roles are not specified, given function directly
  if inspect.isfunction(func) or inspect.ismethod(func):
    return _decorator_func_(func)
  # roles are specified
  return _decorator_func_

def async_mpi(func=None):
  """ This create and asynchronized result using multi-Processing
  Also check: `odin.utils.mpi.async` for multi-Threading
  decorator

  Parameters
  ----------
  func: call-able
      main workload executed in this function and return the
      results.

  callback: call-able
      a callback function triggered when the task finished

  Example
  -------
  >>> from odin.utils import async
  ...
  >>> def test_fn(idx):
  ...   path = '/tmp/tmp%d.txt' % idx
  ...   with open(path, 'w') as f:
  ...     for i in range(100000):
  ...       f.write(str(i))
  ...     print("FINISH!", path)
  ...
  >>> f = async(test_fn)
  >>> x1 = f(1)
  >>> x2 = f(2)
  >>> x3 = f(3)
  ...
  >>> count = 0
  >>> while not x1.finished or not x2.finished or not x3.finished:
  ...   count += 1
  ...   print("Iteration:", count)
  ...   print(x1)
  ...   print(x2)
  ...   print(x3)

  """
  @decorator
  def _decorator_func_(func, *args, **kwargs):
    kwargs['_is_threading'] = False
    task = _async_task(func, *args, **kwargs)
    return task
  # roles are not specified, given function directly
  if inspect.isfunction(func) or inspect.ismethod(func):
    return _decorator_func_(func)
  # roles are specified
  return _decorator_func_

# ===========================================================================
# Multi-processing
# ===========================================================================
_SLEEP_TIME = 0.01

def segment_list(l, size=None, n_seg=None):
  '''
  Example
  -------
  >>> segment_list([1,2,3,4,5],2)
  >>> # [[1, 2, 3], [4, 5]]
  >>> segment_list([1,2,3,4,5],4)
  >>> # [[1], [2], [3], [4, 5]]
  '''
  # by floor, make sure and process has it own job
  if n_seg is None:
    n_seg = int(np.ceil(len(l) / float(size)))
  # start segmenting
  segments = []
  start = 0
  remain_data = len(l)
  remain_seg = n_seg
  while remain_data > 0:
    # ====== adaptive size ====== #
    size = remain_data // remain_seg
    segments.append(l[start:(start + size)])
    # ====== update remain ====== #
    start += size
    remain_data -= size
    remain_seg -= 1
  return segments

class SharedCounter(object):
  """ A multiprocessing syncrhonized counter """

  def __init__(self, initial_value=0):
    self.val = Value('i', initial_value)
    self.lock = Lock()

  def add(self, value=1):
    with self.lock:
      self.val.value += value

  @property
  def value(self):
    with self.lock:
      return self.val.value

  def __del__(self):
    del self.lock
    del self.val

class MPI(object):
  """ MPI - Multi processing interface
  This class use round robin to schedule the tasks to each processes

  Parameters
  ----------
  jobs: list, tuple, numpy.ndarray
      list of works.
  func: call-able
      take a `list of jobs` as input (i.e. map_func([job1, job2, ...])),
      the length of this list is determined by `buffer_size`
      NOTE: the argument of map_func is always a list.
  ncpu: int
      number of processes.
  batch: int
      the amount of samples grouped into list, and feed to each
      process each iteration. (i.e. func([job0, job1, ...]))
      if `batch=1`, each input is feed individually to `func`
      (i.e. func(job0); func(job1]); ...)
  hwm: int
      "high water mark" for SEND socket, is a hard limit on the
      maximum number of outstanding messages ØMQ shall queue
      in memory for any single peer that the specified socket
      is communicating with.
  chunk_scheduler: bool
      if `True`, jobs are grouped into small chunks of `batch`, then each
      chunk will be feed to each process until the jobs are exhausted, hence,
      this approach guarantee that all processes will run until the end, but
      the execution speed will be lower since each processes need to
      continuously receives chunks from main process.
      if `False`, jobs are splited into equal size for each process at the
      beginning, do this if you sure all jobs require same processing time.
  backend: {'pyzmq', 'python'}
      using 'pyzmq' for interprocess communication or default python Queue.
  Note
  ----
  Using pyzmq backend often 3 time faster than python Queue
  """

  def __init__(self, jobs, func,
               ncpu=1, batch=1, hwm=144,
               backend='python'):
    super(MPI, self).__init__()
    backend = str(backend).lower()
    if backend not in ('pyzmq', 'python'):
      raise ValueError("Only support 2 backends: 'pyzmq', and 'python'")
    self._backend = backend
    self._ID = np.random.randint(0, 10e8, dtype=int)
    # ====== check map_func ====== #
    if not hasattr(func, '__call__'):
      raise Exception('"func" must be call-able')
    self._func = func
    # ====== MPI parameters ====== #
    # never use all available CPU
    if ncpu is None:
      ncpu = cpu_count() - 1
    self._ncpu = min(
        np.clip(int(ncpu), 1, cpu_count() - 1),
        len(jobs)
    )
    self._batch = max(1, int(batch))
    self._hwm = max(0, int(hwm))
    # ====== internal states ====== #
    self._nb_working_cpu = self._ncpu
    # processes manager
    self._is_init = False
    self._is_running = False
    self._is_finished = False
    self._terminate_now = False
    # ====== other queue ====== #
    if not isinstance(jobs, (tuple, list, np.ndarray)):
      raise ValueError("`jobs` must be instance of tuple or list.")
    self._jobs = jobs
    self._remain_jobs = SharedCounter(len(self._jobs))
    # Equally split for all processes
    self._tasks = Queue(maxsize=0)
    for i in segment_list(np.arange(len(self._jobs), dtype='int32'),
                          size=self._batch):
      self._tasks.put_nowait(i)
    for i in range(self._ncpu): # ending signal
      self._tasks.put_nowait(None)
    # ====== only 1 iteration is created ====== #
    self._current_iter = None

  # ==================== properties ==================== #
  def __len__(self):
    """ Return the number of remain jobs """
    return max(self._remain_jobs.value, 0)

  @property
  def nb_working_cpu(self):
    return self._nb_working_cpu

  @property
  def is_initialized(self):
    return self._is_init

  @property
  def is_finished(self):
    return self._is_finished

  @property
  def is_running(self):
    return self._is_running

  def terminate(self):
    self._terminate_now = True
    # force everything finished
    if self._current_iter is not None and \
    not self.is_finished:
      try:
        next(self._current_iter)
      except StopIteration:
        pass

  # ==================== helper ==================== #
  def __iter(self):
    # Initialize
    if not self._is_init:
      if self._backend == 'pyzmq':
        init_func = self._init_zmq
        run_func = self._run_pyzmq
      elif self._backend == 'python':
        init_func = self._init_python
        run_func = self._run_python
      init_func()
      self._is_init = True
    yield None # yeild not thing for init
    # Select run function
    self._is_running = True
    for i in run_func():
      if self._terminate_now:
        break
      yield i
    # Finalize
    self._is_running = False
    self._finalize()
    self._is_finished = True

  def __iter__(self):
    if self._current_iter is None:
      self._current_iter = self.__iter()
      next(self._current_iter)
    return self._current_iter

  # ==================== pyzmq ==================== #
  def _init_zmq(self):
    # this is ugly but work well
    import zmq

    # tasks_or_queue only return the indices, need to get it from self._jobs
    def wrapped_map(pID, tasks, remain_jobs):
      # ====== create ZMQ socket ====== #
      ctx = zmq.Context()
      sk = ctx.socket(zmq.PAIR)
      sk.set(zmq.SNDHWM, self._hwm)
      sk.set(zmq.LINGER, -1)
      sk.bind("ipc:///tmp/%d" % (self._ID + pID))

      # ====== Doing the jobs ====== #
      t = tasks.get()
      while t is not None:
        # `t` is just list of indices
        t = [self._jobs[i] for i in t]
        # monitor current number of remain jobs
        remain_jobs.add(-len(t))
        if self._batch == 1: # batch=1, NO need for list of inputs
          ret = self._func(t[0])
        else: # we have input is list of inputs here
          ret = self._func(t)
        # if a generator is return, traverse through the
        # iterator and return each result
        if not isinstance(ret, types.GeneratorType):
          ret = (ret,)
        for r in ret:
          # ignore None values
          if r is not None:
            sk.send_pyobj(r)
        # delete old data (this work, checked)
        del ret
        # ge tne tasks
        t = tasks.get()
      # ending signal
      sk.send_pyobj(None)
      # wait for ending message
      sk.recv()
      sk.close()
      ctx.term()
      sys.exit(0)
    # ====== start the processes ====== #
    self._processes = [Process(target=wrapped_map,
                               args=(i, self._tasks, self._remain_jobs))
                       for i in range(self._ncpu)]
    [p.start() for p in self._processes]
    # ====== pyzmq PULL socket ====== #
    ctx = zmq.Context()
    sockets = []
    for i in range(self._ncpu):
      sk = ctx.socket(zmq.PAIR)
      sk.set(zmq.RCVHWM, 0) # no limit receiving
      sk.connect("ipc:///tmp/%d" % (self._ID + i))
      sockets.append(sk)
    self._ctx = ctx
    self._sockets = sockets
    self._zmq_noblock = zmq.NOBLOCK
    self._zmq_again = zmq.error.Again

  def _run_pyzmq(self):
    while self._nb_working_cpu > 0:
      for sk in self._sockets:
        try:
          r = sk.recv_pyobj(flags=self._zmq_noblock)
          if r is None:
            self._nb_working_cpu -= 1
            sk.send(b'')
            sk.close()
            self._sockets.remove(sk)
          else:
            yield r
        except self._zmq_again:
          pass

  # ==================== python queue ==================== #
  def _init_python(self):
    def worker_func(tasks, queue, counter, remain_jobs):
      hwm = self._hwm
      minimum_update_size = max(hwm // self._ncpu, 1)
      # ====== Doing the jobs ====== #
      t = tasks.get()
      while t is not None:
        # `t` is just list of indices
        t = [self._jobs[i] for i in t]
        remain_jobs.add(-len(t)) # monitor current number of remain jobs
        if self._batch == 1: # batch=1, NO need for list of inputs
          ret = self._func(t[0])
        else: # we have input is list of inputs here
          ret = self._func(t)
        # if a generator is return, traverse through the
        # iterator and return each result
        if not isinstance(ret, types.GeneratorType):
          ret = (ret,)
        nb_returned = 0
        for r in ret:
          if r is not None: # ignore None values
            queue.put(r)
            nb_returned += 1
            # sometime 1 batch get too big, and we need to stop
            # putting too many data into the queue
            if nb_returned >= minimum_update_size:
              counter.add(nb_returned)
              nb_returned = 0
              while counter.value > hwm:
                time.sleep(_SLEEP_TIME)
        del ret # delete old data (this work, checked)
        # increase shared counter (this number must perfectly
        # counted, only 1 mismatch and deadlock will happen)
        if nb_returned > 0:
          counter.add(nb_returned)
        # check if we need to wait for the consumer here
        while counter.value > hwm:
          time.sleep(_SLEEP_TIME)
        # get new tasks
        t = tasks.get()
      # ending signal
      queue.put(None)
      sys.exit(0)
    # ====== multiprocessing variables ====== #
    self._queue = Queue(maxsize=0)
    self._counter = SharedCounter(initial_value=0)
    self._processes = [Process(target=worker_func,
                               args=(self._tasks, self._queue,
                                     self._counter, self._remain_jobs))
                       for i in range(self._ncpu)]
    [p.start() for p in self._processes]

  def _run_python(self):
    while self._nb_working_cpu > 0:
      r = self._queue.get()
      while r is None:
        self._nb_working_cpu -= 1
        if self._nb_working_cpu <= 0:
          break
        r = self._queue.get()
      if r is not None:
        self._counter.add(-1)
        yield r

  # ==================== finalize ==================== #
  def _finalize(self):
    # terminate or join all processes
    if self._terminate_now:
      [p.terminate() for p in self._processes
       if p._popen is not None and p.is_alive()]
    # only join started process which has _popen is not None
    else:
      [p.join() for p in self._processes
      if p._popen is not None]
    self._tasks.close()
    del self._remain_jobs
    # ====== pyzmq ====== #
    if self._backend == 'pyzmq':
      for sk in self._sockets:
        sk.close()
      self._ctx.term()
    # ====== python ====== #
    elif self._backend == 'python':
      self._queue.close()
      del self._counter
