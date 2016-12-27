from __future__ import print_function, division, absolute_import

import os
import time
import types
import inspect
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from multiprocessing import cpu_count, Process, Queue, Value, Lock, current_process

import numpy as np

from odin import SIG_TERMINATE_ITERATOR
# ===========================================================================
# Helper methods
# ===========================================================================
_BACKEND = 'multiprocessing'
_SUPPORT_BACKEND = ('multiprocessing')


def set_backend(backend):
    if backend not in _SUPPORT_BACKEND:
        raise ValueError('%s backend is not supported, the list of supported '
                         'backend is:' % (backend, _SUPPORT_BACKEND))
    global _BACKEND
    _BACKEND = backend


def get_supported_backend():
    return _SUPPORT_BACKEND


def segment_list(l, size=None, n_seg=None):
    '''
    Example
    -------
    >>> segment_list([1,2,3,4,5],2)
    >>> [[1, 2, 3], [4, 5]]
    >>> segment_list([1,2,3,4,5],4)
    >>> [[1], [2], [3], [4, 5]]
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


@add_metaclass(ABCMeta)
class SelfIterator(object):
    """ Extend the implementation of standard iterator
    Allow the iterator to be stopped, and monitoring remain length

    Example
    -------
    >>> class TestIt(SelfIterator):
    >>>     def __init__(self):
    >>>         super(TestIt, self).__init__()
    >>>         self.i = -1
    >>>         self.list = [1, 2, 3, 4]
    >>>     def _next(self):
    >>>         if self.i < 3:
    >>>             self.i += 1
    >>>             return self.list[self.i]
    >>>         raise StopIteration
    >>>     def __len__(self):
    >>>         return len(self.list) - self.i - 1
    ...
    >>> it = TestIt()
    >>> for i in it:
    >>>     print(i, len(it)) # output: 1 3, 2 2, 3 1, 4 0
    >>> print(it.finnished) # True
    >>> for i in it:
    >>>     print(i) # nothing printed out

    Note
    ----
    raise StopIteration at the end of the iteration
    _is_finished is assigned to 2 if stop() is called

    """

    def __init__(self):
        super(SelfIterator, self).__init__()
        self._is_finished = False
        self._is_initialized = False

    def __str__(self):
        return '<%s: length=%d: finished=%s: replicable=%s>' % \
        (self.__class__.__name__, len(self), self.finnished, self.replicable)

    # ==================== general API ==================== #
    def __iter__(self):
        self._is_initialized = True
        self._init()
        return self

    @property
    def finnished(self):
        return self._is_finished

    @property
    def replicable(self):
        """ return True of this Iter is copy-able """
        if 'NotImplementedError' in inspect.getsource(self._copy):
            return False
        return True

    #python 2
    def next(self):
        # check if initilaized
        if not self._is_initialized:
            self._init()
            self._is_initialized = True
        # check if finished
        if self.finnished:
            raise StopIteration
        # run the iteration
        try:
            return self._next()
        except StopIteration: # first time finish
            self._is_finished = True
            self._finalize()
            raise StopIteration

    #python 3
    def __next__(self):
        # check if initilaized
        if not self._is_initialized:
            self._init()
            self._is_initialized = True
        # check if finished
        if self.finnished:
            raise StopIteration
        # run the iteration
        try:
            return self._next()
        except StopIteration: # first time finish
            self._is_finished = True
            self._finalize()
            raise StopIteration

    def stop(self):
        # call finalize if everything is running
        need_to_finalize = not self._is_finished
        self._is_finished = SIG_TERMINATE_ITERATOR
        if need_to_finalize:
            self._finalize()

    def copy(self):
        return self._copy()

    # ==================== abstract methods ==================== #
    def _init(self):
        """ Called before the iteration start """
        pass

    def _finalize(self):
        pass

    def _copy(self):
        raise NotImplementedError

    @abstractmethod
    def _next(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """ Return the remain jobs in the iterator not the
        full length """
        raise NotImplementedError


# ===========================================================================
# Main API
# ===========================================================================
class MPI(SelfIterator):
    """ Multiprocessing interface

    Parameters
    ----------
    jobs: str
        pass
    map_func: callable
        take input as a single job (i.e. map_func(job))
    reduce_func: callable
        take input as a non-None returned object from map_func
    buffer_size: int
        the amount of data each process keep before return to main
        process.
    maximum_queue_size: int (default: 66)
        maximum number of batch will be cached in Queue before main process
        get it and feed to the GPU (if the too many results in Queue, all
        subprocess will be paused)


    Notes
    -----
    If map_func return None, it won't be queued to the results for reduct_func
    If map_func return a Generator, MPI will traverses through it and queues all
    returned values.

    Benchmark
    ---------
    NO LOCK
    2-2: 0.68 0.66 0.66
    4-4: 0.59 0.59 0.62
    LOCK
    2-2: 0.69 0.66 0.66
    4-4: 0.6 0.6 0.58

    Example
    -------
    >>> jobs = list(range(0, 12 * 2, 1))
    >>> def map_func(sub_jobs):
    >>>     ret = [i + 1 for i in sub_jobs]
    >>>     print('Map:', ret)
    >>>     return ret
    >>> def reduce_func(result):
    >>>     return sum(result)
    >>> x = MPI(jobs, map_func, reduce_func,
    >>>         ncpu=3, buffer_size=2, maximum_queue_size=12)
    >>> for i in x.run():
    >>>     print(i)
    >>> print('--------')
    >>> for i in x.run():
    >>>     print(i)
    >>> # Map: [1, 2]
    >>> # Map: [9, 10]
    >>> # Map: [3, 4]
    >>> # Map: [5, 6]
    >>> # Map: [7, 8]
    >>> # 3
    >>> # Map: [17, 18]
    >>> # 7
    >>> # Map: [11, 12]
    >>> # 11
    >>> # 19
    >>> # 15
    >>> # Map: [13, 14]
    >>> # 23
    >>> # 27
    >>> # Map: [15, 16]
    >>> # Map: [19, 20]
    >>> # 35
    >>> # 31
    >>> # Map: [21, 22]
    >>> # 39
    >>> # Map: [23, 24]
    >>> # 43
    >>> # 47
    >>> # ----
    >>> # Exception (cannot re-run the same MPI)
    """

    def __init__(self, jobs, map_func, reduce_func=None,
                 ncpu=1, buffer_size=1, maximum_queue_size=144):
        super(MPI, self).__init__()
        self._jobs = jobs
        # ====== check map_func ====== #
        if not callable(map_func):
            raise Exception('"map_func" must be callable')
        self._map_func = map_func
        # ====== check reduce_func ====== #
        if reduce_func is None: reduce_func = lambda x: x
        if not callable(reduce_func):
            raise Exception('"reduce_func" must be callable or None')
        self._reduce_func = reduce_func
        # ====== MPI parameters ====== #
        self._length = SharedCounter(len(jobs))
        # never use all available CPU
        if ncpu is None:
            ncpu = cpu_count() - 1
        self._ncpu = max(min(ncpu, 2 * cpu_count() - 1), 1)
        self._maximum_queue_size = maximum_queue_size
        self._buffer_size = buffer_size
        # processes manager
        self.__processes_started = False
        self.__shared_counter = SharedCounter()
        self.__results = Queue(maxsize=0)
        self.__nb_working_processes = self._ncpu

    def _copy(self):
        return MPI(self._jobs, self._map_func, self._reduce_func,
                   self._ncpu, self._buffer_size, self._maximum_queue_size)

    def _init(self):
        jobs = segment_list(self._jobs, n_seg=self._ncpu)

        def wrapped_map(tasks, return_queue, counter, length):
            maximum_queue_size = self._maximum_queue_size
            minimum_queue_size = max(maximum_queue_size // self._ncpu, 1)
            for i in range(0, len(tasks), self._buffer_size):
                t = tasks[i:i + self._buffer_size]
                length.add(-len(t)) # monitor current length
                ret = self._map_func(t)
                # if a generator is return, traverse through the
                # iterator and return each result
                if not isinstance(ret, types.GeneratorType):
                    ret = (ret,)
                nb_returned = 0
                for r in ret:
                    if r is not None:
                        return_queue.put(r)
                        nb_returned += 1
                        # sometime 1 batch get too big, and we need to stop
                        # putting too many data into the queue.
                        if nb_returned >= minimum_queue_size:
                            counter.add(nb_returned)
                            nb_returned = 0
                            while counter.value > maximum_queue_size:
                                time.sleep(0.1)
                del ret # delete old data (this work, checked)
                # increase shared counter (this number must perfectly
                # counted, only 1 mismatch and deadlock will happen)
                if nb_returned > 0:
                    counter.add(nb_returned)
                # check if we need to wait for the consumer here
                while counter.value > maximum_queue_size:
                    time.sleep(0.1)
            # ending signal
            return_queue.put(None)
        # ====== multiprocessing variables ====== #
        self.__processes = [Process(target=wrapped_map,
                                    args=(j, self.__results, self.__shared_counter, self._length))
                           for i, j in enumerate(jobs)]

    def _finalize(self):
        self.__nb_working_processes = 0
        if not self.__processes_started:
            return
        # terminate or join all processes
        if self.finnished == SIG_TERMINATE_ITERATOR:
            [p.terminate() for p in self.__processes if p.is_alive()]
        else:
            [p.join() for p in self.__processes]
        self.__results.close()

    def _next(self):
        # if the processes haven't started, start them only once
        if not self.__processes_started:
            [p.start() for p in self.__processes]
            self.__processes_started = True
        # ====== end of iteration ====== #
        if self.__nb_working_processes <= 0:
            raise StopIteration
        # ====== fetch the results ====== #
        r = self.__results.get()
        while r is None:
            self.__nb_working_processes -= 1
            if self.__nb_working_processes <= 0:
                break
            r = self.__results.get()
        # still None, no more tasks to do
        if r is None: raise StopIteration
        # otherwise, something to return and reduce the counter
        self.__shared_counter.add(-1)
        return self._reduce_func(r)

    def __len__(self):
        return max(self._length.value, 0)

    def run(self):
        """"""
        if self.finnished:
            raise Exception('The MPI already finished, call copy() to '
                            'replicate this MPI, and re-run it if you want.')
        return iter(self)
