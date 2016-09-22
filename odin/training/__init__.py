from __future__ import division, absolute_import, print_function

import os
import cPickle
from six.moves import range, zip

import numpy as np

from odin import (SIG_TERMINATE_ITERATOR, SIG_TRAIN_ROLLBACK,
                  SIG_TRAIN_SAVE, SIG_TRAIN_STOP)
from odin.config import RNG_GENERATOR
from odin import fuel
from odin.fuel.dataset import Dataset
from odin.utils import struct, signal_handling
from odin.utils.decorators import terminatable_iterator

from .callbacks import *


# ===========================================================================
# Helper
# ===========================================================================
_SAVE_TASK = struct()
_SAVE_TASK.name = "save"


# ===========================================================================
# Tasks
# ===========================================================================
class Task(object):

    def __init__(self, func, data, epoch, p, batch_size, seed, shuffle_level,
                 name=None):
        super(Task, self).__init__()
        if not callable(func):
            raise ValueError('func must be instance of theano.Function or '
                             'python function, method, or hasattr __call__.')
        if not isinstance(data, (tuple, list)):
            data = [data]
        data = [fuel.data(i) for i in data]

        self._func = func
        self._data = data
        self._epoch = epoch
        self._p = np.clip(p, 0., 1.)

        self.set_batch(batch_size, seed, shuffle_level)
        self._name = name

        self._created_iter = []
        self._stop_all = False

    @property
    def name(self):
        return str(self._name)

    @property
    def epoch(self):
        return self._epoch

    @property
    def samples_per_epoch(self):
        ''' Estimated number of iteration for each epoch '''
        return self._nb_samples_per_epoch

    @property
    def iter_per_epoch(self):
        ''' Estimated number of iteration for each epoch '''
        return int(np.ceil(self._nb_samples_per_epoch / self._batch_size))

    def set_batch(self, batch_size=None, seed=-1, shuffle_level=None):
        if batch_size is not None:
            self._batch_size = batch_size
            self._nb_samples_per_epoch = min([len(i) for i in self._data])
        if seed is None or seed >= 0:
            if seed is not None:
                self._rng = np.random.RandomState(seed)
            else:
                self._rng = struct()
                self._rng.randint = lambda x: None
                self._rng.rand = RNG_GENERATOR.rand
        if shuffle_level is not None:
            self._shuffle_level = min(max(int(shuffle_level), 0), 2)
        return self

    def stop_all(self):
        """ Stop all iterations running for this Task"""
        self._stop_all = True
        for i in self._created_iter:
            for j in i: # just run to end of the iterators
                pass
        self._stop_all = False
        self._created_iter = []

    def __iter(self):
        '''
        Return
        ------
        'task_start':
        'epoch_start' : beginning of epoch
        'epoch_end' : epoch ended
        'task_end' : task ended
        (results, nb_iter, nb_samples, nb_epoch) : results of execute function on data

        Note
        ----
        'end_task' also end of final epoch
        '''
        nb_iter = 0
        p = self._p
        nb_epoch = 0
        nb_samples = 0
        forced_to_terminate = False
        yield None # just for initalize the iterator
        yield 'task_start'
        while nb_epoch < self._epoch:
            nb_epoch += 1
            seed = self._rng.randint(10e8)
            # if only 1 Data, don't need zip or we will mess up
            if len(self._data) == 1:
                data = iter(self._data[0].set_batch(
                    batch_size=self._batch_size, seed=seed,
                    shuffle_level=self._shuffle_level))
                data_it = (data,)
            else:
                data_it = [iter(i.set_batch(batch_size=self._batch_size,
                                            seed=seed,
                                            shuffle_level=self._shuffle_level))
                           for i in self._data]
                data = zip(*data_it)
            yield 'epoch_start'
            # ======  start the iteration ====== #
            nb_samples_per_epoch = 0 # number of iteration for 1 epoch
            for i, x in enumerate(data):
                # alread terminated, try to exhausted the iterator
                # if forced_to_terminate: continue
                # preprocessed the data
                if not isinstance(x, (tuple, list)):
                    x = [x]
                # update some info
                shape0 = x[0].shape[0]
                nb_samples += shape0
                nb_samples_per_epoch += x[0].shape[0]
                nb_iter += 1
                # apply the function
                if p >= 1. or (p < 1 and self._rng.rand() < p):
                    results = self._func(*x)
                else:
                    results = None
                # return results and check TERMINATE signal
                yield (results, nb_iter, nb_samples, nb_epoch)
                if self._stop_all:
                    forced_to_terminate = True
                    # send signal to the data iterators also
                    for i in data_it:
                        i.send(SIG_TERMINATE_ITERATOR)
                    break # break the loop
            # ====== check if terminate ====== #
            if forced_to_terminate:
                break
            # ====== check if we got the right number for epoch iter ====== #
            if nb_samples_per_epoch != self._nb_samples_per_epoch:
                # just for sure should not smaller than the real number
                self._nb_samples_per_epoch = nb_samples_per_epoch
            # ======  end_epoch or task ====== #
            if nb_epoch >= self._epoch:
                yield 'epoch_end'
                yield 'task_end'
            else:
                yield 'epoch_end'
        # ====== end of iteration ====== #

    def __iter__(self):
        it = self.__iter()
        it.next()
        self._created_iter.append(it)
        return it

    def __del__(self):
        self.stop_all()


# ===========================================================================
# MainLoop
# ===========================================================================
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

    """

    def __init__(self, batch_size=256, seed=-1, shuffle_level=0):
        super(MainLoop, self).__init__()
        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability

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

    # ==================== Signal handling ==================== #
    def _signal_handlers(self, sig, frames):
        # do something here
        pass

    # ==================== pickling ==================== #
    def __setstate__(self, value):
        self.set_batch(batch_size=value[0], shuffle_level=value[2])
        self._rng = value[1]

        self._callback = value[-1]

        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability

    def __getstate__(self):
        return (self._batch_size, self._rng, self._shuffle_level,
                self._callback)

    def set_save(self, path, obj, save_hist=True):
        """
        Parameters
        ----------
        path: str
            path to save the obj when the callback return save signal
        obj: object
            any pickle-able object you want to save
        save_hist: boolean
            if True, the History callback will be save together at the
            save path but different file extension: '.hist'
        """
        self._save_path = path
        self._save_obj = obj
        # ====== infer history_path ====== #
        if save_hist:
            base = os.path.basename(path)
            p = path.replace(base, ''); base = base.split('.')
            base = '.'.join(base[:-1] if len(base) > 1 else base)
            self._save_hist = os.path.join(p, base + '.hist')
        else:
            self._save_hist = None

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
        if self._task is not None:
            self._task.set_batch(batch_size=batch_size, seed=seed,
                                 shuffle_level=shuffle_level)
        for i in self._subtask.itervalues():
            i.set_batch(batch_size=batch_size, seed=seed,
                        shuffle_level=shuffle_level)
        for i in self._crosstask.itervalues():
            i.set_batch(batch_size=batch_size, seed=seed,
                        shuffle_level=shuffle_level)

        return self

    @property
    def callback(self):
        return self._callback

    def __str__(self):
        return 'Task'

    def set_callback(self, callback):
        if isinstance(callback, CallbackList):
            self._callback = callback
        else:
            if not isinstance(callback, (tuple, list)):
                callback = [callback]
            self._callback = CallbackList(*callback)
        return self

    def __getitem__(self, key):
        """ Return callback from callback list"""
        return self._callback[key]

    # ==================== main ==================== #
    def _validate_data(self, data):
        if not isinstance(data, (list, tuple)):
            data = [data]
        return [fuel.data(i) for i in data]

    def set_task(self, func, data, epoch=1, p=1., name=None):
        '''
        '''
        self._task = Task(func, self._validate_data(data), epoch, 1.,
                          batch_size=self._batch_size,
                          seed=self._rng.randint(10e8),
                          shuffle_level=self._shuffle_level,
                          name=name)
        return self

    def set_subtask(self, func, data, epoch=float('inf'), p=1., freq=0.5,
                    when=0, name=None):
        '''
        Parameters
        ----------
        when: float
            percentage of epoch of main task before this task is executed
            negative value => execute after final epoch of main task
        freq: float
            percentage of epoch of main task before this task is executed
        '''
        self._subtask[Task(func, self._validate_data(data), epoch, p,
                           batch_size=self._batch_size,
                           seed=self._rng.randint(10e8),
                           shuffle_level=self._shuffle_level,
                           name=name)] = (freq, when)
        return self

    def set_crosstask(self, func, data, epoch=float('inf'), p=0.5,
                      when=0, name=None):
        '''
        Parameters
        ----------
        when: float
            percentage of epoch of main task before this task is executed
            negative value => execute after final epoch of main task
        '''
        self._crosstask[Task(func, self._validate_data(data), epoch, p,
                             batch_size=self._batch_size,
                             seed=self._rng.randint(10e8),
                             shuffle_level=self._shuffle_level,
                             name=name)] = when
        return self

    # ==================== logic ==================== #
    def _save(self):
        if self._save_path is not None and self._save_obj is not None:
            cPickle.dump(self._save_obj, open(self._save_path, 'w'),
                         protocol=cPickle.HIGHEST_PROTOCOL)
            # ====== save history if possible ====== #
            if self._save_hist is not None and 'History' in self._callback:
                cPickle.dump(self._callback['History'], open(self._save_hist, 'w'),
                             protocol=cPickle.HIGHEST_PROTOCOL)

    def _rollback(self):
        if self._save_path is not None and os.path.exists(self._save_path):
            f = open(self._save_path, 'r')
            # the loading process will automatically reload shared variable
            cPickle.load(f)
            f.close()

    def __run(self):
        if self._task is None:
            raise ValueError('You must call set_task and set the main task first.')
        callback = self._callback
        batch_size = self._batch_size
        # ====== prepare subtask ====== #
        # iterator, is_ended=False
        subtask_map = {i: [iter(i), False] for i in self._subtask}
        # iterator, is_ended=False
        crosstask_map = {i: [iter(i), False] for i in self._crosstask}
        # ====== main logics ====== #
        msg = [] # store returned callback messages
        for i in self._task: # each iteration is an batch
            # return signal: start_epoch, end_epoch or end_task
            if isinstance(i, str):
                msg = callback.record(self._task.name, event_type=i,
                                      nb_iter=0, nb_epoch=0, nb_samples=0,
                                      results=None,
                                      samples_size=self._task.samples_per_epoch)
            # return actual results
            else:
                # ====== main task ====== #
                results, nb_iter, nb_samples, nb_epoch = i
                msg = callback.record(self._task.name, event_type='batch_end',
                                      nb_iter=nb_iter, nb_epoch=nb_epoch,
                                      nb_samples=nb_samples, results=results,
                                      samples_size=self._task.samples_per_epoch)
                # ====== run subtask ====== #
                for subtask, (freq, when) in self._subtask.iteritems():
                    subtask_iter, is_end = subtask_map[subtask]
                    if is_end: continue # already ended
                    # check if it is good time to start, if when is negative,
                    # start from last epoch.
                    when = float(when % self._task.epoch) + 1. if when < 0 else when
                    when = int(when * self._task.samples_per_epoch)
                    freq = int(freq * self._task.samples_per_epoch)
                    # OK to run
                    if nb_samples > batch_size and nb_samples >= when and \
                    (nb_samples - when) % freq < batch_size:
                        for x in subtask_iter:
                            if isinstance(x, str): # signal
                                msg = callback.record(subtask.name, x,
                                                      0, 0, 0, None)
                                if x == 'task_end': # task finnished
                                    subtask_map[subtask][-1] = True
                                if x == 'epoch_end': break
                            else: # results
                                msg = callback.record(subtask.name, 'batch_end',
                                                nb_iter=x[1], nb_samples=x[2],
                                                nb_epoch=x[3], results=x[0],
                                                samples_size=subtask.samples_per_epoch)
                            # process callback msg for subtasks
                            if SIG_TRAIN_SAVE in msg: self._save()
                            if SIG_TRAIN_ROLLBACK in msg: self._rollback()
                            if SIG_TRAIN_STOP in msg: break
                # ====== run crosstask ====== #
                callback.mode = 'crosstask'
                for crosstask, when in self._crosstask.iteritems():
                    (crosstask_iter, crosstask_epoch,
                     crosstask_results, is_end) = crosstask_map[crosstask]
                    if is_end: continue # already ended
                    # check if it is good time to start, if when is negative,
                    # start from last epoch.
                    when = float(when % self._task.epoch) + 1. if when < 0 else when
                    when = int(when * self._task.samples_per_epoch)
                    # OK to run
                    if nb_samples > batch_size and nb_samples >= when and not is_end:
                        x = crosstask_iter.next()
                        if isinstance(x, str): # signals
                            msg = callback.record(crosstask.name, x,
                                                  0, 0, 0, None)
                            if x == 'task_end': # finnished crosstask
                                crosstask_map[crosstask][-1] = True
                        else: #results
                            msg = callback.record(crosstask.name, 'batch_end',
                                            nb_iter=x[1], nb_samples=x[2],
                                            nb_epoch=x[3], results=x[0])
            # ====== process callback msg for main task ====== #
            # (this is important order)
            if SIG_TRAIN_SAVE in msg: self._save()
            if SIG_TRAIN_ROLLBACK in msg: self._rollback()
            if SIG_TRAIN_STOP in msg: break
        # ====== end main task ====== #
        self._task.stop_all()
        for t in self._subtask.keys():
            t.stop_all()
        for t in self._crosstask.keys():
            t.stop_all()

    def run(self):
        # with signal_handling(sigint=self._signal_handlers,
        #                      sigtstp=self._signal_handlers,
        #                      sigquit=self._signal_handlers):
        self.__run()
