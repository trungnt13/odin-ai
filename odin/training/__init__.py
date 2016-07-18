from __future__ import division, absolute_import

from six.moves import range, zip

import numpy as np

from odin import SIG_TERMINATE_ITERATOR
from odin.config import RNG_GENERATOR
from odin import fuel
from odin.fuel.dataset import Dataset
from odin.utils import struct
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
                 preprocess=None, name=None):
        super(Task, self).__init__()
        if not hasattr(func, '__call__'):
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
        self._preprocess = preprocess if hasattr(preprocess, '__call__') else lambda x: x
        self._name = name

        self._created_iter = []

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
        for i in self._created_iter:
            try:
                i.send(SIG_TERMINATE_ITERATOR)
            except:
                pass
        self._created_iter = []

    def __iter(self):
        '''
        Return
        ------
        'start_epoch' : beginning of epoch
        'end_epoch' : epoch ended
        'end_task' : task ended
        (results, n_iter, n_samples, n_epoch) : results of execute function on data

        Note
        ----
        'end_task' also end of final epoch
        '''
        n_iter = 0
        p = self._p
        nb_epoch = 0
        nb_samples = 0
        forced_to_terminate = False
        yield None # just for initalize the iterator
        yield 'start_task'
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
            yield 'start_epoch'
            # ======  start the iteration ====== #
            nb_samples_per_epoch = 0 # number of iteration for 1 epoch
            for i, x in enumerate(data):
                # alread terminated, try to exhausted the iterator
                # if forced_to_terminate: continue
                # preprocessed the data
                x = self._preprocess(x)
                if not isinstance(x, (tuple, list)):
                    x = [x]
                # update some info
                shape0 = x[0].shape[0]
                nb_samples += shape0
                nb_samples_per_epoch += x[0].shape[0]
                n_iter += 1
                # apply the function
                if p >= 1. or (p < 1 and self._rng.rand() < p):
                    results = self._func(*x)
                else:
                    results = None
                # return results and check TERMINATE signal
                if (yield (results, n_iter, nb_samples, nb_epoch)) == SIG_TERMINATE_ITERATOR:
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
                yield 'end_task'
            else:
                yield 'end_epoch'
        # keep ending so no Exception
        while not forced_to_terminate:
            yield 'end_task'

    def __iter__(self):
        it = self.__iter()
        it.next()
        self._created_iter.append(it)
        return it


class MainLoop(object):

    """ MainLoop """

    def __init__(self, batch_size=256, dataset=None, seed=-1,
                 shuffle_level=0):
        super(MainLoop, self).__init__()
        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability

        self.set_batch(batch_size=batch_size, seed=seed,
                       shuffle_level=shuffle_level)

        if isinstance(dataset, str):
            dataset = Dataset(dataset)
        elif dataset is not None and not isinstance(dataset, Dataset):
            raise Exception('input dataset can be path (string) or Dataset instance.')
        self._dataset = dataset
        self._callback = CallbackList()

        self._stop_now = False
        self._save_now = False

    # ==================== pickling ==================== #
    def __setstate__(self, value):
        self.set_batch(batch_size=value[0], shuffle_level=value[2])
        self._rng = value[1]

        if isinstance(value[3], str):
            self._dataset = Dataset(value[3])
        else:
            self._dataset = None

        self._callback = value[-1]

        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability

        self._stop_now = False
        self._save_now = False

    def __getstate__(self):
        dataset = self._dataset.path if self._dataset is not None else None
        return (self._batch_size, self._rng, self._shuffle_level,
                dataset, self._callback)

    # ==================== command ==================== #
    def stop(self):
        self._stop_now = True

    def save(self):
        self._save_now = True

    # ==================== properties ==================== #
    @property
    def batch_size(self):
        return self._batch_size

    def set_batch(self, batch_size=None, seed=-1, shuffle_level=None):
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

    def set_callback(self, *callback):
        self._callback = CallbackList(*callback)
        return self

    def __getitem__(self, key):
        return self._callback[key]

    # ==================== main ==================== #
    def _validate_data(self, data):
        if not isinstance(data, (list, tuple)):
            data = [data]
        data = [self._dataset[i] if isinstance(i, (str, tuple, list, dict))
                else fuel.data(i)
                for i in data]
        return data

    def set_task(self, func, data, epoch=1, p=1., preprocess=None, name=None):
        '''
        '''
        self._task = Task(func, self._validate_data(data), epoch, 1.,
                          batch_size=self._batch_size,
                          seed=self._rng.randint(10e8),
                          shuffle_level=self._shuffle_level,
                          preprocess=preprocess,
                          name=name)
        return self

    def add_subtask(self, func, data, epoch=float('inf'), p=1., freq=0.5,
                    when=0, preprocess=None, name=None):
        '''
        Parameters
        ----------
        when: float
            percentage of epoch of main task before this task is executed
            negative value => execute after final epoch of main task
        freq: float
            percentage of epoch of main task before this task is executed
        preprocess: function
            input is list of data, output is transformed data for batch
        '''
        self._subtask[Task(func, self._validate_data(data), epoch, p,
                           batch_size=self._batch_size,
                           seed=self._rng.randint(10e8),
                           shuffle_level=self._shuffle_level,
                           preprocess=preprocess,
                           name=name)] = (freq, when)
        return self

    def add_crosstask(self, func, data, epoch=float('inf'), p=0.5,
                      when=0, preprocess=None, name=None):
        '''
        Parameters
        ----------
        when: float
            percentage of epoch of main task before this task is executed
            negative value => execute after final epoch of main task
        preprocess: function
            input is list of data, output is transformed data for batch
        '''
        self._crosstask[Task(func, self._validate_data(data), epoch, p,
                             batch_size=self._batch_size,
                             seed=self._rng.randint(10e8),
                             shuffle_level=self._shuffle_level,
                             preprocess=preprocess,
                             name=name)] = when
        return self

    # ==================== logic ==================== #
    def run(self):
        if self._task is None:
            raise ValueError('You must call set_task and set the main task first.')
        callback = self._callback
        callback.mainloop = self # set main loop
        epoch_results = []
        task_results = []
        batch_size = self._batch_size
        # ====== prepare subtask ====== #
        # iterator, task_results, is_ended=False
        subtask_map = {i: [iter(i), [], False] for i in self._subtask}
        # iterator, epoch_results, task_results, is_ended=False
        crosstask_map = {i: [iter(i), [], [], False] for i in self._crosstask}
        # ====== main logics ====== #
        for i in self._task: # each iteration is an batch
            # ====== check if stop_now ====== #
            if self._stop_now:
                self._stop_now = False
                break
            # ====== check if save_now ====== #
            # little hacky, so mainloop can signal Callback to save
            if self._save_now:
                self._save_now = False
                callback.mode = 'othertask'
                callback.task = _SAVE_TASK
                callback.task_start()
                callback.task_end() # just end right after start
                callback.reset()
            # ====== Main task ====== #
            callback.mode = 'task' # dirty hack
            callback.reset(); callback.task = self._task
            # return signal: start_epoch, end_epoch or end_task
            if isinstance(i, str):
                if i == 'start_task':
                    callback.task_start()
                elif i == 'start_epoch':
                    callback.epoch_start()
                elif i == 'end_epoch' or i == 'end_task':
                    callback.results = epoch_results
                    callback.epoch_end()
                    epoch_results = []
                    if i == 'end_task':
                        callback.results = task_results
                        callback.task_end()
                        break # end everything
            # return actual results
            else:
                results, niter, nsamples, nepoch = i
                epoch_results.append(results)
                task_results.append(results)
                callback.results = results
                callback.iter = niter
                callback.nb_samples = nsamples
                callback.epoch = nepoch
                callback.batch_end()
                # ====== run subtask ====== #
                callback.mode = 'subtask'
                for subtask, (freq, when) in self._subtask.iteritems():
                    subtask_iter, subtask_results, is_end = subtask_map[subtask]
                    if is_end: continue # already ended
                    # check if it is good time to start, if when is negative,
                    # start from last epoch.
                    when = float(when % self._task.epoch) + 1. if when < 0 else when
                    when = int(when * self._task.samples_per_epoch)
                    freq = int(freq * self._task.samples_per_epoch)
                    # OK to run
                    if nsamples > batch_size and nsamples >= when and \
                    (nsamples - when) % freq <= batch_size:
                        callback.reset(); callback.task = subtask
                        x = subtask_iter.next()
                        if x == 'start_task':
                            callback.task_start()
                            x = subtask_iter.next()
                        if x == 'start_epoch':
                            callback.epoch_start()
                            subepoch_results = []
                            while x != 'end_epoch' and x != 'end_task':
                                x = subtask_iter.next()
                                if isinstance(x, tuple):
                                    subepoch_results.append(x[0])
                                    subtask_results.append(x[0])
                                    callback.results = x[0]
                                    callback.iter = x[1]
                                    callback.nb_samples = x[2]
                                    callback.epoch = x[3]
                                    callback.batch_end()
                            callback.results = subepoch_results
                            callback.epoch_end()
                        if x == 'end_task':
                            callback.results = subtask_results
                            callback.task_end()
                            subtask_map[subtask][-1] = True
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
                    if nsamples > batch_size and nsamples >= when and not is_end:
                        callback.reset(); callback.task = crosstask
                        x = crosstask_iter.next()
                        if x == 'start_task':
                            callback.task_start()
                            x = crosstask_iter.next()
                        if x == 'start_epoch':
                            callback.epoch_start()
                            x = crosstask_iter.next()
                        if isinstance(x, tuple):
                            crosstask_epoch.append(x[0])
                            crosstask_results.append(x[0])
                            callback.results = x[0]
                            callback.iter = x[1]
                            callback.nb_samples = x[2]
                            callback.epoch = x[3]
                            callback.batch_end()
                        elif x == 'end_epoch' or x == 'end_task':
                            callback.results = crosstask_epoch
                            crosstask_map[crosstask][1] = [] # reset epoch results
                            callback.epoch_end()
                            if x == 'end_task':
                                callback.results = crosstask_results
                                callback.task_end()
                                crosstask_map[crosstask][-1] = True
                # ====== end ====== #
        # ====== end main task ====== #
        self._task.stop_all()
        for t in self._subtask.keys():
            t.stop_all()
        for t in self._crosstask.keys():
            t.stop_all()
