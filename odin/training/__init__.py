from __future__ import division, absolute_import

from six.moves import range, zip

import numpy as np

from blocks import RNG_GENERATOR
from blocks import fuel
from blocks.fuel.dataset import Dataset
from blocks.utils import struct

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

    def __init__(self, func, data, epoch, p, batch_size, seed,
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

        self._batch_size = batch_size
        self.set_seed(seed)

        self._preprocess = preprocess if hasattr(preprocess, '__call__') else lambda x: x

        self._iter_per_epoch = int(np.ceil(
            min([len(i) for i in data]) / self._batch_size
        ))
        self._name = name

    @property
    def name(self):
        return str(self._name)

    @property
    def epoch(self):
        return self._epoch

    def set_seed(self, seed):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = struct()
            self._rng.randint = lambda x: None
            self._rng.rand = RNG_GENERATOR.rand
        return self

    @property
    def iter_per_epoch(self):
        ''' Estimated number of iteration for each epoch '''
        return self._iter_per_epoch

    def __iter__(self):
        '''
        Return
        ------
        'start_epoch' : beginning of epoch
        'end_epoch' : epoch ended
        'end_task' : task ended
        (results, n_iter, n_epoch) : results of execute function on data

        Note
        ----
        'end_task' also end of final epoch
        '''
        n_iter = 0
        p = self._p
        _ = 0
        yield 'start_task'
        while _ < self._epoch:
            _ += 1
            seed = self._rng.randint(10e8)
            data = zip(*[iter(i.set_batch(batch_size=self._batch_size, seed=seed))
                         for i in self._data])
            yield 'start_epoch'
            for i, x in enumerate(data):
                x = self._preprocess(x)
                if not isinstance(x, (tuple, list)):
                    x = [x]
                n_iter += 1
                if p >= 1. or (p < 1 and self._rng.rand() < p):
                    results = self._func(*x)
                else:
                    results = None
                yield (results, n_iter, _)
            # end_epoch or task
            if _ >= self._epoch:
                yield 'end_task'
            else:
                yield 'end_epoch'
        # keep ending so no Exception
        while True:
            yield 'end_task'


class MainLoop(object):

    """ MainLoop """

    def __init__(self, batch_size=256, dataset=None, shuffle=True, name=None):
        super(MainLoop, self).__init__()
        self._batch_size = batch_size
        self._name = name
        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability

        if shuffle:
            self._rng = np.random.RandomState(RNG_GENERATOR.randint(10e8))
        else:
            self._rng = struct()
            self._rng.randint = lambda *args, **kwargs: None

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
        self._batch_size = value[0]
        self._name = value[1]
        self._rng = value[2]
        if isinstance(value[3], str):
            self._dataset = Dataset(value[3])
        else:
            self._dataset = None
        self._callback = value[4]

        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability

        self._stop_now = False
        self._save_now = False

    def __getstate__(self):
        dataset = self._dataset.path if self._dataset is not None else None
        return self._batch_size, self._name, self._rng, dataset, self._callback

    # ==================== command ==================== #
    def stop(self):
        self._stop_now = True

    def save(self):
        self._save_now = True

    # ==================== properties ==================== #
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def shuffle(self):
        return isinstance(self._rng, np.random.RandomState)

    @property
    def name(self):
        return self._name

    @property
    def callback(self):
        return self._callback

    def __str__(self):
        return 'Task'

    def add_callback(self, *callback):
        for i in callback:
            if isinstance(i, Callback):
                self._callback.add_callback(i)
        return self

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
                          preprocess=preprocess,
                          name=name)
        return self

    def add_subtask(self, func, data, epoch=float('inf'), p=1., freq=0.5,
                    when=0, preprocess=None, name=None):
        '''
        Parameters
        ----------
        when: float or int
            int => number of main task's iteration before this task is executed
            float => percentage of epoch of main task before this task is executed
            negative value => execute after final epoch of main task
        freq: float or int
            int => number of main task's iteration before this task is executed
            float => percentage of epoch of main task before this task is executed
        preprocess: function
            input is list of data, output is transformed data for batch
        '''
        self._subtask[Task(func, self._validate_data(data), epoch, p,
                           batch_size=self._batch_size,
                           seed=self._rng.randint(10e8),
                           preprocess=preprocess,
                           name=name)] = (freq, when)
        return self

    def add_crosstask(self, func, data, epoch=float('inf'), p=0.5,
                      when=0, preprocess=None, name=None):
        '''
        Parameters
        ----------
        when: float or int
            int => number of main task's iteration before this task is executed
            float => percentage of epoch of main task before this task is executed
            negative value => execute after final epoch of main task
        preprocess: function
            input is list of data, output is transformed data for batch
        '''
        self._crosstask[Task(func, self._validate_data(data), epoch, p,
                             batch_size=self._batch_size,
                             seed=self._rng.randint(10e8),
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
            if self._save_now:
                self._save_now = False
                callback.mode = 'othertask'
                callback.task = _SAVE_TASK
                callback.task_start()
                callback.task_end()
                callback.reset()
            # ====== Main task ====== #
            callback.mode = 'task' # dirty hack
            callback.reset(); callback.task = self._task
            if isinstance(i, str): # start_epoch, end_epoch or end_task
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
            else:
                results, niter, nepoch = i
                epoch_results.append(results)
                task_results.append(results)
                callback.results = results
                callback.iter = niter
                callback.epoch = nepoch
                callback.batch_end()
                # ====== run subtask ====== #
                callback.mode = 'subtask'
                for subtask, (freq, when) in self._subtask.iteritems():
                    subtask_iter, subtask_results, is_end = subtask_map[subtask]
                    if is_end: continue # already ended
                    when = float(when % self._task.epoch) + 1. if when < 0 else when
                    if isinstance(when, float): when = int(when * self._task.iter_per_epoch)
                    if isinstance(freq, float): freq = int(freq * self._task.iter_per_epoch)
                    if niter > 0 and niter >= when and (niter - when) % freq == 0: # OK to run
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
                                    callback.epoch = x[2]
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
                    when = float(when % self._task.epoch) + 1. if when < 0 else when
                    if isinstance(when, float): when = int(when * self._task.iter_per_epoch)
                    crosstask_iter, crosstask_epoch, crosstask_results, is_end = crosstask_map[crosstask]
                    if niter > 0 and niter >= when and not is_end: # OK to run
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
                            callback.epoch = x[2]
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
        # ====== end loop ====== #
