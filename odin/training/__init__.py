from __future__ import division, absolute_import, print_function

import os
from itertools import chain
from collections import defaultdict
from six.moves import range, zip, cPickle

import numpy as np

from odin import (SIG_TRAIN_ROLLBACK, SIG_TRAIN_SAVE, SIG_TRAIN_STOP)
from odin.config import get_rng
from odin import fuel
from odin.fuel import Dataset, as_data
from odin.utils import struct, as_tuple, is_number, Progbar

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


def standard_trainer(train_data, valid_data,
                     X, y_train, y_score, y_target, parameters,
                     test_data=None,
                     cost_train=None, cost_score=None, cost_regu=0,
                     optimizer=None, confusion_matrix=False, gradient_norm=True,
                     batch_size=64, nb_epoch=3, valid_freq=1.,
                     seed=1208, shuffle_level=2, patience=3, earlystop=5,
                     save_path=None, save_obj=None, report_path=None,
                     enable_rollback=True, stop_callback=None, save_callback=None):
    """
    Parameters
    ----------
    cost_train: list of callable, or TensorVariable
        if a function is given, each function will be apply to pair of
        `y_train` and `y_target` (i.e. `zip(y_train, y_target)`)
    cost_score: list of callable, or TensorVariable
        if a function is given, each function will be apply to pair of
        `y_score` and `y_target` (i.e. `zip(y_score, y_target)`)
    cost_regu: list of TensorVariable
        list of all additional cost (for regularization) to add to `cost_train`.
    confusion_matrix: int, list or tuple
        If int is given, it is the number of different classes.
        If a list or tuple is given, it contains all the labels.
    gradient_norm: bool
        if True, record the L2-norm of gradients from all parameters for
        each iteration.

    Return
    ------
    MainLoop, and History

    Note
    ----

    """
    from odin import backend as K
    # ====== prepare variables and cost ====== #
    # check optimizer
    if optimizer is None:
        print("[WARNING] No optimizer is given, use default SGD with "
              "Nesterov momentum (lr=0.00001, momentum=0.9).")
        optimizer = K.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
    elif not isinstance(optimizer, K.optimizers.Optimizer) and \
    not hasattr(optimizer, "get_updates"):
        raise ValueError("Invalid optimizer, the optimizer must be instance of "
                         "backend.optimizers.Optimizer or having function "
                         "get_updates(self, loss_or_grads, params).")
    #  check the cost functions
    if cost_train is None:
        cost_train = K.categorical_crossentropy
    if cost_score is None:
        cost_score = K.categorical_crossentropy
    cost_train = as_tuple(cost_train)
    cost_train_name = [i.name if K.is_variable(i) else i.__name__
                       for i in cost_train]
    cost_score = as_tuple(cost_score)
    cost_score_name = [i.name if K.is_variable(i) else i.__name__
                       for i in cost_score]
    cost_regu = as_tuple(cost_regu)
    # check input X, y, parameters
    X = as_tuple(X)
    y_train = as_tuple(y_train)
    y_score = as_tuple(y_score)
    y_target = as_tuple(y_target)
    parameters = as_tuple(parameters)
    if len(X) == 0 or len(y_train) == 0 or len(y_score) == 0 or \
    len(y_target) == 0 or len(parameters) == 0:
        raise ValueError("X(len=%d), y_train(len=%d), y_score(len=%d), y_target(len=%d),"
                         "and parameters(len=%d) must be list or tuple with length > 0."
                         % (len(X), len(y_train), len(y_score), len(y_target),
                            len(parameters)))
    # get all cost
    cost_train = [f_cost if K.is_variable(f_cost) else K.mean(f_cost(y_, y))
                  for y_, y in zip(y_train, y_target)
                  for f_cost in cost_train]
    cost_score = [f_cost if K.is_variable(f_cost) else K.mean(f_cost(y_, y))
                  for y_, y in zip(y_score, y_target)
                  for f_cost in cost_score]
    # add confusion matrix
    if confusion_matrix:
        if not is_number(confusion_matrix) and \
        not isinstance(confusion_matrix, (tuple, list, np.ndarray)):
            raise ValueError("confusion_matrix must be an integer, or list, tuple"
                             " specifies number of classes, or list of all classes.")
        labels = confusion_matrix
        if is_number(confusion_matrix):
            confusion_matrix = list(range(int(confusion_matrix)))
            labels = confusion_matrix
        elif not is_number(confusion_matrix[0]): # given list of label
            labels = list(range(len(confusion_matrix)))
        if len(labels) == 1:
            raise ValueError("you have to specify the number of labels in 'confusion_matrix'")
        for y_, y in zip(y_score, y_target):
            cost_score.append(K.confusion_matrix(y_pred=y_, y_true=y, labels=labels))
    # get the updates
    training_cost = cost_train[0] + sum(c for c in cost_regu)
    updates = optimizer.get_updates(training_cost, parameters)
    # ====== add gradient norm ====== #
    grad_norm = [] if not gradient_norm or not hasattr(optimizer, 'norm') else \
        [optimizer.norm]
    if len(grad_norm) > 0:
        cost_train_name.append('gradient_norm')
    cost_train = [training_cost] + cost_train[1:] + grad_norm
    # ====== create function ====== #
    print('Building training functions ...')
    f_train = K.function(inputs=X + y_target, outputs=cost_train, updates=updates)
    print('Building scoring functions ...')
    f_score = K.function(inputs=X + y_target, outputs=cost_score)

    # ====== evaluation ====== #
    def evaluation():
        if test_data is not None:
            test = as_data(test_data)
            test.set_batch(batch_size=batch_size, seed=None)
            prog = Progbar(target=len(test), title="Evaluating:")
            _ = []
            for t in test:
                if not isinstance(t, (tuple, list)):
                    t = (t,)
                _.append(f_score(*t))
                prog.add(len(t[0]))
            test = _
            # just 1 result returned
            if not isinstance(test[0], (tuple, list)):
                test = np.mean(test)
            elif confusion_matrix:
                test_cm = sum(i[-1] for i in test)
                test = [np.mean([j[i] for j in test])
                        for i in range(len(test[0]) - 1)]
                test.append(test_cm)
            else:
                test_cm = None
                test = [np.mean([j[i] for j in test])
                        for i in range(len(test[0]))]
            # record the result to history
            history.record('test', 'epoch_end', nb_iter=1, nb_epoch=1,
                nb_samples=test_data.shape[0], results=test)
            print("[Evaluaton] scores:", test[:-1] if confusion_matrix else test)
            if confusion_matrix:
                print("[Evaluaton] confusion matrix:")
                print(test[-1])
        # ====== create report ====== #
        if report_path is None:
            return
        from odin import visual
        from matplotlib import pyplot as plt
        # get train results
        train_epochs = history.get_epoch('train')
        train_results = defaultdict(list)
        for i, name in enumerate(cost_train_name):
            for epoch in train_epochs:
                train_results[name].append([r[i] for r in epoch])
        # get valid results
        valid_epochs = history.get_epoch('valid')
        valid_results = defaultdict(list)
        for i, name in enumerate(cost_score_name):
            for epoch in valid_epochs:
                valid_results[name].append([r[i] for r in epoch])
        # visualize the trianing process
        plt.figure()
        legends = []
        plotted_valid = []
        for name, x in train_results.iteritems():
            x = np.array([np.mean(i) for i in x]).ravel()
            x = (x - x.min()) / (x.max() - x.min())
            nb_train_epoch = len(x)
            legends.append(
                (plt.plot(range(1, nb_train_epoch + 1), x, '-', linewidth=1.2)[0],
                 "[train]" + name))
            recent_color = legends[-1][0].get_color()
            if name in valid_results:
                y = np.array([np.mean(i) for i in valid_results[name]]).ravel()
                y = (y - y.min()) / (y.max() - y.min())
                x = np.linspace(1, nb_train_epoch, num=len(y))
                legends.append(
                    (plt.plot(x, y, '--', linewidth=1.5, color=recent_color)[0],
                     "[valid]" + name))
                plotted_valid.append(name)
        for name, y in valid_results.iteritems(): # plot the remain valid
            if name not in plotted_valid:
                y = np.array([np.mean(i) for i in y]).ravel()
                y = (y - y.min()) / (y.max() - y.min())
                x = np.linspace(1, nb_train_epoch, num=len(y))
                legends.append(
                    (plt.plot(x, y, '--', linewidth=1.5)[0],
                     "[valid]" + name))
        plt.ylim([-0.05, 1.2])
        plt.xlabel("Epoch"); plt.ylabel("Normalized cost")
        plt.legend([i[0] for i in legends], [i[1] for i in legends],
                   loc='upper right', ncol=2, fontsize=8)
        # visualize each training epoch
        for name, X in train_results.iteritems():
            _plot_each_epoch(name, X, "[Train]")
        for name, X in valid_results.iteritems():
            _plot_each_epoch(name, X, "[Valid]")
        # visualize the confusion matrix
        if confusion_matrix:
            # First the validation confusion matrix
            confusion = [sum(i[-1] for i in epoch)
                         for epoch in valid_epochs]
            labels = [str(i) for i in confusion_matrix]
            ncol = 3; nrow = int(np.ceil(len(confusion) / ncol))
            visual.plot_figure(nrow=nrow, ncol=ncol, dpi=180)
            for i, cm in enumerate(confusion):
                ax = plt.subplot(nrow, ncol, i + 1)
                visual.plot_confusion_matrix(cm, labels,
                    axis=ax, fontsize=10, colorbar=False)
                ax.set_xlabel('[Epoch%d]Prediction' % (i + 1), fontsize=10)
            plt.suptitle("[Valid] Confustion matrices", fontsize=12)
            plt.tight_layout()
            # The the test confusion matrix
            if test_data is not None and test_cm is not None:
                plt.figure(figsize=(8, 9), dpi=180)
                visual.plot_confusion_matrix(test_cm, labels,
                    axis=None, fontsize=18, colorbar=False)
                plt.suptitle("[Eval] Confustion matrices", fontsize=20)
                plt.tight_layout()
        # save all the plot
        visual.plot_save(path=report_path, dpi=180, clear_all=True)
        if save_path is not None:
            print("Best checkpoint saved at:", save_path)
    # ====== Create trainer ====== #
    task = MainLoop(batch_size=batch_size, seed=seed, shuffle_level=shuffle_level,
        rollback=enable_rollback)
    if save_path is not None and save_obj is not None:
        task.set_save(save_path, save_obj, save_hist=True)
    # set task
    task.set_task(f_train, train_data, epoch=nb_epoch, name='train')
    task.set_subtask(f_score, valid_data, freq=valid_freq, name='valid')
    task.set_signal_handlers(end=evaluation)
    # format for score
    score_format = 'Results:' + \
        __format_string(len(cost_score) - (len(y_score) if confusion_matrix else 0))
    score_tracking = {(len(cost_score) - i - 1): (lambda x: sum(x))
        for i in range(len(y_score))} if confusion_matrix else []
    # set the callback
    history = History()
    task.set_callback([
        ProgressMonitor(name='train',
            format='Results:' + __format_string(len(cost_train))),
        ProgressMonitor(name='valid', format=score_format, tracking=score_tracking),
        history,
        EarlyStopGeneralizationLoss('valid', threshold=earlystop, patience=patience,
                 get_value=lambda x: np.mean([i[0] for i in x]
                                             if isinstance(x[0], (tuple, list))
                                             else x),
                 stop_callback=stop_callback, save_callback=save_callback
        ) if earlystop is not None else None,
        NaNDetector(('train', 'valid'), patience=patience, rollback=True)
    ])
    return task, history


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
        data = [fuel.as_data(i) for i in data]

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
                self._rng.rand = get_rng().rand
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
                        if hasattr(i, 'stop'):
                            i.stop()
                        else: # just iterate all over
                            for _ in i: pass
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
    rollback: bool
        if True, rollback to the best checkpoint whenever the validation
        performance is degraded.

    """

    def __init__(self, batch_size=256, seed=-1, shuffle_level=0, rollback=True):
        super(MainLoop, self).__init__()
        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability
        self._allow_rollback = bool(rollback)

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

        self._save_func = None
        self._rollback_func = None
        self._end_func = None

    # ==================== pickling ==================== #
    def __setstate__(self, value):
        self.set_batch(batch_size=value[0], shuffle_level=value[2])
        self._rng = value[1]

        self._callback = value[3]
        self._allow_rollback = value[4]

        self._task = None
        self._subtask = {} # run 1 epoch after given frequence
        self._crosstask = {} # randomly run 1 iter given probability

    def __getstate__(self):
        return (self._batch_size, self._rng, self._shuffle_level,
                self._callback, self._allow_rollback)

    # ==================== Signal handling ==================== #
    def set_signal_handlers(self, save=None, rollback=None, end=None):
        """
        Parameters
        ----------
        save: callable
            a function will be execeuted when saving checkpoint.
        rollback: callable
            a function will be called when rollback the saved object
        end: callable
            a function will be called when finish running the `MainLoop`
        """
        self._save_func = save if callable(save) else None
        self._rollback_func = rollback if callable(rollback) else None
        self._end_func = end if callable(end) else None

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
        # ====== save the first checkpoint ====== #
        cPickle.dump(self._save_obj, open(self._save_path, 'w'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
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
            self._callback = CallbackList(*[c for c in callback if c is not None])
        return self

    def __getitem__(self, key):
        """ Return callback from callback list"""
        return self._callback[key]

    # ==================== main ==================== #
    def _validate_data(self, data):
        if not isinstance(data, (list, tuple)):
            data = [data]
        return [fuel.as_data(i) for i in data]

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
        if self._save_func is not None:
            self._save_func()
            return
        # default save procedure
        if self._save_path is not None and self._save_obj is not None:
            cPickle.dump(self._save_obj, open(self._save_path, 'wb'),
                         protocol=cPickle.HIGHEST_PROTOCOL)
            # ====== save history if possible ====== #
            if self._save_hist is not None and 'History' in self._callback:
                cPickle.dump(self._callback['History'], open(self._save_hist, 'w'),
                             protocol=cPickle.HIGHEST_PROTOCOL)
            print("\nCreated checkpoint at path:", self._save_path)

    def _rollback(self):
        if self._rollback_func is not None:
            self._rollback_func()
            return
        # default rollback procedure
        if self._save_path is not None and os.path.exists(self._save_path):
            f = open(self._save_path, 'rb')
            # the loading process will automatically reload shared variable
            cPickle.load(f)
            f.close()

    def _end(self):
        self._rollback()
        if self._end_func is not None:
            self._end_func()

    def run(self):
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
                            if SIG_TRAIN_ROLLBACK in msg and self._allow_rollback:
                                self._rollback()
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
            if SIG_TRAIN_ROLLBACK in msg and self._allow_rollback:
                self._rollback()
            if SIG_TRAIN_STOP in msg: break
        # ====== end main task ====== #
        self._task.stop_all()
        for t in self._subtask.keys():
            t.stop_all()
        for t in self._crosstask.keys():
            t.stop_all()
        # everything finished
        self._end()
