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


def standard_trainer(train_data, valid_data, test_data=None,
                     cost_train='auto', cost_score='auto', cost_regu='auto',
                     parameters='auto', optimizer='auto',
                     confusion_matrix=None, gradient_norm=True,
                     batch_size=64, nb_epoch=3, valid_freq=1.,
                     seed=1208, shuffle_level=2, patience=3, earlystop=5,
                     save_path=None, save_obj=None, report_path=None,
                     enable_rollback=True, stop_callback=None, save_callback=None,
                     labels=None):
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
    patience: int
        number of failures detected by earlystopping before terminating the
        training.
    earlystop: int or None
        if None, turn-off early-stopping. Otherwise, it is percentage of
        generalization loss.

    Return
    ------
    MainLoop, and History

    Note
    ----

    """
    # ====== prepare variables and cost ====== #
    # check optimizer
    if optimizer is None or optimizer == 'auto':
        print("[WARNING] No optimizer is given, use default Adam optimizer.")
        optimizer = K.optimizers.Adam()
    elif not isinstance(optimizer, K.optimizers.Optimizer) and \
    not hasattr(optimizer, "get_updates"):
        raise ValueError("Invalid optimizer, the optimizer must be instance of "
                         "backend.optimizers.Optimizer or having function "
                         "get_updates(self, loss_or_grads, params).")
    #  check the cost train
    if cost_train == 'auto':
        cost_train = K.ComputationGraph().get_roles(K.role.TrainingCost)
    cost_train = as_tuple(cost_train) if cost_train is not None else tuple()
    if len(cost_train) == 0:
        raise ValueError("You must specify cost_train.")
    cost_train = [cost for cost in cost_train if K.is_tensor(cost)]
    cost_train_name = [i.name if K.is_tensor(i) else i.__name__
                       for i in cost_train]
    #  check the cost score
    if cost_score == 'auto':
        graph = K.ComputationGraph()
        cost_score = graph.get_roles(K.role.EarlyStop)
    cost_score = as_tuple(cost_score) if cost_score is not None else tuple()
    cost_score = [cost for cost in cost_score if K.is_tensor(cost)]
    cost_score_name = [i.name if K.is_tensor(i) else i.__name__
                       for i in cost_score]
    #  check the cost regu
    if cost_regu == 'auto':
        cost_regu = K.ComputationGraph().get_roles(K.role.RegularizeLoss)
        if len(cost_regu) == 0:
            cost_regu = None
    cost_regu = as_tuple(cost_regu) if cost_regu is not None else tuple()
    # check parameters
    if parameters == 'auto':
        parameters = K.ComputationGraph().parameters
    parameters = as_tuple(parameters) if parameters is not None else tuple()
    # ====== get the updates ====== #
    training_cost = cost_train[0]
    if len(cost_regu) > 0:
        training_cost += sum(c for c in cost_regu)
    updates = optimizer.get_updates(training_cost, parameters)
    # ====== add gradient norm ====== #
    grad_norm = [] if not gradient_norm or not hasattr(optimizer, 'norm') else \
        [optimizer.norm]
    if len(grad_norm) > 0:
        cost_train_name.append('gradient_norm')
    cost_train = [training_cost] + cost_train[1:] + grad_norm
    # ====== add confusion matrix ====== #
    if confusion_matrix is not None and K.is_tensor(confusion_matrix):
        cost_score.append(confusion_matrix)
        confusion_matrix = True
    else:
        confusion_matrix = False
    # ====== create function ====== #
    print('Building training functions ...')
    train_inputs = K.ComputationGraph(cost_train).placeholders
    f_train = K.function(inputs=train_inputs, outputs=cost_train, updates=updates)
    print('Building scoring functions ...')
    f_score = None
    if valid_data is not None:
        if len(cost_score) == 0:
            print("[WARNING] No scoring cost is specified, using training "
                  "cost for validating!")
            cost_score = cost_train[:-1] if gradient_norm else cost_train
        score_inputs = K.ComputationGraph(cost_score).placeholders
        f_score = K.function(inputs=score_inputs, outputs=cost_score)

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
            confusion_labels = labels
            if labels is None:
                confusion_labels = [str(i) for i in range(confusion[0].shape[0])]
            ncol = 3; nrow = int(np.ceil(len(confusion) / ncol))
            with visual.figure(nrow=nrow, ncol=ncol, dpi=180):
                for i, cm in enumerate(confusion):
                    ax = plt.subplot(nrow, ncol, i + 1)
                    visual.plot_confusion_matrix(cm, confusion_labels,
                        axis=ax, fontsize=10, colorbar=False)
                    ax.set_xlabel('[Epoch%d]Prediction' % (i + 1), fontsize=10)
                plt.suptitle("[Valid] Confustion matrices", fontsize=12)
                plt.tight_layout()
                # The test confusion matrix
                if test_data is not None and test_cm is not None:
                    plt.figure(figsize=(8, 9), dpi=180)
                    visual.plot_confusion_matrix(test_cm, confusion_labels,
                        axis=None, fontsize=18, colorbar=False)
                    plt.suptitle("[Eval] Confustion matrices", fontsize=20)
                    plt.tight_layout()
        # save all the plot
        if report_path == 'show':
            plt.show()
        else:
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
    if f_score is not None:
        task.set_subtask(f_score, valid_data, freq=valid_freq, p=1.,
                         when=0, name='valid')
        # format for score
        score_format = 'Results:' + __format_string(len(cost_score) -
            (1 if confusion_matrix else 0))
        if confusion_matrix:
            score_tracking = {(len(cost_score) - 1): lambda x: sum(x)}
        else:
            score_tracking = []
    task.set_signal_handlers(end=evaluation)
    # set the callback
    history = History()
    task.set_callback([
        history,
        ProgressMonitor(name='train',
            format='Results:' + __format_string(len(cost_train))),
        NaNDetector(('train', 'valid'), patience=patience, rollback=True)
    ] + ([ProgressMonitor(name='valid', format=score_format, tracking=score_tracking),
        EarlyStopGeneralizationLoss('valid', threshold=earlystop, patience=patience,
                 get_value=lambda x: np.mean([i[0] for i in x]
                                             if isinstance(x[0], (tuple, list))
                                             else x),
                 stop_callback=stop_callback, save_callback=save_callback
        ) if earlystop is not None else None,
    ] if f_score is not None else []))
    return task, history


# ===========================================================================
# Tasks
# ===========================================================================
class Task(object):

    def __init__(self, func, data, epoch=1, p=1.0,
                 batch_size=128, seed=None, shuffle_level=2,
                 callbacks=None, name=None):
        super(Task, self).__init__()
        self.set_func(func, data)
        # this Progbar will record the history as well
        self._progbar = Progbar(target=self.nb_samples, name=name,
                                print_report=True, print_summary=True)
        # ====== assign other arguments ====== #
        self._nb_epoch = epoch
        self._p = np.clip(p, 0., 1.)
        self._seed = seed
        self.set_batch(batch_size, seed, shuffle_level)
        self._name = name
        # ====== current info ====== #
        self._curr_epoch = 0
        self._curr_iteration = 0
        self._curr_samples = 0
        self._curr_epoch_iteration = 0
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
        self._curr_iteration = 0
        self._curr_samples = 0
        self._curr_epoch_iteration = 0
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
        if not isinstance(func, K.Function):
            raise ValueError("`func` must be instance of odin.backend.Function")
        self._func = func
        self._output_info = [(o.name, o.get_shape().as_list())
                             for o in self._func.outputs]
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
    def curr_iteration(self):
        """Total number of iteration finished since the beginning of the Task"""
        return self._curr_iteration

    @property
    def curr_samples(self):
        """Total number of samples finished since the beginning of the Task"""
        return self._curr_samples

    @property
    def curr_epoch_iteration(self):
        """Number of iteration within current epoch"""
        return self._curr_epoch_iteration

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
                self._curr_epoch_iteration = 0
                for i, x in enumerate(data):
                    # alread terminated, try to exhausted the iterator
                    # if forced_to_terminate: continue
                    # preprocessed the data
                    if not isinstance(x, (tuple, list)):
                        x = [x]
                    # update some info
                    shape0 = x[0].shape[0]
                    self._curr_samples += shape0
                    self._curr_iteration += 1
                    self._curr_epoch_samples += shape0
                    self._curr_epoch_iteration += 1
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
            self._curr_iteration = 0
            self._curr_samples = 0
            self._curr_epoch_iteration = 0
            self._curr_epoch_samples = 0
            self._callback_msg = []
            # create new iter
            self._created_iter = self.__iter()
            # initialize the iteration
            self._created_iter.next()
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
        task.curr_iteration >= self._iteration:
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
    def set_save(self, path, obj, variables=[]):
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
    def set_train_task(self, func, data, epoch=1, p=1., when=None, name="Train"):
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
                 name=name)
        self._task.append(t)
        self._task_when[t] = when
        self._task_freq[t] = Timer(samples=0)
        # assign main task
        if self._main_task is None:
            self._main_task = t
        return self

    def set_valid_task(self, func, data, freq=Timer(epoch=1),
                       when=Timer(samples=0), name="Valid"):
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
                 seed=None, shuffle_level=0, name=name)
        self._subtask.append(t)
        self._task_when[t] = when
        self._task_freq[t] = freq
        return self

    def set_eval_task(self, func, data, name="Eval"):
        t = Task(func, data, epoch=1, p=1., batch_size=self._batch_size,
                 seed=None, shuffle_level=0, name=name)
        self._subtask.append(t)
        self._task_when[t] = Timer(percentage=1.)
        self._task_freq[t] = Timer(samples=0)
        return self

    # ==================== logic ==================== #
    def _save(self):
        # default save procedure
        if self._save_path is not None and self._save_obj is not None:
            # progbar.add_notification("Creating checkpoint at:" + self._save_path)
            if not os.path.exists(self._save_path):
                os.mkdir(self._save_path)
            elif os.path.isfile(self._save_path):
                raise ValueError("Save path for the model must be a folder.")
            N.serialize(self._save_obj, self._save_path, save_variables=True,
                        variables=self._save_variables, override=True)

    def _rollback(self):
        if not self._allow_rollback: return
        # default rollback procedure
        if self._save_path is not None and os.path.exists(self._save_path):
            # progbar.add_notification("Rollback from:" + self._save_path)
            N.deserialize(self._save_path)

    def _end(self):
        self._rollback()

    def _run(self):
        if self._main_task is None:
            raise ValueError('You must call set_task and set the main task first.')
        for t in self._task + self._subtask:
            t.set_callbacks(self._callback)
        # ====== prepare subtask ====== #
        finished_task = {i: False for i in self._task + self._subtask}
        task_iter = {i: iter(i) for i in self._task + self._subtask}
        for freq in self._task_freq.itervalues():
            freq.set_counter(self._main_task)
        # ====== main logics ====== #
        while not finished_task[self._main_task]:
            for t in self._task:
                # ====== execute training task first ====== #
                if not finished_task[t]:
                    res = task_iter[t].next()
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
                            finished_task[self._main_task] = True
                            break
        # ====== end main task ====== #
        for t in self._task + self._subtask:
            t.stop()
        # everything finished
        self._end()

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
