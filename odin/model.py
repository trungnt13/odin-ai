from __future__ import print_function, division, absolute_import

import os
import inspect
import warnings
import tempfile
import cPickle
from itertools import chain
from numbers import Number
from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import (BaseEstimator, ClassifierMixin,
                          TransformerMixin, RegressorMixin)

from odin import backend as K
from odin.roles import add_role, has_roles, TRAINING, DEPLOYING
from odin.nnet import NNOps, Sequence
from odin.utils.decorators import functionable
from odin.training import (MainLoop, ProgressMonitor, History,
                           EarlyStopPatience, Checkpoint, EarlyStop)


class SequentialModel(BaseEstimator, TransformerMixin):
    """ SequentialModel

    Note
    ----
    Default configuration of this estimator is for Classification task,
    which use categorical_crossentropy loss function and categorical_accuracy
    to measure the performance
    """
    _valid_name = 'Validate'
    _train_name = 'Train'
    _test_name = 'Evaluate'

    def __init__(self, *ops):
        super(SequentialModel, self).__init__()
        self._seq_ops = Sequence(ops, strict_transpose=False)
        # list: (name, dtype, shape)
        self._input_info = []
        self._inputs = []

        self._output_info = []
        self._outputs = []

        self._y_train = None
        self._y_pred = None
        self._functions = {}

        # defaults path
        temp_dir = tempfile.gettempdir()
        self._path = os.path.join(temp_dir,
                                  self.__class__.__name__ + str(id(self)))

        # ====== for training ====== #
        self._loss = K.categorical_crossentropy
        self._metric = K.categorical_accuracy

        self._optimizer = K.optimizers.momentum
        self._train_args = {'learning_rate': 0.001, 'momentum': 0.9}

        self._n_epoch = 10
        self._valid_freq = 0.45

        self._batch_size = 128
        self._seed = None

        self._extensions = [
            ProgressMonitor(title='Results: %.2f'),
            History(),
            # 2 epochs until no more improvement
            EarlyStopPatience(2, task=SequentialModel._valid_name,
                              get_value=lambda x: 1 - np.mean(x)),
            Checkpoint(self._path).set_obj(self)
        ]

    def _check_initialized(self):
        if not self.is_initialized:
            raise Exception("This model haven't initialized yet, "
                            'call "set_inputs" to initialize it.')

    # ==================== getter ==================== #
    @property
    def path(self):
        return self._path

    def is_training(self):
        self._check_initialized()
        for i in self._inputs:
            if has_roles(i, TRAINING):
                return True
        return False

    @property
    def is_initialized(self):
        return len(self._inputs) > 0

    @property
    def input_shape(self):
        self._check_initialized()
        shape = [i[-1] for i in self._input_info]
        if len(shape) == 1:
            return shape[0]
        return shape

    @property
    def output_shape(self):
        self._check_initialized()
        if self.is_training:
            return K.get_shape(self._y_train)
        return K.get_shape(self._y_pred)

    def get_function(self, name):
        """ 3 types of functions are supported:
        'pred': for making prediction or transformation
        'train': for training the model in `fit`
        'score': for scoring the performance on test or validation set
        """
        self._check_initialized()
        return self._functions[name]

    def get_params(self, deep=False):
        parameters = {}
        for ops in self._seq_ops.ops:
            name = ops.name
            params = ops.parameters
            if deep:
                params = [K.get_value(i) for i in params]
            parameters[name] = params
        return parameters

    # ==================== setter ==================== #
    def set_params(self, **params):
        # mapping: ops.name -> ops
        ops = {i.name: i for i in self._seq_ops.ops}
        for name, p in params.iteritems():
            if name in ops:
                for param_old, param_new in zip(ops[name].parameters, p):
                    if not isinstance(param_new, np.ndarray):
                        param_new = K.get_value(param_new)
                    K.set_value(param_old, param_new)
        # parameters changed, which mean re-train everything,
        # reset the earlystop's history (dirty hack)
        for i in self._extensions:
            if isinstance(i, EarlyStop):
                i.reset_history()
        return self

    def set_path(self, path):
        self._path = str(path)
        for i in self._extensions:
            if isinstance(i, Checkpoint):
                i.path = self._path
        return self

    def set_training_info(self, loss=None, metric=None, optimizer=None,
                          batch_size=None, seed=None, n_epoch=None,
                          valid_freq=None, extensions=None,
                          **kwargs):
        if loss is not None and hasattr(loss, '__call__'):
            self._loss = loss
        if metric is not None and hasattr(metric, '__call__'):
            self._metric = metric
        if optimizer is not None and hasattr(optimizer, '__call__'):
            self._optimizer = optimizer
        if batch_size is not None:
            self._batch_size = batch_size
        if seed is not None:
            self._seed = seed
        if isinstance(n_epoch, Number):
            self._n_epoch = int(n_epoch)
        if isinstance(valid_freq, Number):
            self._valid_freq = valid_freq
        if extensions is not None:
            if not isinstance(extensions, (tuple, list)):
                extensions = [extensions]
            extensions = [i.set_obj(self) if isinstance(i, Checkpoint) else i
                          for i in extensions]
            self._extensions = extensions
        self._train_args.update(kwargs)
        return self

    def set_inputs(self, *inputs):
        self._input_info = []
        self._inputs = []
        for i in inputs:
            if not K.is_placeholder(i):
                raise ValueError('Only accept input which is placeholder.')
            name, dtype, shape = i.name, i.dtype, K.get_shape(i)
            self._input_info.append([name, dtype, shape])
            self._inputs.append(i)
        # ====== Try to check if the inputs match the Ops ====== #
        try:
            # call this to initialize the parameters and get
            # estimated output shape (we assume training and deploying
            # mode get the same shape).
            for i in self._inputs:
                add_role(i, TRAINING)
            self._y_train = self._seq_ops(*self._inputs)

            for i in self._inputs:
                add_role(i, DEPLOYING)
            self._y_pred = self._seq_ops(*self._inputs)

            # create default output
            if len(self._output_info) == 0:
                shape = K.get_shape(self._y_train)
                self._outputs = [K.placeholder(shape=shape,
                                              dtype=self._y_train.dtype,
                                              name='output1')]
                self._output_info = [('output1', self._y_train.dtype, shape)]

            # reset all functions
            for i, j in self._functions.items():
                del self._functions[i]
                del j
            self._functions = {}
        except Exception, e:
            warnings.warn('Inputs do not match the Ops requirements, '
                          'error: ' + str(e))
            self._input_info = []
            self._inputs = []
        return self

    def set_outputs(self, *outputs):
        self._output_info = []
        self._outputs = []
        for i in outputs:
            if not K.is_placeholder(i):
                raise ValueError('Only accept input which is placeholder.')
            name, dtype, shape = i.name, i.dtype, K.get_shape(i)
            self._output_info.append((name, dtype, shape))
            self._outputs.append(i)
        return self

    # ==================== sklearn methods ==================== #
    def _create_function(self):
        self._check_initialized()
        # ====== prediction function ====== #
        if 'pred' not in self._functions:
            f_pred = K.function(self._inputs, self._y_pred)
            self._functions['pred'] = f_pred
        # ====== training function ====== #
        if 'train' not in self._functions:
            # update optimizer arguments
            _ = inspect.getargspec(self._optimizer)
            optimizer_kwargs = {i: j for i, j in zip(reversed(_.args), reversed(_.defaults))}
            optimizer_kwargs.update(self._train_args)

            # update loss_function arguments
            _ = inspect.getargspec(self._loss)
            if _.defaults is not None:
                loss_kwargs = {i: j for i, j in zip(reversed(_.args), reversed(_.defaults))}
                loss_kwargs.update(self._train_args)
            else:
                loss_kwargs = {}

            # create cost, updates and fucntion
            cost_train = K.mean(
                self._loss(self._y_train, self._outputs[0], **loss_kwargs))
            parameters = self._seq_ops.parameters
            updates = self._optimizer(cost_train, parameters, **optimizer_kwargs)

            f_train = K.function(self._inputs + self._outputs, cost_train,
                                 updates=updates)
            self._functions['train'] = f_train
        # ====== scoring function ====== #
        if 'score' not in self._functions:
            cost_pred = K.mean(self._metric(self._y_pred, self._outputs[0]))
            f_score = K.function(self._inputs + self._outputs, cost_pred)
            self._functions['score'] = f_score

    def fit(self, X, X_valid=None):
        """ This is very standard procedure """
        self._create_function()

        mainloop = MainLoop(batch_size=self._batch_size, seed=self._seed)
        mainloop.set_task(self._functions['train'], data=X,
                          epoch=self._n_epoch,
                          name=SequentialModel._train_name)
        if X_valid is not None:
            mainloop.add_subtask(self._functions['score'], data=X_valid,
                                freq=self._valid_freq,
                                name=SequentialModel._valid_name)
        mainloop.set_callback(*self._extensions)
        mainloop.run()

    def predict(self, *args):
        proba = self.predict_proba(*args)
        return np.argmax(proba, -1)

    def predict_proba(self, *args):
        self._create_function()
        x = self._functions['pred'](*args)
        _min = np.min(x, axis=-1)[:, None]
        _max = np.max(x, axis=-1)[:, None]
        x = (x - _min) / (_max - _min)
        return x / x.sum(-1)[:, None]

    def transform(self, *args):
        self._create_function()
        return self._functions['pred'](*args)

    # ==================== pickling methods ==================== #
    def __repr__(self):
        s = '========  %s  ========\n' % self.__class__.__name__
        s += 'Path: %s\n' % self._path
        s += 'Inputs:\n'
        for name, dtype, shape in self._input_info:
            s += ' - name:%s, dtype:%s, shape:%s\n' % (name, dtype, shape)
        s += 'Outputs:\n'
        for name, dtype, shape in self._output_info:
            s += ' - name:%s, dtype:%s, shape:%s\n' % (name, dtype, shape)
        s += 'Loss: %s\n' % str(self._loss.__name__)
        s += 'Optimizer: %s\n' % str(self._optimizer.__name__)
        s += 'Metric: %s\n' % str(self._metric.__name__)
        s += 'TrainArgs: \n'
        for i, j in self._train_args.iteritems():
            s += ' - %s: %s\n' % (i, str(j))
        s += '#Epoch: %d, ValidFreq: %.2f, BatchSize: %d, Seed:%s\n' % \
        (self._n_epoch, self._valid_freq, self._batch_size, str(self._seed))
        s += 'Extensions: %s' % ', '.join([str(i.__class__.__name__)
                                          for i in self._extensions])
        return s

    def __setstate__(self, states):
        (self._seq_ops,
         self._input_info, # list: (name, dtype, shape)
         self._output_info,
         self._path,
         self._loss,
         self._optimizer,
         self._train_args,
         self._metric,
         self._n_epoch,
         self._valid_freq,
         self._batch_size,
         self._seed,
         self._extensions) = states
        # ====== create some default values ====== #
        self._y_train = None
        self._y_pred = None
        self._functions = {}
        # ====== initialize ====== #
        self.set_inputs(*[K.placeholder(shape=shape, dtype=dtype, name=name)
                          for name, dtype, shape in self._input_info])
        self.set_outputs(*[K.placeholder(shape=shape, dtype=dtype, name=name)
                           for name, dtype, shape in self._output_info])
        # ====== set obj for Checkpoint ====== #
        for i in self._extensions:
            if isinstance(i, Checkpoint):
                i.set_obj(self)

    def __getstate__(self):
        self._check_initialized()
        return (
            self._seq_ops,
            self._input_info, # list: (name, dtype, shape)
            self._output_info,
            self._path,
            self._loss,
            self._optimizer,
            self._train_args,
            self._metric,
            self._n_epoch,
            self._valid_freq,
            self._batch_size,
            self._seed,
            self._extensions
        )


class SequentialClassifier(SequentialModel, ClassifierMixin):

    pass


class SequentialRegressor(SequentialModel, RegressorMixin):

    def __init__(self, *ops):
        super(SequentialRegressor, self).__init__(*ops)
        self._loss = K.squared_error
        self._metric = K.squared_error

    def predict(self, *args):
        return self.transform(*args)
