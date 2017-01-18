from __future__ import print_function, division, absolute_import

import os
import inspect
import warnings
import tempfile
from six.moves import cPickle
from itertools import chain
from numbers import Number
from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import (BaseEstimator, ClassifierMixin,
                          TransformerMixin, RegressorMixin)

from odin import backend as K
from odin.basic import add_role, has_roles
from odin.nnet import Sequence
from odin.fuel import Data, speech_features_extraction
from odin.training import (MainLoop, ProgressMonitor, History,
                           EarlyStopGeneralizationLoss)
from odin.utils import Progbar
from odin.utils.decorators import autoinit, functionable
from odin.preprocessing import speech


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
        # info: (name, dtype, shape)
        self._input_info = []
        self._inputs = []

        # info: (name, dtype, shape)
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
            ProgressMonitor(name=SequentialModel._train_name,
                            format='Results: %.2f'),
            ProgressMonitor(name=SequentialModel._valid_name,
                            format='Results: %.2f'),
            History(),
            # 2 epochs until no more improvement
            EarlyStopGeneralizationLoss(
                name=SequentialModel._valid_name,
                threshold=5, patience=2,
                get_value=lambda x: 1 - np.mean(x)),
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
        """
        Parameters
        ----------
        deep: boolean
            if True, return the numpy array (i.e. the real values of
            each parameters)
        """
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
        return self

    def set_path(self, path):
        self._path = str(path)
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
    def _auto_create_inputs(self, X):
        if len(self._inputs) > 0:
            return
        if not isinstance(X, (tuple, list)):
            X = (X,)
        X = [K.placeholder(shape=(None,) + i.shape[1:],
                           dtype=i.dtype,
                           name='input%d' % _)
             for _, i in enumerate(X)]
        self.set_inputs(*X)

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

    def fit(self, X, y=None, X_valid=None, y_valid=None):
        """ This is very standard procedure """
        # we assume always only 1 outputs, hence,
        # the last variable must be output
        self._auto_create_inputs(X)
        self._create_function()

        mainloop = MainLoop(batch_size=self._batch_size, seed=self._seed)
        mainloop.set_save(self._path, self)
        mainloop.set_callback(self._extensions)
        # training task
        data = [X, y] if y is not None else [X]
        mainloop.set_task(self._functions['train'], data=data,
                          epoch=self._n_epoch,
                          name=SequentialModel._train_name)
        # validation task
        if X_valid is not None:
            data = [X_valid, y_valid] if y_valid is not None else [X_valid]
            mainloop.set_subtask(self._functions['score'], data=data,
                                 freq=self._valid_freq,
                                 name=SequentialModel._valid_name)
        # run the training process
        mainloop.run()

    def predict(self, *args):
        """
        Return
        ------
        Raw prediction which is the output directly from the network

        """
        self._auto_create_inputs(args)
        self._check_initialized()

        proba = self.predict_proba(*args)
        return np.argmax(proba, -1)

    def predict_proba(self, *args):
        self._auto_create_inputs(args)
        self._create_function()

        n = 0
        nb_samples = args[0].shape[0]
        batch_size = self._batch_size
        prediction = []
        prog = Progbar(target=nb_samples, title='Predicting')
        while n < nb_samples:
            end = min(n + batch_size, nb_samples)
            x = [i[n:end] for i in args]
            x = self._functions['pred'](*x)
            _min = np.min(x, axis=-1)[:, None]
            _max = np.max(x, axis=-1)[:, None]
            x = (x - _min) / (_max - _min)
            x = x / x.sum(-1)[:, None]
            prediction.append(x)
            n = end
            prog.update(n)

        return np.concatenate(prediction, axis=0)

    def transform(self, *args):
        self._auto_create_inputs(args)
        self._create_function()

        n = 0
        nb_samples = args[0].shape[0]
        batch_size = self._batch_size
        prediction = []
        while n < nb_samples:
            end = min(n + batch_size, nb_samples)
            x = [i[n:end] for i in args]
            x = self._functions['pred'](*x)
            prediction.append(x)
            n = end

        return np.concatenate(prediction, axis=0)

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

    def __getstate__(self):
        self._check_initialized()
        return (
            self._seq_ops, # main NNOps
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

    def __init__(self, *ops):
        super(SequentialClassifier, self).__init__(*ops)


class SequentialRegressor(SequentialModel, RegressorMixin):

    def __init__(self, *ops):
        super(SequentialRegressor, self).__init__(*ops)
        self._loss = K.squared_error
        self._metric = K.squared_error

        self._extensions = [
            ProgressMonitor(name=SequentialModel._train_name,
                            format='Results: %.2f'),
            ProgressMonitor(name=SequentialModel._valid_name,
                            format='Results: %.2f'),
            History(),
            # 2 epochs until no more improvement
            EarlyStopGeneralizationLoss(
                name=SequentialModel._valid_name,
                threshold=5, patience=2,
                get_value=lambda x: np.mean(x)),
        ]

    def predict(self, *args):
        return self.transform(*args)


# ===========================================================================
# Transformer
# ===========================================================================
class SpeechTransform(BaseEstimator, TransformerMixin):
    """ SpeechTransform
    Parameters
    ----------
    feature_type: "mspec", "spec", "mfcc", "pitch"

    Note
    ----
    if your want to set the default fs (sample frequency), set
    SpeechTransform.DEFAULT_FS
    """

    DEFAULT_FS = None

    @autoinit
    def __init__(self, feature_type, fs=8000, win=0.025, shift=0.01,
                 n_filters=40, n_ceps=13, delta_order=2,
                 energy=True, vad=True, downsample='sinc_best',
                 pitch_threshold=0):
        super(SpeechTransform, self).__init__()
        if not isinstance(feature_type, (tuple, list)):
            feature_type = [feature_type, ]
        if any(i not in ['mspec', 'spec', 'mfcc', 'pitch']
               for i in feature_type):
            raise ValueError('Feature type must be "mspec", "spec", '
                             '"mfcc", or "pitch".')
        self.feature_type = feature_type

    def fit(*args):
        pass

    def transform(self, X, start=0., end=-1, channel=0):
        """
        fs : int
            original sample frequency of data (in case reading pcm file
            we don't know the original sample frequency)
        """
        get_mspec = False
        get_spec = False
        get_mfcc = False
        get_pitch = False
        if 'mspec' in self.feature_type:
            get_mspec = True
        if 'spec' in self.feature_type:
            get_spec = True
        if 'mfcc' in self.feature_type:
            get_mfcc = True
        if 'pitch' in self.feature_type:
            get_pitch = True
        # ====== Read audio ====== #
        if isinstance(X, str) and os.path.exists(X):
            X, orig_fs = speech.read(X)
        elif isinstance(X, np.ndarray):
            orig_fs = None
        elif isinstance(X, Data):
            X = X[:]
            orig_fs = None
        else:
            raise ValueError('Cannot process data type %s' % type(X))
        # if specifed DEFAULT_FS
        if orig_fs is None:
            orig_fs = (SpeechTransform.DEFAULT_FS
                       if SpeechTransform.DEFAULT_FS is not None else self.fs)
        # ====== check if downsample necessary ====== #
        if self.fs < orig_fs: # downsample
            from scikits.samplerate import resample
            X = resample(X, self.fs / orig_fs, 'sinc_best')
        elif self.fs > orig_fs:
            raise ValueError('Cannot perform upsample from frequency: '
                             '{}Hz to {}Hz'.format(orig_fs, self.fs))
        fs = orig_fs if self.fs is None else self.fs
        # ====== preprocessing ====== #
        N = len(X)
        start = int(float(start) * fs)
        end = int(N if end < 0 else end * fs)
        X = X[start:end, channel] if X.ndim > 1 else X[start:end]
        data = speech_features_extraction(X.ravel(), fs=fs,
            n_filters=self.n_filters, n_ceps=self.n_ceps,
            win=self.win, shift=self.shift, delta_order=self.delta_order,
            energy=self.energy, vad=self.vad, dtype='float32',
            pitch_threshold=self.pitch_threshold, get_pitch=get_pitch,
            get_spec=get_spec, get_mspec=get_mspec, get_mfcc=get_mfcc)
        # ====== return results ====== #
        if data is None:
            return None
        results = {}
        if get_spec:
            X, sum1, sum2 = data[0]
            results['spec'] = X
        if get_mspec:
            X, sum1, sum2 = data[1]
            results['mspec'] = X
        if get_mfcc:
            X, sum1, sum2 = data[2]
            results['mfcc'] = X
        if get_pitch:
            X, sum1, sum2 = data[3]
            results['pitch'] = X
        results = [results[i] for i in self.feature_type]
        if len(results) == 1: results = results[0]
        if self.vad:
            return (results + [data[-1]]
                    if isinstance(results, (tuple, list))
                    else [results, data[-1]])
        return results

    def __setstate__(self, states):
        (SpeechTransform.DEFAULT_FS,
         self.feature_type,
         self.fs,
         self.win,
         self.shift,
         self.n_filters,
         self.n_ceps,
         self.delta_order,
         self.energy,
         self.vad,
         self.pitch_threshold,
         self.downsample) = states

    def __getstate__(self):
        return (
            SpeechTransform.DEFAULT_FS,
            self.feature_type,
            self.fs,
            self.win,
            self.shift,
            self.n_filters,
            self.n_ceps,
            self.delta_order,
            self.energy,
            self.vad,
            self.pitch_threshold,
            self.downsample
        )


class Transform(BaseEstimator, TransformerMixin):
    """ Transform """

    def __init__(self, func, *args, **kwarg):
        super(Transform, self).__init__()
        self._func = functionable(func, *args, **kwarg)

    def fit(*args):
        pass

    def transform(self, X):
        return self._func(X)

    def __getstate__(self):
        return self._func

    def __setstate__(self, states):
        self._func = states
