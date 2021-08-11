import glob
import inspect
import os
import pickle
import tempfile
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union, \
  TextIO, Sequence
import datetime

import numpy as np
from scipy import signal
import tensorflow as tf
from six import string_types
from tensorflow import Tensor
from tensorflow.python import keras
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.data.ops.iterator_ops import OwnedIterator
from tensorflow.python.eager.def_function import Function as TFFunction
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm import tqdm

__all__ = ['Trainer', 'get_current_trainer']

# ===========================================================================
# Helpers
# ===========================================================================
Callback = Callable[[], Optional[Dict[str, Any]]]

_BEST_WEIGHTS = {}
_BEST_OPTIMIZER = {}
_CHECKPOINT_MANAGER = {}
_CURRENT_TRAINER = None


def _ema(x, w):
  """`s[0] = x[0]` and `s[t] = w * x[t] + (1-w) * s[t-1]`"""
  b = [w]
  a = [1, w - 1]
  zi = signal.lfilter_zi(b, a)
  return signal.lfilter(b, a, x, zi=zi * x[0])[0]


def _is_text(x):
  return isinstance(x, string_types) or \
         (isinstance(x, np.ndarray) and isinstance(x[0], string_types))


def _save_summary(loss, metrics, prefix="", flush=False):
  tf.summary.scalar(f"{prefix}loss", loss)
  for k, v in metrics.items():
    k = f"{prefix}{k}"
    if isinstance(v, (tuple, list)):
      v = tf.convert_to_tensor(v)
    # text
    if _is_text(v):
      tf.summary.text(k, v)
    # scalar
    elif tf.less(tf.rank(v), 1):
      tf.summary.scalar(k, v)
    # images
    elif v.dtype == tf.uint8:
      tf.summary.image(k, v)
    # histogram
    else:
      tf.summary.histogram(k, v)
  if flush:
    tf.summary.flush()


def _print_summary(print_fn: Callable,
                   log_tag: str,
                   loss: Union[Tensor, float],
                   metrics: dict,
                   n_iter: int,
                   is_valid: bool = False):
  log_tag = ("" if len(log_tag) == 0 else f"{log_tag} ")
  if is_valid:
    log_tag = f" * [VALID]{log_tag}"
  print_fn(f"{log_tag} #{int(n_iter)} loss:{loss:.4f}")
  for k, v in metrics.items():
    if _is_text(v):
      v = str(v)
    elif hasattr(v, 'shape'):
      v = f"{tf.reduce_mean(v):.4f}"
    else:
      v = f"{v:.4f}"
    print_fn(f"{len(log_tag) * ' '} {k}:{v}")


def _process_callback_returns(print_fn: Callable,
                              log_tag: str,
                              callback_name: str,
                              n_iter: int,
                              metrics: Optional[dict] = None):
  if metrics is not None and isinstance(metrics, dict) and len(metrics) > 0:
    print_fn(f"{log_tag} [{callback_name}#{int(n_iter)}]:")
    for k, v in metrics.items():
      if isinstance(v, (tuple, list)):
        v = tf.convert_to_tensor(v)
      # text
      if _is_text(v):
        tf.summary.text(k, v)
      # scalar
      elif tf.less(tf.rank(v), 1):
        tf.summary.scalar(k, v)
      # images
      elif v.dtype == tf.uint8:
        tf.summary.image(k, v)
        v = f'image{v.shape}'
      # histogram
      else:
        tf.summary.histogram(k, v)
      print_fn(f" {k}:{v}")


def _create_callback(callbacks):
  if not isinstance(callbacks, (tuple, list)):
    callbacks = [callbacks]

  def _callback():
    results = {}
    for fn in callbacks:
      r = fn()
      if isinstance(r, dict):
        results.update(r)
    return None if len(results) == 0 else results

  return _callback


def read_tensorboard(logdir: str) -> Dict[Text, Tuple[float, int, float]]:
  r""" Read Tensorboard event files from a `logdir`

  Return:
    a dictionary mapping from `tag` (string) to list of tuple
    `(wall_time, step, value)`
  """
  all_log = defaultdict(list)
  for f in sorted(glob.glob(f"{logdir}/event*"),
                  key=lambda x: int(os.path.basename(x).split('.')[3])):
    for event in summary_iterator(f):
      t = event.wall_time
      step = event.step
      summary = event.summary
      for value in event.summary.value:
        tag = value.tag
        meta = value.metadata
        dtype = meta.plugin_data.plugin_name
        data = tf.make_ndarray(value.tensor)
        # scalars values
        if dtype == "scalars":
          pass
        # text values
        elif dtype == "text":
          if len(value.tensor.tensor_shape.dim) == 0:
            data = str(data.tolist(), 'utf-8')
          else:
            data = np.array([str(i, 'utf-8') for i in data])
        # image
        elif dtype == "images":
          data = data  # byte string
        # histogram
        elif dtype == "histograms":
          data = data  # array
        else:
          raise NotImplementedError(f"Unknown data type: {dtype}-{data}")
        all_log[tag].append((t, step, data))
  all_log = {i: sorted(j, key=lambda x: x[1]) for i, j in all_log.items()}
  return all_log


# ===========================================================================
# Main
# ===========================================================================
class Trainer(object):
  """Simple training procedure"""

  @staticmethod
  def save_weights(models, optimizers=None):
    r""" Fast store the weights of models in-memory, only the lastest
    version of the weights are kept """
    for i in tf.nest.flatten(models):
      if not isinstance(i, keras.layers.Layer):
        continue
      _BEST_WEIGHTS[id(i)] = i.get_weights()
    if optimizers is not None:
      for i in tf.nest.flatten(optimizers):
        if isinstance(i, tf.optimizers.Optimizer):
          _BEST_OPTIMIZER[id(i)] = i.get_weights()

  @staticmethod
  def restore_weights(models, optimizers=None):
    r""" Fast recover the weights of models, only the lastest version
    of the weights are kept """
    for i in tf.nest.flatten(models):
      if not isinstance(i, keras.layers.Layer):
        continue
      if id(i) in _BEST_WEIGHTS:
        i.set_weights(_BEST_WEIGHTS[id(i)])
    if optimizers is not None:
      for i in tf.nest.flatten(optimizers):
        if isinstance(i, tf.optimizers.Optimizer):
          if id(i) in _BEST_OPTIMIZER:
            i.set_weights(_BEST_OPTIMIZER[id(i)])

  @staticmethod
  def save_checkpoint(dir_path,
                      models,
                      optimizers=None,
                      trainer=None,
                      max_to_keep=5):
    r""" Save checkpoint """
    if optimizers is None:
      optimizers = []
    optimizers = tf.nest.flatten(optimizers)
    assert all(isinstance(opt, tf.optimizers.Optimizer) for opt in optimizers), \
      "optimizer must be instance of tf.optimizers.Optimizer"
    dir_path = os.path.abspath(dir_path)
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)
    elif os.path.isfile(dir_path):
      raise ValueError("dir_path must be path to a folder")
    models = [
      i for i in tf.nest.flatten(models)
      if isinstance(i, (tf.Variable, keras.layers.Layer))
    ]
    footprint = dir_path + \
                ''.join(sorted([str(id(i)) for i in optimizers])) + \
                ''.join(sorted([str(id(i)) for i in models]))
    if footprint in _CHECKPOINT_MANAGER:
      manager, cp = _CHECKPOINT_MANAGER[footprint]
    else:
      models = {"model%d" % idx: m for idx, m in enumerate(models)}
      models.update(
        {"optimizer%d" % idx: m for idx, m in enumerate(optimizers)})
      cp = tf.train.Checkpoint(**models)
      manager = tf.train.CheckpointManager(cp,
                                           directory=dir_path,
                                           max_to_keep=max_to_keep)
      _CHECKPOINT_MANAGER[footprint] = (manager, cp)
    manager.save()
    with open(os.path.join(dir_path, 'optimizers.pkl'), 'wb') as f:
      pickle.dump(
        [(opt.__class__.__name__, opt.get_config()) for opt in optimizers], f)
    with open(os.path.join(dir_path, 'max_to_keep'), 'wb') as f:
      pickle.dump(max_to_keep, f)
    if trainer is not None:
      with open(os.path.join(dir_path, 'trainer.pkl'), 'wb') as f:
        pickle.dump(trainer, f)

  @staticmethod
  def restore_checkpoint(dir_path, models=None, optimizers=None, index=-1):
    r""" Restore saved checkpoint

    Returns:
      models : list of `keras.Model` or `tf.Variable`
      optimizers : list of `tf.optimizers.Optimizer`
      trainer : `odin.backend.Trainer` or `None`
    """
    dir_path = os.path.abspath(dir_path)
    kwargs = {}
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)
    elif os.path.isfile(dir_path):
      raise ValueError("dir_path must be path to a folder")
    # check models and optimizers
    if models is None and optimizers is None:
      footprint = [name for name in _CHECKPOINT_MANAGER \
                   if dir_path in name]
      if len(footprint) == 0:
        raise ValueError("Cannot find checkpoint information for path: %s" %
                         dir_path)
      footprint = footprint[0]
      manager, cp = _CHECKPOINT_MANAGER[footprint]
    # given models
    elif models is not None:
      models = [
        i for i in tf.nest.flatten(models)
        if isinstance(i, (tf.Variable, keras.layers.Layer))
      ]
      models_ids = ''.join(sorted([str(id(i)) for i in models]))
      with open(os.path.join(dir_path, 'max_to_keep'), 'rb') as f:
        max_to_keep = pickle.load(f)
      # not provided optimizers
      if optimizers is None:
        footprint = [name for name in _CHECKPOINT_MANAGER \
                     if dir_path in name and models_ids in name]
        if len(footprint) > 0:
          manager, cp = _CHECKPOINT_MANAGER[footprint[0]]
        else:  # create optimizer
          with open(os.path.join(dir_path, 'optimizers.pkl'), 'rb') as f:
            optimizers = pickle.load(f)
          optimizers = [
            tf.optimizers.get(name).from_config(config)
            for name, config in optimizers
          ]
          kwargs = {"model%d" % idx: m for idx, m in enumerate(models)}
          kwargs.update(
            {"optimizer%d" % idx: m for idx, m in enumerate(optimizers)})
          cp = tf.train.Checkpoint(**kwargs)
          manager = tf.train.CheckpointManager(cp,
                                               directory=dir_path,
                                               max_to_keep=max_to_keep)
      # both models and optimizers available
      else:
        optimizers = tf.nest.flatten(optimizers)
        optimizers_ids = ''.join(sorted([str(id(i)) for i in optimizers]))
        footprint = [name for name in _CHECKPOINT_MANAGER \
                     if dir_path in name and \
                     optimizers_ids in name and \
                     models_ids in name]
        if len(footprint) > 0:
          manager, cp = _CHECKPOINT_MANAGER[footprint[0]]
        else:
          kwargs = {"model%d" % idx: m for idx, m in enumerate(models)}
          kwargs.update(
            {"optimizer%d" % idx: m for idx, m in enumerate(optimizers)})
          cp = tf.train.Checkpoint(**kwargs)
          manager = tf.train.CheckpointManager(cp,
                                               directory=dir_path,
                                               max_to_keep=max_to_keep)
    # only given optimizers, cannot do anything
    else:
      raise ValueError("Not support for models=%s optimizers=%s" %
                       (str(models), str(optimizers)))
    cp.restore(manager.checkpoints[int(index)])
    # restore trainer (if exist)
    trainer = None
    trainer_path = os.path.join(dir_path, 'trainer.pkl')
    if os.path.exists(trainer_path):
      with open(trainer_path, 'rb') as f:
        trainer = pickle.load(f)
    # return
    models = [getattr(cp, k) for k in sorted(dir(cp)) if 'model' in k[:5]]
    optimizers = [
      getattr(cp, k) for k in sorted(dir(cp)) if 'optimizer' in k[:9]
    ]
    return models, optimizers, trainer

  @staticmethod
  def apply_gradients(tape, optimizer, loss, model_or_weights):
    r"""
    Arguments:
      tape : GradientTape (optional). If not given, no optimization is
        performed.
      optimizer : Instance of `tf.optimizers.Optimizer`.
      loss : a Tensor value with rank 0.
      model_or_weights : List of keras.layers.Layer or tf.Variable, only
        `trainable` variables are optimized.

    Returns:
      gradients : List of gradient Tensor
    """
    if tape is None:
      return
    assert isinstance(tape, tf.GradientTape), "tape must be tf.GradientTape"
    assert isinstance(optimizer, tf.optimizers.Optimizer)
    model_or_weights = tf.nest.flatten(model_or_weights)
    weights = []
    for i in model_or_weights:
      if not i.trainable:
        continue
      if isinstance(i, tf.Variable):
        weights.append(i)
      else:
        weights += i.trainable_variables
    with tape.stop_recording():
      grads = tape.gradient(loss, weights)
      optimizer.apply_gradients(grads_and_vars=zip(grads, weights))
    return grads

  #######################################################
  def __init__(self, logdir: Optional[str] = None, trace_on: bool = False):
    super().__init__()
    if logdir is None:
      logdir = tempfile.mkdtemp()
    logdir = os.path.abspath(os.path.expanduser(logdir))
    self.logdir = logdir
    self.trace_on = bool(trace_on)
    self._n_iter = 0
    self._summary_writer = None
    # default attributes
    self._last_valid_loss = None
    self._last_valid_metrics = {}
    self._last_train_loss = None
    self._last_train_metrics = {}
    # others
    self._current_train_progress = None
    self._cached_tensorboard = None
    self._is_training = tf.Variable(False,
                                    trainable=False,
                                    dtype=tf.bool,
                                    name='is_training')

  @property
  def n_iter(self) -> int:
    return self._n_iter

  @property
  def last_train_loss(self) -> Optional[float]:
    return self._last_train_loss

  @property
  def last_valid_loss(self) -> Optional[float]:
    return self._last_valid_loss

  @property
  def last_train_metrics(self) -> Dict[str, Any]:
    return self._last_train_metrics

  @property
  def last_valid_metrics(self) -> Dict[str, Any]:
    return self._last_valid_metrics

  @property
  def tensorboard(self) -> Dict[Text, Tuple[float, int, float]]:
    """Return data stored in the Tensorboard
    `Dict['metric_name', Tuple['time', 'step', 'values']]`
    """
    if self._cached_tensorboard is None:
      if not os.path.exists(self.logdir):
        self._cached_tensorboard = dict()
      else:
        self._cached_tensorboard = read_tensorboard(self.logdir)
    return self._cached_tensorboard

  @property
  def log_file(self) -> TextIO:
    if not hasattr(self, '_log_file'):
      path = os.path.join(self.logdir, 'log.txt')
      self._log_file = open(path, mode='a')
    return self._log_file

  def get_train_losses(self) -> List[float]:
    losses = self.tensorboard.get('train/loss', [])
    losses = [i[-1] for i in losses]
    return losses

  def get_valid_losses(self) -> List[float]:
    losses = self.tensorboard.get('valid/loss', [])
    losses = [i[-1] for i in losses]
    return losses

  def get_train_metrics(self) -> Dict[str, List[float]]:
    metrics = dict()
    for key, val in self.tensorboard.items():
      if "train/" == key[:6] and key != "train/loss":
        key = key[6:]
        val = [i[-1] for i in val]
        metrics[key] = val
    return metrics

  def get_valid_metrics(self) -> Dict[str, List[float]]:
    metrics = dict()
    for key, val in self.tensorboard.items():
      if "valid/" == key[:6] and key != "valid/loss":
        key = key[6:]
        val = [i[-1] for i in val]
        metrics[key] = val
    return metrics

  @property
  def summary_writer(self) -> tf.summary.SummaryWriter:
    if self._summary_writer is None:
      self._summary_writer = tf.summary.create_file_writer(self.logdir)
    return self._summary_writer

  @property
  def is_training(self) -> bool:
    return self._is_training.numpy()

  def terminate(self):
    self.log_file.flush()
    self._is_training.assign(False)

  def __getstate__(self):
    self.log_file.flush()
    return (self.logdir, self.trace_on, self.n_iter)

  def __setstate__(self, states):
    (self.logdir, self.trace_on, self._n_iter) = states
    # default attributes
    self._summary_writer = None
    self._current_train_progress = None
    self._cached_tensorboard = None
    self._is_training = tf.Variable(False,
                                    trainable=False,
                                    dtype=tf.bool,
                                    name='is_training')
    # default attributes
    self._last_valid_loss = None
    self._last_valid_metrics = {}
    self._last_train_loss = None
    self._last_train_metrics = {}

  def set_optimize_fn(self,
                      optimize: Callable[..., Tuple[Tensor, Dict[str, Any]]],
                      compile_graph: bool = True,
                      autograph: bool = False):
    assert callable(optimize), \
      'optimize function must be callable with input arguments ' \
      '(inputs, training)'
    if isinstance(optimize, partial):
      func = optimize.func
    elif isinstance(optimize, TFFunction):
      func = optimize._python_function
    else:
      func = optimize
    if inspect.ismethod(func):
      func_obj = func.__self__
    else:
      func_obj = None
    func_name = func.__name__

    def fn_step(inputs, training):
      if isinstance(inputs, dict):
        loss, metrics = optimize(training=training, **inputs)
      else:
        loss, metrics = optimize(inputs, training=training)
      if not isinstance(metrics, dict):
        raise RuntimeError(
          f"Metrics must be instance of dictionary, but return: {metrics}")
      return loss, metrics

    if compile_graph and not isinstance(optimize, TFFunction):
      fn_step = tf.function(fn_step, autograph=autograph)
    self._fn_step = fn_step
    self._func_obj = func_obj
    self._func_name = func_name
    return self

  def fit(self,
          train_ds: DatasetV2,
          optimize: Callable[..., Tuple[Tensor, Dict[str, Any]]],
          valid_ds: Optional[DatasetV2] = None,
          valid_freq: int = 0,
          valid_interval: float = 120,
          compile_graph: bool = True,
          autograph: bool = False,
          logging_interval: float = 5,
          log_tag: str = '',
          max_iter: int = -1,
          on_batch_end: Union[Callback, Sequence[Callback]] = lambda: None,
          on_valid_end: Union[Callback, Sequence[Callback]] = lambda: None):
    """ A simplified fitting API

    Parameters
    ----------
    train_ds : tf.data.Dataset.
        Training dataset.
    optimize : Callable.
        Optimization function, return loss and a list of metrics.
        The input arguments must be:
          - ('inputs', 'training');
        and the function must returns:
          - ('loss': `tf.Tensor`, 'metrics': `dict(str, tf.Tensor)`)
    valid_ds : tf.data.Dataset.
        Validation dataset
    valid_freq : an Integer.
        The frequency of validation task, based on the current number
        of iteration.
    valid_interval : a Scalar.
        The number of second until next validation.
    compile_graph : bool
        compile the function to graph for computational optimization.
    autograph : Boolean.
        Enable static graph for the `optimize` function.
    logging_interval : Scalar. Interval for print out log information
        (in second).
    log_tag : str
        string for tagging the training process
    max_iter : An Integer or `None`. Maximum number of iteration for
        training. If `max_iter <= 0`, iterate the training data until the end.
    on_batch_end : Callable take no input arguments.
        The callback will be called after every fixed number of iteration
        according to `valid_freq`, or fixed duration defined by `valid_interval`
    on_valid_end : Callable take no input arguments.
        The callback will be called after every fixed number of iteration
        according to `valid_freq`, or fixed duration defined by `valid_interval`

    Example
    -------
    ```
    def optimize(inputs, tape, n_iter, training):
      return loss, dict(llk=0, div=0, elbo=0)
    ```
    """
    self.set_optimize_fn(optimize,
                         compile_graph=compile_graph,
                         autograph=autograph)
    # reset last stored valid losses
    if len(log_tag) > 0:
      log_tag += " "
    if hasattr(train_ds, '__len__') and max_iter <= 0:
      max_iter = len(train_ds)
    ### check the callbacks
    on_valid_end = _create_callback(on_valid_end)
    on_batch_end = _create_callback(on_batch_end)
    ### Prepare the data
    assert isinstance(train_ds, (tf.data.Dataset, OwnedIterator)), \
      'train_ds must be instance of tf.data.Datasets'
    if valid_ds is not None:
      assert isinstance(valid_ds, (tf.data.Dataset, OwnedIterator)), \
        'valid_ds must be instance of tf.data.Datasets'
    valid_freq = max(1, int(valid_freq))
    valid_interval = float(valid_interval)
    if valid_interval > 0:  # prefer the interval
      valid_freq = 1

    ### validating function
    def valid():
      epoch_loss = []
      epoch_metrics = defaultdict(list)
      valid_progress = tqdm(
        enumerate(valid_ds.repeat(1)),
        desc=f"Validating {valid_freq}(it) or {valid_interval:.1f}(s)")
      for it, inputs in valid_progress:
        _loss, _metrics = self._fn_step(inputs, training=False)
        # store for calculating average
        epoch_loss.append(_loss)
        for k, v in _metrics.items():
          epoch_metrics[k].append(v)
      epoch_loss = tf.reduce_mean(epoch_loss, axis=0)
      epoch_metrics = {
        k: tf.reduce_mean(v, axis=0) for k, v in epoch_metrics.items()
      }
      return epoch_loss, epoch_metrics

    ### training function
    def train():
      global _CURRENT_TRAINER
      _CURRENT_TRAINER = self
      self._is_training.assign(True)
      progress = tqdm(train_ds, desc=f"Training {max_iter}(its)")
      self._current_train_progress = progress
      start_time = progress.start_t
      last_print_time = 0
      last_valid_time = start_time
      self.print("*** Start training: "
                 f"{datetime.datetime.now().strftime(r'%H:%M:%S %d/%m/%Y')}")
      for cur_iter, inputs in enumerate(progress):
        self._n_iter += 1
        tf.summary.experimental.set_step(self.n_iter)
        # ====== check maximum iteration ====== #
        if 0 < max_iter <= cur_iter:
          break
        # the tensorboard will change after each iteration
        self._cached_tensorboard = None
        # ====== train ====== #
        loss, metrics = self._fn_step(inputs, training=True)
        self._last_train_loss = loss
        self._last_train_metrics = dict(metrics)
        _process_callback_returns(self.print,
                                  log_tag=log_tag,
                                  callback_name='BatchEnd',
                                  n_iter=self.n_iter,
                                  metrics=on_batch_end())
        # metric could be hidden by add '_' to the beginning
        metrics = {k: v for k, v in metrics.items() if '_' != k[0]}
        # do not record the loss and metrics at every iteration, the
        # performance will drop about 40%
        # ====== logging ====== #
        interval = progress._time() - last_print_time
        if interval >= logging_interval:
          self.log_file.flush()
          # summarize the batch loss and metrics
          _save_summary(loss, metrics, prefix="train/")
          _print_summary(self.print,
                         log_tag,
                         loss,
                         metrics,
                         self.n_iter,
                         is_valid=False)
          last_print_time = progress._time()
        # ====== validation ====== #
        interval = progress._time() - last_valid_time
        if cur_iter == 0 or \
            (self.n_iter % valid_freq == 0 and interval >= valid_interval):
          if valid_ds is not None:
            # finish the validation
            val_loss, val_metrics = valid()
            self._last_valid_loss = val_loss
            self._last_valid_metrics = val_metrics
            _save_summary(val_loss, val_metrics, prefix="valid/", flush=True)
            _print_summary(self.print,
                           log_tag,
                           val_loss,
                           val_metrics,
                           self.n_iter,
                           is_valid=True)
          # callback always called
          _process_callback_returns(self.print,
                                    log_tag=log_tag,
                                    callback_name='ValidEnd',
                                    n_iter=self.n_iter,
                                    metrics=on_valid_end())
          last_valid_time = progress._time()
        # ====== terminate training ====== #
        if not self.is_training:
          self.print('Terminate training!')
          self.print(f' Loss: {np.asarray(loss)}')
          for k, v in metrics.items():
            self.print(f' {k}: {np.asarray(v)}')
          break
      # Final callback to signal train ended
      self._is_training.assign(False)
      _process_callback_returns(self.print,
                                log_tag=log_tag,
                                callback_name='ValidEnd',
                                n_iter=self.n_iter,
                                metrics=on_valid_end())
      # end the progress
      progress.clear()
      progress.close()
      self.log_file.flush()

    ### train and return
    with self.summary_writer.as_default():
      if self.trace_on:
        tf.summary.trace_on(graph=True, profiler=False)
      else:
        tf.summary.trace_off()
      train()  # training
      if self.trace_on:
        tf.summary.trace_export(name=self._func_name,
                                step=0,
                                profiler_outdir=self.logdir)
        tf.summary.trace_off()
    if isinstance(self._func_obj, (Model, Sequential)):
      self.write_keras_graph(self._func_obj,
                             name=self._func_obj.__class__.__name__)
    self.summary_writer.flush()
    self._current_train_progress = None
    return self

  def print(self, *msg):
    """Print log message without interfering the current `tqdm`
    progress bar"""
    for m in msg:
      if self._current_train_progress is None:
        print(m)
      else:
        self._current_train_progress.write(m)
      self.log_file.write(f'{m}\n')
    return self

  def write_keras_graph(self,
                        model: Union[Model, Sequential],
                        step: int = 0,
                        name: str = "keras"):
    """Writes Keras graph networks to TensorBoard. """
    with self.summary_writer.as_default():
      with summary_ops_v2.always_record_summaries():
        if not model.run_eagerly:
          summary_ops_v2.graph(keras.backend.get_graph(), step=step)
        summary_writable = (model._is_graph_network or
                            model.__class__.__name__ == 'Sequential')
        if summary_writable:
          summary_ops_v2.keras_model(name=str(name), data=model, step=step)
    return self

  def plot_learning_curves(self,
                           path: str = "/tmp/tmp.png",
                           smooth: float = 0.3,
                           dpi: int = 200,
                           n_col: int = 4,
                           title: Optional[str] = None):
    """ Plot learning curves

    Parameters
    ----------
    path : str, optional
        path to save image file, by default "/tmp/tmp.png"
    smooth : float, optional
        exponential moving average smoothing, by default 0.3
    dpi : int, optional
        dot-per-inches, the resolution of the figures, by default 200
    n_col : int, optional
        number of subplot column, by default 2
    title : Optional[str], optional
        title of the figure, by default None

    """
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set()
    smooth = float(smooth)
    assert 0. < smooth < 1., f"EMA smooth must be in [0, 1] but given {smooth}"
    # prepare
    all_data = []
    for name, data in sorted(self.tensorboard.items(),
                             key=lambda x: x[0].split('/')[-1]):
      if data[0][-1].ndim != 0:  # skip non-scalar metrics
        continue
      x, y = [], []
      for _, step, val in data:
        x.append(step)
        y.append(val)
      all_data.append((name, np.array(x, dtype=np.int32), np.array(y)))
    # create the figure
    n_metrics = len(all_data)
    ncol = int(n_col)
    nrow = int(np.ceil(n_metrics / ncol))
    fig = plt.figure(figsize=(ncol * 3, nrow * 3), dpi=dpi)
    # plotting
    subplots = []
    for idx, (name, x, y) in enumerate(all_data):
      ax = plt.subplot(nrow, ncol, idx + 1)
      subplots.append(ax)
      vmin, vmax = np.min(y), np.max(y)
      y = _ema(y, w=1 - smooth)
      plt.plot(x, y)
      plt.plot(x[np.argmin(y)],
               vmin,
               marker='o',
               color='green',
               alpha=0.5,
               label=f'Min:{vmin:.2f}')
      plt.plot(x[np.argmax(y)],
               vmax,
               marker='o',
               color='red',
               alpha=0.5,
               label=f'Max:{vmax:.2f}')
      plt.title(name)
      plt.legend(fontsize=8)
      plt.tick_params(axis='both', labelsize=8)
      ticks = np.linspace(x[0], x[-1], num=5, dtype=np.int32)
      plt.xticks(ticks, ticks)
    # set the title
    if title is not None:
      fig.suptitle(str(title))
      fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    else:
      fig.tight_layout()
    if path is None:
      return fig
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return self


# ===========================================================================
# Helpers
# ===========================================================================
def get_current_trainer() -> Trainer:
  """Return the current operating Trainer, this function is progress safe but not
  thread safe.

  Returns
  -------
  Trainer
      the Trainer that is training.
  """
  return _CURRENT_TRAINER
