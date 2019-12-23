from __future__ import absolute_import, division, print_function

import inspect
import os
import pickle
from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.eager.def_function import Function
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Layer
from tensorflow.python.util import nest

__all__ = [
    'copy_keras_metadata', 'has_keras_meta', 'add_trainable_weights',
    'layer2text'
]


def has_keras_meta(tensor):
  return hasattr(tensor, '_keras_history') and hasattr(tensor, '_keras_mask')


def copy_keras_metadata(keras_tensor, *new_tensors):
  if not hasattr(keras_tensor, '_keras_history') or \
  not hasattr(keras_tensor, '_keras_mask'):
    pass
  else:
    new_tensors = nest.flatten(new_tensors)
    history = keras_tensor._keras_history
    mask = keras_tensor._keras_mask
    for t in new_tensors:
      setattr(t, '_keras_history', history)
      setattr(t, '_keras_mask', mask)
  return new_tensors[0] if len(new_tensors) == 1 else new_tensors


def add_trainable_weights(layer, *variables):
  from odin.backend import is_variable
  variables = nest.flatten(variables)
  assert all(is_variable(v) for v in variables), \
  "All objects from variables must be instance of tensorflow.Variable"

  assert isinstance(layer, Layer), \
  "layer must be instance of tensorflow.python.keras.layers.Layer"

  variables = [v for v in variables if v not in layer._trainable_weights]
  layer._trainable_weights = layer._trainable_weights + variables
  return layer


def layer2text(layer):
  assert isinstance(layer, keras.layers.Layer)
  from tensorflow.python.keras.layers.convolutional import Conv
  text = str(layer)
  name = '[%s:"%s"]' % (layer.__class__.__name__, layer.name)
  if isinstance(layer, keras.layers.Dense):
    text = '%sunits:%d bias:%s activation:%s' % \
      (name, layer.units, layer.use_bias, layer.activation.__name__)
  elif isinstance(layer, Conv):
    text = '%sfilter:%d kernel:%s stride:%s dilation:%s pad:%s bias:%s activation:%s' % \
      (name, layer.filters, layer.kernel_size, layer.strides,
       layer.dilation_rate, layer.padding, layer.use_bias,
       layer.activation.__name__)
  elif isinstance(layer, keras.layers.Activation):
    text = '%s %s' % (name, layer.activation.__name__)
  elif isinstance(layer, keras.layers.Dropout):
    text = '%s p=%.2f' % (name, layer.rate)
  elif isinstance(layer, keras.layers.BatchNormalization):
    text = '[%s:"%s"]axis=%s center:%s scale:%s trainable:%s' % \
      ('BatchRenorm' if layer.renorm else 'BatchNorm', layer.name,
       [i for i in tf.nest.flatten(layer.axis)],
       layer.center, layer.scale, layer.trainable)
  elif isinstance(layer, keras.layers.Lambda):
    spec = inspect.getfullargspec(layer.function)
    kw = dict(layer.arguments)
    if spec.defaults is not None:
      kw.update(spec.defaults)
    text = '%s <%s>(%s) default:%s' % \
      (name, layer.function.__name__,
       ', '.join(spec.args), kw)
  elif isinstance(layer, keras.layers.Reshape):
    text = '%s %s' % (name, layer.target_shape)
  elif isinstance(layer, keras.layers.Flatten):
    text = '%s %s' % (name, layer.data_format)
  return text


# ===========================================================================
# For training schedule
# ===========================================================================
_BEST_WEIGHTS = {}
_BEST_OPTIMIZER = {}
_CHECKPOINT_MANAGER = {}


class Trainer(object):
  r"""
  """
  SIGNAL_TERMINATE = '__signal_terminate__'
  SIGNAL_BEST = '__signal_best__'

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
  def save_checkpoint(dir_path, optimizer, models, trainer=None, max_to_keep=5):
    r""" Save checkpoint """
    assert isinstance(optimizer, tf.optimizers.Optimizer), \
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
    footprint = dir_path + str(id(optimizer)) + \
      ''.join(sorted([str(id(i)) for i in models]))
    if footprint in _CHECKPOINT_MANAGER:
      manager, cp = _CHECKPOINT_MANAGER[footprint]
    else:
      models = {"model%d" % idx: m for idx, m in enumerate(models)}
      cp = tf.train.Checkpoint(optimizer=optimizer, **models)
      manager = tf.train.CheckpointManager(cp,
                                           directory=dir_path,
                                           max_to_keep=max_to_keep)
      _CHECKPOINT_MANAGER[footprint] = (manager, cp)
    with open(os.path.join(dir_path, 'trainer.pkl'), 'wb') as f:
      pickle.dump(trainer, f)
    with open(os.path.join(dir_path, 'optimizer.pkl'), 'wb') as f:
      pickle.dump([optimizer.__class__.__name__, optimizer.get_config()], f)
    with open(os.path.join(dir_path, 'max_to_keep'), 'wb') as f:
      pickle.dump(max_to_keep, f)
    manager.save()

  @staticmethod
  def restore_checkpoint(dir_path, models=None, optimizer=None, index=-1):
    r""" Restore saved checkpoint """
    dir_path = os.path.abspath(dir_path)
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)
    elif os.path.isfile(dir_path):
      raise ValueError("dir_path must be path to a folder")
    #
    if models is None and optimizer is None:
      footprint = [name for name in _CHECKPOINT_MANAGER \
                   if dir_path in name]
      if len(footprint) == 0:
        raise ValueError("Cannot find checkpoint information for path: %s" %
                         dir_path)
      footprint = footprint[0]
      manager, cp = _CHECKPOINT_MANAGER[footprint]
    #
    elif models is not None:
      models = [
          i for i in tf.nest.flatten(models)
          if isinstance(i, (tf.Variable, keras.layers.Layer))
      ]
      models_ids = ''.join(sorted([str(id(i)) for i in models]))
      with open(os.path.join(dir_path, 'max_to_keep'), 'rb') as f:
        max_to_keep = pickle.load(f)
      if optimizer is None:
        footprint = [name for name in _CHECKPOINT_MANAGER \
                     if dir_path in name and models_ids in name]
        if len(footprint) > 0:
          manager, cp = _CHECKPOINT_MANAGER[footprint[0]]
        else:  # create optimizer
          with open(os.path.join(dir_path, 'optimizer.pkl'), 'rb') as f:
            name, config = pickle.load(f)
          optimizer_class = tf.optimizers.get(name)
          optimizer = optimizer_class.from_config(config)
          cp = tf.train.Checkpoint(
              optimizer=optimizer,
              **{"model%d" % idx: m for idx, m in enumerate(models)})
          manager = tf.train.CheckpointManager(cp,
                                               directory=dir_path,
                                               max_to_keep=max_to_keep)
      else:
        footprint = [name for name in _CHECKPOINT_MANAGER \
                     if dir_path in name and \
                       str(id(optimizer)) in name and \
                         models_ids in name]
        if len(footprint) > 0:
          manager, cp = _CHECKPOINT_MANAGER[footprint[0]]
        else:
          cp = tf.train.Checkpoint(
              optimizer=optimizer,
              **{"model%d" % idx: m for idx, m in enumerate(models)})
          manager = tf.train.CheckpointManager(cp,
                                               directory=dir_path,
                                               max_to_keep=max_to_keep)
    else:
      raise ValueError("Not support for models=%s optimizers=%s" %
                       (str(models), str(optimizer)))
    cp.restore(manager.checkpoints[int(index)])
    # restore trainer (if exist)
    trainer_path = os.path.join(dir_path, 'trainer.pkl')
    if os.path.exists(trainer_path):
      with open(trainer_path, 'rb') as f:
        return pickle.load(f)
    return None

  @staticmethod
  def early_stop(losses,
                 threshold=0.2,
                 progress_length=5,
                 patience=5,
                 min_epoch=-np.inf,
                 terminate_on_nan=True,
                 verbose=0):
    r""" Early stopping based on generalization loss and the three rules:
      - Stop when generalization error exceeds threshold in a number of
          successive steps.
      - Stop as soon as the generalization loss exceeds a certain threshold.
      - Supress stopping if the training is still progress rapidly.

    Generalization loss: `GL(t) = losses[-1] / min(losses[:-1]) - 1`

    Progression: `PG(t) = 10 * sum(L) / (k * min(L))` where `L = losses[-k:]`

    The condition for early stopping: `GL(t) / PG(t) >= threshold`

    Arugments:
      losses : List of loss values (smaller is better)
      threshold : Float. Determine by `generalization_error / progression`
      progress_length : Integer. Number of steps to look into the past for
        estimating the training progression. If smaller than 2, turn-off
        progression for early stopping.
      patience : Integer. Number of successive steps that yield loss exceeds
        `threshold / 2` until stopping. If smaller than 2, turn-off patience
        for early stopping
      min_epoch: Minimum number of epoch until early stop kicks in.
        Note, all the metrics won't be updated until the given epoch.
      terminate_on_nan : Boolean. Terminate the training progress if NaN or Inf
        appear in the losses.

    Return:
      Trainer.SIGNAL_TERMINATE : stop training
      Trainer.SIGNAL_BEST : best model achieved

    Reference:
      Prechelt, L. (1998). "Early Stopping | but when?".
    """
    if terminate_on_nan and (np.isnan(losses[-1]) or np.isinf(losses[-1])):
      return Trainer.SIGNAL_TERMINATE
    if len(losses) < max(2., min_epoch):
      tf.print("[EarlyStop] First 2 warmup epochs.")
      return Trainer.SIGNAL_BEST
    current = losses[-1]
    best = np.min(losses[:-1])
    # patience
    if patience > 1 and len(losses) > patience and \
      all((i / best - 1) >= (threshold / 2) for i in losses[-patience:]):
      if verbose:
        tf.print("[EarlyStop] %d successive step without enough improvement." %
                 patience)
      return Trainer.SIGNAL_TERMINATE
    # generalization error (smaller is better)
    generalization = current / best - 1
    if generalization <= 0:
      if verbose:
        tf.print("[EarlyStop] Best model, generalization improved: %.4f" %
                 (best / current - 1.))
      return Trainer.SIGNAL_BEST
    # progression (bigger is better)
    if progress_length > 1:
      progress = losses[-progress_length:]
      progression = 10 * \
        (np.sum(progress) / (progress_length * np.min(progress)) - 1)
    else:
      progression = 1.
    # thresholding
    error = generalization / progression
    if error >= threshold:
      tf.print(
          "[EarlyStop] Exceed threshold:%.2f  generalization:%.4f progression:%.4f"
          % (threshold, generalization, progression))
      return Trainer.SIGNAL_TERMINATE

  @staticmethod
  def validate_optimize(func):
    assert callable(func), "optimize must be callable."
    if isinstance(func, Function):
      args = func.function_spec.arg_names
    else:
      args = inspect.getfullargspec(func).args
    template = ['tape', 'training', 'n_iter']
    assert 'tape' in args, \
      "tape (i.e. GradientTape) must be in arguments list of optimize function."
    assert all(a in template for a in args[1:]), \
      "optimize function must has the following arguments: %s; but given: %s"\
        % (template, args)
    return args

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
    """
    if tape is None:
      return
    assert isinstance(tape, tf.GradientTape)
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

  @staticmethod
  def prepare(ds,
              preprocess=None,
              postprocess=None,
              batch_size=128,
              epochs=1,
              cache=True,
              drop_remainder=False,
              shuffle=None,
              parallel_preprocess=0,
              parallel_postprocess=0):
    r""" A standarlized procedure for preparing `tf.data.Dataset` for training
    or evaluation.

    Arguments:
      ds : `tf.data.Dataset`
      preprocess : Callable (optional). Call at the beginning when a single
        example is fetched.
      postprocess : Callable (optional). Called after `.batch`, which processing
        the whole minibatch.
      parallel_preprocess, parallel_postprocess : Integer.
        - if value is 0, process sequentially
        - if value is -1, using `tf.data.experimental.AUTOTUNE`
        - if value > 0, explicitly specify the number of process for running
            map task
    """
    parallel_preprocess = None if parallel_preprocess == 0 else \
      (tf.data.experimental.AUTOTUNE if parallel_preprocess == -1 else
       int(parallel_preprocess))
    parallel_postprocess = None if parallel_postprocess == 0 else \
      (tf.data.experimental.AUTOTUNE if parallel_postprocess == -1 else
       int(parallel_postprocess))

    if not isinstance(ds, tf.data.Dataset):
      ds = tf.data.Dataset.from_tensor_slices(ds)

    if shuffle:
      if isinstance(shuffle, Number):
        ds = ds.shuffle(buffer_size=int(shuffle), reshuffle_each_iteration=True)
      else:
        ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

    if preprocess is not None:
      ds = ds.map(preprocess, num_parallel_calls=parallel_preprocess)
      if cache:
        ds = ds.cache()
    ds = ds.repeat(epochs).batch(batch_size, drop_remainder=drop_remainder)

    if postprocess is not None:
      ds = ds.map(postprocess, num_parallel_calls=parallel_postprocess)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  #######################################################
  def __init__(self):
    super().__init__()
    self.n_iter = tf.Variable(0,
                              dtype=tf.float32,
                              trainable=False,
                              name='n_iter')
    self.train_loss = []
    self.train_metrics = []
    self.valid_loss = []
    self.valid_metrics = []

  def __getstate__(self):
    return self.n_iter.numpy(), self.train_loss, self.train_metrics, \
      self.valid_loss, self.valid_metrics

  def __setstate__(self, states):
    self.n_iter, self.train_loss, self.train_metrics, \
      self.valid_loss, self.valid_metrics = states
    self.n_iter = tf.Variable(self.n_iter,
                              dtype=tf.float32,
                              trainable=False,
                              name='n_iter')

  def fit(self,
          ds,
          optimize,
          valid_ds=None,
          valid_freq=1000,
          persistent_tape=True,
          autograph=True,
          logging_interval=2,
          callback=lambda: None):
    r""" A simplified fitting API

    Arugments:
      ds : tf.data.Dataset. Training dataset
      optimize : Callable. Optimization function, return loss and a list of
        metrics.
      valid_ds : tf.data.Dataset. Validation dataset
      valid_freq : Integer. The frequency of validation task, based on number
        of iteration in training.
      persistent_tape : Boolean. Using persistent GradientTape, so multiple
        call to gradient is feasible.
      autograph : Boolean. Enable static graph for optimize function
      logging_interval : Scalar. Interval for print out log information
        (in second)
      callback : Callable. The callback will be called after every validation
        epoch. If `valid_ds=None`, it is called at `logging_interval`.
    """
    autograph = int(autograph)
    ### Prepare the data
    assert isinstance(ds, tf.data.Dataset), \
      'ds must be instance of tf.data.Datasets'
    if valid_ds is not None:
      assert isinstance(valid_ds, tf.data.Dataset), \
        'valid_ds must be instance of tf.data.Datasets'
    valid_freq = int(valid_freq)
    ### optimizing function
    optimize_args = Trainer.validate_optimize(optimize)

    ### helper function for training iteration
    def step(n_iter, inputs, training):
      kw = dict()
      if 'n_iter' in optimize_args:
        kw['n_iter'] = n_iter
      if 'training' in optimize_args:
        kw['training'] = training
      if training:  # for training
        with tf.GradientTape(persistent=persistent_tape) as tape:
          loss, metrics = optimize(inputs, tape=tape, **kw)
      else:  # for validation
        loss, metrics = optimize(inputs, tape=None, **kw)
      return loss, metrics

    if autograph and not isinstance(optimize, Function):
      step = tf.function(step)

    def valid():
      avg_loss = 0.
      avg_metrics = 0.
      start_time = tf.timestamp()
      last_it = 0.
      for it, inputs in enumerate(valid_ds.repeat(1)):
        it = tf.cast(it, tf.float32)
        _loss, _metrics = step(it, inputs, training=False)
        # moving average
        avg_loss = (_loss + it * avg_loss) / (it + 1)
        avg_metrics = (_metrics + it * avg_metrics) / (it + 1)
        # print log
        end_time = tf.timestamp()
        if end_time - start_time >= logging_interval:
          it_per_sec = tf.cast(
              (it - last_it) / tf.cast(end_time - start_time, tf.float32),
              tf.int32)
          tf.print(" [Valid] #", it + 1, " ", it_per_sec, "(it/s)", sep="")
          start_time = tf.timestamp()
          last_it = it
      return avg_loss, avg_metrics

    def train():
      total_time = 0.
      start_time = tf.timestamp()
      last_iter = tf.identity(self.n_iter)
      for inputs in ds:
        self.n_iter.assign_add(1.)
        # ====== validation ====== #
        if valid_ds is not None:
          if self.n_iter % valid_freq == 0:
            total_time += tf.cast(tf.timestamp() - start_time, tf.float32)
            # finish the validation
            valid_loss, valid_metrics = valid()
            self.valid_loss.append(valid_loss.numpy())
            self.valid_metrics.append(np.array(tf.nest.flatten(valid_metrics)))
            tf.print(" [Valid] loss:",
                     valid_loss,
                     " metr:",
                     valid_metrics,
                     sep="")
            # reset start_time
            start_time = tf.timestamp()
            signal = callback()
            if signal == Trainer.SIGNAL_TERMINATE:
              break
        # ====== train ====== #
        loss, metrics = step(self.n_iter, inputs, training=True)
        self.train_loss.append(loss.numpy())
        self.train_metrics.append(np.array(tf.nest.flatten(metrics)))
        interval = tf.cast(tf.timestamp() - start_time, tf.float32)
        # ====== logging ====== #
        if interval >= logging_interval:
          if valid_ds is None:
            signal = callback()
            if signal == Trainer.SIGNAL_TERMINATE:
              break
          total_time += interval
          tf.print("#",
                   self.n_iter,
                   " loss:",
                   loss,
                   " metr:",
                   metrics,
                   "  ",
                   tf.cast(total_time, tf.int32),
                   "(s) ",
                   tf.cast((self.n_iter - last_iter) / interval, tf.int32),
                   "(it/s)",
                   sep="")
          start_time = tf.timestamp()
          last_iter = tf.identity(self.n_iter)

    ### train and return
    # train = tf.function(train)
    train()
