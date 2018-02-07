from collections import Mapping

import numpy as np
import tensorflow as tf

from .trainer import Task, Timer, MainLoop
from .callbacks import *
from odin.utils import as_tuple, is_string, is_number


# ===========================================================================
# Helper
# ===========================================================================
def _parse_optimizer(name):
  from odin import backend as K
  name = str(name).lower()
  for key in dir(K.optimizers):
    obj = getattr(K.optimizers, key)
    if isinstance(obj, type) and issubclass(obj, K.optimizers.Optimizer):
      # more robust in handling the name
      if name in key.lower():
        return obj
  return None

def _preprocessing_data(train, valid):
  from odin import fuel as F
  train = F.as_data(train)
  if is_number(valid):
    start_train = 0.
    end_train = 1. - valid
    start_valid = 1. - valid
    end_valid = 1.
    valid = F.DataGroup(train.data).set_batch(start=start_valid, end=end_valid)
    train = F.DataGroup(train.data).set_batch(start=start_train, end=end_train)
  elif valid is not None:
    valid = F.as_data(valid)
  return train, valid

def _preprocessing_losses(losses, y_true, y_pred, is_differentiable):
  """ Can be used for both objectives and metrics """
  from odin import backend as K
  cost = []
  for fn, yt, yp in zip(losses, y_true, y_pred):
    weight = 1
    kwargs = {}
    # preprocess
    if isinstance(fn, (tuple, list)):
      if len(fn) == 1:
        fn = fn[0]
      else:
        weight = [i for i in fn if is_number(i)][0]
        kwargs = [i for i in fn if isinstance(i, Mapping)]
        if len(kwargs) == 0:
          kwargs = {}
        fn = [i for i in fn if i != weight and i != kwargs][0]
    # apply the loss
    if K.is_tensor(fn):
      obj = fn
    elif hasattr(fn, '__call__'):
      obj = fn(yt, yp, **kwargs)
    cost.append((weight, obj))
  # ====== reduce ====== #
  if is_differentiable:
    cost = sum(w * c for w, c in cost)
  else:
    cost = [c if w == 1 else w * c for w, c in cost]
  return cost

# ===========================================================================
# Main methods
# ===========================================================================
def train(X, y_true, y_pred, train_data,
          valid_data=None, valid_freq=1.,
          patience=3, threshold=5, rollback=True,
          objectives=[], metrics=[], parameters=[],
          batch_size=256, epochs=6, shuffle=True,
          optimizer='rmsprop', lr=0.001,
          decay_steps=None, decay_rate=0.96, staircase=True,
          clipnorm=None, clipvalue=None,
          labels=None, seed=5218, verbose=1):
  """
  Parameters
  ----------
  rollback : bool (default: True)
    if True, allow rollback to the best checkpoint during training
  objectives : {callable, tensorflow.Tensor}
    if `callable`, the function must take `y_true`, and `y_pred`
    The objectives must be differentiable and used for training.
  metrics : {callable, tensorflow.Tensor}
    if `callable`, the function must take `y_true`, and `y_pred`
    The `metrics` is for monitoring the training process.
  labels : {None, list of string}
    Given labels for classification task
  seed : int
    specific random seed for reproducible
  verbose : int
    0 - Turn off all log
    1 - Only show important log and summary
    2 - Show progress and summary

  """
  from odin import backend as K
  # ====== preprocess inputs ====== #
  X = as_tuple(X, t=K.is_tensor)
  y_true = as_tuple(y_true, t=K.is_tensor)
  y_pred = as_tuple(y_pred, t=K.is_tensor)
  if len(y_true) != len(y_pred):
    raise ValueError("There are %d `y_true` variables but %d `y_pred` variables"
                     % (len(y_true), len(y_pred)))
  # ====== parsing objectives and metrics ====== #
  if objectives is None or len(objectives) == 0:
    objectives = [tf.losses.softmax_cross_entropy]
  objectives = _preprocessing_losses(objectives, y_true, y_pred,
                                     is_differentiable=True)
  if metrics is None or len(metrics) == 0:
    metrics = [K.metrics.categorical_accuracy]
    get_value = lambda x: 1 - np.mean(x)
  else:
    get_value = lambda x: np.mean(x)
  metrics = _preprocessing_losses(metrics, y_true, y_pred,
                                  is_differentiable=False)
  # ====== preprocess optimizer ====== #
  if is_string(optimizer):
    optimizer = _parse_optimizer(optimizer)
    optimizer = optimizer(lr=lr, decay_rate=decay_rate, decay_steps=decay_steps,
                          staircase=staircase,
                          clipnorm=clipnorm, clipvalue=clipvalue)
  elif not isinstance(optimizer, K.optimizers.Optimizer):
    raise ValueError("`optimizer` must be string - name of algorithm or instance "
                     "of odin.backend.optimizers.Optimizer")
  # ====== parameters ====== #
  if parameters is None or len(parameters) == 0:
    parameters = K.ComputationGraph(objectives).parameters
  else:
    parameters = as_tuple(parameters, t=K.is_variable)
  updates = optimizer.get_updates(objectives, parameters)
  # ====== build function ====== #
  inputs = list(X)
  for i in y_true: # no duplicated inputs (e.g. autoencoder X == y)
    if i not in inputs:
      inputs.append(i)
  f_train = K.function(inputs=inputs,
                       outputs=as_tuple(objectives) + as_tuple(metrics),
                       updates=updates, training=True)
  f_test = K.function(inputs=inputs, outputs=metrics,
                      training=False)
  # ====== preprocessing data ====== #
  train_data, valid_data = _preprocessing_data(train_data, valid_data)
  # ====== create trainer ====== #
  callback_log = True if verbose > 0 else False
  trainer = MainLoop(batch_size=batch_size,
                     seed=seed if shuffle else None,
                     shuffle_level=2 if shuffle else 0,
                     allow_rollback=True,
                     verbose=verbose, labels=labels)
  trainer.set_checkpoint(path=None, obj=None, variables=parameters)
  trainer._rollback()
  trainer.set_callbacks([
      NaNDetector(patience=patience, log=callback_log),
      EarlyStopGeneralizationLoss('valid', metrics[0],
                                  threshold=threshold, patience=patience,
                                  log=callback_log, get_value=get_value)
  ])
  trainer.set_train_task(func=f_train, data=train_data,
                         epoch=epochs, name='train')
  if valid_data is not None:
    trainer.set_valid_task(func=f_test, data=valid_data,
                           freq=Timer(percentage=valid_freq),
                           name='valid')
  if rollback:
    trainer._rollback()
  trainer.run()

def train_network():
  pass
