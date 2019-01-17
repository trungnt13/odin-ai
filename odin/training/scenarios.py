from __future__ import print_function, division, absolute_import

from collections import Mapping

import tensorflow as tf

from odin.utils import (as_tuple, is_string, is_number,
                        wprint, one_hot, ctext)

from odin.training.trainer import Timer, MainLoop
from odin.training.callbacks import *

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

def _preprocessing_losses(losses, y_true, y_pred, inherit_losses=None,
                          sample_weights=None):
  """ Can be used for both objectives and metrics """
  from odin import backend as K
  # ====== special cases, only one inputs outputs, and multiple loss ====== #
  nb_losses = len(losses)
  if len(y_true) == 0:
    y_true = [None] * nb_losses
  elif len(y_true) == 1:
    y_true = y_true * nb_losses
  if len(y_pred) == 0:
    y_pred = [None] * nb_losses
  elif len(y_pred) == 1:
    y_pred = y_pred * nb_losses
  # ====== applying ====== #
  cost = []
  for idx, fn in enumerate(as_tuple(losses)):
    weight = 1
    kwargs = {}
    # preprocess
    if isinstance(fn, (tuple, list)):
      if len(fn) == 1:
        fn = fn[0]
      else:
        weight = [i for i in fn if is_number(i)]
        weight = 1 if len(weight) == 0 else weight[0]
        kwargs = [i for i in fn if isinstance(i, Mapping)]
        kwargs = {} if len(kwargs) == 0 else kwargs[0]
        fn = [i for i in fn if i != weight and i != kwargs][0]
    # apply the loss
    if is_number(fn):
      if inherit_losses is None or fn >= len(inherit_losses):
        raise ValueError("Cannot find losses at index: '%d'" % fn)
      obj = inherit_losses[fn]
    elif K.is_tensor(fn):
      obj = fn
    elif hasattr(fn, '__call__'):
      try:
        sign = inspect.signature(fn)
        if 'weights' in sign.parameters and sample_weights is not None:
          kwargs['weights'] = sample_weights
      except ValueError:
        pass
      finally:
        obj = fn(y_true[idx], y_pred[idx], **kwargs)
      if isinstance(obj, (tuple, list)):
        wprint("function: '%s' return %d outputs (%s), only pick the first one"
               % (fn.__name__,
                  len(obj),
                  '; '.join([str(i) for i in obj])))
        obj = obj[0]
    cost.append((weight, obj))
  # ====== reduce ====== #
  return [c if w == 1 else w * c for w, c in cost]

def _preprocess_prior_weights(y_true, prior_weights):
  if prior_weights is None:
    return None
  from odin import backend as K
  # ====== everything must be list ====== #
  if not isinstance(prior_weights, (tuple, list)):
    prior_weights = (prior_weights,)
  elif is_number(prior_weights[0]):
    prior_weights = (prior_weights,)
  # ====== matching indices and prior_weights ====== #
  pw = 0
  for yt, w in zip(y_true, prior_weights):
    if w is not None:
      pw += K.to_sample_weights(indices=yt, weights=w)
  return pw

# ===========================================================================
# Main methods
# ===========================================================================
def train(X, y_true, y_pred, train_data,
          valid_data=None, valid_freq=1.,
          patience=3, threshold=5, rollback=True,
          objectives=[tf.losses.softmax_cross_entropy],
          metrics=[0], training_metrics=[],
          l1_regu=0., l2_regu=0., parameters=[],
          prior_weights=None, sample_weights=None,
          batch_size=256, epochs=8, shuffle=True,
          optimizer='rmsprop', optz_kwargs={'lr': 0.001}, updates=None,
          init_vars=True, labels=None, seed=5218, verbose=2):
  """

  Parameters
  ----------
  rollback : bool (default: True)
    if True, allow rollback to the best checkpoint during training
  objectives : {callable, tensorflow.Tensor}
    if `callable`, the function must take `y_true`, and `y_pred`
    The objectives must be differentiable and used for training.
  metrics : {callable, tensorflow.Tensor, int}
    if `callable`, the function must take `y_true`, and `y_pred`
    The `metrics` is for monitoring the training process.
    if `int`, it is the index of the loss in `objectives`
    NOTE: the first metrics in the list will be used for
    early-stopping (smaller is better).
  training_metrics : {callable, tensorflow.Tensor, int}
    if `int`, it is the index of the loss in `metrics`
  parameters : {list or tensorflow.Variables}
    All the parameters will be updated by the `optimizer`, if None
    or empty list is given, use ComputationalGraph to get
    all variables with Parameters roles related to the objectives
  init_vars : bool (default: True)
    automatically initialize all variables
  labels : {None, list of string}
    Given labels for classification task
  seed : int
    specific random seed for reproducible
  verbose : int
    0 - Turn off all log
    1 - only show notification
    2 - show notification, important log and summary
    3 - Show progress, summary, notification and logging
    4 - Show debug information and everything

  Return
  ------
  Function used for prediction
  """
  from odin import backend as K
  # ====== preprocess inputs ====== #
  X = as_tuple(X, t=K.is_tensor)
  y_true = as_tuple(y_true, t=K.is_tensor)
  y_pred = as_tuple(y_pred, t=K.is_tensor)
  # ====== parsing objectives and metrics ====== #
  # for training
  prior_weights = _preprocess_prior_weights(y_true=y_true,
                                            prior_weights=prior_weights)
  if prior_weights is not None:
    if sample_weights is not None:
      sample_weights = sample_weights + prior_weights
    else:
      sample_weights = prior_weights
  objectives = _preprocessing_losses(as_tuple(objectives), y_true, y_pred,
                                     sample_weights=sample_weights)
  # metrics for monitoring
  metrics = as_tuple(metrics)
  get_value = lambda x: np.mean(x)
  if len(metrics) > 0 and \
  (metrics[0] == tf.metrics.accuracy or
   metrics[0] == K.metrics.categorical_accuracy):
    get_value = lambda x: 1 - np.mean(x)
  metrics = _preprocessing_losses(metrics, y_true, y_pred,
                                  inherit_losses=objectives)
  # training_metrics
  training_metrics = _preprocessing_losses(as_tuple(training_metrics),
                                           y_true, y_pred,
                                           inherit_losses=metrics)
  # sum the objectives for differentiable
  if len(objectives) > 0:
    objectives = [sum(objectives) if len(objectives) > 1 else objectives[0]]
  # ====== preprocess optimizer and get updates====== #
  if updates is None: # not given updates
    if is_string(optimizer):
      optimizer = _parse_optimizer(optimizer)
      optimizer = optimizer(**optz_kwargs)
    elif not isinstance(optimizer, K.optimizers.Optimizer):
      raise ValueError("`optimizer` must be string - name of algorithm or instance "
                       "of odin.backend.optimizers.Optimizer")
    parameters = K.ComputationGraph(objectives).parameters\
    if len(parameters) == 0 else as_tuple(parameters, t=K.is_variable)
    # check objectives
    if len(objectives) == 0:
      raise RuntimeError("`objectives` must be given due to `updates=None`")
    weights = [p for p in parameters if K.role.has_roles(p, roles=K.role.Weight)]
    # l1 regularization
    if l1_regu > 0.:
      l1_norm = sum(tf.norm(w, ord=1) for w in weights)
      objectives[0] += l1_norm
    # l2 regularization
    if l2_regu > 0.:
      l2_norm = sum(tf.norm(w, ord=2) for w in weights)
      objectives[0] += l2_norm
    # update rules
    updates = optimizer.get_updates(objectives[0], parameters)
    # adding global norm and learning rate
    training_metrics.append(optimizer.norm)
    training_metrics.append(optimizer.lr)
  elif K.is_operation(updates): # given updates
    optimizer = None
  else:
    raise ValueError("`updates` can be None or tensorflow Operation, but given "
      "type: %s" % str(type(updates)))
  # ====== placeholders ====== #
  inputs_plh = []
  for plh in X:
    for i in (K.ComputationGraph(plh).placeholders
              if not K.is_placeholder(plh)
              else as_tuple(plh)):
      inputs_plh.append(i)
  outputs_plh = []
  for plh in y_true: # no duplicated inputs (e.g. autoencoder X == y)
    if not K.is_placeholder(plh):
      plh = K.ComputationGraph(plh).placeholders
    for i in as_tuple(plh):
      if i not in inputs_plh:
        outputs_plh.append(i)
  inputs = inputs_plh + outputs_plh
  # ====== initialize variables ====== #
  if bool(init_vars):
    K.initialize_all_variables()
  # ====== creating function ====== #
  # training function
  f_train = K.function(inputs=inputs,
                       outputs=objectives + training_metrics,
                       updates=updates, training=True)
  # scoring function
  f_score = None
  if len(metrics) > 0:
    f_score = K.function(inputs=inputs, outputs=metrics,
                         training=False)
  # prediction function
  f_pred = K.function(inputs=inputs_plh,
                      outputs=y_pred[0] if len(y_pred) == 1 else y_pred,
                      training=False)
  # ====== preprocessing data ====== #
  train_data, valid_data = _preprocessing_data(train_data, valid_data)
  # print some debug information if necessary
  if verbose >= 4:
    print("%s %s %s" % (
        ctext("============", 'cyan'),
        ctext("Prepare for Training", 'red'),
        ctext("============", 'cyan')))
    print(ctext("Input placeholders:", 'yellow'))
    for i in inputs_plh:
      print(" * ", str(i))
    print(ctext("Output placeholders:", 'yellow'))
    for i in outputs_plh:
      print(" * ", str(i))
    print(ctext("Parameters:", 'yellow'))
    for p in parameters:
      print(" * ", p.name, '-', p.shape, ';', p.dtype.name)
    print(ctext("Optimizer:", 'yellow'))
    print(" * ", str(optimizer))
    print(" * Optimizer kwargs:", optz_kwargs)
    print(" * L1:", l1_regu)
    print(" * L2:", l2_regu)
    print(ctext("Training:", 'yellow'))
    print(" * Valid freq:", valid_freq)
    print(" * Patience:", patience)
    print(" * Threshold:", threshold)
    print(" * Rollback:", rollback)
    print(" * Batch size:", batch_size)
    print(" * Epoch:", epochs)
    print(" * Shuffle:", shuffle)
    print(" * Seed:", seed)
    print(ctext("Objectives:", 'yellow'))
    for o in objectives:
      print(" * ", str(o))
    print(ctext("Weights:", 'yellow'))
    print(" * Prior:", str(prior_weights))
    print(" * Sample:", str(sample_weights))
    print(ctext("Metrics:", 'yellow'))
    for m in metrics:
      print(" * ", str(m))
    print(ctext("Training metrics:", 'yellow'))
    for t in training_metrics:
      print(" * ", str(t))
    print(ctext("Training Data:", 'yellow'), str(train_data))
    print(ctext("Validating Data:", 'yellow'), str(valid_data))
    print(ctext("Labels:", 'yellow'), labels)
  # ====== create trainer ====== #
  callback_log = True if verbose > 0 else False
  trainer = MainLoop(batch_size=batch_size,
                     seed=seed if shuffle else None,
                     shuffle_level=2 if shuffle else 0,
                     allow_rollback=rollback,
                     verbose=verbose, labels=labels)
  trainer.set_checkpoint(path=None, obj=None,
                         variables=parameters)
  # create callback
  callbacks = [NaNDetector(patience=patience, log=callback_log)]
  if valid_data is not None and f_score is not None:
    callbacks.append(
        EarlyStopGeneralizationLoss(task_name='valid', output_name=metrics[0],
                                    threshold=threshold, patience=patience,
                                    log=callback_log, get_value=get_value))
  trainer.set_callbacks(callbacks)
  # set the tasks
  trainer.set_train_task(func=f_train, data=train_data,
                         epoch=epochs, name='train')
  if valid_data is not None and f_score is not None:
    trainer.set_valid_task(func=f_score, data=valid_data,
                           freq=Timer(percentage=valid_freq),
                           name='valid')
  # running
  trainer.run()
  return f_pred
