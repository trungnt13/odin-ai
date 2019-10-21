from __future__ import absolute_import, division, print_function

import inspect
import os
from typing import Text, Type, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import torch
from six import string_types
from tensorflow.python import keras

from odin.backend.maths import sqrt
from odin.backend.tensor import (concatenate, moments, reduce_max, reduce_mean,
                                 reduce_min, reduce_sum)

__all__ = [
    'parse_activation', 'parse_attention', 'parse_constraint',
    'parse_initializer', 'parse_normalizer', 'parse_regularizer',
    'parse_optimizer', 'parse_reduction'
]


# ===========================================================================
# Helper
# ===========================================================================
def _linear_function(x):
  return x


def _invalid(msg, obj):
  if isinstance(obj, string_types):
    pass
  elif obj is None:
    obj = 'None'
  else:
    obj = str(type(obj))
  raise ValueError("%s, given type: %s" % (msg, obj))


def _is_tensorflow(framework):
  if isinstance(framework, string_types):
    framework = framework.lower()
    if any(i in framework for i in ('tf', 'tensorflow', 'tensor')):
      return True
  if not inspect.isclass(framework):
    framework = type(framework)
  cls_desc = str(framework) + ''.join(str(i) for i in type.mro(framework))
  if 'tensorflow' in cls_desc:
    return True
  return False


# ===========================================================================
# Network basics
# ===========================================================================
def parse_activation(activation, framework):
  """
  Parameters
  ----------
  activation : `str`
    alias for activation function
  framework : `str`
    'tensorflow' or 'pytorch'
  """
  if activation is None:
    activation = 'linear'
  if callable(activation):
    return activation
  if isinstance(activation, string_types):
    if activation.lower() == 'linear':
      return _linear_function

    if _is_tensorflow(framework):
      return keras.activations.get(activation)
    else:
      for i in dir(torch.nn.functional):
        if i.lower() == activation.lower():
          fn = getattr(torch.nn.functional, i)
          if inspect.isfunction(fn):
            return fn
  _invalid("No support for activation", activation)


def parse_initializer(initializer, framework):
  if _is_tensorflow(framework):
    return keras.initializers.get(initializer)
  else:
    if callable(initializer):
      return initializer
    if isinstance(initializer, string_types):
      initializer = initializer.lower().replace('glorot_', 'xavier_').replace(
          'he_', 'kaiming_')
      for i in dir(torch.nn.init):
        if i.lower() == initializer.lower() + '_':
          fn = getattr(torch.nn.init, i)
          if inspect.isfunction(fn):
            return fn
  _invalid("No support for initializer", initializer)


def parse_optimizer(optimizer, framework) -> Type:
  """ Return the class for given optimizer alias """
  if _is_tensorflow(framework):
    all_classes = {
        'adadelta': keras.optimizers.adadelta_v2.Adadelta,
        'adagrad': keras.optimizers.adagrad_v2.Adagrad,
        'adam': keras.optimizers.adam_v2.Adam,
        'adamax': keras.optimizers.adamax_v2.Adamax,
        'nadam': keras.optimizers.nadam_v2.Nadam,
        'rmsprop': keras.optimizers.rmsprop_v2.RMSprop,
        'sgd': keras.optimizers.gradient_descent_v2.SGD,
    }
  else:
    all_classes = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'nadam': None,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
  opt = all_classes.get(str(optimizer).lower(), None)
  if opt is not None:
    return opt
  _invalid("No support for optimizer", optimizer)


def parse_regularizer(regularizer, framework):
  if regularizer is None:
    return None
  if _is_tensorflow(framework):
    return keras.regularizers.get(regularizer)
  else:
    pass
  _invalid("No support for regularizer", regularizer)


def parse_constraint(constraint, framework):
  if constraint is None:
    return None
  if _is_tensorflow(framework):
    return keras.constraints.get(constraint)
  else:
    pass
  _invalid("No support for constraint", constraint)


# ===========================================================================
# Layers
# ===========================================================================
def parse_reduction(reduce: Text, framework=None):
  """ Return a reduce function """
  if reduce is None:
    reduce = 'none'
  if isinstance(reduce, string_types):
    if "min" in reduce:
      return reduce_min
    if "max" in reduce:
      return reduce_max
    if "avg" in reduce or "mean" in reduce:
      return reduce_mean
    if "sum" in reduce:
      return reduce_sum
    if "none" in reduce or reduce == "":
      return lambda x, *args, **kwargs: x

    if "stat" in reduce:

      def stat_reduce(x, axis=None, keepdims=None):
        m, v = moments(x, axis=axis, keepdims=keepdims)
        return concatenate([m, sqrt(v)], axis=-1)

      return stat_reduce
  _invalid("No support for reduce", reduce)


def parse_attention(attention, framework):
  pass


def parse_normalizer(normalizer, framework):
  if _is_tensorflow(framework):
    if isinstance(normalizer, string_types):
      normalizer = normalizer.strip().lower()
      if normalizer == 'batchnorm':
        return keras.layers.BatchNormalization
      elif normalizer == 'batchrenorm':
        from odin.networks.util_layers import BatchRenormalization
        return keras.layers.BatchRenormalization
      elif normalizer == 'layernorm':
        return keras.layers.LayerNormalization
      elif normalizer == 'instancenorm':
        return tfa.layers.InstanceNormalization
      elif normalizer == 'groupnorm':
        return tfa.layers.GroupNormalization
  else:
    pass
  _invalid("No support for normalizer", normalizer)


def parse_layer(layer, framework):
  pass


# ===========================================================================
# Loss and metric
# ===========================================================================
def parse_loss(loss, framework):
  pass


def parse_metric(loss, framework):
  pass
