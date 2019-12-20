from __future__ import absolute_import, division, print_function

from functools import partial

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow.python.ops import init_ops_v2

from odin.backend.alias import (parse_activation, parse_constraint,
                                parse_initializer, parse_regularizer)
from odin.bay.helpers import coercible_tensor


class StochasticVariable(keras.layers.Layer, tf.initializers.Initializer):

  def __init__(self, sample_shape=(), seed=None):
    super().__init__()
    self._sample_shape = sample_shape
    self._seed = seed

  @property
  def sample_shape(self):
    return self._sample_shape

  @sample_shape.setter
  def sample_shape(self, shape):
    self._sample_shape = shape

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    distribution = self.call()
    assert isinstance(distribution, tfp.distributions.Distribution), \
      'StochasticVariable.call must return Distribution'
    distribution = coercible_tensor(distribution,
                                    convert_to_tensor_fn=partial(
                                        tfp.distributions.Distribution.sample,
                                        sample_shape=self.sample_shape))
    return distribution


class TrainableNormal(StochasticVariable):

  def __init__(self,
               loc_initializer='truncated_normal',
               scale_initializer='truncated_normal',
               loc_regularizer=None,
               scale_regularizer=None,
               loc_activation=None,
               scale_activation='softplus',
               shared_scale=False,
               **kwargs):
    super().__init__(**kwargs)
    self.loc_initializer = parse_initializer(loc_initializer, 'tf')
    self.scale_initializer = parse_initializer(scale_initializer, 'tf')
    self.loc_regularizer = parse_regularizer(loc_regularizer, 'tf')
    self.scale_regularizer = parse_regularizer(scale_regularizer, 'tf')
    self.loc_activation = parse_activation(loc_activation, 'tf')
    self.scale_activation = parse_activation(scale_activation, 'tf')
    self.shared_scale = bool(shared_scale)

  def build(self, shape, dtype=None):
    super().build(shape)
    self.loc = self.add_weight(
        name='loc',
        shape=shape,
        dtype=dtype,
        initializer=self.loc_initializer,
        regularizer=self.loc_regularizer,
        constraint=None,
        trainable=True,
    )
    self.scale = self.add_weight(
        name='scale',
        shape=() if self.shared_scale else shape,
        dtype=dtype,
        initializer=self.scale_initializer,
        regularizer=self.scale_regularizer,
        constraint=None,
        trainable=True,
    )

  def call(self):
    dist = tfp.distributions.Independent(
        tfp.distributions.Normal(loc=self.loc_activation(self.loc),
                                 scale=self.scale_activation(self.scale)), 1)
    return dist


class TrainableNormalSharedScale(TrainableNormal):

  def __init__(self,
               loc_initializer='glorot_normal',
               scale_initializer='truncated_normal',
               loc_regularizer=None,
               scale_regularizer=None,
               loc_activation=None,
               scale_activation='softplus',
               **kwargs):
    super().__init__(loc_initializer,
                     scale_initializer,
                     loc_regularizer,
                     scale_regularizer,
                     loc_activation,
                     scale_activation,
                     shared_scale=True,
                     **kwargs)


trainable_normal = TrainableNormal
trainable_normal_shared_scale = TrainableNormalSharedScale

# NOTE: this only hijack the keras.initializers if you import odin.bay
init_ops_v2.trainable_normal = TrainableNormal
init_ops_v2.trainable_normal_shared_scale = TrainableNormalSharedScale

get = keras.initializers.get
