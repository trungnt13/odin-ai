from __future__ import absolute_import, division, print_function

import collections

import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from tensorflow.python import keras
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Dense
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers.distribution_layer import (
    DistributionLambda, _get_convert_to_tensor_fn, _serialize,
    _serialize_function)
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible as dtc
from tensorflow_probability.python.layers.internal import \
    tensor_tuple as tensor_tuple

__all__ = ['MixtureDensityNetwork']

_COV_TYPES = ('none', 'diag', 'full', 'tril')


class MixtureDensityNetwork(Dense):
  """A mixture of Gaussian Keras layer.

  Parameters
  ----------
  units : `int`
    number of output features for each component.
  n_components : `int` (default=`2`)
    The number of mixture components.
  covariance_type : {'none', 'diag', 'full', 'tril'}
    String describing the type of covariance parameters to use.
    Must be one of:
      'none' (each component has its own single variance).
      'diag' (each component has its own diagonal covariance matrix),
      'tril' (lower triangle matrix),
      'full' (each component has its own general covariance matrix),

  """

  def __init__(self,
               units,
               n_components=2,
               covariance_type='none',
               convert_to_tensor_fn=tfd.Distribution.sample,
               softplus_scale=True,
               validate_args=False,
               activation='linear',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    covariance_type = str(covariance_type).lower()
    assert covariance_type in _COV_TYPES, \
    "No support for covariance_type: '%s', the support value are: %s" % \
      (covariance_type, ', '.join(_COV_TYPES))
    self._covariance_type = covariance_type
    self._n_components = int(n_components)
    self._validate_args = bool(validate_args)
    self._convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)
    self._softplus_scale = bool(softplus_scale)
    # We'll need to keep track of who's calling who since the functional
    # API has a different way of injecting `_keras_history` than the
    # `keras.Sequential` way.
    self._enter_dunder_call = False
    # ====== calculating the number of parameters ====== #
    if covariance_type == 'none':
      component_params_size = 2 * units
    elif covariance_type == 'diag':  # only the diagonal
      component_params_size = units + units
    elif covariance_type == 'tril':  # lower triangle
      component_params_size = units + units * (units + 1) // 2
    elif covariance_type == 'full':  # full matrix
      component_params_size = units + units * units
    else:
      raise NotImplementedError
    self._component_params_size = component_params_size
    params_size = self.n_components + self.n_components * component_params_size
    self._event_size = units

    super(MixtureDensityNetwork,
          self).__init__(units=params_size,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)

  @property
  def event_size(self):
    return self._event_size

  @property
  def covariance_type(self):
    return self._covariance_type

  @property
  def n_components(self):
    return self._n_components

  @property
  def component_params_size(self):
    return self._component_params_size

  def __call__(self, inputs, *args, **kwargs):
    self._enter_dunder_call = True
    distribution, _ = super(MixtureDensityNetwork,
                            self).__call__(inputs, *args, **kwargs)
    self._enter_dunder_call = False
    return distribution

  def call(self, inputs, *args, **kwargs):
    dense_kwargs = dict(kwargs)
    dense_kwargs.pop('training', None)
    params = super(MixtureDensityNetwork, self).call(inputs, *args,
                                                     **dense_kwargs)
    n_components = tf.convert_to_tensor(value=self.n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    # ====== mixture weights ====== #
    mixture_coefficients = params[..., :n_components]
    mixture_dist = tfd.Categorical(logits=mixture_coefficients,
                                   validate_args=self._validate_args,
                                   name="MixtureWeights")
    # ====== initialize the components ====== #
    params = tf.reshape(
        params[..., n_components:],
        tf.concat([tf.shape(input=params)[:-1], [n_components, -1]], axis=0))
    if bool(self._softplus_scale):
      scale_fn = lambda x: tf.math.softplus(x) + tfd.softplus_inverse(1.0)
    else:
      scale_fn = lambda x: x

    if self.covariance_type == 'none':
      cov = 'IndependentNormal'
      loc_params, scale_params = tf.split(params, 2, axis=-1)
      scale_params = scale_params
      components_dist = tfd.Independent(tfd.Normal(
          loc=loc_params,
          scale=scale_fn(scale_params),
          validate_args=self._validate_args),
                                        reinterpreted_batch_ndims=1)
    #
    elif self.covariance_type == 'diag':
      cov = 'MultivariateNormalDiag'
      loc_params, scale_params = tf.split(params, 2, axis=-1)
      components_dist = tfd.MultivariateNormalDiag(
          loc=loc_params,
          scale_diag=scale_fn(scale_params),
          validate_args=self._validate_args)
    #
    elif self.covariance_type == 'tril':
      cov = 'MultivariateNormalTriL'
      loc_params = params[..., :self.event_size]
      scale_params = scale_fn(params[..., self.event_size:])
      scale_tril = tfb.ScaleTriL(diag_shift=np.array(
          1e-5, params.dtype.as_numpy_dtype()),
                                 validate_args=self._validate_args)
      components_dist = tfd.MultivariateNormalTriL(
          loc=loc_params,
          scale_tril=scale_tril(scale_params),
          validate_args=self._validate_args)
    #
    elif self.covariance_type == 'full':
      cov = 'MultivariateNormalFull'
      loc_params = params[..., :self.event_size]
      scale_params = tf.reshape(
          scale_fn(params[..., self.event_size:]),
          tf.concat(
              [tf.shape(input=params)[:-1], (self.event_size, self.event_size)],
              axis=0))
      components_dist = tfd.MultivariateNormalFullCovariance(
          loc=loc_params,
          covariance_matrix=scale_params,
          validate_args=self._validate_args)
    else:
      raise NotImplementedError
    # ====== finally the mixture ====== #
    d = tfd.MixtureSameFamily(mixture_distribution=mixture_dist,
                              components_distribution=components_dist,
                              validate_args=False,
                              name="Mixture%s" % cov)
    # Wraps the distribution to return both dist and concrete value."""
    value_is_seq = isinstance(d.dtype, collections.Sequence)
    maybe_composite_convert_to_tensor_fn = (
        (lambda d: tensor_tuple.TensorTuple(self._convert_to_tensor_fn(d)))
        if value_is_seq else self._convert_to_tensor_fn)
    distribution = dtc._TensorCoercible(  # pylint: disable=protected-access
        distribution=d,
        convert_to_tensor_fn=maybe_composite_convert_to_tensor_fn)
    value = distribution._value()  # pylint: disable=protected-access
    value._tfp_distribution = distribution  # pylint: disable=protected-access
    if value_is_seq:
      value.shape = value[-1].shape
      value.get_shape = value[-1].get_shape
      value.dtype = value[-1].dtype
      distribution.shape = value[-1].shape
      distribution.get_shape = value[-1].get_shape
    else:
      distribution.shape = value.shape
      distribution.get_shape = value.get_shape
    if self._enter_dunder_call:
      # Its critical to return both distribution and concretization
      # so Keras can inject `_keras_history` to both. This is what enables
      # either to be used as an input to another Keras `Model`.
      return distribution, value
    return distribution

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    # the number of output units is equal to event_size, not number of
    # hidden units
    return input_shape[:-1].concatenate(self.event_size)

  def get_config(self):
    """Returns the config of this layer. """
    config = {
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
        'covariance_type': self._covariance_type,
        'validate_args': self._validate_args,
        'n_components': self._n_components,
        'softplus_scale': self._softplus_scale,
    }
    base_config = super(MixtureDensityNetwork, self).get_config()
    base_config.update(config)
    return base_config
