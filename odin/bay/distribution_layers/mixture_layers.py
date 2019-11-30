from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import \
    distribution_util as dist_util
from tensorflow_probability.python.layers import (
    CategoricalMixtureOfOneHotCategorical, MixtureLogistic, MixtureNormal,
    MixtureSameFamily)

from odin.backend import parse_activation
from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated

__all__ = [
    'MixtureLogisticLayer', 'MixtureNormalLayer', 'MixtureSameFamilyLayer',
    'CategoricalMixtureOfOneHotCategorical', 'MixtureNegativeBinomial'
]
MixtureLogisticLayer = MixtureLogistic
MixtureNormalLayer = MixtureNormal
MixtureSameFamilyLayer = MixtureSameFamily


class MixtureNegativeBinomial(tfp.layers.DistributionLambda):
  r"""Initialize the `MixtureNegativeBinomial` distribution layer.

  Arguments:
    n_components: Number of component distributions in the mixture
      distribution.
    mean_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures in default parameterization, or
      `mean` in alternative approach.
    disp_activation: activation function for the success rate (default
      parameterization), or the non-negative dispersion (alternative approach).
    alternative_paramerization: `bool`, using default parameterization of
      `total_count` and `probs_success`, or the alternative with `mean` and
      `dispersion`. Default: `False`
    dispersion : {'full', 'share', 'single'}
      'full' creates a dispersion value for each individual data point,
      'share' creates a single dispersion vector of `event_shape` for all examples,
      and 'single' uses a single value as dispersion for all data points.
    convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
      instance and returns a `tf.Tensor`-like object.
      Default value: `tfd.Distribution.sample`.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
      Default value: `False`.
    **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
  """

  def __init__(self,
               event_shape=(),
               n_components=2,
               mean_activation='softplus',
               disp_activation='linear',
               dispersion='full',
               alternative_parameterization=False,
               is_zero_inflated=False,
               convert_to_tensor_fn=tfp.distributions.Distribution.sample,
               validate_args=False,
               **kwargs):
    super().__init__(
        lambda params: MixtureNegativeBinomial.new(
            params, event_shape, n_components,
            parse_activation(mean_activation, self),
            parse_activation(disp_activation, self), dispersion,
            alternative_parameterization, is_zero_inflated, validate_args),
        convert_to_tensor_fn, **kwargs)
    self.event_shape = event_shape
    self.n_components = n_components
    self.is_zero_inflated = is_zero_inflated

  @staticmethod
  def new(
      params,
      event_shape=(),
      n_components=2,
      mean_activation=tf.exp,
      disp_activation=lambda x: x,
      dispersion='full',
      alternative_parameterization=False,
      is_zero_inflated=False,
      validate_args=False,
  ):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    n_components = tf.convert_to_tensor(value=n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        [n_components],
        event_shape,
    ],
                             axis=0)
    mixture = tfp.distributions.Categorical(logits=params[..., :n_components])
    if is_zero_inflated:
      mean, disp, rate = tf.split(params[..., n_components:], 3, axis=-1)
      rate = tf.reshape(rate, output_shape)
    else:
      mean, disp = tf.split(params[..., n_components:], 2, axis=-1)
      rate = None
    mean = tf.reshape(mean, output_shape)
    disp = tf.reshape(disp, output_shape)

    if dispersion == 'single':
      disp = tf.reduce_mean(disp)
    elif dispersion == 'share':
      disp = tf.reduce_mean(disp,
                            axis=tf.range(0,
                                          output_shape.shape[0] - 1,
                                          dtype='int32'),
                            keepdims=True)
    mean = mean_activation(mean)
    disp = disp_activation(disp)

    NB = NegativeBinomialDisp if alternative_parameterization else \
      tfp.distributions.NegativeBinomial
    components = tfp.distributions.Independent(
        NB(mean, disp, validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        validate_args=validate_args)
    if is_zero_inflated:
      components = ZeroInflated(count_distribution=components,
                                logits=rate,
                                validate_args=False)
    return tfp.distributions.MixtureSameFamily(mixture,
                                               components,
                                               validate_args=False)

  def params_size(self):
    r"""Number of `params` needed to create a `MixtureNegativeBinomial`
    distribution.

    Returns:
     params_size: The number of parameters needed to create the mixture
       distribution.
    """
    n_components = tf.convert_to_tensor(value=self.n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    params_size = tf.convert_to_tensor(value=tf.reduce_prod(self.event_shape) *
                                       (3 if self.is_zero_inflated else 2),
                                       name='params_size')
    num_components = dist_util.prefer_static_value(n_components)
    params_size = dist_util.prefer_static_value(params_size)
    return num_components + num_components * params_size
