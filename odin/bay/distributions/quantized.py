from __future__ import absolute_import, division, print_function

from typing import Optional, Any
from typing_extensions import Literal

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.bijectors import Shift
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.distributions import (
    Categorical, Distribution, Independent, Logistic, MixtureSameFamily,
    NegativeBinomial, Normal, QuantizedDistribution, TransformedDistribution,
    Uniform, NOT_REPARAMETERIZED)
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

__all__ = [
    "PixelCNNpp",
    "MixtureQLogistic",
    "qUniform",
    "qNormal",
]


# ===========================================================================
# Mixture Quantized Logistic
# ===========================================================================
class PixelCNNpp(Distribution):
  """ PixelCNN++ Mixture of Quantized Logistic Distribution

  Builds a mixture of quantized logistic distributions.

  Note: this distribution assumes:

    - ELU activated `params`, and
    - The inputs to `log_prob` function could be sigmoid, tanh or pixel domain

  Parameters
  ----------
  params :
    component_logits: 4D `Tensor` of logits for the Categorical distribution
      over Quantized Logistic mixture components. Dimensions are `[batch_size,
      height, width, num_logistic_mix]`.
    locs: 4D `Tensor` of location parameters for the Quantized Logistic
      mixture components. Dimensions are `[batch_size, height, width,
      num_logistic_mix, num_channels]`.
    scales: 4D `Tensor` of location parameters for the Quantized Logistic
      mixture components. Dimensions are `[batch_size, height, width,
      num_logistic_mix, num_channels]`.

  Example
  -------
  ```
  pcnn = _PixelCNNNetwork(
      num_resnet=1,
      num_hierarchies=1,
      num_filters=32,
      num_logistic_mix=10,
      dropout_p=.3,
      use_weight_norm=False,
  )
  pcnn.build((None,) + image_shape)
  image_input = tf.keras.Input(shape=image_shape)
  x = keras.layers.Lambda(lambda x: 2. * x / 255. - 1.)(image_input)
  params = pcnn(x, training=True)
  dist = PixelCNNpp(params, inputs_domain='pixel')
  log_prob = dist.log_prob(image_input)
  ```

  """

  def __init__(
      self,
      params: tf.Tensor,
      n_components: int = 10,
      n_channels: int = 3,
      high: int = 255,
      low: int = 0,
      inputs_domain: Literal['sigmoid', 'tanh', 'pixel'] = 'sigmoid',
      dtype: Any = tf.float32,
      name: str = 'PixelCNNpp',
  ):
    parameters = dict(locals())
    super().__init__(dtype=dtype,
                     reparameterization_type=NOT_REPARAMETERIZED,
                     validate_args=False,
                     allow_nan_stats=True,
                     parameters=parameters,
                     name=name)
    self.inputs_domain = inputs_domain
    self.low = low
    self.high = high
    self.n_channels = n_channels
    self.n_components = n_components
    self.n_coeffs = n_channels * (n_channels - 1) // 2
    self.n_out = n_channels * 2 + self.n_coeffs + 1
    if isinstance(params, (tuple, list)):
      pass
    else:
      assert params.shape[-1] == self.n_out * n_components
      # prepare the parameters
      splits = (3 if n_channels == 1 else
                [1, n_channels, n_channels, self.n_coeffs])
      params = tf.convert_to_tensor(params, dtype=self.dtype)
      params = tf.reshape(
          params,
          tf.concat([tf.shape(params)[:-1], [n_components, self.n_out]],
                    axis=0))
      params = tf.split(params, splits, axis=-1)
      # Squeeze singleton dimension from component logits
      params[0] = tf.squeeze(params[0], axis=-1)
      # Ensure scales are positive and do not collapse to near-zero
      params[2] = tf.nn.softplus(params[2]) + tf.cast(tf.exp(-7.), self.dtype)
    self._params = params
    self.image_shape = list(params[0].shape[-3:-1]) + [n_channels]

  def transform_tanh(self, x):
    """ Transform the image [0, 255] to [-1, 1] """
    return (2. * (x - self.low) / (self.high - self.low)) - 1.

  def _log_prob(self, value: tf.Tensor):
    """ expect `value` is output from ELU function """
    params = self._params
    ## prepare the parameters
    if self.n_channels == 1:
      component_logits, locs, scales = params
    else:
      if self.inputs_domain == 'sigmoid':
        transformed_value = 2. * value - 1.
        value = value * self.high
      elif self.inputs_domain == 'pixel':
        transformed_value = self.transform_tanh(value)
      else:
        transformed_value = value
        value = (value + 1) / 2 * self.high
      channel_tensors = tf.split(transformed_value, self.n_channels, axis=-1)
      # If there is more than one channel, we create a linear autoregressive
      # dependency among the location parameters of the channels of a single
      # pixel (the scale parameters within a pixel are independent). For a pixel
      # with R/G/B channels, the `r`, `g`, and `b` saturation values are
      # distributed as:
      #
      # r ~ Logistic(loc_r, scale_r)
      # g ~ Logistic(coef_rg * r + loc_g, scale_g)
      # b ~ Logistic(coef_rb * r + coef_gb * g + loc_b, scale_b)
      component_logits, locs, scales, coeffs = params
      loc_tensors = tf.split(locs, self.n_channels, axis=-1)
      coef_tensors = tf.split(coeffs, self.n_coeffs, axis=-1)
      coef_count = 0
      for i in range(self.n_channels):
        channel_tensors[i] = channel_tensors[i][..., tf.newaxis, :]
        for j in range(i):
          loc_tensors[i] += channel_tensors[j] * coef_tensors[coef_count]
          coef_count += 1
      locs = tf.concat(loc_tensors, axis=-1)
    ## create the distrubtion
    mixture_distribution = Categorical(logits=component_logits)
    # Convert distribution parameters for pixel values in
    # `[self._low, self._high]` for use with `QuantizedDistribution`
    locs = self.low + 0.5 * (self.high - self.low) * (locs + 1.)
    scales = scales * 0.5 * (self.high - self.low)
    logistic_dist = QuantizedDistribution(
        distribution=TransformedDistribution(
            distribution=Logistic(loc=locs, scale=scales),
            bijector=Shift(shift=tf.cast(-0.5, self.dtype))),
        low=self.low,
        high=self.high,
    )
    dist = MixtureSameFamily(mixture_distribution=mixture_distribution,
                             components_distribution=Independent(
                                 logistic_dist, reinterpreted_batch_ndims=1))
    dist = Independent(dist, reinterpreted_batch_ndims=2)
    return dist.log_prob(value)

  def _mean(self, **kwargs):
    params = self._params
    if self.n_channels == 1:
      component_logits, locs, scales = params
    else:
      # r ~ Logistic(loc_r, scale_r)
      # g ~ Logistic(coef_rg * r + loc_g, scale_g)
      # b ~ Logistic(coef_rb * r + coef_gb * g + loc_b, scale_b)
      component_logits, locs, scales, coeffs = params
      loc_tensors = tf.split(locs, self.n_channels, axis=-1)
      coef_tensors = tf.split(coeffs, self.n_coeffs, axis=-1)
      coef_count = 0
      for i in range(self.n_channels):
        for j in range(i):
          loc_tensors[i] += loc_tensors[j] * coef_tensors[coef_count]
          coef_count += 1
      locs = tf.concat(loc_tensors, axis=-1)
    ## create the distrubtion
    mixture_distribution = Categorical(logits=component_logits)
    # Convert distribution parameters for pixel values in
    # `[self._low, self._high]` for use with `QuantizedDistribution`
    locs = self.low + 0.5 * (self.high - self.low) * (locs + 1.)
    scales = scales * 0.5 * (self.high - self.low)
    logistic_dist = TransformedDistribution(
        distribution=Logistic(loc=locs, scale=scales),
        bijector=Shift(shift=tf.cast(-0.5, self.dtype)))
    dist = MixtureSameFamily(mixture_distribution=mixture_distribution,
                             components_distribution=Independent(
                                 logistic_dist, reinterpreted_batch_ndims=1))
    mean = Independent(dist, reinterpreted_batch_ndims=2).mean()
    ## normalize the data back to input domain
    if self.inputs_domain == 'sigmoid':
      mean = mean / self.high
    elif self.inputs_domain == 'tanh':
      mean = 2. * mean / self.high - 1.
    return mean



  def _sample_n(self, n, seed=None, conditional_input=None, training=False):
    # TODO
    shape = ps.concat([[n], self.image_shape], axis=0)
    return tf.random.uniform(shape=shape, dtype=self.dtype)

  def _batch_shape(self):
    return tf.TensorShape([])

  def _event_shape(self):
    return tf.TensorShape(self.image_shape)


def MixtureQLogistic(
    locs: tf.Tensor,
    scales: tf.Tensor,
    logits: Optional[tf.Tensor] = None,
    probs: Optional[tf.Tensor] = None,
    batch_ndims: int = 0,
    low: int = 0,
    bits: int = 8,
    name: str = 'MixtureQuantizedLogistic') -> MixtureSameFamily:
  """ Mixture of quantized logistic distribution

  Parameters
  ----------
  locs : tf.Tensor
      locs of all logistics components, shape `[batch_size, n_components, event_size]`
  scales : tf.Tensor
      scales of all logistics components, shape `[batch_size, n_components, event_size]`
  logits, probs : tf.Tensor
      probability for the mixture Categorical distribution, shape `[batch_size, n_components]`
  low : int, optional
      minimum quantized value, by default 0
  bits : int, optional
      number of bits for quantization, the maximum will be `2^bits - 1`, by default 8
  name : str, optional
      distribution name, by default 'MixtureQuantizedLogistic'

  Returns
  -------
  MixtureSameFamily
      the mixture of quantized logistic distribution

  Example
  -------
  ```
  d = MixtureQLogistic(np.ones((12, 3, 8)).astype('float32'),
                       np.ones((12, 3, 8)).astype('float32'),
                       logits=np.random.rand(12, 3).astype('float32'),
                       batch_ndims=1)
  ```

  Reference
  ---------
  Salimans, T., Karpathy, A., Chen, X., Kingma, D.P., 2017.
    PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture
    Likelihood and Other Modifications. arXiv:1701.05517 [cs, stat].

  """
  cats = Categorical(probs=probs, logits=logits)
  dists = Logistic(loc=locs, scale=scales)
  dists = TransformedDistribution(distribution=dists,
                                  bijector=Shift(shift=-0.5))
  dists = QuantizedDistribution(dists, low=low, high=2**bits - 1.)
  dists = Independent(dists, reinterpreted_batch_ndims=batch_ndims)
  dists = MixtureSameFamily(mixture_distribution=cats,
                            components_distribution=dists,
                            name=name)
  return dists


# ===========================================================================
# Others
# ===========================================================================
class qNormal(QuantizedDistribution):

  def __init__(self,
               loc=0.,
               scale=1.,
               min_value=None,
               max_value=None,
               validate_args=False,
               allow_nan_stats=True,
               name="qNormal"):
    super(qNormal,
          self).__init__(distribution=Normal(loc=loc,
                                             scale=scale,
                                             validate_args=validate_args,
                                             allow_nan_stats=allow_nan_stats),
                         low=min_value,
                         high=max_value,
                         name=name)


class qUniform(QuantizedDistribution):

  def __init__(self,
               low=0.,
               high=1.,
               min_value=None,
               max_value=None,
               validate_args=False,
               allow_nan_stats=True,
               name="qUniform"):
    super(qUniform,
          self).__init__(distribution=Uniform(low=low,
                                              high=high,
                                              validate_args=validate_args,
                                              allow_nan_stats=allow_nan_stats),
                         low=min_value,
                         high=max_value,
                         name=name)
