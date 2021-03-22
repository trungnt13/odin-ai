from typing import Any, Optional, Union
from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.bijectors import Shift
from tensorflow_probability.python.distributions import (
  Categorical, Independent, MixtureSameFamily, Normal, TransformedDistribution,
  Uniform, QuantizedDistribution, Logistic, Distribution, NOT_REPARAMETERIZED)
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from typing_extensions import Literal

__all__ = [
  "QuantizedLogistic",
  "MixtureQuantizedLogistic",
  "MixtureQLogistic",
  "qUniform",
  "qNormal",
]


# ===========================================================================
# Quantized Logistic
# ===========================================================================
def _switch_domain(value, inputs_domain, low, high):
  if inputs_domain == 'sigmoid':
    transformed_value = 2. * value - 1.
    value = value * high
  elif inputs_domain == 'pixel':
    # Transform the image [0, 255] to [-1, 1]
    transformed_value = (2. * (value - low) / (high - low)) - 1.
  elif inputs_domain == 'tanh':
    transformed_value = value
    value = (value + 1) / 2 * high
  else:
    raise NotImplementedError
  return transformed_value, value


def _pixels_to(x, inputs_domain, low, high):
  """normalize the pixels data back to the input domain"""
  if inputs_domain == 'sigmoid':
    x = (x - low) / high
  elif inputs_domain == 'tanh':
    x = 2. * (x - low) / high - 1.
  return x


class QuantizedLogistic(Distribution):
  """ PixelCNN Quantized Logistic Distribution

  Builds a mixture of quantized logistic distributions.

  The inputs to this distribution should be logits values

  Note: this distribution assumes the input and output value is pixel value
  (i.e. [0, 255]), this could be changed by `inputs_domain` argument.

  Parameters
  ----------
  loc: Floating point tensor
      the means of the distribution(s).
  scale: Floating point tensor
      the scales of the distribution(s). Must contain only positive values.
  low: `Tensor` with same `dtype` as this distribution and shape
      that broadcasts to that of samples but does not result in additional
      batch dimensions after broadcasting. Should be a whole number. Default
      `0`. If provided, base distribution's `prob` should be defined at
      `low`. For example, a pixel take value `0`.
  high: `Tensor` with same `dtype` as this distribution and shape
      that broadcasts to that of samples but does not result in additional
      batch dimensions after broadcasting. Should be a whole number. Default
      `255`. If provided, base distribution's `prob` should be defined at
      `high - 1`. `high` must be strictly greater than `low`. For example,
      a pixel take value `2**8 - 1`.
  """

  def __init__(self,
               loc: Union[tf.Tensor, np.ndarray],
               scale: Union[tf.Tensor, np.ndarray],
               low: Union[None, Number] = 0,
               high: Union[None, Number] = 2 ** 8 - 1,
               inputs_domain: Literal['sigmoid', 'tanh', 'pixel'] = 'sigmoid',
               reinterpreted_batch_ndims: Optional[int] = None,
               validate_args: bool = False,
               allow_nan_stats: bool = True,
               name: str = 'QuantizedLogistic'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale, low, high],
                                      dtype_hint=tf.float32)
      self._low = low
      self._high = high
      # Convert distribution parameters for pixel values in
      # `[self._low, self._high]` for use with `QuantizedDistribution`
      if low is not None and high is not None:
        support = 0.5 * (high - low)
        loc = low + support * (loc + 1.)
        scale = scale * support
      self._logistic = Logistic(loc=loc, scale=scale,
                                validate_args=validate_args,
                                allow_nan_stats=allow_nan_stats,
                                name=name)
      self._dist = QuantizedDistribution(
        distribution=TransformedDistribution(
          distribution=self._logistic,
          bijector=Shift(tf.cast(-0.5, dtype=dtype))),
        low=low,
        high=high,
        validate_args=validate_args,
        name=name)
      if reinterpreted_batch_ndims is not None:
        self._dist = Independent(
          self._dist,
          reinterpreted_batch_ndims=reinterpreted_batch_ndims)
      self.inputs_domain = inputs_domain
      super(QuantizedLogistic, self).__init__(
        dtype=dtype,
        reparameterization_type=NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @property
  def distribution(self) -> QuantizedDistribution:
    """Base Quantized distribution"""
    return self._dist

  @property
  def logistic(self) -> Logistic:
    """Base Logistic distribution"""
    return self._logistic

  @property
  def low(self) -> Optional[tf.Tensor]:
    """Lowest value that quantization returns."""
    return self._low

  @property
  def high(self) -> Optional[tf.Tensor]:
    """Highest value that quantization returns."""
    return self._high

  @property
  def loc(self) -> tf.Tensor:
    """Distribution parameter for the location."""
    return self.logistic.loc

  @property
  def scale(self) -> tf.Tensor:
    """Distribution parameter for scale."""
    return self.logistic.scale

  def _batch_shape_tensor(self):
    return self.distribution.batch_shape_tensor()

  def _batch_shape(self):
    return self.distribution.batch_shape

  def _event_shape_tensor(self):
    return self.distribution.event_shape_tensor()

  def _event_shape(self):
    return self.distribution.event_shape

  def _log_prob(self, y):
    _, y = _switch_domain(y, self.inputs_domain, low=self.low, high=self.high)
    return self.distribution._log_prob(y)

  def _prob(self, y):
    return self.distribution._prob(y)

  def _log_cdf(self, y, low=None, high=None):
    return self.distribution._log_cdf(y, low=low, high=high)

  def _cdf(self, y, low=None, high=None):
    return self.distribution._cdf(y, low=low, high=high)

  def _sample_n(self, n, seed=None):
    x = self.distribution._sample_n(n=n, seed=seed)
    return _pixels_to(x, self.inputs_domain, self.low, self.high)

  def _mean(self):
    x = self.logistic._mean()
    return _pixels_to(x, self.inputs_domain, self.low, self.high)

  def _entropy(self):
    return self._logistic._entropy()

  def _stddev(self):
    return self._logistic._stddev()

  def _mode(self):
    return self._logistic._mode()

  def _z(self, x):
    """Standardize input `x` to a unit logistic."""
    return self._logistic._z(x)

  def _quantile(self, x):
    return self._logistic._quantile(x)


class MixtureQuantizedLogistic(Distribution):
  """ PixelCNN++ Mixture of Quantized Logistic Distribution

  The inputs to this distribution should be logits values

  Parameters
  ----------
  params : Tensor
    Concatenation of three parameters:

    - component_logits: 4D `Tensor` of logits for the Categorical distribution
        over Quantized Logistic mixture components. Dimensions are `[batch_size,
        height, width, num_logistic_mix]`.
    - locs: 4D `Tensor` of location parameters for the Quantized Logistic
        mixture components. Dimensions are `[batch_size, height, width,
        num_logistic_mix, num_channels]`.
    - scales: 4D `Tensor` of location parameters for the Quantized Logistic
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
  dist = MixtureQuantizedLogistic(params, inputs_domain='pixel')
  log_prob = dist.log_prob(image_input)
  ```

  """

  def __init__(
      self,
      params: tf.Tensor,
      n_components: int = 10,
      n_channels: int = 3,
      low: int = 0,
      high: int = 255,
      inputs_domain: Literal['sigmoid', 'tanh', 'pixel'] = 'sigmoid',
      dtype: Any = tf.float32,
      name: str = 'MixtureQuantizedLogistic',
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
      assert params.shape[-1] == self.n_out * n_components, \
        (f'Mixture of Quantized Logistic require {n_components} components of '
         f'{self.n_out} each, but given inputs with shape: {params.shape}')
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

  @staticmethod
  def params_size(n_components: int, n_channels: int) -> int:
    n_coeffs = n_channels * (n_channels - 1) // 2
    n_out = n_channels * 2 + n_coeffs + 1
    return int(n_out * n_components)

  def _log_prob(self, value: tf.Tensor):
    """ expect `value` is output from ELU function """
    params = self._params
    transformed_value, value = _switch_domain(
      value,
      inputs_domain=self.inputs_domain,
      low=self.low,
      high=self.high)
    ## prepare the parameters
    if self.n_channels == 1:
      component_logits, locs, scales = params
    else:
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
    ## normalize the data back to the input domain
    return _pixels_to(mean, self.inputs_domain, self.low, self.high)

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
  dists = TransformedDistribution(
    distribution=dists,
    bijector=Shift(shift=tf.cast(-0.5, dists.dtype)))
  dists = QuantizedDistribution(dists, low=low, high=2 ** bits - 1.)
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
