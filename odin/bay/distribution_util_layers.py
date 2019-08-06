from __future__ import absolute_import, division, print_function

from typing import Optional, Type, Union

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import Dense, Lambda, Layer
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible as dtc

from odin.bay.helpers import Statistic


class ReduceMean(Layer):
  """ ReduceMean
  """

  def __init__(self, axis=None, keepdims=None, **kwargs):
    super(ReduceMean, self).__init__(**kwargs)
    self.axis = axis
    self.keepdims = keepdims

  def call(self, x):
    return [tf.reduce_mean(i, axis=self.axis, keepdims=self.keepdims) for i in x] \
    if isinstance(x, (tuple, list)) else \
    tf.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)


class Sampling(Layer):
  """ Sample the output from tensorflow-probability
  distribution layers """

  def __init__(self, n_samples=None, **kwargs):
    super(Sampling, self).__init__(**kwargs)
    self.n_samples = n_samples

  def call(self, x, n_samples=-1):
    assert isinstance(x, Distribution), \
    "Input to this layer must be instance of tensorflow_probability Distribution"
    if n_samples < 0:
      n_samples = self.n_samples
    return x.sample() if n_samples is None else x.sample(n_samples)


class Moments(Layer):
  """ Moments """

  def __init__(self, mean=True, variance=True, **kwargs):
    super(Moments, self).__init__(**kwargs)
    self.mean = bool(mean)
    self.variance = bool(variance)
    assert self.mean or self.variance, "This layer must return mean or variance"

  def call(self, x):
    assert isinstance(x, Distribution), \
    "Input to this layer must be instance of tensorflow_probability Distribution"
    outputs = []
    if self.mean:
      outputs.append(x.mean())
    if self.variance:
      outputs.append(x.variance())
    return outputs[0] if len(outputs) == 1 else tuple(outputs)

  def compute_output_shape(self, input_shape):
    return [input_shape, input_shape
           ] if self.mean and self.variance else input_shape


class Stddev(Layer):

  def __init__(self, **kwargs):
    super(Stddev, self).__init__(**kwargs)

  def call(self, x):
    assert isinstance(x, Distribution), \
    "Input to this layer must be instance of tensorflow_probability Distribution"
    return x.stddev()

  def compute_output_shape(self, input_shape):
    return input_shape


class GetDistributionAttr(Layer):
  """ GetAttr """

  def __init__(self,
               attr_name,
               convert_to_tensor_fn=Distribution.sample,
               **kwargs):
    super(GetDistributionAttr, self).__init__(**kwargs)
    self.attr_name = str(attr_name)
    if isinstance(convert_to_tensor_fn, property):
      convert_to_tensor_fn = convert_to_tensor_fn.fget
    self.convert_to_tensor_fn = convert_to_tensor_fn

  def call(self, x):
    attrs = self.attr_name.split('.')
    for a in attrs:
      x = getattr(x, a)
    # special case a distribution is returned
    if isinstance(x, Distribution) and not isinstance(x, dtc._TensorCoercible):
      dist = dtc._TensorCoercible(
          distribution=x, convert_to_tensor_fn=self.convert_to_tensor_fn)
      value = tf.convert_to_tensor(value=dist)
      value._tfp_distribution = dist
      dist.shape = value.shape
      dist.get_shape = value.get_shape
      x = dist
    return x


class DistributionLayer(Model):
  """ DistributionLayer

  Parameters
  ----------
  n_units : int
    number of output units.
  posterior : `tensorflow_probability.DistributionLambda`
    posterior distribution, the class is given for later
    initialization
  prior : {`None`, `tensorflow_probability.Distribution`}
    prior distribution, used for calculating KL divergence later.
  use_bias : `bool` (default=`True`)

  call_mode : `odin.bay.helpers.Statistic` (default=Statistic.SAMPLE)

  name : `str` (default='DistributionLayer')

  Return
  ------
  sample, mean, variance, stddev : [n_samples, batch_size, n_units]
    depend on the `call_mode`, multiple statistics could be returned
  """

  def __init__(self,
               n_units,
               posterior: Union[DistributionLambda, Type[DistributionLambda]],
               prior: Optional[Distribution] = None,
               use_bias=True,
               call_mode: Statistic = Statistic.SAMPLE,
               name="DistributionLayer"):
    super(DistributionLayer, self).__init__(name=name)
    assert isinstance(posterior, DistributionLambda) or\
       (isinstance(posterior, type) and issubclass(posterior, DistributionLambda)),\
         "posterior must be instance or subclass of DistributionLambda"
    self._n_units = int(n_units)
    layers = [
        Dense(posterior.params_size(self.n_units),
              activation='linear',
              use_bias=bool(use_bias)),
        posterior if isinstance(posterior, DistributionLambda) else posterior(
            self.n_units),
    ]
    if isinstance(posterior, DistributionLambda):
      distribution_type = type(posterior)
    else:
      distribution_type = posterior
    self._distribution = Sequential(layers,
                                    name="%s%s" %
                                    (name, distribution_type.__name__))
    self._last_distribution = None
    # check the prior, this could be given later
    assert prior is None or isinstance(prior, Distribution), \
      "prior can be None or instance of tensorflow_probability.Distribution"
    self._prior = prior
    # statistics extraction layers
    self._fn_mean = Moments(mean=True, variance=False)
    self._fn_var = Moments(mean=False, variance=True)
    # call mode decide what to return in `Layer.call` method
    assert isinstance(call_mode, Statistic), \
      "call_mode must be instance of odin.bay.helpers.Statistic"
    self._call_mode = call_mode

  @property
  def prior(self):
    return self._prior

  @property
  def posterior(self):
    return self._last_distribution

  @property
  def n_units(self):
    return self._n_units

  def __apply_distribution(self, x):
    if hasattr(x, '_distribution') and \
      x._distribution == self._last_distribution:
      dist = x._distribution
    else:
      dist = self._distribution(x)
    return dist

  def mean(self, x):
    dist = self.__apply_distribution(x)
    y = self._fn_mean(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def variance(self, x):
    dist = self.__apply_distribution(x)
    y = self._fn_var(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def stddev(self, x):
    v = self.variance(x)
    y = Lambda(tf.math.sqrt)(v)
    setattr(y, '_distribution', v._distribution)
    return y

  def sample(self, x, n_samples=1):
    if n_samples is None or n_samples <= 0:
      n_samples = 1
    dist = self.__apply_distribution(x)
    y = Sampling(n_samples=n_samples)(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def build(self, input_shape):
    self._distribution.build(input_shape)
    return super(DistributionLayer, self).build(input_shape)

  def call(self, x, training=None, n_samples=1, mode=None):
    """
    Parameters
    ----------
    x : {`numpy.ndarray`, `tensorflow.Tensor`}

    training : {`None`, `bool`} (default=`None`)

    n_samples : {`None`, `int`} (default=`1`)

    mode : {`None`, `odin.bay.helpers.Statistic`} (default=`None`)
      decide which of the statistics will be return from the distribution,
      this value will overide the default value of the class
    """
    dtype = tuple(set([w.dtype for w in self._distribution.weights]))[0]
    if x.dtype != dtype:
      raise RuntimeError(
          "Given input with %s, but the layers were created with %s" %
          (str(x.dtype), str(dtype)))

    results = []
    variance = None
    call_mode = self._call_mode if not isinstance(mode, Statistic) else mode
    # special case only need the distribution
    if Statistic.DIST == call_mode:
      dist = self._distribution(x)
      self._last_distribution = dist
      return dist

    if Statistic.SAMPLE in call_mode:
      results.append(self.sample(x, n_samples=n_samples))
    if Statistic.MEAN in call_mode:
      results.append(self.mean(x))
    if Statistic.VAR in call_mode or Statistic.STDDEV in call_mode:
      variance = self.variance(x)
      if Statistic.VAR in call_mode:
        results.append(variance)
      if Statistic.STDDEV in call_mode:
        y = Lambda(tf.math.sqrt)(variance)
        setattr(y, '_distribution', variance._distribution)
        results.append(y)
    if Statistic.DIST in call_mode:
      assert len(results) > 0
      results.append(results[0]._distribution)
    return results[0] if len(results) == 1 else tuple(results)

  def kl_divergence(self, prior=None, analytic_kl=True, n_samples=1):
    """
    Parameters
    ---------
    prior : instance of `tensorflow_probability.Distribution`
      prior distribution of the latent

    analytic_kl : `bool` (default=`True`)
      using closed form solution for calculating divergence,
      otherwise, sampling with MCMC

    n_samples : `int` (default=`1`)
      number of MCMC sample if `analytic_kl=False`

    Return
    ------
    kullback_divergence : Tensor [n_samples, batch_size, ...]
    """
    from odin.bay.helpers import kl_divergence
    if n_samples is None:
      n_samples = 1
    if prior is None:
      prior = self._prior
    assert isinstance(prior, Distribution), "prior is not given!"
    if self.posterior is None:
      raise RuntimeError(
          "DistributionLayer must be called to create the distribution before "
          "calculating the kl-divergence.")
    kullback_div = kl_divergence(q=self.posterior,
                                 p=prior,
                                 use_analytic_kl=bool(analytic_kl),
                                 q_sample=int(n_samples),
                                 auto_remove_independent=True)
    if analytic_kl:
      kullback_div = tf.expand_dims(kullback_div, axis=0)
      if n_samples > 1:
        ndims = kullback_div.shape.ndims
        kullback_div = tf.tile(kullback_div, [n_samples] + [1] * (ndims - 1))
    return kullback_div

  def log_prob(self, x):
    assert self.n_units == x.shape[-1], \
      "Number of features mismatch, n_units=%d  input_shape=%s" % \
        (self.n_units, str(x.shape))
    if self._last_distribution is None:
      raise RuntimeError(
          "DistributionLayer must be called to create the distribution before "
          "calculating the log-likelihood.")
    dist = self._last_distribution
    return dist.log_prob(x)
