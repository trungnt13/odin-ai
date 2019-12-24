from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.internal import \
    distribution_util as dist_util
from tensorflow_probability.python.layers.distribution_layer import _event_size

from odin.bay.distributions import ZeroInflated

__all__ = [
    'OneHotCategoricalLayer', 'CategoricalLayer', 'RelaxedSoftmaxLayer',
    'RelaxedBernoulliLayer', 'BernoulliLayer', 'ZIBernoulliLayer'
]


class BernoulliLayer(tfl.DistributionLambda):
  r"""An Independent-Bernoulli Keras layer from `prod(event_shape)` params.

  Typical choices for `convert_to_tensor_fn` include:

  - `tfd.Distribution.sample`
  - `tfd.Distribution.mean`
  - `tfd.Distribution.mode`
  - `tfd.Bernoulli.logits`

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
      instance and returns a `tf.Tensor`-like object. For examples, see
      `class` docstring.
      Default value: `tfd.Distribution.sample`.
    sample_dtype: `dtype` of samples produced by this distribution.
      Default value: `None` (i.e., previous layer's `dtype`).
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
      Default value: `False`.
    **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               sample_dtype=None,
               validate_args=False,
               **kwargs):
    # If there is a 'make_distribution_fn' keyword argument (e.g., because we
    # are being called from a `from_config` method), remove it.  We pass the
    # distribution function to `DistributionLambda.__init__` below as the first
    # positional argument.
    kwargs.pop('make_distribution_fn', None)
    super().__init__(
        lambda t: BernoulliLayer.new(t, event_shape, sample_dtype, validate_args
                                    ), convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params, event_shape=(), dtype=None, validate_args=False, name=None):
    r"""Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype_hint=tf.int32),
                                             tensor_name='event_shape')
    new_shape = tf.concat([
        tf.shape(input=params)[:-tf.size(event_shape)],
        event_shape,
    ],
                          axis=0)
    dist = tfd.Independent(tfd.Bernoulli(logits=tf.reshape(params, new_shape),
                                         dtype=dtype or params.dtype.base_dtype,
                                         validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args)
    dist._logits = dist.distribution._logits
    dist._probs = dist.distribution._probs
    dist.logits = tfd.Bernoulli.logits
    dist.probs = tfd.Bernoulli.probs
    return dist

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype_hint=tf.int32)
    return _event_size(event_shape, name=name or 'BernoulliLayer_params_size')


class ZIBernoulliLayer(tfl.DistributionLambda):
  r"""A Independent zero-inflated bernoulli keras layer

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
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
               given_logits=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(ZIBernoulliLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, given_logits, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          given_logits=True,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-tf.size(event_shape)],
        event_shape,
    ],
                             axis=0)
    (bernoulli_params, rate_params) = tf.split(params, 2, axis=-1)
    bernoulli_params = tf.reshape(bernoulli_params, output_shape)
    bern = tfd.Bernoulli(logits=bernoulli_params if given_logits else None,
                         probs=bernoulli_params if not given_logits else None,
                         validate_args=validate_args)
    zibern = ZeroInflated(count_distribution=bern,
                          logits=tf.reshape(rate_params, output_shape),
                          validate_args=validate_args)
    return tfd.Independent(zibern,
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 2 * _event_size(event_shape,
                           name=name or 'ZeroInflatedBernoulli_params_size')


class CategoricalLayer(tfl.DistributionLambda):

  def __init__(self,
               event_size=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               probs_input=False,
               sample_dtype=None,
               validate_args=False,
               **kwargs):
    super(CategoricalLayer, self).__init__(
        lambda t: type(self).new(t, probs_input, sample_dtype, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params, probs_input=False, dtype=None, validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    return tfd.Categorical(
        logits=params if not probs_input else None,
        probs=tf.clip_by_value(params, 1e-8, 1 - 1e-8) \
          if probs_input else None,
        dtype=dtype or params.dtype,
        validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """The number of `params` needed to create a single distribution."""
    return event_size


class OneHotCategoricalLayer(tfl.DistributionLambda):
  r""" A `d`-variate OneHotCategorical Keras layer from `d` params.
  a.k.a. Multinoulli distribution is a generalization of the Bernoulli
  distribution

  Arguments:
    convert_to_tensor_fn: Callable. Function takes a `tfd.Distribution`
      instance and returns a `tf.Tensor`-like object. For examples,
      see `class` docstring.
      Default value: `tfd.Distribution.sample`.
    probs_input : `boolean`. Indicate the input is probability value or logit
      value.
    sample_dtype: `dtype`. Type of samples produced by this distribution.
      Default value: `None` (i.e., previous layer's `dtype`).
    validate_args: `bool` (default `False`). When `True` distribution parameters
      are checked for validity despite possibly degrading runtime performance.
      When `False` invalid inputs may silently render incorrect outputs.
      Default value: `False`.

  **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.

  Note
  ----
  If input as probability values is given, it will be clipped by value
  [1e-8, 1 - 1e-8]

  """

  def __init__(self,
               event_size=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               probs_input=False,
               sample_dtype=None,
               validate_args=False,
               **kwargs):
    super(OneHotCategoricalLayer, self).__init__(
        lambda t: type(self).new(t, probs_input, sample_dtype, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params, probs_input=False, dtype=None, validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    return tfd.OneHotCategorical(
        logits=params if not probs_input else None,
        probs=tf.clip_by_value(params, 1e-8, 1 - 1e-8) \
          if probs_input else None,
        dtype=dtype or params.dtype,
        validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """The number of `params` needed to create a single distribution."""
    return event_size


# ===========================================================================
# Continous approximation
# ===========================================================================
class RelaxedSoftmaxLayer(tfl.DistributionLambda):
  r""" The RelaxedOneHotCategorical is a distribution over random probability
  vectors, vectors of positive real values that sum to one, which continuously
  approximates a OneHotCategorical. The degree of approximation is controlled by
  a temperature: as the temperature goes to 0 the RelaxedOneHotCategorical
  becomes discrete with a distribution described by the `logits` or `probs`
  parameters, as the temperature goes to infinity the RelaxedOneHotCategorical
  becomes the constant distribution that is identically the constant vector of
  (1/event_size, ..., 1/event_size).

  Arguments:
    temperature: An 0-D `Tensor`, representing the temperature
      of a set of RelaxedOneHotCategorical distributions. The temperature
      should be positive.
    probs_input : A `boolean` indicates the input is probability value or logit
      value.

  References:
    Eric J., et al., 2016. Categorical Reparameterization with
      Gumbel-Softmax.
    Chris J.M., et al., 2016. The Concrete Distribution: A Continuous
      Relaxation of Discrete Random Variables.
  """

  def __init__(self,
               event_size=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               temperature=0.5,
               probs_input=False,
               validate_args=False,
               **kwargs):
    super().__init__(make_distribution_fn=lambda t: RelaxedSoftmaxLayer.new(
        params=t,
        temperature=temperature,
        probs_input=probs_input,
        validate_args=validate_args),
                     convert_to_tensor_fn=convert_to_tensor_fn,
                     **kwargs)

  @staticmethod
  def new(params,
          temperature,
          probs_input=False,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    return tfd.RelaxedOneHotCategorical(temperature=temperature,
                                        logits=None if probs_input else params,
                                        probs=params if probs_input else None,
                                        validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """The number of `params` needed to create a single distribution."""
    return event_size


class RelaxedBernoulliLayer(tfl.DistributionLambda):
  r"""An Independent-Relaxed-Bernoulli Keras layer from `prod(event_shape)`
  params.

  The RelaxedBernoulli is a distribution over the unit interval (0,1), which
  continuously approximates a Bernoulli. The degree of approximation is
  controlled by a temperature: as the temperature goes to 0 the
  RelaxedBernoulli becomes discrete with a distribution described by the
  `logits` or `probs` parameters, as the temperature goes to infinity the
  RelaxedBernoulli becomes the constant distribution that is identically 0.5.

  The RelaxedBernoulli distribution is a reparameterized continuous
  distribution that is the binary special case of the RelaxedOneHotCategorical
  distribution (Maddison et al., 2016; Jang et al., 2016). For details on the
  binary special case see the appendix of Maddison et al. (2016) where it is
  referred to as BinConcrete. If you use this distribution, please cite both
  papers.

  Some care needs to be taken for loss functions that depend on the
  log-probability of RelaxedBernoullis, because computing log-probabilities of
  the RelaxedBernoulli can suffer from underflow issues. In many case loss
  functions such as these are invariant under invertible transformations of
  the random variables. The KL divergence, found in the variational autoencoder
  loss, is an example. Because RelaxedBernoullis are sampled by a Logistic
  random variable followed by a `tf.sigmoid` op, one solution is to treat
  the Logistic as the random variable and `tf.sigmoid` as downstream. The
  KL divergences of two Logistics, which are always followed by a `tf.sigmoid`
  op, is equivalent to evaluating KL divergences of RelaxedBernoulli samples.
  See Maddison et al., 2016 for more details where this distribution is called
  the BinConcrete.

  An alternative approach is to evaluate Bernoulli log probability or KL
  directly on relaxed samples, as done in Jang et al., 2016. In this case,
  guarantees on the loss are usually violated. For instance, using a Bernoulli
  KL in a relaxed ELBO is no longer a lower bound on the log marginal
  probability of the observation. Thus care and early stopping are important.

  Arguments:
    temperature: An 0-D `Tensor`, representing the temperature
      of a set of RelaxedBernoulli distributions. The temperature should be
      positive.
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    probs_input : A `boolean` indicates the input is probability value or logit
      value.

  References:
    Eric J., et al., 2016. Categorical Reparameterization with
      Gumbel-Softmax.
    Chris J.M., et al., 2016. The Concrete Distribution: A Continuous
      Relaxation of Discrete Random Variables.
  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               temperature=0.5,
               probs_input=False,
               validate_args=False,
               **kwargs):
    super().__init__(make_distribution_fn=lambda t: RelaxedBernoulliLayer.new(
        params=t,
        temperature=temperature,
        probs_input=probs_input,
        event_shape=event_shape,
        validate_args=validate_args),
                     convert_to_tensor_fn=convert_to_tensor_fn,
                     **kwargs)

  @staticmethod
  def new(params,
          temperature,
          probs_input=False,
          event_shape=(),
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype_hint=tf.int32),
                                             tensor_name='event_shape')
    new_shape = tf.concat([
        tf.shape(input=params)[:-tf.size(event_shape)],
        event_shape,
    ],
                          axis=0)
    params = tf.reshape(params, new_shape)
    dist = tfd.Independent(tfd.RelaxedBernoulli(
        temperature=temperature,
        logits=None if probs_input else params,
        probs=params if probs_input else None,
        validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args)
    return dist

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype_hint=tf.int32)
    return _event_size(event_shape,
                       name=name or 'RelaxedBernoulliLayer_params_size')
