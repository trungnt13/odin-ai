"""The Alternative parameterization for Negative Binomial distribution class."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import (assert_util,
                                                    distribution_util,
                                                    dtype_util,
                                                    reparameterization,
                                                    tensor_util)
from tensorflow_probability.python.util.seed_stream import SeedStream


class NegativeBinomialDisp(distribution.Distribution):
  r"""Alternate parameterization for NegativeBinomial distribution using
  mean and dispersion as often mentioned in infobiomatic literatures.

  The NegativeBinomial distribution is related to the experiment of performing
  Bernoulli trials in sequence. Given a Bernoulli trial with probability `p` of
  success, the NegativeBinomial distribution represents the distribution over
  the number of successes `s` that occur until we observe `f` failures.

  The probability mass function (pmf) is,

  ```none
  pmf(s; f, p) = p**s (1 - p)**f / Z
  Z = s! (f - 1)! / (s + f - 1)!
  ```

  where:
  * `total_count = f`,
  * `probs = p`,
  * `Z` is the normalizaing constant, and,
  * `n!` is the factorial of `n`.
  """

  def __init__(self,
               loc,
               disp,
               eps=1e-7,
               validate_args=False,
               allow_nan_stats=True,
               name="NegativeBinomialDisp"):
    r"""Construct NegativeBinomial distributions.

    Args:
      loc: Non-negative floating-point `Tensor` with shape
        broadcastable to `[B1,..., Bb]` with `b >= 0` and the same dtype as
        `probs` or `logits`. Defines this as a batch of `N1 x ... x Nm`
        different Negative Binomial distributions. In practice, this represents
        the number of negative Bernoulli trials to stop at (the `total_count`
        of failures), but this is still a valid distribution when
        `total_count` is a non-integer.
      disp: Positive floating-point `Tensor` with shape broadcastable to
        `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
        Each entry represents logits for the probability of success for
        independent Negative Binomial distributions and must be in the open
        interval `(-inf, inf)`. Only one of `logits` or `probs` should be
        specified. (also known as `theta`)
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """

    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, disp], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(value=loc,
                                                       name="loc",
                                                       dtype_hint=dtype)
      self._disp = tensor_util.convert_nonref_to_tensor(value=disp,
                                                        name="disp",
                                                        dtype_hint=dtype)
      self._eps = tensor_util.convert_nonref_to_tensor(eps,
                                                       name="eps",
                                                       dtype_hint=dtype)
    super(NegativeBinomialDisp, self).__init__(
        dtype=self._loc.dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, disp=0)

  @property
  def loc(self):
    r""" Mean."""
    return self._loc

  @property
  def disp(self):
    r""" Dispersion"""
    return self._disp

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(tf.shape(input=self.loc),
                                      tf.shape(input=self.disp))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.loc.shape, self.disp.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    # Here we use the fact that if:
    # p = loc / (loc + disp)
    # r = disp
    # concentration = r
    # rate = (1-p)/p
    # lam ~ Gamma(concentration, rate
    # then X ~ Poisson(lam) is Negative Binomially distributed.
    stream = SeedStream(seed, salt="NegativeBinomialDisp")
    p = self.loc / (self.loc + self.disp)
    r = self.disp
    concentration = r
    rate = (1 - p) / p
    rate = tf.random.gamma(shape=[n],
                           alpha=concentration,
                           beta=rate,
                           dtype=self.dtype,
                           seed=stream())
    return tf.random.poisson(lam=rate,
                             shape=[],
                             dtype=self.dtype,
                             seed=stream())

  def _cdf(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    raise NotImplementedError()

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization(x)

  def _log_unnormalized_prob(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    log_loc_disp_eps = tf.math.log(self.disp + self.loc + self._eps)
    return self.disp * (tf.math.log(self.disp + self._eps) - log_loc_disp_eps) \
            + x * (tf.math.log(self.loc + self._eps) - log_loc_disp_eps)

  def _log_normalization(self, x):
    if self.validate_args:
      x = distribution_util.embed_check_nonnegative_integer_form(x)
    return tf.math.lgamma(self.disp) \
      + tf.math.lgamma(x + 1) \
        - tf.math.lgamma(x + self.disp)

  def _mean(self):
    return self._loc

  def _mode(self):
    raise NotImplementedError()

  def _variance(self):
    mean = self._mean()
    return mean + tf.square(mean) / self.disp

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    loc = self._loc
    disp = self._disp
    if is_init != tensor_util.is_ref(loc):
      assertions.extend([
          assert_util.assert_non_negative(
              loc, message='`loc` has components less than 0.'),
          distribution_util.assert_integer_form(
              loc, message='`loc` has fractional components.')
      ])
    if is_init != tensor_util.is_ref(disp):
      assertions.extend([
          assert_util.assert_positive(
              disp, message='`disp` has components less than or equal to 0.'),
          distribution_util.assert_integer_form(
              disp, message='`disp` has fractional components.')
      ])
    return assertions

  def _sample_control_dependencies(self, x):
    """Check counts for proper shape and values, then return tensor version."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.extend(distribution_util.assert_nonnegative_integer_form(x))
    return assertions
