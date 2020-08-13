from typing import Optional

import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow_probability.python.distributions import (Distribution,
                                                         Multinomial,
                                                         kullback_leibler)
from tensorflow_probability.python.internal import reparameterization


class VectorQuantized(Distribution):
  r""" A wrapper for storing the output of VQ-VAE latents """

  def __init__(self,
               codes: tf.Tensor,
               assignments: tf.Tensor,
               nearest_codes: tf.Tensor,
               validate_args: bool = False,
               allow_nan_stats: bool = True,
               name: str = "VectorQuantized"):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._codes = codes
      self._assignments = assignments
      self._nearest_codes = nearest_codes
      tf.assert_equal(tf.shape(codes)[:-1], tf.shape(assignments)[:-1])
      tf.assert_equal(tf.shape(codes), tf.shape(nearest_codes))
    super().__init__(
        dtype=self.codes.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @property
  def commitment_loss(self):
    r""" Calculate the commitment loss, i.e. the distance from the input code
    to its nearest neighbor """
    return tf.reduce_mean(
        tf.square(self.codes - tf.stop_gradient(self.nearest_codes)))

  @property
  def codes(self):
    return self._codes

  @property
  def assignments(self):
    return self._assignments

  @property
  def nearest_codes(self):
    return self._nearest_codes

  def _batch_shape_tensor(self):
    return tf.shape(self.codes)[:-1]

  def _batch_shape(self):
    return self.codes.shape[:-1]

  def _event_shape_tensor(self):
    return tf.shape(self.codes)[-1:]

  def _event_shape(self):
    return self.codes.shape[-1:]

  def _log_prob(self, x=None):
    raise NotImplementedError

  def _mean(self):
    raise NotImplementedError

  def _stddev(self):
    raise NotImplementedError

  def _sample_n(self, n, seed=None):
    # This trick stop the gradient propagate to the codebook, but allow it
    # to pass straight to the encoder
    codes_straight_through = self.codes + tf.stop_gradient(self.nearest_codes -
                                                           self.codes)
    samples = tf.expand_dims(codes_straight_through, axis=0)
    return tf.cond(tf.greater(n, 1),
                   true_fn=lambda: tf.repeat(samples, n, axis=0),
                   false_fn=lambda: samples)

  def update_codebook(self,
                      codebook: tf.Variable,
                      counts: tf.Variable,
                      means: tf.Variable,
                      decay: float = 0.99,
                      perturb: float = 1e-5):
    r"""Add control dependencies to the commitment loss to update the codebook.
    These updates should be performed after calculating the commitment loss.

    Args:
      codebook: A `float`-like `Tensor`,
        the codebook for code embedding, shape `[n_codes, code_size]`.
      counts: A `float`-like `Tensor`,
        stores the occurrences counts for each code in the codebook
        the codebook for code embedding, shape `[n_codes]`.
      means: A `float`-like `Tensor`,
        stores the moving average of each code in the codebook,
        shape `[n_codes, code_size]`.

    Returns:
      updated_codebook: the updated codebook, shape  `[n_codes, code_size]`
      updated_counts: the moving average updated counts, shape  `[code_size]`
      updated_means: the moving average updated means, shape  `[n_codes, code_size]`
    """
    input_ndim = len(self.codes.shape) - 2
    axes = range(input_ndim + 1)  # the batch axes
    # Use an exponential moving average to update the codebook.
    updated_ema_count = moving_averages.assign_moving_average(
        variable=counts,
        value=tf.reduce_sum(self.assignments, axis=axes),
        decay=decay,
        zero_debias=False)
    updated_ema_means = moving_averages.assign_moving_average(
        variable=means,
        value=tf.reduce_sum(tf.expand_dims(self.codes, axis=-2) *
                            tf.expand_dims(self.assignments, axis=-1),
                            axis=axes),
        decay=decay,
        zero_debias=False)
    # Add small value to avoid dividing by zero.
    perturbed_ema_count = updated_ema_count + perturb
    codebook.assign(updated_ema_means / perturbed_ema_count[..., tf.newaxis])
    return codebook, updated_ema_count, updated_ema_means


@kullback_leibler.RegisterKL(VectorQuantized, Multinomial)
def _kl_vectorquantized_multinomial(a: VectorQuantized,
                                    b: Multinomial,
                                    name: Optional[str] = None):
  r"""Calculate the batched KL divergence KL(a || b) with a and b Normal.

  Args:
    a: instance of a VectorQuantized distribution object.
    b: instance of a Multinomial distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_normal_normal'`).

  Returns:
    kl_div: Batchwise KL(a || b)
  """
  assert isinstance(a, VectorQuantized) and isinstance(b, Multinomial), \
    ("Only support VectorQuantized posterior and Multinomial prior, "
     f"but given: {a} and {b}")
  with tf.name_scope(name or 'kl_vectorquantized_multinomial'):
    one_hot_assignments = a.assignments
    batch_size = tf.shape(one_hot_assignments)[0]
    # [batch_size, ...]
    llk = b.log_prob(one_hot_assignments)
    llk = tf.reshape(llk, (batch_size, -1))
    llk = tf.reduce_mean(tf.reduce_sum(llk, axis=1))
    return -llk
