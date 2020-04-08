import tensorflow as tf
from tensorflow.python.training import moving_averages

from odin.bay.vi.autoencoder import BetaVAE




def add_ema_control_dependencies(vector_quantizer, one_hot_assignments, codes,
                                 commitment_loss, decay):
  """Add control dependencies to the commmitment loss to update the codebook.

  Args:
    vector_quantizer: An instance of the VectorQuantizer class.
    one_hot_assignments: The one-hot vectors corresponding to the matched
      codebook entry for each code in the batch.
    codes: A `float`-like `Tensor` containing the latent vectors to be compared
      to the codebook.
    commitment_loss: The commitment loss from comparing the encoder outputs to
      their neighboring codebook entries.
    decay: Decay factor for exponential moving average.

  Returns:
    commitment_loss: Commitment loss with control dependencies.
  """
  # Use an exponential moving average to update the codebook.
  updated_ema_count = moving_averages.assign_moving_average(
      vector_quantizer.ema_count,
      tf.reduce_sum(input_tensor=one_hot_assignments, axis=[0, 1]),
      decay,
      zero_debias=False)
  updated_ema_means = moving_averages.assign_moving_average(
      vector_quantizer.ema_means,
      tf.reduce_sum(input_tensor=tf.expand_dims(codes, 2) *
                    tf.expand_dims(one_hot_assignments, 3),
                    axis=[0, 1]),
      decay,
      zero_debias=False)

  # Add small value to avoid dividing by zero.
  perturbed_ema_count = updated_ema_count + 1e-5
  with tf.control_dependencies([commitment_loss]):
    update_means = tf.compat.v1.assign(
        vector_quantizer.codebook,
        updated_ema_means / perturbed_ema_count[..., tf.newaxis])
    with tf.control_dependencies([update_means]):
      return tf.identity(commitment_loss)


class VectorQuantizer(object):
  """Creates a vector-quantizer.

  It quantizes a continuous vector under a codebook. The codebook is also known
  as "embeddings" or "memory", and it is learned using an exponential moving
  average.
  """

  def __init__(self, num_codes, code_size):
    self.num_codes = num_codes
    self.code_size = code_size
    self.codebook = tf.compat.v1.get_variable(
        "codebook",
        [num_codes, code_size],
        dtype=tf.float32,
    )
    self.ema_count = tf.compat.v1.get_variable(
        name="ema_count",
        shape=[num_codes],
        initializer=tf.compat.v1.initializers.constant(0),
        trainable=False)
    self.ema_means = tf.compat.v1.get_variable(
        name="ema_means",
        initializer=self.codebook.initialized_value(),
        trainable=False)

  def __call__(self, codes):
    """Uses codebook to find nearest neighbor for each code.

    Args:
      codes: A `float`-like `Tensor` containing the latent
        vectors to be compared to the codebook. These are rank-3 with shape
        `[batch_size, latent_size, code_size]`.

    Returns:
      nearest_codebook_entries: The 1-nearest neighbor in Euclidean distance for
        each code in the batch.
      one_hot_assignments: The one-hot vectors corresponding to the matched
        codebook entry for each code in the batch.
    """
    distances = tf.norm(
        tensor=tf.expand_dims(codes, 2) -
        tf.reshape(self.codebook, [1, 1, self.num_codes, self.code_size]),
        axis=3)
    assignments = tf.argmin(input=distances, axis=2)
    one_hot_assignments = tf.one_hot(assignments, depth=self.num_codes)
    nearest_codebook_entries = tf.reduce_sum(
        input_tensor=tf.expand_dims(one_hot_assignments, -1) *
        tf.reshape(self.codebook, [1, 1, self.num_codes, self.code_size]),
        axis=2)
    return nearest_codebook_entries, one_hot_assignments


class VQVAE(BetaVAE):
  r"""

  Reference:
    Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. "Neural Discrete
      Representation Learning". In _Conference on Neural Information Processing
      Systems_, 2017. https://arxiv.org/abs/1711.00937
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
