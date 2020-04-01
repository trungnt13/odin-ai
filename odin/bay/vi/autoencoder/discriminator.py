import tensorflow as tf
from tensorflow.python import keras
from tensorflow_probability.python.distributions import Distribution

from odin.bay.vi.utils import permute_dims
from odin.networks import DenseNetwork


class FactorDiscriminator(DenseNetwork):
  r""" The main goal is minimizing the total correlation (the mutual information
    which quantifies the redundancy or dependency among latent variables).

    We use a discriminator to estimate TC

  Arguments:
    latent_dim : an Integer, the number of latent units used in VAE.
    hdim : an Integer, the number of hidden units for the discriminator.
    nlayer : an Integer, the number of layers.
    activation : Callable or String, activation function of each layer.

  Reference:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """

  def __init__(self,
               input_shape,
               batchnorm=True,
               units=1024,
               nlayers=6,
               noutputs=2,
               activation=tf.nn.leaky_relu):
    # 1: real sample for q(z) (or last unit in case noutputs > 2) and
    # 0: fake sample from q(z-)
    super().__init__(
        units=[int(units)] * nlayers,
        activation=activation,
        batchnorm=bool(batchnorm),
        flatten=True,
        end_layers=[keras.layers.Dense(int(noutputs), activation='linear')],
        input_shape=input_shape,
        name="Discriminator",
    )
    self.input_ndim = len(self.input_shape) - 1

  def _to_samples(self, qZ_X):
    qZ_X = tf.nest.flatten(qZ_X)
    z = tf.concat([tf.convert_to_tensor(q) for q in qZ_X], axis=-1)
    z = tf.reshape(z, tf.concat([(-1,), z.shape[-self.input_ndim:]], axis=0))
    return z

  def total_correlation(self, qZ_X, training=None):
    r""" Total correlation Eq(3)
      `TC(z) = KL(q(z)||q(z-)) = E_q(z)[log(q(z) / q(z-))]`

    Note:
      In many implementation, `log(q(z-)) - log(q(z))` is returned as `total
      correlation loss`, here, we return `log(q(z)) - log(q(z-))` as for
      the construction of the ELBO in Eq(2)

    Arguments:
      z : a Tensor, [batch_dim, latent_dim]

    Return:
      TC(z) : a scalar, approximation of the density-ratio that arises in the
        KL-term.
    """
    # adding log_softmax here could provide more stable loss than minimizing
    # the logit directly
    # tf.nn.log_softmax
    d_z = self(self._to_samples(qZ_X), training=training)
    return tf.reduce_mean(d_z[..., -1] - tf.reduce_sum(d_z[..., :-1], axis=-1))

  def dtc_loss(self, qZ_X, training=None):
    r""" Discriminated total correlation loss Algorithm(2)

      Minimize the probablity of:
       - `q(z)` misclassified as `D(z)[:, 0]`
       - `q(z-)` misclassified as `D(z)[:, 1]`

    """
    # we don't want the gradient to be propagated to the encoder
    z = self._to_samples(qZ_X)
    z = tf.stop_gradient(z)
    d_z = tf.nn.log_softmax(self(z, training=training), axis=-1)
    z_perm = permute_dims(z)
    d_zperm = tf.nn.log_softmax(self(z_perm, training=training), axis=-1)
    # reduce the negative of d_z, and the positive of d_zperm
    loss = 0.5 * tf.reduce_mean(
        tf.reduce_sum(d_z[..., :-1], axis=-1) + d_zperm[..., -1])
    return loss
