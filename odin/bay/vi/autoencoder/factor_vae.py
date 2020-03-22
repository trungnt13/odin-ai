import tensorflow as tf
from tensorflow.python import keras

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.utils import permute_dims
from odin.networks import DenseNetwork


class FactorDiscriminator(keras.Model):
  r""" The main goal is minimizing the total correlation (the mutual information
    which quantifies the redundancy or dependency among latent variables).

    We use a discriminator to estimate TC

  Arguments:
    zdim : an Integer, the number of latent units used in VAE.
    hdim : an Integer, the number of hidden units for the discriminator.
    nlayer : an Integer, the number of layers.
    activation : Callable or String, activation function of each layer.

  Reference:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """

  def __init__(self, zdim, hdim=1024, nlayer=6, activation=tf.nn.leaky_relu):
    super().__init__()
    # 1: real sample for q(z) and 0: fake sample from q(z-)
    self.discriminator = DenseNetwork(
        units=[hdim] * nlayer,
        activation=activation,
        batchnorm=True,
        flatten=True,
        end_layers=[keras.layers.Dense(2, activation='linear')],
        input_shape=(zdim,),
        name="Discriminator")

  def tc(self, z, training=None):
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
    d_z = self.discriminator(z, training=training)
    return tf.reduce_mean(d_z[:, 1] - d_z[:, 0])

  def dtc_loss(self, z, training=None):
    r""" Discriminated total correlation loss Algorithm(2)

      Minimize the probablity of:
       - `q(z)` misclassified as `D(z)[:, 0]`
       - `q(z-)` misclassified as `D(z)[:, 1]`

    """
    # we don't want the gradient to be propagated to the encoder
    z = tf.stop_gradient(z)
    d_z = tf.nn.log_softmax(self.discriminator(z, training=training), axis=-1)
    z_perm = permute_dims(z)
    d_zperm = tf.nn.log_softmax(self.discriminator(z_perm, training=training),
                                axis=-1)
    loss = 0.5 * tf.reduce_mean(d_z[:, 0] + d_zperm[:, 1])
    return loss

  def call(self, inputs, training=None):
    return self.discriminator(inputs, training=training)


class FactorVAE(VariationalAutoencoder):
  pass
