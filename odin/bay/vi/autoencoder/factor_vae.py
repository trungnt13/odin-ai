import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from odin.bay.random_variable import RandomVariable
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.utils import permute_dims
from odin.networks import DenseNetwork, NetworkConfig, ReshapeMCMC


class FactorDiscriminator(ReshapeMCMC):
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
               latent_dim,
               batchnorm=True,
               units=1024,
               nlayers=6,
               activation=tf.nn.leaky_relu):
    # 1: real sample for q(z) and 0: fake sample from q(z-)
    super().__init__(
        layer=DenseNetwork(
            units=[int(units)] * nlayers,
            activation=activation,
            batchnorm=bool(batchnorm),
            flatten=True,
            end_layers=[keras.layers.Dense(2, activation='linear')],
            input_shape=(latent_dim,)),
        sample_ndim=1,
        keepdims=True,
        name="Discriminator",
    )

  def tc(self, qZ_X, training=None):
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
    d_z = self(qZ_X, training=training)
    return tf.reduce_mean(d_z[..., 1] - d_z[..., 0])

  def dtc_loss(self, qZ_X, training=None):
    r""" Discriminated total correlation loss Algorithm(2)

      Minimize the probablity of:
       - `q(z)` misclassified as `D(z)[:, 0]`
       - `q(z-)` misclassified as `D(z)[:, 1]`

    """
    # we don't want the gradient to be propagated to the encoder
    shape = tf.concat([qZ_X.batch_shape, qZ_X.event_shape], axis=0)
    z = tf.stop_gradient(qZ_X)
    sample_ndim = tf.rank(z) - shape.shape[0]
    d_z = tf.nn.log_softmax(self(z, training=training, sample_ndim=sample_ndim),
                            axis=-1)
    z_perm = permute_dims(z)
    d_zperm = tf.nn.log_softmax(self(z_perm,
                                     training=training,
                                     sample_ndim=sample_ndim),
                                axis=-1)
    loss = 0.5 * tf.reduce_mean(d_z[..., 0] + d_zperm[..., 1])
    return loss


class FactorVAE(BetaVAE):
  r""" The default encoder and decoder configuration is the same as proposed
  in (Kim et. al. 2018).

  The training procedure of FactorVAE is as follows:

  ```
    foreach iter:
      X = minibatch()
      pX_Z, qZ_X = vae(x, trainining=True)
      loss = -vae.elbo(X, pX_Z, qZ_X, training=True)
      vae_optimizer.apply_gradients(loss, vae.parameters)

      dtc_loss = vae.dtc_loss(qZ_X, training=True)
      dis_optimizer.apply_gradients(dtc_loss, dis.parameters)
  ```

  Reference:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """

  def __init__(self,
               discriminator=dict(units=1000, nlayers=6),
               gamma=1.0,
               beta=1.0,
               **kwargs):
    super().__init__(beta=beta, **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    self.discriminator = FactorDiscriminator(
        latent_dim=int(np.prod(self.latent_layer.event_shape)),
        **discriminator,
    )

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc, training=None):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    total_correlation = self.discriminator.tc(qZ_X, training=training)
    div = self.gamma * total_correlation
    return llk, div

  def dtc_loss(self, qZ_X, training=None):
    r""" Discriminated total correlation loss Algorithm(2) """
    return self.discriminator.dtc_loss(qZ_X, training=training)

  def __str__(self):
    text = super().__str__()
    text += "\n Discriminator:\n  "
    text += "\n  ".join(str(self.discriminator).split('\n'))
    return text
