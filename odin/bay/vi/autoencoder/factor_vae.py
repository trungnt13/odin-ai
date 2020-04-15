import numpy as np
import tensorflow as tf

from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.discriminator import FactorDiscriminator
from odin.bay.vi.autoencoder.variational_autoencoder import TrainStep


class FactorStep(TrainStep):

  def __call__(self, training=True):
    inputs, qZ_X = self.inputs
    qZ_Xprime = self.vae.encode(inputs,
                                training=training,
                                sample_shape=self.sample_shape)
    dtc_loss = self.vae.dtc_loss(qZ_X, qZ_Xprime, training=training)
    return dtc_loss, dict(dtc=dtc_loss)


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
               discriminator=dict(units=1000, n_layers=5),
               gamma=10.0,
               beta=1.0,
               **kwargs):
    super().__init__(beta=beta, **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    # all latents will be concatenated
    latent_dim = np.prod(
        sum(np.array(layer.event_shape) for layer in self.latent_layers))
    self.discriminator = FactorDiscriminator(input_shape=(latent_dim,),
                                             **discriminator)
    # VAE and discriminator must be trained separatedly so we split
    # their params here
    self.disc_params = self.discriminator.trainable_variables
    exclude = set(id(p) for p in self.disc_params)
    self.vae_params = [
        p for p in self.trainable_variables if id(p) not in exclude
    ]

  def _elbo(self,
            X,
            pX_Z,
            qZ_X,
            analytic,
            reverse,
            sample_shape,
            training=None):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, sample_shape)
    div['tc'] = self.total_correlation(qZ_X, training=training)
    return llk, div

  def total_correlation(self, qZ_X, scaled_by_gamma=True, training=None):
    r""" Using the discriminator output to estimate total correlation of
    the latents
    """
    tc = self.discriminator.total_correlation(qZ_X, training=training)
    if scaled_by_gamma:
      tc = self.gamma * tc
    return tc

  def dtc_loss(self, qZ_X, qZ_Xprime=None, training=None):
    r""" Discrimination loss between real and permuted codes Algorithm (2) """
    return self.discriminator.dtc_loss(qZ_X,
                                       qZ_Xprime=qZ_Xprime,
                                       training=training)

  def train_steps(self,
                  inputs,
                  sample_shape=(),
                  iw=False,
                  elbo_kw=dict()) -> TrainStep:
    r""" Facilitate multiple steps training for each iteration (smilar to GAN)

    Example:
    ```
    vae = FactorVAE()
    x = vae.sample_data()
    vae_step, discriminator_step = list(vae.train_steps(x))
    # optimizer VAE with total correlation loss
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vae_step.parameters)
      loss, metrics = vae_step()
      tape.gradient(loss, vae_step.parameters)
    # optimizer the discriminator
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(discriminator_step.parameters)
      loss, metrics = discriminator_step()
      tape.gradient(loss, discriminator_step.parameters)
    ```
    """
    self.step.assign_add(1.)
    # split inputs into 2 mini-batches here
    if tf.is_tensor(inputs):
      x1, x2 = tf.split(inputs, 2, axis=0)
    else:
      inputs = [tf.split(x, 2, axis=0) for x in tf.nest.flatten(inputs)]
      x1 = [i[0] for i in inputs]
      x2 = [i[1] for i in inputs]
    # first step optimize VAE with total correlation loss
    step1 = TrainStep(vae=self,
                      inputs=x1,
                      sample_shape=sample_shape,
                      iw=iw,
                      elbo_kw=elbo_kw,
                      parameters=self.vae_params)
    yield step1
    # second step optimize the discriminator for discriminate permuted code
    step2 = FactorStep(vae=self,
                       inputs=[x2, step1.qZ_X],
                       sample_shape=sample_shape,
                       parameters=self.disc_params)
    yield step2

  def __str__(self):
    text = super().__str__()
    text += "\n Discriminator:\n  "
    text += "\n  ".join(str(self.discriminator).split('\n'))
    return text
