import numpy as np
import tensorflow as tf

from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.discriminator import FactorDiscriminator
from odin.bay.vi.autoencoder.variational_autoencoder import TrainStep


class FactorStep(TrainStep):

  def __call__(self, training=True):
    dtc_loss = self.vae.dtc_loss(self.inputs, training=training)
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
               discriminator=dict(units=1000, nlayers=6),
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

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc, training=None):
    llk, div = super()._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc)
    div['tc'] = self.gamma * self.discriminator.total_correlation(
        qZ_X, training=training)
    return llk, div

  def dtc_loss(self, qZ_X, training=None):
    r""" Discrimination loss between real and permuted codes Algorithm(2) """
    return self.discriminator.dtc_loss(qZ_X, training=training)

  def train_steps(self,
                  inputs,
                  n_mcmc=(),
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
    # first step optimize VAE with total correlation loss
    step1 = TrainStep(vae=self,
                      inputs=inputs,
                      n_mcmc=n_mcmc,
                      iw=iw,
                      elbo_kw=elbo_kw,
                      parameters=self.vae_params)
    yield step1
    # second step optimize the discriminator for discriminate permuted code
    step2 = FactorStep(vae=self, inputs=step1.qZ_X, parameters=self.disc_params)
    yield step2

  def __str__(self):
    text = super().__str__()
    text += "\n Discriminator:\n  "
    text += "\n  ".join(str(self.discriminator).split('\n'))
    return text
