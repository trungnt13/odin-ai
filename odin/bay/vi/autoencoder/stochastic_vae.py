from __future__ import absolute_import, division, print_function

import tensorflow as tf

from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import TrainStep


class PosteriorStep(TrainStep):

  def __call__(self, training=True):
    analytic = self.elbo_kw.pop('analytic', False)
    reverse = self.elbo_kw.pop('reverse', True)
    qZ_X = self.vae.encode(self.inputs,
                           training=training,
                           sample_shape=self.sample_shape)
    if len(self.vae.latent_layers) == 1:
      qZ_X = [qZ_X]
    #
    metrics = {}
    kl_loss = tf.convert_to_tensor(0., dtype=self.vae.dtype)
    for name, qZ in zip(self.vae.latent_names, qZ_X):
      kl = tf.reduce_mean(qZ.KL_divergence(analytic=analytic, reverse=reverse))
      metrics["kl_%s" % name] = kl
      kl_loss += self.vae.beta * kl
    return kl_loss, metrics


class LikelihoodStep(TrainStep):

  def __call__(self, training=True):
    prior = self.vae.sample_prior(self.sample_shape)
    pX_Z = self.vae.decode(prior,
                           training=training,
                           sample_shape=self.sample_shape)
    if len(self.vae.output_layers) == 1:
      pX_Z = [pX_Z]
    inputs = tf.nest.flatten(self.inputs)
    #
    metrics = {}
    llk_loss = tf.convert_to_tensor(0., dtype=self.vae.dtype)
    for name, X, pX in zip(self.vae.variable_names, inputs, pX_Z):
      llk = tf.reduce_mean(pX.log_prob(X))
      metrics["llk_%s" % name] = llk
      llk_loss += -llk
    return llk_loss, metrics


class StochasticVAE(BetaVAE):

  def __init__(self, kl_steps=1, llk_steps=1, beta=1.0, **kwargs):
    super().__init__(beta=beta, **kwargs)
    self.kl_steps = max(int(kl_steps), 1)
    self.llk_steps = max(int(llk_steps), 1)
    ## parameters for each step
    kl_params = self.encoder.trainable_variables
    for layer in self.latent_layers:
      kl_params += layer.trainable_variables
    #
    llk_params = self.decoder.trainable_variables
    for layer in self.output_layers:
      llk_params += layer.trainable_variables
    #
    self.kl_params = kl_params
    self.llk_params = llk_params

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
    self.step.assign_add(1)
    for _ in range(self.kl_steps):
      yield PosteriorStep(vae=self,
                          inputs=inputs,
                          sample_shape=sample_shape,
                          iw=iw,
                          elbo_kw=elbo_kw,
                          parameters=self.kl_params)
    for _ in range(self.llk_steps):
      yield LikelihoodStep(vae=self,
                           inputs=inputs,
                           sample_shape=sample_shape,
                           iw=iw,
                           elbo_kw=elbo_kw,
                           parameters=self.llk_params)
