from __future__ import absolute_import, division, print_function

import tensorflow as tf
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import TrainStep
from odin.utils import as_tuple

__all__ = ['ImputeVAE', 'StochasticVAE']


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
    if len(self.vae.observation) == 1:
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
    for layer in self.observation:
      llk_params += layer.trainable_variables
    #
    self.kl_params = kl_params
    self.llk_params = llk_params

  def train_steps(self,
                  inputs,
                  training=None,
                  mask=None,
                  sample_shape=(),
                  iw=False,
                  elbo_kw=dict()) -> TrainStep:
    r""" Facilitate multiple steps training for each iteration (smilar to GAN)

    Example:
    ```
    model = factorVAE()
    x = model.sample_data()
    vae_step, discriminator_step = list(model.train_steps(x))
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
                          mask=mask,
                          training=training,
                          sample_shape=sample_shape,
                          iw=iw,
                          elbo_kw=elbo_kw,
                          parameters=self.kl_params)
    for _ in range(self.llk_steps):
      yield LikelihoodStep(vae=self,
                           inputs=inputs,
                           mask=mask,
                           training=training,
                           sample_shape=sample_shape,
                           iw=iw,
                           elbo_kw=elbo_kw,
                           parameters=self.llk_params)


class ImputeVAE(BetaVAE):
  r""" Iteratively imputing VAE outputs for a fixed number of steps

  Arguments:
    sequential : a Boolean. If True, using the outputs from previous step
      as inputs for the next step when calculating ELBO.
      This could be interpreted as a scheme for data augmentation.

  Example:
  ```
  ds = MNIST()
  train = ds.create_dataset(partition='train')
  model = ImputeVAE(
      encoder='mnist',
      outputs=RV((28, 28, 1), 'bern', name="Image"),
      impute_steps=3,
      sequential=True)
  model.fit(train, epochs=-1, max_iter=8000, compile_graph=True)
  ```
  """

  def __init__(self,
               beta=1.,
               impute_steps=3,
               impute_llk_weights=[1.0, 0.8, 0.4],
               impute_kl_weights=[1.0, 0.8, 0.4],
               sequential=True,
               **kwargs):
    super().__init__(beta=beta, **kwargs)
    assert impute_steps >= 1
    self.impute_steps = int(impute_steps)
    self.impute_kl_weights = as_tuple(impute_kl_weights,
                                      t=float,
                                      N=self.impute_steps)
    self.impute_llk_weights = as_tuple(impute_llk_weights,
                                       t=float,
                                       N=self.impute_steps)
    self.sequential = bool(sequential)

  def _elbo(self,
            X,
            pX_Z,
            qZ_X,
            analytic,
            reverse,
            sample_shape=None,
            mask=None,
            training=None,
            **kwargs):
    if sample_shape is None:
      sample_shape = []
    X = [X] * self.impute_steps
    all_llk = {}
    all_div = {}
    prev_px = None
    for step, (inputs, px, qz, w_llk, w_div) in enumerate(
        zip(X, pX_Z, qZ_X, self.impute_llk_weights, self.impute_kl_weights)):
      if self.sequential and prev_px is not None:
        inputs = [p.mean() for p in prev_px]
      px = tf.nest.flatten(px)
      qz = tf.nest.flatten(qz)
      llk, div = super()._elbo(X=inputs,
                               pX_Z=px,
                               qZ_X=qz,
                               analytic=analytic,
                               reverse=reverse,
                               sample_shape=sample_shape,
                               mask=mask,
                               training=training,
                               **kwargs)
      all_llk.update({'%s_%d' % (k, step): w_llk * v for k, v in llk.items()})
      all_div.update({'%s_%d' % (k, step): w_div * v for k, v in div.items()})
      prev_px = px
    return all_llk, all_div

  def call(self, inputs, training=None, mask=None, sample_shape=()):
    sample_shape = tf.nest.flatten(sample_shape)
    pX_Z, qZ_X = super().call(inputs,
                              training=training,
                              mask=mask,
                              sample_shape=sample_shape)
    results = [[pX_Z], [qZ_X]]
    for _ in range(1, self.impute_steps):
      pX_Z = tf.nest.flatten(pX_Z)
      inputs = [p.mean() for p in pX_Z]
      if len(sample_shape) > 0:
        inputs = [
            tf.reduce_mean(i, axis=list(range(len(sample_shape))))
            for i in inputs
        ]
      pX_Z, qZ_X = super().call(inputs[0] if len(inputs) == 1 else inputs,
                                training=training,
                                mask=mask,
                                sample_shape=sample_shape)
      results[0].append(pX_Z)
      results[1].append(qZ_X)
    return results[0], results[1]
