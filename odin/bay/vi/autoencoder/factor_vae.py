from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from odin.bay.random_variable import RandomVariable as RV
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.networks import FactorDiscriminator
from odin.bay.vi.autoencoder.variational_autoencoder import TrainStep


class FactorDiscriminatorStep(TrainStep):

  def call(self, inputs, mask, qZ_X, qZ_Xprime, training):
    dtc_loss = self.vae.dtc_loss(qZ_X, qZ_Xprime, training=training)
    metrics = dict(dtc_loss=dtc_loss)
    ## applying the classifier loss
    classifier_loss = 0
    if len(tf.nest.flatten(inputs)) > len(self.vae.output_layers):
      labels = inputs[len(self.vae.output_layers):]
      classifier_loss = self.vae.classifier_loss(labels,
                                                 qZ_X,
                                                 mask=mask,
                                                 training=training,
                                                 apply_alpha=True)
      metrics['classifier_loss'] = classifier_loss
    return dtc_loss + classifier_loss, metrics

  def __call__(self):
    inputs, qZ_X = self.inputs
    mask = self.mask
    training = self.training
    qZ_Xprime = self.vae.encode(inputs,
                                training=training,
                                mask=mask,
                                sample_shape=self.sample_shape)
    return self.call(inputs, mask, qZ_X, qZ_Xprime, training)


class Factor2DiscriminatorStep(FactorDiscriminatorStep):

  def __call__(self):
    inputs, qZ_X = self.inputs
    mask = self.mask
    training = self.training
    qZ_Xprime = self.vae.encode(inputs,
                                training=training,
                                mask=mask,
                                sample_shape=self.sample_shape)
    # only select the last latent space
    return self.call(inputs, mask, qZ_X[-1], qZ_Xprime[-1], training)


# ===========================================================================
# Main FactorVAE
# ===========================================================================
class FactorVAE(BetaVAE):
  r""" The default encoder and decoder configuration is the same as proposed
  in (Kim et. al. 2018).

  The training procedure of FactorVAE is as follows:

  ```
  foreach iter:
    X = minibatch()
    X1, X2 = split(X, 2, axis=0)

    pX_Z, qZ_X = vae(X1, trainining=True)
    loss = -vae.elbo(X1, pX_Z, qZ_X, training=True)
    vae_optimizer.apply_gradients(loss, vae.parameters)

    qZ_Xprime = vae.encode(X2, training=True)
    dtc_loss = vae.dtc_loss(qZ_X, qZ_Xprime, training=True)
    dis_optimizer.apply_gradients(dtc_loss, dis.parameters)
  ```

  Arguments:
    discriminator : a Dictionary or `keras.layers.Layer`.
      Keywords arguments for creating the `FactorDiscriminator`
    maximize_tc : a Boolean. If True, instead of minimize total correlation
      for more factorized latents, try to maximize the divergence.

  Note:
    You should use double the `batch_size` since the minibatch will be splitted
    into 2 partitions for `X` and `X_prime`.

    It is recommended to use the same optimizers configuration like in the
    paper: `Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)` for the VAE
    and `Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)` for the
    discriminator.

    Discriminator's Adam has learning rate `1e-4` for dSprites and `1e-5` for
    Shapes3D and other colored image datasets.

  Reference:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """

  def __init__(self,
               discriminator=dict(units=[1000, 1000, 1000, 1000, 1000]),
               gamma=6.0,
               beta=1.0,
               maximize_tc=False,
               **kwargs):
    latent_dim = kwargs.pop('latent_dim', None)
    super().__init__(beta=beta,
                     reduce_latent=kwargs.pop('reduce_latent', 'concat'),
                     **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    # all latents will be concatenated (a bit spooky to customize latent_dim)
    if latent_dim is None:
      latent_dim = np.prod(
          sum(np.array(layer.event_shape) for layer in self.latent_layers))
    # init discriminator
    if not isinstance(discriminator, keras.layers.Layer):
      discriminator = FactorDiscriminator(input_shape=(latent_dim,),
                                          **discriminator)
    self.discriminator = discriminator
    # VAE and discriminator must be trained separated so we split
    # their params here
    self.disc_params = self.discriminator.trainable_variables
    exclude = set(id(p) for p in self.disc_params)
    self.vae_params = [
        p for p in self.trainable_variables if id(p) not in exclude
    ]
    self.maximize_tc = bool(maximize_tc)
    # store class for training factor discriminator, this allow later
    # modification without re-writing the train_steps method
    self._FactorStep = FactorDiscriminatorStep

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training):
    llk, div = super()._elbo(X,
                             pX_Z,
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training)
    # by default, this support multiple latents by concatenating all latents
    tc = self.total_correlation(qZ_X, apply_gamma=True, training=training)
    if self.maximize_tc:
      tc = -tc
    div['total_corr'] = tc
    return llk, div

  def total_correlation(self, qZ_X, training=None, apply_gamma=False):
    r""" Using the discriminator output to estimate total correlation of
    the latents

    """
    tc = self.discriminator.total_correlation(qZ_X, training=training)
    if apply_gamma:
      tc = self.gamma * tc
    return tc

  def dtc_loss(self, qZ_X, qZ_Xprime=None, training=None):
    r""" Discrimination loss between real and permuted codes Algorithm (2) """
    return self.discriminator.dtc_loss(qZ_X,
                                       qZ_Xprime=qZ_Xprime,
                                       training=training)

  def _prepare_steps(self, inputs, mask):
    r""" Split the data into 2 partitions for training the VAE and Discriminator """
    self.step.assign_add(1.)
    # split inputs into 2 mini-batches here
    if tf.is_tensor(inputs):
      x1, x2 = tf.split(inputs, 2, axis=0)
    else:
      inputs = [tf.split(x, 2, axis=0) for x in tf.nest.flatten(inputs)]
      x1 = [i[0] for i in inputs]
      x2 = [i[1] for i in inputs]
    # split the mask
    mask1 = None
    mask2 = None
    if mask is not None:
      if tf.is_tensor(mask):
        mask1, mask2 = tf.split(mask, 2, axis=0)
      else:
        mask = [tf.split(m, 2, axis=0) for m in tf.nest.flatten(mask)]
        mask1 = [i[0] for i in mask]
        mask2 = [i[1] for i in mask]
    return (x1, mask1), (x2, mask2)

  def train_steps(self,
                  inputs,
                  training=None,
                  mask=None,
                  sample_shape=(),
                  iw=False,
                  elbo_kw=dict()) -> TrainStep:
    r""" Facilitate multiple steps training for each iteration (similar to GAN)

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
    (x1, mask1), (x2, mask2) = self._prepare_steps(inputs, mask)
    # first step optimize VAE with total correlation loss
    step1 = TrainStep(vae=self,
                      inputs=x1,
                      training=training,
                      mask=mask1,
                      sample_shape=sample_shape,
                      iw=iw,
                      elbo_kw=elbo_kw,
                      parameters=self.vae_params)
    yield step1
    # second step optimize the discriminator for discriminate permuted code
    step2 = self._FactorStep(vae=self,
                             inputs=[x2, step1.qZ_X],
                             training=training,
                             mask=mask2,
                             sample_shape=sample_shape,
                             parameters=self.disc_params)
    yield step2

  def fit(
      self,
      train: tf.data.Dataset,
      valid: Optional[tf.data.Dataset] = None,
      valid_freq=1000,
      valid_interval=0,
      optimizer=[
          tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
          tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
      ],
      learning_rate=1e-3,
      clipnorm=None,
      epochs=2,
      max_iter=-1,
      sample_shape=(),  # for ELBO
      analytic=False,  # for ELBO
      iw=False,  # for ELBO
      callback=lambda: None,
      compile_graph=True,
      autograph=False,
      logging_interval=2,
      log_tag='',
      log_path=None,
      **kwargs):
    kw = dict(locals())
    del kw['self']
    del kw['__class__']
    kwargs = kw.pop('kwargs')
    kw.update(kwargs)
    return super().fit(**kw)

  def __str__(self):
    text = super().__str__()
    text += "\n Discriminator:\n  "
    text += "\n  ".join(str(self.discriminator).split('\n'))
    return text


# ===========================================================================
# Same as Factor VAE but with multi-task semi-supervised extension
# ===========================================================================
class SemiFactorVAE(FactorVAE):

  def __init__(self,
               n_labels=10,
               discriminator=dict(units=[1000, 1000, 1000, 1000, 1000]),
               alpha=10.,
               ss_strategy='logsumexp',
               **kwargs):
    if isinstance(discriminator, dict):
      discriminator['n_outputs'] = n_labels
      discriminator['ss_strategy'] = ss_strategy
    super().__init__(discriminator=discriminator, **kwargs)
    assert self.discriminator.output_shape[-1] == n_labels, \
      "The discriminator has output shape %s, but need (n_labels + 1)=%d outputs" \
        % (self.discriminator.output_shape, n_labels)
    self.n_labels = n_labels
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name='alpha')

  def encode(self, inputs, training=None, mask=None, sample_shape=(), **kwargs):
    inputs = tf.nest.flatten(inputs)[:len(self.output_layers)]
    if len(inputs) == 1:
      inputs = inputs[0]
    return super().encode(inputs,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape,
                          **kwargs)

  def classify(self, inputs, proba=False, training=None):
    qZ_X = self.encode(inputs, training=training)
    if hasattr(self.discriminator, '_to_samples'):
      z = self.discriminator._to_samples(qZ_X)
    else:
      z = qZ_X
    logits = self.discriminator(z, training=training)
    if proba:
      return tf.nn.softmax(logits, axis=-1)
    return logits

  def classifier_loss(self,
                      labels,
                      qZ_X,
                      mask=None,
                      training=None,
                      apply_alpha=True):
    r""" The semi-supervised classifier loss, `mask` is given to indicate
    labelled examples (i.e. `mask=1`), and otherwise, unlabelled examples.
    """
    loss = self.discriminator.classifier_loss(labels=labels,
                                              qZ_X=qZ_X,
                                              mask=mask,
                                              training=training)
    if apply_alpha:
      loss = self.alpha * loss
    return loss

  @property
  def is_semi_supervised(self):
    return True


# ===========================================================================
# Separated latents for TC factorization
# ===========================================================================
class Factor2VAE(FactorVAE):
  r""" The same architecture as `FactorVAE`, however, utilize two different
  latents `Z` for contents generalizability and `C` for disentangling of
  invariant factors.
  """

  def __init__(self,
               latents=RV(5, 'diag', projection=True, name='Latents'),
               factors=RV(5, 'diag', projection=True, name="Factors"),
               **kwargs):
    latents = tf.nest.flatten(latents)
    assert isinstance(factors, RV), \
      "factors must be instance of RandomVariable, but given: %s" % \
        str(type(factors))
    latents.append(factors)
    super().__init__(latents=latents,
                     latent_dim=int(np.prod(factors.event_shape)),
                     **kwargs)
    self.factors = factors
    self._FactorStep = Factor2DiscriminatorStep

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training):
    llk, div = super(BetaVAE, self)._elbo(X,
                                          pX_Z,
                                          qZ_X,
                                          analytic,
                                          reverse,
                                          sample_shape=sample_shape,
                                          mask=mask,
                                          training=training)
    # only use the assumed factors space for total correlation
    tc = self.total_correlation(qZ_X[-1], apply_gamma=True, training=training)
    if self.maximize_tc:
      tc = -tc
    div['tc_%s' % self.factors.name] = tc
    return llk, div


class SemiFactor2VAE(SemiFactorVAE, Factor2VAE):
  r""" Combination of Semi-supervised VAE and Factor-2 VAE which leverages
  both labelled samples and the use of 2 latents space (1 for contents, and
  1 for factors)

  Example:
  ```
  from odin.fuel import MNIST
  from odin.bay.vi.autoencoder import SemiFactor2VAE

  # load the dataset
  ds = MNIST()
  train = ds.create_dataset(partition='train', inc_labels=0.3, batch_size=128)
  valid = ds.create_dataset(partition='valid', inc_labels=1.0, batch_size=128)

  # construction of SemiFactor2VAE for MNIST dataset
  vae = SemiFactor2VAE(encoder='mnist',
                       outputs=RV((28, 28, 1), 'bern', name="Image"),
                       latents=RV(10, 'diag', projection=True, name='Latents'),
                       factors=RV(10, 'diag', projection=True, name='Factors'),
                       alpha=10.,
                       n_labels=10,
                       ss_strategy='logsumexp')
  vae.fit(
      train,
      valid=valid,
      valid_freq=500,
      compile_graph=True,
      epochs=-1,
      max_iter=8000,
  )
  ```
  """

  def __init__(self,
               n_labels=10,
               latents=RV(5, 'diag', projection=True, name='Latents'),
               factors=RV(5, 'diag', projection=True, name='Factors'),
               **kwargs):
    super().__init__(n_labels=n_labels,
                     latents=latents,
                     factors=factors,
                     **kwargs)
