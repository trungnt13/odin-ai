import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Sequence

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution
from typing_extensions import Literal

from odin.backend import TensorType
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.beta_vae import AnnealingVAE
from odin.bay.vi.autoencoder.factor_discriminator import FactorDiscriminator
from odin.bay.vi.autoencoder.variational_autoencoder import TrainStep, VAEStep
from odin.bay.vi.utils import prepare_ssl_inputs
from odin.utils import as_tuple


# ===========================================================================
# Helpers
# ===========================================================================
def _split_if_tensor(x):
  if tf.is_tensor(x):
    x1, x2 = tf.split(x, 2, axis=0)
  else:
    x1 = x
    x2 = x
  return x1, x2


def _split_inputs(inputs, mask, call_kw):
  """ Split the data into 2 partitions for training the VAE and Discriminator"""
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
  # split the call_kw
  call_kw1 = {}
  call_kw2 = {}
  for k, v in call_kw.items():
    if isinstance(v, collections.Sequence):
      v = [_split_if_tensor(i) for i in v]
      call_kw1[k] = [i[0] for i in v]
      call_kw2[k] = [i[1] for i in v]
    else:
      v1, v2 = _split_if_tensor(v)
      call_kw1[k] = v1
      call_kw2[k] = v2
  return (x1, mask1, call_kw1), (x2, mask2, call_kw2)


@dataclass
class FactorDiscriminatorStep(VAEStep):
  vae: 'FactorVAE' = None

  def call(self):
    px_z, qz_x = self.vae.last_outputs
    # if only inputs is provided without labels, error for ssl model,
    # need to flatten the list here.
    qz_xprime = self.vae.encode(self.inputs,
                                training=self.training,
                                mask=self.mask,
                                **self.call_kw)
    # discriminator loss
    dtc_loss = self.vae.dtc_loss(qz_x=qz_x,
                                 qz_xprime=qz_xprime,
                                 training=self.training)
    metrics = dict(dtc_loss=dtc_loss)
    ## applying the classifier loss,
    # if model is semi-supervised and the labels is given
    supervised_loss = 0.
    inputs = as_tuple(self.inputs)
    if self.vae.__class__.is_semi_supervised() and len(inputs) > 1:
      labels = inputs[1:]
      supervised_loss = self.vae.supervised_loss(labels,
                                                 qz_x=qz_x,
                                                 mask=self.mask,
                                                 training=self.training)
      metrics['supv_loss'] = supervised_loss
    return dtc_loss + supervised_loss, metrics


# ===========================================================================
# Main FactorVAE
# ===========================================================================
class FactorVAE(AnnealingVAE):
  """ The default encoder and decoder configuration is the same as proposed
  in (Kim et. al. 2018).

  The training procedure of FactorVAE is as follows:

  ```
  foreach iter:
    X = minibatch()
    X1, X2 = split(X, 2, axis=0)

    pX_Z, qz_x = vae(X1, training=True)
    loss = -vae.elbo(X1, pX_Z, qz_x, training=True)
    vae_optimizer.apply_gradients(loss, vae.parameters)

    qz_xprime = vae.encode(X2, training=True)
    dtc_loss = vae.dtc_loss(qz_x, qz_xprime, training=True)
    dis_optimizer.apply_gradients(dtc_loss, dis.parameters)
  ```

  Parameters
  ------------
  discriminator : a Dictionary or `keras.layers.Layer`.
      Keywords arguments for creating the `FactorDiscriminator`
  maximize_tc : a Boolean. If True, instead of minimize total correlation
      for more factorized latents, try to maximize the divergence.
  tc_coef : float.
      Weight for minimizing total correlation. According to (Kim et al. 2018),
      for dSprites dataset `tc_coef=35`, for `3DShapes` dataset `tc_coef=7`,
      and for `CelebA` dataset `tc_coef=6.4`.

  Note
  ------
  You should use double the `batch_size` since the minibatch will be splitted
  into 2 partitions for `X` and `X_prime`.

  It is recommended to use the same optimizers configuration like in the
  paper: `Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999)` for the VAE
  and `Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)` for the
  discriminator.

  Discriminator's Adam has learning rate `1e-4` for dSprites and `1e-5` for
  Shapes3D and other colored image datasets.

  Reference
  -----------
  Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """

  def __init__(self,
               discriminator_units: Sequence[int] = (
               1000, 1000, 1000, 1000, 1000),
               activation: Union[str, Callable[[], Any]] = tf.nn.relu,
               batchnorm: bool = False,
               tc_coef: float = 7.0,
               maximize_tc: bool = False,
               name: str = 'FactorVAE',
               **kwargs):
    ss_strategy = kwargs.pop('ss_strategy', 'logsumexp')
    labels = kwargs.pop(
      'labels', RVconf(1, 'bernoulli', projection=True, name="discriminator"))
    super().__init__(name=name, **kwargs)
    self.tc_coef = tf.convert_to_tensor(tc_coef,
                                        dtype=self.dtype,
                                        name='tc_coef')
    ## init discriminator
    self.discriminator = FactorDiscriminator(
      units=as_tuple(discriminator_units),
      activation=activation,
      batchnorm=batchnorm,
      ss_strategy=ss_strategy,
      observation=labels)
    ## Discriminator and VAE must be trained separately
    self.disc_params = []
    self.vae_params = []
    self.maximize_tc = bool(maximize_tc)
    ## For training
    # store class for training factor discriminator, this allow later
    # modification without re-writing the train_steps method
    self._is_pretraining = False

  def build(self, input_shape=None) -> 'FactorVAE':
    super().build(input_shape)
    zdim = int(sum(np.prod(z.event_shape) for z in as_tuple(self.latents)))
    self.discriminator.build((None, zdim))
    # split the parameters
    self.disc_params = self.discriminator.trainable_variables
    exclude = set(id(p) for p in self.disc_params)
    self.vae_params = [
      p for p in self.trainable_variables if id(p) not in exclude
    ]
    return self

  @property
  def is_pretraining(self):
    return self._is_pretraining

  def pretrain(self):
    r""" Pretraining only train the VAE without the factor discriminator """
    self._is_pretraining = True
    return self

  def finetune(self):
    self._is_pretraining = False
    return self

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    px_z, qz_x = self.last_outputs
    # by default, this support multiple latents by concatenating all latents
    if self.is_pretraining and training:
      tc = 0.
    else:
      tc = self.total_correlation(qz_x=qz_x, training=training)
    if self.maximize_tc:
      tc = -tc
    kl['tc'] = tc
    return llk, kl

  def total_correlation(self,
                        qz_x: Distribution,
                        training: Optional[bool] = None) -> tf.Tensor:
    return self.tc_coef * self.discriminator.total_correlation(
      qz_x, training=training)

  def dtc_loss(self,
               qz_x: Distribution,
               qz_xprime: Optional[Distribution] = None,
               training: Optional[bool] = None) -> tf.Tensor:
    """ Discrimination loss between real and permuted codes Algorithm (2) """
    return self.discriminator.dtc_loss(qz_x,
                                       qz_xprime=qz_xprime,
                                       training=training)

  def train_steps(self,
                  inputs: Union[TensorType, List[TensorType]],
                  training: bool = True,
                  mask: Optional[TensorType] = None,
                  name: str = '',
                  **kwargs) -> TrainStep:
    """ Facilitate multiple steps training for each iteration (similar to GAN)

    Example
    -------

    ```
    model = FactorVAE()
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
    # split the data
    (x1, mask1, call_kw1), \
    (x2, mask2, call_kw2) = _split_inputs(inputs, mask, kwargs)
    # first step optimize VAE with total correlation loss
    yield VAEStep(vae=self,
                  inputs=x1,
                  training=training,
                  mask=mask1,
                  call_kw=call_kw1,
                  parameters=self.vae_params,
                  name=f'elbo{name}')
    # second step optimize the discriminator for discriminate permuted code
    # skip training Discriminator of pretraining
    if not self.is_pretraining:
      yield FactorDiscriminatorStep(vae=self,
                                    inputs=x2,
                                    training=training,
                                    mask=mask2,
                                    call_kw=call_kw2,
                                    parameters=self.disc_params,
                                    name=f'disc{name}')

  def fit(self,
          train,
          *,
          valid=None,
          optimizer=(
              tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
              tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
          ),
          **kwargs):
    """ Override the original fit method of keras to provide simplified
    procedure with `VariationalAutoencoder.optimize` and
    `VariationalAutoencoder.train_steps` """
    assert isinstance(optimizer, (tuple, list)) and len(optimizer) == 2, \
      ("Two different optimizer must be provided, "
       "one for VAE, and one of FactorDiscriminator")
    return super().fit(train=train, valid=valid, optimizer=optimizer, **kwargs)

  def __str__(self):
    text = super().__str__()
    text += "\n Discriminator:\n  "
    text += "\n  ".join(str(self.discriminator).split('\n'))
    return text


# ===========================================================================
# Same as Factor VAE but with multi-task semi-supervised extension
# ===========================================================================
class SemiFactorVAE(FactorVAE):
  """Semi-supervised Factor VAE

  Note:
    The classifier won't be optimized during the training, with an unstable
    latent space.

    But if a VAE is pretrained, then, the extracted latents  are feed into
    the classifier for training, then it could reach > 90% accuracy easily.
  """

  def __init__(
      self,
      labels: RVconf = RVconf(10, 'onehot', projection=True, name="Labels"),
      alpha: float = 10.,
      ss_strategy: Literal['sum', 'logsumexp', 'mean', 'max',
                           'min'] = 'logsumexp',
      name: str = 'SemiFactorVAE',
      **kwargs,
  ):
    super().__init__(ss_strategy=ss_strategy,
                     labels=labels,
                     name=name,
                     **kwargs)
    self.n_labels = self.discriminator.n_outputs
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name='alpha')

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    return super().encode(X[0], training=training, mask=None, **kwargs)

  def classify(self,
               inputs: Union[TensorType, List[TensorType]],
               training: Optional[bool] = None) -> Distribution:
    qz_x = self.encode(inputs, training=training)
    if hasattr(self.discriminator, '_to_samples'):
      z = self.discriminator._to_samples(qz_x)
    else:
      z = qz_x
    y = self.discriminator(z, training=training)
    assert isinstance(y, Distribution), \
      f"Discriminator must return a Distribution, but returned: {y}"
    return y

  def supervised_loss(self,
                      labels: tf.Tensor,
                      qz_x: Distribution,
                      mask: Optional[TensorType] = None,
                      training: Optional[bool] = None) -> tf.Tensor:
    """The semi-supervised classifier loss, `mask` is given to indicate
    labelled examples (i.e. `mask=1`), and otherwise, unlabelled examples.
    """
    return self.alpha * self.discriminator.supervised_loss(
      labels=labels, qz_x=qz_x, mask=mask, training=training)

  @classmethod
  def is_semi_supervised(self) -> bool:
    return True


# ===========================================================================
# Separated latents for TC factorization
# ===========================================================================
class Factor2VAE(FactorVAE):
  """The same architecture as `FactorVAE`, however, utilize two different
  latents `Z` for contents generalizability and `C` for disentangling of
  invariant factors."""

  def __init__(self,
               latents: RVconf = RVconf(5, 'mvndiag',
                                        projection=True,
                                        name='Latents'),
               factors: RVconf = RVconf(5,
                                        'mvndiag',
                                        projection=True,
                                        name="Factors"),
               **kwargs):
    latents = tf.nest.flatten(latents)
    assert isinstance(factors, RVconf), \
      "factors must be instance of RVmeta, but given: %s" % \
      str(type(factors))
    latents.append(factors)
    super().__init__(latents=latents,
                     latent_dim=int(np.prod(factors.event_shape)),
                     **kwargs)
    self.factors = factors

  def _elbo(self, inputs, pX_Z, qz_x, mask, training):
    llk, div = super(AnnealingVAE, self)._elbo(
      inputs,
      pX_Z,
      qz_x,
      mask=mask,
      training=training,
    )
    # only use the assumed factors space for total correlation
    tc = self.total_correlation(qz_x[-1], training=training)
    if self.maximize_tc:
      tc = -tc
    div[f'tc_{self.factors.name}'] = tc
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
  train = ds.create_dataset(partition='train', label_percent=0.3, batch_size=128)
  valid = ds.create_dataset(partition='valid', label_percent=1.0, batch_size=128)

  # construction of SemiFactor2VAE for MNIST dataset
  vae = SemiFactor2VAE(encoder='mnist',
                       outputs=RVmeta((28, 28, 1), 'bern', name="Image"),
                       latents=RVmeta(10, 'mvndiag', projection=True, name='Latents'),
                       factors=RVmeta(10, 'mvndiag', projection=True, name='Factors'),
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
               latents=RVconf(5, 'mvndiag', projection=True, name='Latents'),
               factors=RVconf(5, 'mvndiag', projection=True, name='Factors'),
               **kwargs):
    super().__init__(latents=latents, factors=factors, **kwargs)
