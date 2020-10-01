import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from odin.bay.random_variable import RandomVariable
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.factor_discriminator import FactorDiscriminator
from odin.bay.vi.autoencoder.variational_autoencoder import (DatasetV2,
                                                             OptimizerV2,
                                                             TensorTypes,
                                                             TrainStep, VAEStep)
from tensorflow.python import keras
from tensorflow_probability.python.distributions import Distribution
from typing_extensions import Literal


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
  r""" Split the data into 2 partitions for training the VAE and
  Discriminator """
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
    is_list = False
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
  qZ_X: Distribution

  def optimize(self, inputs, mask, qZ_X, qZ_Xprime,
               training) -> Tuple[tf.Tensor, Dict[str, Any]]:
    dtc_loss = self.vae.dtc_loss(qZ_X,
                                 qZ_Xprime,
                                 training=training,
                                 apply_lamda=True)
    metrics = dict(dtc_loss=dtc_loss)
    ## applying the classifier loss
    supervised_loss = 0.
    if hasattr(self.vae, 'supervised_loss'):
      labels = inputs[len(self.vae.output_layers):]
      supervised_loss = self.vae.supervised_loss(labels,
                                                 qZ_X,
                                                 mask=mask,
                                                 training=training,
                                                 apply_alpha=True)
      metrics['supervised_loss'] = supervised_loss
    return dtc_loss + supervised_loss, metrics

  def call(self) -> Tuple[tf.Tensor, Dict[str, Any]]:
    inputs = self.inputs
    qZ_X = self.qZ_X
    mask = self.mask
    training = self.training
    call_kw = self.call_kw
    qZ_Xprime = self.vae.encode(inputs, training=training, mask=mask, **call_kw)
    return self.optimize(inputs, mask, qZ_X, qZ_Xprime, training)


@dataclass
class Factor2DiscriminatorStep(FactorDiscriminatorStep):

  def call(self) -> Tuple[tf.Tensor, Dict[str, Any]]:
    inputs = self.inputs
    qZ_X = self.qZ_X
    mask = self.mask
    training = self.training
    call_kw = self.call_kw
    qZ_Xprime = self.vae.encode(inputs, training=training, mask=mask, **call_kw)
    # only select the last latent space
    return self.optimize(inputs, mask, qZ_X[-1], qZ_Xprime[-1], training)


# ===========================================================================
# Main factorVAE
# ===========================================================================
class factorVAE(betaVAE):
  r""" The default encoder and decoder configuration is the same as proposed
  in (Kim et. al. 2018).

  The training procedure of factorVAE is as follows:

  ```
  foreach iter:
    X = minibatch()
    X1, X2 = split(X, 2, axis=0)

    pX_Z, qZ_X = vae(X1, training=True)
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
    gamma : a Scalar. Weight for minimizing total correlation
    beta : a Scalar. Weight for minimizing Kl-divergence to the prior
    lamda : a Scalar. Weight for minimizing the discriminator loss

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
               discriminator: Union[FactorDiscriminator, Dict[str, Any]] = dict(
                   units=[1000, 1000, 1000, 1000, 1000]),
               gamma: float = 6.0,
               beta: float = 1.0,
               lamda: float = 1.0,
               maximize_tc: bool = False,
               **kwargs):
    super().__init__(beta=beta, reduce_latent='concat', **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    self.lamda = tf.convert_to_tensor(lamda, dtype=self.dtype, name='lamda')
    # all latents will be concatenated (a bit spooky to customize latent_dim)
    latent_dim = np.prod(
        sum(np.array(layer.event_shape) for layer in self.latent_layers))
    ## init discriminator
    if not isinstance(discriminator, FactorDiscriminator):
      discriminator = FactorDiscriminator(input_shape=(latent_dim,),
                                          **discriminator)
    self.discriminator = discriminator
    assert hasattr(self.discriminator, 'total_correlation') and \
      hasattr(self.discriminator, 'dtc_loss'), \
        (f"discriminator of type: {type(self.discriminator)} "
         "must has method total_correlation and dtc_loss.")
    # VAE and discriminator must be trained separated so we split
    # their params here
    self.disc_params = self.discriminator.trainable_variables
    exclude = set(id(p) for p in self.disc_params)
    self.vae_params = [
        p for p in self.trainable_variables if id(p) not in exclude
    ]
    self.maximize_tc = bool(maximize_tc)
    ## For training
    # store class for training factor discriminator, this allow later
    # modification without re-writing the train_steps method
    self._factor_step = FactorDiscriminatorStep
    self._is_pretraining = False

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

  def _elbo(self, inputs, pX_Z, qZ_X, mask, training):
    llk, div = super()._elbo(inputs, pX_Z, qZ_X, mask=mask, training=training)
    # by default, this support multiple latents by concatenating all latents
    tc = self.total_correlation(qZ_X, apply_gamma=True, training=training)
    if self.maximize_tc:
      tc = -tc
    div['total_corr'] = tc
    return llk, div

  def total_correlation(self,
                        qZ_X: Distribution,
                        training: Optional[bool] = None,
                        apply_gamma: bool = False) -> tf.Tensor:
    r""" Using the discriminator output to estimate total correlation of
    the latents """
    if self.is_pretraining:
      return 0.
    tc = self.discriminator.total_correlation(qZ_X, training=training)
    if apply_gamma:
      tc = self.gamma * tc
    return tc

  def dtc_loss(self,
               qZ_X: Distribution,
               qZ_Xprime: Optional[Distribution] = None,
               training: Optional[bool] = None,
               apply_lamda: bool = False) -> tf.Tensor:
    r""" Discrimination loss between real and permuted codes Algorithm (2) """
    loss = self.discriminator.dtc_loss(qZ_X,
                                       qZ_Xprime=qZ_Xprime,
                                       training=training)
    if apply_lamda:
      loss = self.lamda * loss
    return loss

  def train_steps(self,
                  inputs: Union[TensorTypes, List[TensorTypes]],
                  training: bool = True,
                  mask: Optional[TensorTypes] = None,
                  call_kw: Dict[str, Any] = {}) -> TrainStep:
    r""" Facilitate multiple steps training for each iteration (similar to GAN)

    Example:
    ```
    vae = factorVAE()
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
    (x1, mask1, call_kw1), \
      (x2, mask2, call_kw2) = _split_inputs(inputs, mask, call_kw)
    # first step optimize VAE with total correlation loss
    step1 = VAEStep(vae=self,
                    inputs=x1,
                    training=training,
                    mask=mask1,
                    call_kw=call_kw1,
                    parameters=self.vae_params)
    yield step1
    # second step optimize the discriminator for discriminate permuted code
    # skip training Discriminator of pretraining
    if not self.is_pretraining:
      step2 = self._factor_step(vae=self,
                                inputs=x2,
                                qZ_X=step1.qZ_X,
                                training=training,
                                mask=mask2,
                                call_kw=call_kw2,
                                parameters=self.disc_params)
      yield step2

  def fit(self,
          train: Union[TensorTypes, DatasetV2],
          valid: Optional[Union[TensorTypes, DatasetV2]] = None,
          optimizer: Tuple[OptimizerV2, OptimizerV2] = [
              tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
              tf.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
          ],
          **kwargs):
    r""" Override the original fit method of keras to provide simplified
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
class SemifactorVAE(factorVAE):
  r""" Semi-supervised Factor VAE

  Note:
    The classifier won't be optimized during the training, with an unstable
    latent space.

    But if a VAE is pretrained, then, the extracted latents  are feed into
    the classifier for training, then it could reach > 90% accuracy easily.
  """

  def __init__(self,
               labels: RandomVariable = RandomVariable(10,
                                                       'onehot',
                                                       projection=True,
                                                       name="Labels"),
               discriminator: Union[FactorDiscriminator, Dict[str, Any]] = dict(
                   units=[1000, 1000, 1000, 1000, 1000]),
               alpha: float = 10.,
               ss_strategy: Literal['sum', 'logsumexp', 'mean', 'max',
                                    'min'] = 'logsumexp',
               **kwargs):
    if isinstance(discriminator, dict):
      discriminator['outputs'] = labels
      discriminator['ss_strategy'] = ss_strategy
    super().__init__(discriminator=discriminator, **kwargs)
    self.n_labels = self.discriminator.n_outputs
    self.n_unsupervised = len(self.output_layers)
    self.labels = labels
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name='alpha')

  def encode(self,
             inputs: Union[TensorTypes, List[TensorTypes]],
             training: Optional[bool] = None,
             mask: Optional[TensorTypes] = None,
             **kwargs) -> Union[Distribution, List[Distribution]]:
    inputs = tf.nest.flatten(inputs)[:self.n_unsupervised]
    if len(inputs) == 1:
      inputs = inputs[0]
    return super().encode(inputs, training=training, mask=mask, **kwargs)

  def classify(self,
               inputs: Union[TensorTypes, List[TensorTypes]],
               training: Optional[bool] = None) -> Distribution:
    qZ_X = self.encode(inputs, training=training)
    if hasattr(self.discriminator, '_to_samples'):
      z = self.discriminator._to_samples(qZ_X)
    else:
      z = qZ_X
    y = self.discriminator(z, training=training)
    assert isinstance(
        y, Distribution
    ), f"Discriminator must return a Distribution, but returned: {y}"
    return y

  def supervised_loss(self,
                      labels: tf.Tensor,
                      qZ_X: Distribution,
                      mask: Optional[TensorTypes] = None,
                      training: Optional[bool] = None,
                      apply_alpha: bool = False):
    r""" The semi-supervised classifier loss, `mask` is given to indicate
    labelled examples (i.e. `mask=1`), and otherwise, unlabelled examples.
    """
    loss = self.discriminator.supervised_loss(labels=labels,
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
class Factor2VAE(factorVAE):
  r""" The same architecture as `factorVAE`, however, utilize two different
  latents `Z` for contents generalizability and `C` for disentangling of
  invariant factors.

  TODO:
  """

  def __init__(self,
               latents: RandomVariable = RandomVariable(5,
                                                        'diag',
                                                        projection=True,
                                                        name='Latents'),
               factors: RandomVariable = RandomVariable(5,
                                                        'diag',
                                                        projection=True,
                                                        name="Factors"),
               **kwargs):
    latents = tf.nest.flatten(latents)
    assert isinstance(factors, RandomVariable), \
      "factors must be instance of RandomVariable, but given: %s" % \
        str(type(factors))
    latents.append(factors)
    super().__init__(latents=latents,
                     latent_dim=int(np.prod(factors.event_shape)),
                     **kwargs)
    self.factors = factors
    self._factor_step = Factor2DiscriminatorStep

  def _elbo(self, inputs, pX_Z, qZ_X, mask, training):
    llk, div = super(betaVAE, self)._elbo(
        inputs,
        pX_Z,
        qZ_X,
        mask=mask,
        training=training,
    )
    # only use the assumed factors space for total correlation
    tc = self.total_correlation(qZ_X[-1], apply_gamma=True, training=training)
    if self.maximize_tc:
      tc = -tc
    div[f'tc_{self.factors.name}'] = tc
    return llk, div


class SemiFactor2VAE(SemifactorVAE, Factor2VAE):
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
                       outputs=RandomVariable((28, 28, 1), 'bern', name="Image"),
                       latents=RandomVariable(10, 'diag', projection=True, name='Latents'),
                       factors=RandomVariable(10, 'diag', projection=True, name='Factors'),
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
               latents=RandomVariable(5,
                                      'diag',
                                      projection=True,
                                      name='Latents'),
               factors=RandomVariable(5,
                                      'diag',
                                      projection=True,
                                      name='Factors'),
               **kwargs):
    super().__init__(latents=latents, factors=factors, **kwargs)
