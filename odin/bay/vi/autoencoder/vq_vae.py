import warnings
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python.distributions import (Distribution,
                                                         Multinomial)
from tensorflow_probability.python.layers import DistributionLambda

from odin.bay.distributions import VectorQuantized
from odin.bay.helpers import KLdivergence
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import TrainStep


# ===========================================================================
# Helpers
# ===========================================================================
class VQVAEStep(TrainStep):

  def __call__(self):
    vae: VQVAE = self.vae
    assert isinstance(vae, VQVAE), \
      f"VQVAEStep only applied for VQVAE but, given VAE type: {type(vae)}"
    pX_Z, qZ_X = vae(self.inputs,
                     training=self.training,
                     mask=self.mask,
                     sample_shape=self.sample_shape,
                     **self.call_kw)
    ## Calculate the ELBO
    llk, div = self.vae.elbo(self.inputs,
                             pX_Z,
                             qZ_X,
                             training=self.training,
                             mask=self.mask,
                             return_components=True,
                             **self.elbo_kw)
    # sum all the components log-likelihood and divergence
    llk_sum = tf.constant(0., dtype=self.vae.dtype)
    div_sum = tf.constant(0., dtype=self.vae.dtype)
    for x in llk.values():
      llk_sum += x
    for x in div.values():
      div_sum += x
    elbo = llk_sum - div_sum
    if self.iw and tf.rank(elbo) > 1:
      elbo = self.vae.importance_weighted(elbo, axis=0)
    loss = -tf.reduce_mean(elbo)
    # metrics
    metrics = llk
    metrics.update(div)
    ## update the codebook
    if self.training:
      vae.quantizer.update_codebook(qZ_X)
    return loss, metrics


class VectorQuantizer(Layer):
  r"""

  Arguments:
    n_codes : int (default=64),
      Number of discrete codes in codebook.
    input_ndim : int (default=1),
      Number of dimension for a single input example.
  """

  def __init__(self,
               n_codes: int = 64,
               commitment_weight: float = 0.25,
               distance_metric: str = 'euclidean',
               trainable_prior=False,
               ema_decay: float = 0.99,
               ema_update: bool = False,
               epsilon: float = 1e-5,
               name: str = "VectorQuantizer"):
    super().__init__(name=name)
    self.n_codes = int(n_codes)
    self.distance_metric = str(distance_metric)
    self.trainable_prior = bool(trainable_prior)
    self.commitment_weight = tf.convert_to_tensor(commitment_weight,
                                                  dtype=self.dtype)
    self.ema_decay = tf.convert_to_tensor(ema_decay, dtype=self.dtype)
    self.ema_update = bool(ema_update)
    self.epsilon = tf.convert_to_tensor(epsilon, dtype=self.dtype)

  def build(self, input_shape):
    self.input_ndim = len(input_shape) - 2
    self.code_size = input_shape[-1]
    self.event_size = int(np.prod(input_shape[1:]))
    self.codebook: tf.Variable = self.add_weight(
        name="codebook",
        shape=[self.n_codes, self.code_size],
        initializer=tf.initializers.variance_scaling(distribution="uniform"),
        dtype=tf.float32,
        trainable=not self.ema_update)
    # exponential moving average
    if self.ema_update:
      self.ema_counts = self.add_weight(name="ema_counts",
                                        shape=[self.n_codes],
                                        initializer=tf.initializers.constant(0),
                                        trainable=False)
      self.ema_means = self.add_weight(name="ema_means",
                                       initializer=tf.initializers.constant(0),
                                       shape=self.codebook.shape,
                                       trainable=False)
      self.ema_means.assign(self.codebook)
    # create the prior and posterior
    prior_logits = self.add_weight(
        name="prior_logits",
        shape=input_shape[1:-1] + [self.n_codes],
        dtype=self.dtype,
        initializer=tf.initializers.constant(0),
        trainable=self.trainable_prior,
    )
    self._prior = Multinomial(total_count=1.0,
                              logits=prior_logits,
                              name="VectorQuantizerPrior")
    self._posterior = DistributionLambda(
        make_distribution_fn=lambda params: VectorQuantized(
            codes=params[0],
            assignments=params[1],
            nearest_codes=params[2],
            commitment=self.commitment_weight),
        convert_to_tensor_fn=Distribution.sample)
    return super().build(input_shape)

  @property
  def prior(self) -> Distribution:
    return self._prior

  @property
  def posterior(self) -> DistributionLambda:
    return self._posterior

  def call(self, codes, training=None, *args, **kwargs) -> VectorQuantized:
    r""" Uses codebook to find nearest neighbor for each code.

    Args:
      codes: A `float`-like `Tensor`,
        containing the latent vectors to be compared to the codebook.
        These are rank-3 with shape `[batch_size, ..., code_size]`.

    Returns:
      codes_straight_through: A `float`-like `Tensor`,
        the nearest entries with stopped gradient to the codebook,
        shape `[batch_size, ..., code_size]`.
    """
    indices = self.sample_indices(codes, one_hot=False)
    nearest_codebook_entries = self.sample_nearest(indices)
    dist: VectorQuantized = self.posterior(
        (
            codes,
            tf.one_hot(indices, depth=self.n_codes, axis=-1),
            nearest_codebook_entries,
        ),
        training=training,
    )
    dist.KL_divergence = KLdivergence(posterior=dist,
                                      prior=self.prior,
                                      analytic=True)
    # tf.debugging.assert_near(dist,
    #                          nearest_codebook_entries,
    #                          rtol=1e-5,
    #                          atol=1e-5)
    return dist

  def sample_indices(self, codes, one_hot=True) -> tf.Tensor:
    r""" Uses codebook to find nearest neighbor index for each code.

    Args:
      codes: A `float`-like `Tensor`,
        containing the latent vectors to be compared to the codebook.
        These are rank-3 with shape `[batch_size, ..., code_size]`.

    Returns:
      one_hot_assignments: a Tensor with shape `[batch_size, ..., n_codes]`
        The one-hot vectors corresponding to the matched codebook entry for
        each code in the batch.
    """
    tf.assert_equal(tf.shape(codes)[-1], self.code_size)
    input_shape = tf.shape(codes)
    codes = tf.reshape(codes, [-1, self.code_size])
    codebook = tf.transpose(self.codebook)
    distances = (tf.reduce_sum(codes**2, 1, keepdims=True) -
                 2 * tf.matmul(codes, codebook) +
                 tf.reduce_sum(codebook**2, 0, keepdims=True))
    assignments = tf.argmax(-distances, axis=1)
    assignments = tf.reshape(assignments, input_shape[:-1])
    if one_hot:
      assignments = tf.one_hot(assignments, depth=self.n_codes, axis=-1)
    return assignments

  def sample_nearest(self, indices) -> tf.Tensor:
    r""" Sample from the code book the nearest codes based on calculated
    one-hot assignments.

    Args:
      indices: A `int`-like `Tensor` containing the assignment
        vectors, shape `[batch_size, ...]`.

    Returns:
      nearest_codebook_entries: a Tensor with shape `[batch_size, ..., code_size]`
        The 1-nearest neighbor in Euclidean distance for each code in the batch.
    """
    return tf.nn.embedding_lookup(self.codebook, indices)

  def sample(self, sample_shape=(), seed=None) -> tf.Tensor:
    r""" Sampling from Multinomial prior """
    # [batch_dims, ..., n_codes]
    sample_shape = tf.nest.flatten(sample_shape)
    if len(sample_shape) == 0:
      sample_shape = [1]
    samples = self.prior.sample(sample_shape, seed)
    codebook = tf.reshape(self.codebook, [1] * (self.input_ndim + 1) +
                          [self.n_codes, self.code_size])
    return tf.reduce_sum(tf.expand_dims(samples, -1) * codebook, axis=-2)

  def update_codebook(self, vq_dist: VectorQuantized):
    assert isinstance(vq_dist, VectorQuantized), \
      ("vq_dist must be instance of VectorQuantized distribution, "
       f"but given: {type(vq_dist)}")
    assert self.ema_update, \
      "Exponential moving average update is not enable for VectorQuantizer"
    vq_dist.update_codebook(codebook=self.codebook,
                            counts=self.ema_counts,
                            means=self.ema_means,
                            decay=self.ema_decay,
                            epsilon=self.epsilon)
    return self

  def __str__(self):
    if self.built:
      input_shape = self.input_shape[1:]
    else:
      input_shape = None
    return (f"<VectorQuantizer input:{input_shape}"
            f" codebook:({self.n_codes}, {self.code_size})"
            f" commitment:{self.commitment_weight}"
            f" ema:(enable={self.ema_update}, decay={self.decay})"
            f" metric:{self.distance_metric}>")


# ===========================================================================
# Main
# ===========================================================================
class VQVAE(betaVAE):
  r"""

  Arguments:
    commitment_loss : float (default=0.25),
      or `beta`, weight for the commitment loss

  Reference:
    Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. "Neural Discrete
      Representation Learning". In _Conference on Neural Information Processing
      Systems_, 2017. https://arxiv.org/abs/1711.00937
  """

  def __init__(self,
               n_codes: int = 64,
               commitment_weight: float = 0.25,
               distance_metric: str = 'euclidean',
               trainable_prior: bool = False,
               ema_decay: float = 0.99,
               ema_update=False,
               beta=1.0,
               **kwargs):
    latents = kwargs.pop('latents', None)
    if latents is not None and not isinstance(latents, VectorQuantizer):
      warnings.warn(
          f"VQVAE uses VectorQuantizer latents, ignore: {type(latents)}")
    latents = VectorQuantizer(n_codes=n_codes,
                              commitment_weight=commitment_weight,
                              trainable_prior=trainable_prior,
                              distance_metric=distance_metric,
                              ema_decay=ema_decay,
                              ema_update=ema_update,
                              name="VQLatents")
    analytic = kwargs.pop('analytic', True)
    if not analytic:
      raise ValueError("VQVAE only support analytic KL-divergence.")
    super().__init__(beta=beta, latents=latents, analytic=analytic, **kwargs)
    self._quantizer = latents

  @property
  def quantizer(self) -> VectorQuantizer:
    return self._quantizer

  @property
  def codebook(self) -> tf.Variable:
    return self._quantizer.codebook

  @property
  def ema_update(self) -> bool:
    return self._quantizer.ema_update

  @property
  def ema_counts(self) -> tf.Variable:
    return self._quantizer.ema_counts

  @property
  def ema_means(self) -> tf.Variable:
    return self._quantizer.ema_means

  def _elbo(self, inputs, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training, **kwargs):
    llk, div = super()._elbo(inputs,
                             pX_Z,
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training,
                             **kwargs)
    for i, (qzx, name) in enumerate(zip(qZ_X, self.latent_names)):
      if isinstance(qzx, VectorQuantized):
        qzx: VectorQuantized
        if not self.ema_update:
          div[f'latents_{name}'] = qzx.latents_loss
        div[f'commitment_{name}'] = qzx.commitment_loss
    return llk, div

  def train_steps(self,
                  inputs,
                  training=True,
                  mask=None,
                  sample_shape=(),
                  iw=False,
                  elbo_kw={},
                  call_kw={}) -> VQVAEStep:
    self.step.assign_add(1)
    args = dict(vae=self,
                inputs=inputs,
                training=training,
                mask=mask,
                sample_shape=sample_shape,
                iw=iw,
                elbo_kw=elbo_kw,
                call_kw=call_kw)
    if self.quantizer.ema_update:
      yield VQVAEStep(**args)
    else:
      yield TrainStep(**args)
