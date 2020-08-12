import warnings
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python.distributions import (Distribution,
                                                         Multinomial)
from tensorflow_probability.python.layers import DistributionLambda

from odin.bay.distributions import VectorQuantized
from odin.bay.helpers import KLdivergence
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
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
    ## update the codebook
    qZ_X: VectorQuantized
    qZ_X.update_codebook(codebook=vae.codebook,
                         counts=vae.ema_counts,
                         means=vae.ema_means,
                         decay=vae.decay,
                         perturb=vae.ema_perturb)
    # store so it could be reused
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
               distance_metric: str = 'euclidean',
               trainable_prior=False,
               name: str = "VectorQuantizer"):
    super().__init__(name=name)
    self.n_codes = int(n_codes)
    self.distance_metric = str(distance_metric)
    self.trainable_prior = bool(trainable_prior)

  def build(self, input_shape):
    self.input_ndim = len(input_shape) - 2
    self.code_size = input_shape[-1]
    self.event_size = int(np.prod(input_shape[1:]))
    self.codebook: tf.Variable = self.add_weight(
        name="codebook",
        shape=[self.n_codes, self.code_size],
        initializer=tf.initializers.variance_scaling(distribution="uniform"),
        dtype=tf.float32,
        trainable=True)
    self.ema_counts: tf.Variable = self.add_weight(
        initializer=tf.initializers.constant(0),
        shape=[self.n_codes],
        name="ema_counts",
        trainable=False)
    self.ema_means: tf.Variable = self.add_weight(name="ema_means",
                                                  shape=self.codebook.shape,
                                                  trainable=False)
    self.ema_means.assign(tf.convert_to_tensor(self.codebook))
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
            codes=params[0], assignments=params[1], nearest_codes=params[2]),
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
    batch_size = tf.shape(codes)[0]
    #
    one_hot_assignments = self.sample_index(codes, one_hot=True)
    nearest_codebook_entries = self.sample_nearest(one_hot_assignments)
    dist: VectorQuantized = self.posterior(
        (codes, one_hot_assignments, nearest_codebook_entries),
        training=training)
    dist.KL_divergence = KLdivergence(posterior=dist,
                                      prior=self.prior,
                                      analytic=True)
    return dist

  def sample_index(self, codes, one_hot=True) -> tf.Tensor:
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
    # make codebook broadcastable
    codebook = tf.reshape(self.codebook, [1] * (self.input_ndim + 1) +
                          [self.n_codes, self.code_size])
    codes = tf.expand_dims(codes, axis=-2)
    # Euclidean distances (in case of 2D images)
    # [batch_size, w, h, 1, code_size] - [1, 1, 1, n_codes, code_size]
    # -> distance: [batch_size, w, h, n_codes]
    distances = tf.norm(codes - codebook, axis=-2, ord=self.distance_metric)
    # get the assignment to each code
    assignments = tf.argmin(input=distances, axis=-1)
    if one_hot:
      return tf.one_hot(assignments, depth=self.n_codes)
    return assignments

  def sample_nearest(self, one_hot_assignments) -> tf.Tensor:
    r""" Sample from the code book the nearest codes based on calculated
    one-hot assignments.

    Args:
      one_hot_assignments: A `int`-like `Tensor` containing the assignment
        vectors. These are rank-3 with shape
        `[batch_size, ..., n_codes]`.

    Returns:
      nearest_codebook_entries: a Tensor with shape `[batch_size, ..., code_size]`
        The 1-nearest neighbor in Euclidean distance for each code in the batch.
    """
    tf.assert_equal(tf.shape(one_hot_assignments)[-1], self.n_codes)
    # make codebook broadcastable
    codebook = tf.reshape(self.codebook, [1] * (self.input_ndim + 1) +
                          [self.n_codes, self.code_size])
    # get the nearest codebook entries
    # [batch_size, ..., n_codes, 1] * [1, ..., n_codes, code_size]
    # -> reduce_sum([batch_size, ..., n_codes, code_size], axis=-2)
    # -> [batch_size, ..., code_size]
    nearest_codebook_entries = tf.reduce_sum(
        tf.expand_dims(one_hot_assignments, -1) * codebook, axis=-2)
    return nearest_codebook_entries

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

  def __str__(self):
    if self.built:
      input_shape = self.input_shape[1:]
    else:
      input_shape = None
    return (f"<VectorQuantizer input:{input_shape}"
            f" codebook:({self.n_codes}, {self.code_size})"
            f" metric:{self.distance_metric}>")


# ===========================================================================
# Main
# ===========================================================================
class VQVAE(BetaVAE):
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
               n_codes=64,
               commitment_weight: float = 0.25,
               distance_metric: str = 'euclidean',
               trainable_prior: bool = False,
               decay: float = 0.99,
               ema_perturb: float = 1e-5,
               beta=1.0,
               **kwargs):
    latents = kwargs.pop('latents', None)
    if latents is not None and not isinstance(latents, VectorQuantizer):
      warnings.warn(
          f"VQVAE uses VectorQuantizer latents, ignore: {type(latents)}")
    latents = VectorQuantizer(n_codes=n_codes,
                              trainable_prior=trainable_prior,
                              distance_metric=distance_metric,
                              name="VQVAE_latents")
    analytic = kwargs.pop('analytic', True)
    if not analytic:
      raise ValueError("VQVAE only support analytic KL-divergence.")
    super().__init__(beta=beta, latents=latents, analytic=analytic, **kwargs)
    self.vq_latents = latents
    self.ema_perturb = tf.convert_to_tensor(ema_perturb, dtype=self.dtype)
    self.decay = tf.convert_to_tensor(decay, dtype=self.dtype)
    self.commitment_weight = tf.convert_to_tensor(commitment_weight,
                                                  dtype=self.dtype)

  @property
  def codebook(self) -> tf.Variable:
    return self.vq_latents.codebook

  @property
  def ema_counts(self) -> tf.Variable:
    return self.vq_latents.ema_counts

  @property
  def ema_means(self) -> tf.Variable:
    return self.vq_latents.ema_means

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
        div[f'commitment_{name}'] = self.commitment_weight * qzx.commitment_loss
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
    yield VQVAEStep(vae=self,
                    inputs=inputs,
                    training=training,
                    mask=mask,
                    sample_shape=sample_shape,
                    iw=iw,
                    elbo_kw=elbo_kw,
                    call_kw=call_kw)
