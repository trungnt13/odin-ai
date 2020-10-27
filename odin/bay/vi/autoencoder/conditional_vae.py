from __future__ import absolute_import, annotations, division, print_function

import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from odin import backend as bk
from odin.backend.keras_helpers import layer2text
from odin.bay.helpers import coercible_tensor, kl_divergence
from odin.bay.layers.distribution_util_layers import ConditionalTensorLayer
from odin.bay.random_variable import RandomVariable, RVmeta
from odin.bay.vi._base import VariationalModel
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.factor_discriminator import FactorDiscriminator
from odin.bay.vi.autoencoder.variational_autoencoder import LayerCreator
from odin.bay.vi.utils import marginalize_categorical_labels
from odin.networks import TensorTypes
from odin.networks.conditional_embedding import (DictionaryEmbedding, Embedder,
                                                 IdentityEmbedding,
                                                 ProjectionEmbedding)
from odin.utils import as_tuple
from tensorflow.python import keras
from tensorflow_probability.python.distributions import (
    Categorical, Deterministic, Distribution, JointDistributionSequential,
    OneHotCategorical, VectorDeterministic)
from typing_extensions import Literal

__all__ = ['conditionalM2VAE']


# ===========================================================================
# Helpers
# ===========================================================================
def _batch_size(x):
  batch_size = x.shape[0]
  if batch_size is None:
    batch_size = tf.shape(x)[0]
  return batch_size


def prepare_ssl_inputs(
    inputs: Union[TensorTypes, List[TensorTypes]],
    mask: TensorTypes,
    n_unsupervised_inputs: int,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], tf.Tensor]:
  """Prepare the inputs for the semi-supervised learning,
  three cases are considered:

    - Only the unlabeled data given
    - Only the labeled data given
    - A mixture of both unlabeled and labeled data, indicated by mask

  Parameters
  ----------
  inputs : Union[TensorTypes, List[TensorTypes]]
  n_unsupervised_inputs : int
  mask : TensorTypes
      The `mask` is given as indicator, `1` for labeled sample and
      `0` for unlabeled samples

  Returns
  -------
  Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
      - List of inputs tensors
      - List of labels tensors
      - mask tensor
  """
  inputs = tf.nest.flatten(as_tuple(inputs))
  batch_size = _batch_size(inputs[0])
  ## no labels provided
  if len(inputs) == n_unsupervised_inputs:
    X = inputs
    y = []
    mask = tf.cast(tf.zeros(shape=(batch_size, 1)), dtype=tf.bool)
  ## labels is provided
  else:
    X = inputs[:n_unsupervised_inputs]
    y = inputs[n_unsupervised_inputs:]
    if mask is None:  # all data is labelled
      mask = tf.cast(tf.ones(shape=(batch_size, 1)), tf.bool)
  y = [i for i in y if i is not None]
  return X, y, mask


def split_ssl_inputs(
    X: List[tf.Tensor],
    y: List[tf.Tensor],
    mask: tf.Tensor,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
  """Split semi-supervised inputs into unlabelled and labelled data

  Parameters
  ----------
  X : List[tf.Tensor]
  y : List[tf.Tensor]
  mask : tf.Tensor

  Returns
  -------
  Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], tf.Tensor]
      - list of unlablled inputs
      - list of labelled inputs
      - list of labels
  """
  if not isinstance(X, (tuple, list)):
    X = [X]
  if y is None:
    y = []
  elif not isinstance(y, (tuple, list)):
    y = [y]
  if mask is None:
    mask = tf.cast(tf.zeros(shape=(_batch_size(X[0]), 1)), dtype=tf.bool)
  # flatten the mask
  mask = tf.reshape(mask, (-1,))
  # split into unlabelled and labelled data
  X_unlabelled = [tf.boolean_mask(i, tf.logical_not(mask), axis=0) for i in X]
  X_labelled = [tf.boolean_mask(i, mask, axis=0) for i in X]
  y_labelled = [tf.boolean_mask(i, mask, axis=0) for i in y]
  return X_unlabelled, X_labelled, y_labelled


# ===========================================================================
# main classes
# ===========================================================================
class conditionalM2VAE(betaVAE):
  """Implementation of M2 model (Kingma et al. 2014). The default
  configuration of this layer is optimized for MNIST.

  The inference model:
  ```
  q(z|y,x) = N(z|f_mu(y,x),f_sig(x)))
  q(y|x) = Cat(y|pi(x))
  q(pi|x) = g(x)
  ```

  The generative model:
  ```
  p(x,y,z) = p(x|y,z;theta)p(y)p(z)
  p(y) = Cat(y|pi)
  p(z) = N(z|0,I)
  ```

  Parameters
  ------------
  conditional_embedding : {'repeat', 'embed', 'project'}.
      Strategy for concatenating one-hot encoded labels to inputs.
  alpha : a Scalar.
      The weight of discriminative objective added to the labelled data objective.
      In the paper, it is recommended:
      `alpha = 0.1 * (n_total_samples / n_labelled_samples)`

  Example
  ---------
  ```
  from odin.fuel import MNIST
  from odin.bay.vi.autoencoder import conditionalM2VAE
  ds = MNIST()
  train = ds.create_dataset(partition='train', inc_labels=0.1)
  test = ds.create_dataset(partition='test', inc_labels=True)
  encoder, decoder = create_image_autoencoder(image_shape=(28, 28, 1),
                                              input_shape=(28, 28, 2),
                                              center0=True,
                                              latent_shape=20)
  vae = conditionalM2VAE(encoder=encoder,
                         decoder=decoder,
                         conditional_embedding='embed',
                         alpha=0.1 * 10)
  vae.fit(train, compile_graph=True, epochs=-1, max_iter=8000, sample_shape=5)
  ```

  References
  ------------
  Kingma, D.P., Rezende, D.J., Mohamed, S., Welling, M., 2014.
    "Semi-Supervised Learning with Deep Generative Models".
    arXiv:1406.5298 [cs, stat].
  """

  def __init__(
      self,
      classifier: keras.layers.Layer,
      n_classes: int,
      labels_embedder: Tuple[Embedder, Embedder] = IdentityEmbedding,
      alpha: float = 0.1,
      beta: float = 1.,
      temperature: float = 10.,
      name: str = 'ConditionalM2VAE',
      marginalize: bool = False,
      **kwargs,
  ):
    super().__init__(beta=beta, name=name, **kwargs)
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name="alpha")
    self.marginalize = bool(marginalize)
    self.n_classes = int(n_classes)
    self.classifier = classifier
    if marginalize:
      temperature = 0
    if temperature == 0.:
      posterior = 'onehot'
      dist_kw = dict()
      self.relaxed = False
    else:
      posterior = 'relaxedonehot'
      dist_kw = dict(temperature=temperature)
      self.relaxed = True
    self.labels = RVmeta(n_classes,
                         posterior,
                         projection=True,
                         prior=OneHotCategorical(probs=[1. / n_classes] *
                                                 n_classes),
                         name='labels',
                         kwargs=dist_kw).create_posterior()
    # note: we assume only 1 observation variable and 1 latents variable for
    # this implementation
    shape = self.latents[0].event_shape
    self.labels_embedder = [
        e(n_classes=n_classes, event_shape=shape) if isinstance(e, type) else e
        for e in as_tuple(labels_embedder, N=2)
    ]

  def build(self, input_shape):
    self.classifier.build(input_shape=input_shape)
    return super().build(input_shape=input_shape)

  def classify(self,
               inputs: TensorTypes,
               training: bool = False) -> Distribution:
    """Return the prediction of labels"""
    if isinstance(inputs, (tuple, list)) and len(inputs) == 1:
      inputs = inputs[0]
    h = self.classifier(inputs, training=training)
    return self.labels(h, training=training)

  def sample_labels(self,
                    sample_shape: List[int] = (),
                    seed: int = 1) -> tf.Tensor:
    """Sample labels from prior of the labels distribution"""
    return bk.atleast_2d(
        self.labels.prior.sample(sample_shape=sample_shape, seed=seed))

  def encode(self,
             inputs: Union[TensorTypes, List[TensorTypes]],
             training: Optional[bool] = None,
             mask: Optional[TensorTypes] = None,
             **kwargs) -> JointDistributionSequential:
    X, y, mask = prepare_ssl_inputs(inputs,
                                    mask=mask,
                                    n_unsupervised_inputs=self.n_observation)
    if len(y) == 0:
      py = self.classify(X, training=training)
    else:  # only support single labels model
      py = coercible_tensor(VectorDeterministic(loc=y[0]))
    # encode normally
    h_e = [
        bk.flatten(fe(X[0], training=training, mask=mask), n_outdim=2)
        for fe in self.encoder
    ]
    if len(h_e) > 1:
      h_e = tf.concat(h_e, axis=-1)
    else:
      h_e = h_e[0]
    # conditional embedding y
    y_embedded = self.labels_embedder[0](py)
    h_e = tf.concat([h_e, y_embedded], axis=-1)
    qz_x = [fz(h_e, training=training) for fz in self.latents]
    qz_x.append(py)
    return qz_x

  def decode(self, latents, training=None, mask=None, **kwargs):
    qz_x, py = latents[:-1], latents[-1]
    y_embedded = self.labels_embedder[1](py)
    z = tf.concat(qz_x + [y_embedded], axis=-1)
    return super().decode(z, training=training, mask=mask)

  def elbo_components(self, inputs, training=None, mask=None):
    X, y, mask = prepare_ssl_inputs(inputs,
                                    mask=mask,
                                    n_unsupervised_inputs=self.n_observation)
    X_u, X_l, y_l = split_ssl_inputs(X, y, mask)
    # for simplication only 1 inputs and 1 labels are supported
    X_u, X_l = X_u[0], X_l[0]
    if len(y_l) > 0:
      y_l = y_l[0]
    else:
      y_l = None
    # marginalize the unsupervised data
    if self.marginalize:
      X_u, y_u = marginalize_categorical_labels(
          X=X_u,
          n_classes=self.n_classes,
          dtype=self.dtype,
      )
    else:
      y_u = None
    ### for unlabelled data (assumed always available)
    llk_u, kl_u = super().elbo_components(inputs=[X_u, y_u], training=training)
    p_u, q_u = self.last_outputs
    if self.marginalize:
      qy_xu = self.classify(X_u)
    else:
      qy_xu = q_u[-1]
    p = qy_xu.probs_parameter()
    llk_u = {
        k + '_u': tf.reduce_sum(p * tf.expand_dims(v, axis=-1), axis=-1)
        for k, v in llk_u.items()
    }
    kl_u = {
        k + '_u':
        tf.reduce_sum(tf.expand_dims(p, axis=0) * tf.expand_dims(v, axis=-1),
                      axis=-1) for k, v in kl_u.items()
    }
    # the entropy
    entropy = -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(p), p),
                             axis=-1)
    llk_u['H_qy'] = entropy
    ### for labelled data, add the discriminative objective
    if y_l is not None:
      llk_l, kl_l = super().elbo_components(inputs=[X_l, y_l],
                                            training=training)
      p_l, q_l = self.last_outputs
      is_ss = tf.shape(y_l)[0] > 0
      llk_l = {
          k + '_l': tf.cond(is_ss, lambda: v, lambda: 0.)
          for k, v in llk_l.items()
      }
      kl_l = {
          k + '_l': tf.cond(is_ss, lambda: v, lambda: 0.)
          for k, v in kl_l.items()
      }
      qy_xl = self.classify(X_l)
      if self.relaxed:
        y_l = tf.clip_by_value(y_l, 1e-8, 1. - 1e-8)
      llk_l['llk_classifier'] = self.alpha * tf.cond(
          is_ss, lambda: qy_xl.log_prob(y_l), lambda: 0.)
    else:
      llk_l = {}
      kl_l = {}
    ### merge everything
    llk = {k: tf.reduce_mean(v) for k, v in dict(**llk_u, **llk_l).items()}
    kl = {k: tf.reduce_mean(v) for k, v in dict(**kl_u, **kl_l).items()}
    return llk, kl

  @property
  def is_semi_supervised(self):
    return True

  def __str__(self):
    text = super().__str__()
    text += "\nClassifier:  "
    text += f"\n {self.classifier}"
    text += f"\n {self.labels}"
    # text += "\n Conditional Embedder:\n  "
    # text += "\n  ".join(layer2text(self.embedder).split('\n'))
    return text


class StructuredSemiVAE(betaVAE):
  r"""
  Reference:
    Siddharth, N., Paige, B., et al., 2017. "Learning Disentangled
      Representations with Semi-Supervised Deep Generative Models".
      arXiv:1706.00400 [cs, stat].
  """
