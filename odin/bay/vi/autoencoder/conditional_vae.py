from __future__ import absolute_import, division, print_function

from typing import List

import numpy as np
import tensorflow as tf
from odin import backend as bk
from odin.backend import TensorTypes
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import (LayerCreator,
                                                             _parse_layers)
from odin.bay.vi.utils import (marginalize_categorical_labels,
                               prepare_ssl_inputs, split_ssl_inputs)
from odin.networks import NetConf
from odin.networks.conditional_embedding import get_embedding
from odin.utils import as_tuple
from tensorflow.python import keras
from tensorflow.python.keras.layers import (BatchNormalization, Dense, Dropout,
                                            Layer)
from tensorflow_probability.python.distributions import (Distribution,
                                                         OneHotCategorical)
from typing_extensions import Literal

__all__ = ['conditionalM2VAE']


# ===========================================================================
# main classes
# ===========================================================================
class conditionalM2VAE(betaVAE):
  """Implementation of M2 model (Kingma et al. 2014). The default
  configuration of this layer is optimized for MNIST.

  The inference model:
  ```
  q(xyz) = q(z|xy)q(y|x)
  q(z|xy) = N(z|f_mu(xy),f_sig(x)))
  q(y|x) = Cat(y|pi(x))
  ```

  The generative model:
  ```
  p(x,y,z) = p(x|y,z;theta)p(y)p(z)
  p(y) = Cat(y|pi)
  p(z) = N(z|0,I)
  ```

  where: `x` is observed inputs, `y` is categorical labels and `z` is
  continuous latents

  Parameters
  ------------
    n_classes : int
        number of supervised labels.
    classifier : LayerCreator, optional
        classifier `q(y|x)`
    xy_to_qz : LayerCreator, optional
        a network transforming the joint variable `x,y` for modeling `q(z)`
        distribution
    zy_to_px : LayerCreator, optional
        a network transforming the joint variable `z,y` for modeling `p(x)`
        distribution
    embedding_dim : int, optional
        embedding dimension, by default 128
    embedding_method : {'repetition', 'projection', 'dictionary',
                        'sequential', 'identity'}
        embedding method, by default 'sequential'
    batchnorm : str, optional
        if True, applying batch normalization on the joint variables `x,y`
        and `z, y`, by default 0., by default False
    dropout : float, optional
        if greater than zeros, applying dropout on the joint variables `x,y`
        and `z, y`, by default 0.
    alpha : float, optional
        The weight of discriminative objective added to the labelled
        data objective. In the paper, it is recommended:
        `alpha = 0.1 * (n_total_samples / n_labelled_samples)`, by default 0.05
    beta : float, optional
        [description], by default 1.
    temperature : float, optional
        [description], by default 10.
    marginalize : bool, optional
        marginalizing the labels (i.e. `y`), otherwide, use Gumbel-Softmax for
        reparameterization, by default True

  References
  ------------
  Kingma, D.P., Rezende, D.J., Mohamed, S., Welling, M., 2014.
    "Semi-Supervised Learning with Deep Generative Models".
    arXiv:1406.5298 [cs, stat].

  Notes
  ------
  `batchnorm=True` is not recommended, sometimes training return NaNs values
    for KL-divergence.
  The default arguments are for MNIST.

  """

  def __init__(
      self,
      labels: RVmeta = RVmeta(10, 'onehot', name='digits'),
      observation: RVmeta = RVmeta((28, 28, 1),
                                   'bernoulli',
                                   projection=True,
                                   name='image'),
      latents: RVmeta = RVmeta(64, 'mvndiag', projection=True, name='latents'),
      classifier: LayerCreator = NetConf([128, 128],
                                               flatten_inputs=True,
                                               name='classifier'),
      encoder: LayerCreator = NetConf([512, 512],
                                            flatten_inputs=True,
                                            name='encoder'),
      decoder: LayerCreator = NetConf([512, 512],
                                            flatten_inputs=True,
                                            name='decoder'),
      xy_to_qz: LayerCreator = NetConf([128, 128], name='xy_to_qz'),
      zy_to_px: LayerCreator = NetConf([128, 128], name='zy_to_px'),
      embedding_dim: int = 128,
      embedding_method: Literal['repetition', 'projection', 'dictionary',
                                'sequential', 'identity'] = 'sequential',
      batchnorm: str = False,
      dropout: float = 0.,
      alpha: float = 0.05,
      beta: float = 1.,
      temperature: float = 10.,
      marginalize: bool = True,
      name: str = 'ConditionalM2VAE',
      **kwargs,
  ):
    super().__init__(latents=latents,
                     observation=observation,
                     encoder=encoder,
                     decoder=decoder,
                     beta=beta,
                     name=name,
                     **kwargs)
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name="alpha")
    self.marginalize = bool(marginalize)
    self.n_classes = int(np.prod(labels.event_shape))
    assert labels.posterior == 'onehot', \
      f'only support Categorical distribution for labels, given {labels.posterior}'
    self.embedding_dim = int(embedding_dim)
    self.embedding_method = str(embedding_method)
    self.batchnorm = bool(batchnorm)
    self.dropout = float(dropout)
    # the networks
    self.classifier = _parse_layers(classifier)
    self.xy_to_qz_net = _parse_layers(xy_to_qz)
    self.zy_to_px_net = _parse_layers(zy_to_px)
    # labels distribution
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
    self.labels = RVmeta(self.n_classes,
                         posterior,
                         projection=True,
                         prior=OneHotCategorical(probs=[1. / self.n_classes] *
                                                 self.n_classes),
                         name=labels.name,
                         kwargs=dist_kw).create_posterior()
    # create embedder
    embedder = get_embedding(self.embedding_method)
    # q(z|xy)
    self.y_to_qz = embedder(n_classes=self.n_classes,
                            event_shape=self.embedding_dim,
                            name='y_to_qz')
    self.x_to_qz = Dense(embedding_dim, activation='linear', name='x_to_qz')
    # p(x|zy)
    self.y_to_px = embedder(n_classes=self.n_classes,
                            event_shape=self.embedding_dim,
                            name='y_to_px')
    self.z_to_px = Dense(embedding_dim, activation='linear', name='z_to_px')
    # batch normalization
    if self.batchnorm:
      self.qz_xy_norm = BatchNormalization(axis=-1, name='qz_xy_norm')
      self.px_zy_norm = BatchNormalization(axis=-1, name='px_zy_norm')
    if 0.0 < self.dropout < 1.0:
      self.qz_xy_drop = Dropout(rate=self.dropout, name='qz_xy_drop')
      self.px_zy_drop = Dropout(rate=self.dropout, name='px_zy_drop')

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

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    X = X[0]  # only accept single inputs now
    # prepare the label embedding
    qy_x = self.classify(X, training=training)
    h_y = self.y_to_qz(qy_x, training=training)
    # encode normally
    h_x = self.encoder(X, training=training, mask=mask)
    h_x = bk.flatten(h_x, n_outdim=2)
    h_x = self.x_to_qz(h_x, training=training)
    # combine into q(z|xy)
    h_xy = h_x + h_y
    if self.batchnorm:
      h_xy = self.qz_xy_norm(h_xy, training=training)
    if 0.0 < self.dropout < 1.0:
      h_xy = self.qz_xy_drop(h_xy, training=training)
    # conditional embedding y
    h_xy = self.xy_to_qz_net(h_xy, training=training, mask=mask)
    qz_xy = self.latents(h_xy, training=training, mask=mask)
    return (qz_xy, qy_x)

  def decode(self, latents, training=None, mask=None, **kwargs):
    qz_xy, qy_x = latents
    h_z = self.z_to_px(qz_xy, training=training)
    h_y = self.y_to_px(qy_x, training=training)
    h_zy = h_z + h_y
    if self.batchnorm:
      h_zy = self.px_zy_norm(h_zy, training=training)
    if 0.0 < self.dropout < 1.0:
      h_zy = self.px_zy_drop(h_zy, training=training)
    h_zy = self.zy_to_px_net(h_zy, training=training, mask=mask)
    return super().decode(h_zy, training=training, mask=mask)

  ##################### Helper methods for ELBO
  def _prepare_elbo(self, inputs, training=None, mask=None):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
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
    return X_u, y_u, X_l, y_l

  def _unlabelled_loss(self, X_u, y_u, training):
    llk_u, kl_u = super().elbo_components(inputs=[X_u, y_u], training=training)
    P_u, Q_u = self.last_outputs
    # Note: qy_x is always expected to be the last value
    qy_x_u = Q_u[-1]
    probs = qy_x_u.probs_parameter()
    # weighed the loss by qy_x
    llk_u = {
        k + '_u': tf.reduce_sum(probs * tf.expand_dims(v, axis=-1), axis=-1)
        for k, v in llk_u.items()
    }
    kl_u = {
        k + '_u': tf.reduce_sum(tf.expand_dims(probs, axis=0) *
                                tf.expand_dims(v, axis=-1),
                                axis=-1) for k, v in kl_u.items()
    }
    # the entropy
    entropy = -tf.reduce_sum(tf.math.multiply_no_nan(tf.math.log(probs), probs),
                             axis=-1)
    llk_u['H_qy'] = entropy
    return P_u, Q_u, llk_u, kl_u

  def _labelled_loss(self, X_l, y_l, training):
    if y_l is not None:
      llk_l, kl_l = super().elbo_components(inputs=[X_l, y_l],
                                            training=training)
      P_l, Q_l = self.last_outputs
      # we need this condition since NaNs are returned if y_l.shape[0] == 0
      is_ss = tf.shape(y_l)[0] > 0
      llk_l = {
          k + '_l': tf.cond(is_ss, lambda: v, lambda: 0.)
          for k, v in llk_l.items()
      }
      kl_l = {
          k + '_l': tf.cond(is_ss, lambda: v, lambda: 0.)
          for k, v in kl_l.items()
      }
      qy_x_l = Q_l[-1]
      if self.relaxed:
        y_l = tf.clip_by_value(y_l, 1e-8, 1. - 1e-8)
      llk_l['llk_qy'] = self.alpha * tf.cond(
          is_ss, lambda: qy_x_l.log_prob(y_l), lambda: 0.)
    else:
      P_l = None
      Q_l = None
      llk_l = {}
      kl_l = {}
    return P_l, Q_l, llk_l, kl_l

  def elbo_components(self, inputs, training=None, mask=None):
    X_u, y_u, X_l, y_l = self._prepare_elbo(inputs,
                                            training=training,
                                            mask=mask)
    ### for unlabelled data (assumed always available)
    P_u, Q_u, llk_u, kl_u = self._unlabelled_loss(X_u, y_u, training)
    ### for labelled data, add the discriminative objective
    P_l, Q_l, llk_l, kl_l = self._labelled_loss(X_l, y_l, training)
    ### merge everything
    llk = {k: tf.reduce_mean(v) for k, v in dict(**llk_u, **llk_l).items()}
    kl = {k: tf.reduce_mean(v) for k, v in dict(**kl_u, **kl_l).items()}
    return llk, kl

  @classmethod
  def is_semi_supervised(self) -> bool:
    return True

  def __str__(self):
    text = super().__str__()
    text += f"\nEmbedding:"
    text += f"\n Dim      : {self.embedding_dim}"
    text += f"\n Method   : '{self.embedding_method}'"
    text += f"\n Batchnorm: {self.batchnorm}"
    text += f"\n Dropout  : {self.dropout}"
    text += "\nClassifier:\n "
    text += '\n '.join(str(self.classifier).split('\n'))
    text += f"\n {self.labels}"
    text += "\nq(z|xy) network:\n "
    text += '\n '.join(str(self.xy_to_qz_net).split('\n'))
    text += "\np(x|zy) network:\n "
    text += '\n '.join(str(self.zy_to_px_net).split('\n'))
    return text


class StructuredSemiVAE(betaVAE):
  r"""
  Reference:
    Siddharth, N., Paige, B., et al., 2017. "Learning Disentangled
      Representations with Semi-Supervised Deep Generative Models".
      arXiv:1706.00400 [cs, stat].
  """
