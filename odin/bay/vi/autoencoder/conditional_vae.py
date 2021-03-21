from __future__ import absolute_import, division, print_function

from typing import List
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow_probability.python.distributions import (Distribution, Normal,
                                                         OneHotCategorical)
from tensorflow_probability.python.layers import IndependentNormal
from typing_extensions import Literal
from tensorflow_probability.python.internal import prefer_static as ps

from odin import backend as bk
from odin.backend import TensorType
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import (LayerCreator,
                                                             _parse_layers)
from odin.bay.vi.utils import (marginalize_categorical_labels,
                               prepare_ssl_inputs, split_ssl_inputs)
from odin.networks import NetConf
from odin.networks.conditional_embedding import get_embedding

__all__ = [
  'ConditionalM2VAE',
    'reparamsM3VAE',
]


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


# ===========================================================================
# main classes
# ===========================================================================
class ConditionalM2VAE(BetaVAE):
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
      beta value in BetaVAE, by default 1.
  temperature : float, optional
      temperature in case using relaxed onehot distribution, by default 10.
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
      labels: RVconf = RVconf(10, 'onehot', name='digits'),
      observation: RVconf = RVconf((28, 28, 1),
                                   'bernoulli',
                                   projection=True,
                                   name='image'),
      latents: RVconf = RVconf(64, 'mvndiag', projection=True, name='latents'),
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
    self.embedding_dim = int(embedding_dim)
    self.embedding_method = str(embedding_method)
    self.batchnorm = bool(batchnorm)
    self.dropout = float(dropout)
    ## the networks
    self.classifier = _parse_layers(classifier)
    self.xy_to_qz_net = _parse_layers(xy_to_qz)
    self.zy_to_px_net = _parse_layers(zy_to_px)
    ## check the labels distribution
    if hasattr(labels, 'posterior'):
      posterior_name = str(labels.posterior)
    if hasattr(labels, 'posterior_layer'):
      posterior_name = str(labels.posterior_layer).lower()
    if 'onehot' not in posterior_name:
      warnings.warn(
          'Conditional VAE only support one-hot or relaxed one-hot distribution, '
          f'but given: {labels}')
    self.n_classes = int(np.prod(labels.event_shape))
    self.marginalize = bool(marginalize)
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
    self.labels = RVconf(self.n_classes,
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
               inputs: TensorType,
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
    X_u, y_u, X_l, y_l = _prepare_elbo(self,
                                       inputs,
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


class StructuredSemiVAE(BetaVAE):
  r"""
  Reference:
    Siddharth, N., Paige, B., et al., 2017. "Learning Disentangled
      Representations with Semi-Supervised Deep Generative Models".
      arXiv:1706.00400 [cs, stat].
  """


# ===========================================================================
# M3 Reparameterized VAE
# ===========================================================================
class PriorRegressor(keras.layers.Layer):

  def __init__(self, n_classes: int, **kwargs):
    super().__init__(**kwargs)
    self.n_classes = int(n_classes)

  def build(self, input_shape=None):
    dim = self.n_classes
    self.diag_loc_true = tf.Variable(tf.zeros((dim), dtype=self.dtype))
    self.diag_loc_false = tf.Variable(tf.zeros((dim), dtype=self.dtype))
    self.diag_scale_true = tf.Variable(tf.ones((dim), dtype=self.dtype))
    self.diag_scale_false = tf.Variable(tf.ones((dim), dtype=self.dtype))
    self.dist = IndependentNormal(event_shape=(dim,))
    return super().build((None, self.n_classes))

  def call(self, x, training=None, mask=None):
    loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
    scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
    scale = tf.clip_by_value(tf.nn.softplus(scale), 1e-3, 1e12)
    return self.dist(tf.concat([loc, scale], axis=-1))


class reparamsM3VAE(BetaVAE):

  def __init__(
      self,
      labels: RVconf = RVconf(10, 'relaxedonehot', name='digits'),
      observation: RVconf = RVconf((28, 28, 1),
                                   'bernoulli',
                                   projection=True,
                                   name='image'),
      latents: RVconf = RVconf(54, 'mvndiag', projection=True, name='latents'),
      classifier: LayerCreator = NetConf([128, 128],
                                         flatten_inputs=True,
                                         name='classifier'),
      encoder: LayerCreator = NetConf([512, 512],
                                      flatten_inputs=True,
                                      name='encoder'),
      decoder: LayerCreator = NetConf([512, 512],
                                      flatten_inputs=True,
                                      name='decoder'),
      n_resamples: int = 128,
      alpha: float = 0.05,
      temperature: float = 10.,
      name: str = 'ReparameterizedM3VAE',
      **kwargs,
  ):
    super().__init__(latents=latents,
                     observation=observation,
                     encoder=encoder,
                     decoder=decoder,
                     name=name,
                     **kwargs)
    assert labels.posterior == 'relaxedonehot', \
      f"only support 'relaxedonehot' distribution for labels, given {labels.posterior}"
    self.marginalize = False
    self.n_classes = int(np.prod(labels.event_shape))
    self.n_resamples = int(n_resamples)
    self.regressor = PriorRegressor(self.n_classes)
    self.labels = RVconf(
        self.n_classes,
        posterior='relaxedonehot',
        projection=True,
        prior=OneHotCategorical(probs=[1. / self.n_classes] * self.n_classes),
        name=labels.name,
        kwargs=dict(temperature=temperature)).create_posterior()
    self.denotations = RVconf(event_shape=(self.n_classes,),
                              posterior='normal',
                              projection=True,
                              name='denotations').create_posterior()
    self.classifier = _parse_layers(classifier)

  def build(self, input_shape):
    self.regressor.build()
    self.classifier.build((None, self.n_classes))
    return super().build(input_shape)

  def classify(self,
               inputs: TensorType,
               training: bool = False) -> Distribution:
    """Return the prediction of labels"""
    if isinstance(inputs, (tuple, list)) and len(inputs) == 1:
      inputs = inputs[0]
    h = self.classifier(inputs, training=training)
    return self.labels(h, training=training)

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    X = X[0]  # only accept single inputs now
    # encode normally
    h_x = self.encoder(X, training=training, mask=mask)
    qz_x = self.latents(h_x, training=training, mask=mask)
    qzc_x = self.denotations(h_x, training=training, mask=mask)
    # prepare the label embedding
    z_c = tf.convert_to_tensor(qzc_x)
    qy_zx = self.classify(z_c, training=training)
    return (qz_x, qzc_x, qy_zx)

  def decode(self, latents, training=None, mask=None, **kwargs):
    qz_x, qzc_x, qy_zx = latents
    z = tf.concat([qz_x, qzc_x], axis=-1)
    return super().decode(z, training=training, mask=mask, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None):
    X_u, y_u, X_l, y_l = _prepare_elbo(self,
                                       inputs,
                                       training=training,
                                       mask=mask)
    y_l = tf.clip_by_value(y_l, 1e-8, 1. - 1e-8)
    px_z_u, (qz_x_u, qzc_x_u, qy_zx_u) = self(X_u, training=training)
    px_z_l, (qz_x_l, qzc_x_l, qy_zx_l) = self(X_l, training=training)
    z_exc = tf.concat(
        [tf.convert_to_tensor(qz_x_u),
         tf.convert_to_tensor(qz_x_l)], axis=0)
    z_c = tf.concat(
        [tf.convert_to_tensor(qzc_x_u),
         tf.convert_to_tensor(qzc_x_l)], axis=0)
    # Convert y to one-hot vector and Sample y for those without labels
    y_sup = y_l
    y_uns = tf.convert_to_tensor(qy_zx_u)
    y = tf.concat((y_uns, y_sup), axis=0)
    # log q(y|z_c)
    h = tf.concat([qy_zx_u.logits, qy_zx_l.logits], axis=0)
    log_q_y_zc = tf.reduce_sum(h * y, axis=1)
    # log p(x|z)
    log_p_x_z = tf.concat([px_z_u.log_prob(X_u), px_z_l.log_prob(X_l)], axis=0)
    # log p(z_c|y)
    pzc_y = self.regressor(y)
    log_p_zc_y = pzc_y.log_prob(z_c)
    # log p(z_\c)
    dist = Normal(tf.cast(0., self.dtype), 1.)
    log_p_zexc = tf.reduce_sum(dist.log_prob(z_exc), axis=-1)
    # log p(z|y)
    log_p_z_y = log_p_zc_y + log_p_zexc
    # log q(y|x)  (Draw 128 points from q(z_c|x). Supervised samples only)
    h = qzc_x_l.sample(self.n_resamples)
    h = tf.reshape(h, (-1, h.shape[-1]))
    qy_x = self.classify(h, training=training)
    qy_x_logits = tf.reshape(qy_x.logits, (self.n_resamples, -1, h.shape[-1]))
    h = tf.reduce_logsumexp(h, axis=0) - tf.math.log(128.)
    log_q_y_x = tf.reduce_sum(h * y_l, axis=1)
    # log q(z|x)
    log_qz_x = tf.concat([qz_x_u.log_prob(qz_x_u),
                          qz_x_l.log_prob(qz_x_l)],
                         axis=0)
    log_qzc_x = tf.concat(
        [qzc_x_u.log_prob(qzc_x_u),
         qzc_x_l.log_prob(qzc_x_l)], axis=0)
    log_q_z_x = log_qz_x + log_qzc_x
    # Calculate the lower bound
    n_uns = ps.shape(X_u)[0]
    h = log_p_x_z + log_p_z_y - log_q_y_zc - log_q_z_x
    coef_sup = tf.math.exp(log_q_y_zc[n_uns:] - log_q_y_x)
    coef_uns = tf.ones((n_uns,), dtype=self.dtype)
    coef = tf.concat((coef_uns, coef_sup), axis=0)
    zeros = tf.zeros((n_uns,), dtype=self.dtype)
    lb = coef * h + tf.concat((zeros, log_q_y_x), axis=0)
    return {'elbo': lb}, {}

  def _elbo_components(self, inputs, training=None, mask=None):
    X_u, y_u, X_l, y_l = _prepare_elbo(self,
                                       inputs,
                                       training=training,
                                       mask=mask)
    y_l = tf.clip_by_value(y_l, 1e-8, 1. - 1e-8)
    ## ELBO unsupervised examples
    elbo_u = super().elbo_components(X_u, training=training, mask=mask)
    P_u, Q_u = self.last_outputs
    ## ELBO supervised examples
    elbo_l = super().elbo_components(X_l, training=training, mask=mask)
    P_l, Q_l = self.last_outputs
    ## The classifier loss
    qy_zx_u = Q_u[-1]
    qy_zx_l = Q_l[-1]
    y_zx_u = tf.convert_to_tensor(qy_zx_u)
    y = tf.concat([y_zx_u, y_l], axis=0)
    log_qy_zx = tf.concat([qy_zx_u.log_prob(y_zx_u),
                           qy_zx_l.log_prob(y_l)],
                          axis=0)
    ## The conditional prior (reparameterized regressor, sec 4.1)
    pzc_y = self.regressor(y)
    z_c = tf.concat([Q_u[1], Q_l[1]], axis=0)
    log_pzc_y = pzc_y.log_prob(z_c)
    ## MCMC sample marginalize z_c to estimate q(y|x), B2
    z_c = Q_l[1].sample(self.n_resamples)
    qy_x = self.classify(tf.reshape(z_c, (-1, z_c.shape[-1])),
                         training=training)
    log_qy_x = qy_x.log_prob(tf.repeat(y_l, self.n_resamples, axis=0))
    log_qy_x = tf.reshape(log_qy_x, (self.n_resamples, -1))
    log_qy_x = (tf.reduce_logsumexp(log_qy_x, axis=0) -
                tf.math.log(tf.cast(self.n_resamples, self.dtype)))
    ## coefficients
    ## the final elbo
    llk = {}
    kl = {}
    for k, v in elbo_u[0].items():
      llk[f'{k}_u'] = v
    for k, v in elbo_l[0].items():
      llk[f'{k}_u'] = v
    for k, v in elbo_u[1].items():
      kl[f'{k}_u'] = v
    for k, v in elbo_l[1].items():
      kl[f'{k}_u'] = v
    llk['log_pzc_y'] = log_pzc_y
    print(llk)
    print(kl)
    exit()

  @classmethod
  def is_semi_supervised(self) -> bool:
    return True
