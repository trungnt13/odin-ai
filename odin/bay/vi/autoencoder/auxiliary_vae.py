from typing import Optional

import numpy as np
import tensorflow as tf
from odin import backend as bk
from odin.bay.helpers import coercible_tensor
from odin.bay.layers import DistributionNetwork
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.conditional_vae import (ConditionalM2VAE,
                                                     prepare_ssl_inputs)
from odin.bay.vi.autoencoder.variational_autoencoder import (LayerCreator,
                                                             _parse_layers)
from odin.networks import NetConf, get_embedding
from odin.utils import as_tuple
from tensorflow.python.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow_probability.python.distributions import (Distribution,
                                                         VectorDeterministic)
from typing_extensions import Literal


class auxiliaryVAE(ConditionalM2VAE):
  """
  The inference model:
  ```
  q(xayz)=q(z|axy)q(y|xa)q(a|x)
  ```

  The generative model:
  ```
  # If skip_connection=False
  p(xayz) = p(x|yz)p(a|yz)p(y)p(z)
  # If skip_connection=True
  p(xayz) = p(x|ayz)p(a|yz)p(y)p(z)
  ```

  The key point of the ADGM is that the auxiliary unit a introduce a
  latent feature extractor to the inference model giving a richer mapping
  between `x` and `y`.

  References
  ------------
  Maaløe, L., Sønderby, C. K., Sønderby, S. K. & Winther, O. Auxiliary
    Deep Generative Models. arXiv:1602.05473 [cs, stat] (2016).
  Lucas, T. & Verbeek, J. Auxiliary Guided Autoregressive Variational
    Autoencoders. arXiv:1711.11479 [cs] (2017).
  """

  def __init__(
      self,
      n_classes: int = 10,
      observation=RVconf((28, 28, 1),
                         'bernoulli',
                         projection=True,
                         name='image'),
      latents: RVconf = RVconf(64, 'mvndiag', projection=True, name='latents'),
      classifier: LayerCreator = NetConf([128, 128],
                                               flatten_inputs=True,
                                               name='classifier'),
      auxiliary: RVconf = RVconf(64,
                                 'mvndiag',
                                 projection=True,
                                 name='auxiliary'),
      encoder_a: LayerCreator = NetConf([512, 512],
                                              flatten_inputs=True,
                                              name='encoder_a'),
      decoder_a: LayerCreator = NetConf([512, 512],
                                              flatten_inputs=True,
                                              name='decoder_a'),
      encoder: LayerCreator = NetConf([512, 512],
                                            flatten_inputs=True,
                                            name='encoder'),
      decoder: LayerCreator = NetConf([512, 512],
                                            flatten_inputs=True,
                                            name='decoder'),
      axy_to_qz: LayerCreator = NetConf([128, 128], name='axy_to_qz'),
      azy_to_px: LayerCreator = NetConf([128, 128], name='azy_to_px'),
      embedding_dim: int = 128,
      embedding_method: Literal['repetition', 'projection', 'dictionary',
                                'sequential', 'identity'] = 'sequential',
      batchnorm: bool = False,
      dropout: float = 0.,
      skip_connection: bool = True,
      alpha: float = 1.0,
      beta: float = 1.0,
      temperature: float = 10.,
      marginalize: bool = True,
      name='AuxiliaryVAE',
      **kwargs,
  ):
    super().__init__(n_classes=n_classes,
                     observation=observation,
                     latents=latents,
                     classifier=classifier,
                     encoder=encoder,
                     decoder=decoder,
                     xy_to_qz=axy_to_qz,
                     zy_to_px=azy_to_px,
                     embedding_dim=embedding_dim,
                     embedding_method=embedding_method,
                     batchnorm=batchnorm,
                     dropout=dropout,
                     alpha=alpha,
                     beta=beta,
                     temperature=temperature,
                     marginalize=marginalize,
                     name=name,
                     **kwargs)
    self.skip_connection = bool(skip_connection)
    self.batchnorm = bool(batchnorm)
    self.qa_dist = auxiliary.create_posterior(name='qa_x')
    self.pa_dist = auxiliary.create_posterior(name='pa_xz')
    self.encoder_a = _parse_layers(encoder_a)
    self.decoder_a = _parse_layers(decoder_a)
    # labels connections
    self.x_to_qy = Dense(units=self.embedding_dim, activation='linear')
    self.a_to_qy = Dense(units=self.embedding_dim, activation='linear')
    # auxiliary connections
    self.a_to_qz = Dense(units=self.embedding_dim, activation='linear')
    self.a_to_px = Dense(units=self.embedding_dim, activation='linear')
    # for p(a|yz)
    self.y_to_pa = Dense(units=self.embedding_dim, activation='linear')
    self.z_to_pa = Dense(units=self.embedding_dim, activation='linear')
    # batchnorm and dropout
    if self.batchnorm:
      self.qy_ax_norm = BatchNormalization(axis=-1, name='qy_ax_norm')
      self.pa_zy_norm = BatchNormalization(axis=-1, name='pa_zy_norm')
    if 0.0 < self.dropout < 1.0:
      self.qy_ax_drop = Dropout(rate=self.dropout, name='qy_ax_drop')
      self.pa_zy_drop = Dropout(axis=-1, name='pa_zy_drop')

  def classify(self,
               inputs,
               training=False,
               qa_x: Optional[Distribution] = None) -> Distribution:
    """Return the prediction of labels"""
    # prepare x
    if isinstance(inputs, (tuple, list)):
      inputs = inputs[0]  # only support a single inputs Tensor
    h_x = self.x_to_qy(bk.flatten(inputs, n_outdim=2), training=training)
    # prepare a
    if qa_x is None:
      qa_x = self.qa_dist(self.encoder_a(inputs, training=training),
                          training=training)
    h_a = self.a_to_qy(qa_x, training=training)
    # final combination
    h_ax = h_a + h_x
    if self.batchnorm:
      h_ax = self.qy_ax_norm(h_ax, training=training)
    if 0.0 < self.dropout < 1.0:
      h_ax = self.qy_ax_drop(h_ax, training=training)
    h = self.classifier(h_ax, training=training)
    return self.labels(h, training=training)

  def encode(self, inputs, training=None, mask=None):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    X = X[0]  # only accept single inputs now
    ## encode normally
    h_x = self.encoder(X, training=training, mask=mask)
    h_x = bk.flatten(h_x, n_outdim=2)
    ## prepare the auxiliary
    qa_x = self.qa_dist(self.encoder_a(X, training=training),
                        training=training,
                        mask=mask)
    ## prepare the label embedding
    qy_ax = self.classify(X, training=training, qa_x=qa_x)
    ## combine into q(z|axy)
    h_x = self.x_to_qz(h_x, training=training)
    h_a = self.a_to_qz(qa_x, training=training)
    h_y = self.y_to_qz(qy_ax, training=training)
    h_axy = h_x + h_y + h_a
    if self.batchnorm:
      h_axy = self.qz_xy_norm(h_axy, training=training)
    if 0.0 < self.dropout < 1.0:
      h_axy = self.qz_xy_drop(h_axy, training=training)
    # conditional embedding y
    h_axy = self.xy_to_qz_net(h_axy, training=training, mask=mask)
    qz_axy = self.latents(h_axy, training=training, mask=mask)
    return (qz_axy, qa_x, qy_ax)

  def decode(self, latents, training=None, mask=None):
    # skip_connection=False: p(xayz) = p(x|yz)p(a|yz)p(y)p(z)
    # skip_connection=True: p(xayz) = p(x|ayz)p(a|yz)p(y)p(z)
    qz_axy, qa_x, qy_ax = latents
    h_z = self.z_to_px(qz_axy, training=training)
    h_y = self.y_to_px(qy_ax, training=training)
    # skip connection to auxiliary variable
    if self.skip_connection:
      h_a = self.a_to_px(qa_x, training=training)
    else:
      h_a = 0.
    # combining all latent states
    h_ = h_z + h_y + h_a
    if self.batchnorm:
      h_ = self.px_zy_norm(h_, training=training)
    if 0.0 < self.dropout < 1.0:
      h_ = self.px_zy_drop(h_, training=training)
    h_ = self.zy_to_px_net(h_, training=training, mask=mask)
    px_ayz = super(ConditionalM2VAE, self).decode(h_,
                                                  training=training,
                                                  mask=mask)
    ## generate the auxiliary variable
    h_y = self.y_to_pa(qy_ax, training=training)
    h_z = self.z_to_pa(qz_axy, training=training)
    h_ = h_y + h_z
    if self.batchnorm:
      h_ = self.pa_zy_norm(h_, training=training)
    if 0.0 < self.dropout < 1.0:
      h_ = self.pa_zy_drop(h_, training=training)
    h_ = self.decoder_a(h_, training=training)
    pa_zy = self.pa_dist(h_, training=training, mask=mask)
    return px_ayz, pa_zy

  def elbo_components(self, inputs, training=None, mask=None):
    X_u, y_u, X_l, y_l = self._prepare_elbo(inputs,
                                            training=training,
                                            mask=mask)
    ### for unlabelled data (assumed always available)
    P_u, Q_u, llk_u, kl_u = self._unlabelled_loss(X_u, y_u, training)
    qa_x = Q_u[1]
    pa_zy = P_u[1]
    a = tf.convert_to_tensor(qa_x)
    kl_qp_a = qa_x.log_prob(a) - pa_zy.log_prob(a)
    if self.free_bits is not None:
      kl_qp_a = tf.maximum(kl_qp_a, self.free_bits)
    kl_u['kl_aux_u'] = kl_qp_a
    ### for labelled data, add the discriminative objective
    P_l, Q_l, llk_l, kl_l = self._labelled_loss(X_l, y_l, training)
    if P_l is not None:
      is_ss = tf.shape(y_l)[0] > 0
      qa_x = Q_l[1]
      pa_zy = P_l[1]
      a = tf.convert_to_tensor(qa_x)
      kl_qp_a = qa_x.log_prob(a) - pa_zy.log_prob(a)
      if self.free_bits is not None:
        kl_qp_a = tf.maximum(kl_qp_a, self.free_bits)
      kl_l['kl_aux_l'] = tf.cond(is_ss, lambda: kl_qp_a, lambda: 0.)
    # l_qa_x, l_qa_x_mu, l_qa_x_logvar = stochastic_layer(l_qa_x)
    # l_pa_zy, l_pa_zy_mu, l_pa_zy_logvar = stochastic_layer(l_pa_zy)
    # l_log_pa = log_prob(self.l_qa, self.l_pa_mu, self.l_pa_logvar)
    # l_log_qa = log_prob(self.l_qa, self.l_qa_mu, self.l_qa_logvar)
    # lb = log_px + log_py + log_pz + log_pa - log_qa - log_qz
    ### merge everything
    llk = {k: tf.reduce_mean(v) for k, v in dict(**llk_u, **llk_l).items()}
    kl = {k: tf.reduce_mean(v) for k, v in dict(**kl_u, **kl_l).items()}
    return llk, kl

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True

  def __str__(self):
    s = super().__str__()
    s += '\nAuxiliary:  '
    s += f'\n Skip-connection:{self.skip_connection}'
    s += '\n '
    s += '\n  '.join(str(self.encoder_a).split('\n'))
    s += '\n '
    s += '\n  '.join(str(self.decoder_a).split('\n'))
    return s
