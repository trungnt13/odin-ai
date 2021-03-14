from typing import List, Union, Optional
from six import string_types

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow_probability.python import distributions as tfd

from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import (LayerCreator,
                                                             NetConf,
                                                             _parse_layers)
from odin.bay.helpers import kl_divergence
from odin.bay.vi.utils import prepare_ssl_inputs
from odin.utils import as_tuple


class MultitaskVAE(BetaVAE):
  """Multi-tasks VAE for semi-supervised learning

  Parameters
  ----------
  labels : Union[RVmeta, List[RVmeta]], optional
      distribution description of the label(s)
  skip_decoder : bool, optional
      the supervised outputs are directly connected to the latents,
      by default True
  alpha : float, optional
      coefficient of the supervised objective, by default 10.
  n_samples_semi : int
      if greater than 0, then sample from the posterior for training an
      auxiliary classifier of labelled-unlabelled data points.
      This extra loss reduce the within-clusters variance, and reduce the
      confusion between closely related classes.
  name : str, optional
      by default 'MultitaskVAE'

  Reference
  -----------
  Trong, T. N. et al. Semisupervised Generative Autoencoder for Single-Cell Data.
      Journal of Computational Biology 27, 1190–1203 (2019).
  """

  def __init__(
      self,
      labels: Union[RVmeta, List[RVmeta]] = RVmeta(10,
                                                   'onehot',
                                                   projection=True,
                                                   name="digits"),
      encoder_y: Optional[Union[LayerCreator, str]] = None,
      decoder_y: Union[LayerCreator, str] = None,
      probabilistic_encoder_y: bool = True,
      alpha: float = 10.,
      skip_decoder: bool = False,
      n_samples_semi: int = 0,
      name: str = 'MultitaskVAE',
      **kwargs,
  ):
    super().__init__(name=name, **kwargs)
    self.labels = [_parse_layers(y) for y in as_tuple(labels)]
    self.alpha = alpha
    self.skip_decoder = bool(skip_decoder)
    self.n_samples_semi = int(n_samples_semi)
    # labelled - unlabelled discriminator
    if self.n_samples_semi > 0:
      self.semi_discriminator = keras.layers.Dense(1,
                                                   activation='linear',
                                                   name='SemiDiscriminator')
      self.flatten = keras.layers.Flatten()
    else:
      self.semi_discriminator = None
      self.flatten = None
    ## prepare encoder for Y
    if encoder_y is not None:
      units_z = sum(
        np.prod(
          z.event_shape if hasattr(z, 'event_shape') else z.output_shape)
        for z in as_tuple(self.latents))
      if isinstance(encoder_y, string_types):  # copy
        layers = [
          keras.models.clone_model(self.encoder),
          keras.layers.Flatten()
        ]
      else:  # different network
        layers = [_parse_layers(encoder_y)]
      if probabilistic_encoder_y:
        layers.append(
          RVmeta(units_z, 'mvndiag', projection=True,
                 name='qzy_x').create_posterior())
      else:
        layers.append(keras.layers.Dense(units_z, activation='linear'))
      encoder_y = keras.Sequential(layers, name='encoder_y')
    self.encoder_y = encoder_y
    self.probabilistic_encoder_y = probabilistic_encoder_y
    ## prepare decoder for Y
    if decoder_y is not None:
      decoder_y = _parse_layers(decoder_y)
    self.decoder_y = decoder_y

  def build(self, input_shape):
    super().build(input_shape)
    if self.semi_discriminator is not None:
      units_z = sum(
        np.prod(
          z.event_shape if hasattr(z, 'event_shape') else z.output_shape)
        for z in as_tuple(self.latents))
      # units_y = sum(
      #     np.prod(
      #         y.event_shape if hasattr(y, 'event_shape') else y.output_shape)
      #     for y in self.labels)
      self.semi_discriminator.build((None, units_z))
    return self

  @property
  def alpha(self):
    return self._alpha

  @alpha.setter
  def alpha(self, a):
    self._alpha = tf.convert_to_tensor(a, dtype=self.dtype, name='alpha')

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    # don't condition on the labels, only accept inputs
    X = X[0]
    qz_x = super().encode(X, training=training, mask=None, **kwargs)
    if self.encoder_y is not None:
      X_y = self.encoder_y(X, training=training, mask=mask)
    else:
      X_y = None
    if X_y is not None:
      return as_tuple(qz_x) + (X_y,)
    return qz_x

  def decode(self, latents, training=None, mask=None, **kwargs):
    qzy_x = None  # q(z_y|x)
    if self.encoder_y is not None:
      qzy_x = latents[-1]
      latents = latents[:-1]
      if len(latents) == 1:
        latents = latents[0]
    h_d = super().decode(latents,
                         training=training,
                         mask=mask,
                         only_decoding=True,
                         **kwargs)
    px_z = self.observation(h_d, training=training, mask=mask)
    if isinstance(latents, (tuple, list)):
      latents = tf.concat(latents, axis=-1)
    ## decode py_zx
    h_y = latents if self.skip_decoder else h_d
    if self.decoder_y is not None:
      h_y = self.decoder_y(h_y, training=training, mask=mask)
    if qzy_x is not None:  # add skip connection
      h_y = tf.concat([h_y, qzy_x], axis=-1)
    py_z = [fy(h_y, training=training, mask=mask) for fy in self.labels]
    return as_tuple(px_z) + tuple(py_z)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    # unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super().elbo_components(X[0], mask=mask, training=training)
    P, Q = self.last_outputs
    qzy_x = None
    if self.encoder_y is not None:
      qzy_x = Q[-1]
      Q = Q[:-1]
    else:
      Q = as_tuple(Q)
    ## supervised log-likelihood
    if len(y) > 0:
      # iterate over each pair
      for layer, yi, py in zip(self.labels, y, P[1:]):
        name = layer.name
        llk_y = py.log_prob(yi)
        if mask is not None:
          # take into account the sample_shape by transpose the batch dim to
          # the first dimension
          # need to check the mask here, otherwise the loss can be NaN
          llk_y = tf.cond(
            tf.reduce_all(tf.logical_not(mask)),
            lambda: 0.,
            lambda: tf.transpose(
              tf.boolean_mask(tf.transpose(llk_y), mask, axis=0)),
          )
        # this is important, if loss=0 when using one-hot log_prob,
        # the gradient is NaN
        llk_y = tf.reduce_mean(self.alpha * llk_y)
        llk_y = tf.cond(tf.abs(llk_y) < 1e-8,
                        true_fn=lambda: tf.stop_gradient(llk_y),
                        false_fn=lambda: llk_y)
        llk[f"llk_{name}"] = llk_y
    ## semi-supervised labelled-unlabelled discriminator
    if self.semi_discriminator is not None and mask is not None:
      z = tf.concat([i.sample(self.n_samples_semi) for i in Q[:self.n_latents]],
                    axis=-1)
      z = tf.reshape(z, (-1, z.shape[-1]))
      y_true = tf.tile(tf.cast(tf.expand_dims(mask, axis=-1), self.dtype),
                       (self.n_samples_semi, 1))
      # y_pred = tf.reduce_logsumexp(self.semi_discriminator(z), axis=0) - \
      #   tf.math.log(tf.cast(self.n_samples_semi, self.dtype))
      y_pred = self.semi_discriminator(z)
      loss = tf.reduce_mean(
        tf.losses.binary_crossentropy(y_true=y_true,
                                      y_pred=y_pred,
                                      from_logits=True))
    else:
      loss = 0.
    # have to minimize this loss
    kl['discr_loss'] = loss
    if self.encoder_y is not None and self.probabilistic_encoder_y:
      qz_x = Q[0]  # prior
      kl['kl_qzy_x'] = self.beta * kl_divergence(
        q=qzy_x, p=qz_x, analytic=False, free_bits=self.free_bits)
      # (qzy_x.log_prob(z) - tf.stop_gradient(qz_x.log_prob(z)))
      # self._last_xy.KL_divergence(
      #     analytic=self.analytic,
      #     free_bits=self.free_bits,
      #     reverse=self.reverse)
    return llk, kl

  @classmethod
  def is_hierarchical(cls) -> bool:
    return False

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True

  def __str__(self):
    text = super().__str__()
    text += "\nEncoderY:\n "
    if self.encoder_y is None:
      text += ' None'
    else:
      text += '\n '.join(str(self.encoder_y).split('\n'))
    text += "\nDecoderY:\n "
    if self.decoder_y is None:
      text += ' None'
    else:
      text += '\n '.join(str(self.decoder_y).split('\n'))
    return text


class SkiptaskVAE(MultitaskVAE):
  """The supervised outputs, skip the decoder, and directly connect to
  the latents"""

  def __init__(self,
               encoder_y: Optional[LayerCreator] = None,
               name: str = 'SkiptaskVAE',
               **kwargs):
    kwargs.pop('skip_decoder', None)
    kwargs.pop('decoder_y', None)
    super().__init__(encoder_y=encoder_y,
                     decoder_y=None,
                     skip_decoder=True,
                     name=name,
                     **kwargs)


class MultiheadVAE(MultitaskVAE):
  """Similar to skiptaskVAE, the supervised outputs, skip the decoder,
  and directly connect to the via non-linear layers latents"""

  def __init__(
      self,
      decoder_y: LayerCreator = NetConf((256, 256),
                                        flatten_inputs=True,
                                        name='decoder_y'),
      name: str = 'MultiheadVAE',
      **kwargs,
  ):
    kwargs.pop('skip_decoder', None)
    kwargs.pop('encoder_y', None)
    super().__init__(encoder_y=None,
                     decoder_y=decoder_y,
                     skip_decoder=True,
                     name=name,
                     **kwargs)
    self.decoder_y = _parse_layers(decoder_y)
