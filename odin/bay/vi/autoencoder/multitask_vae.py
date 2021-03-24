from typing import List, Union, Optional

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.python import keras
from typing_extensions import Literal
from tensorflow_probability.python.distributions import Distribution

from odin.bay.layers import DistributionDense
from odin.bay.random_variable import RVconf
from odin.bay.vi.autoencoder.beta_vae import AnnealingVAE
from odin.bay.vi.autoencoder.variational_autoencoder import (LayerCreator,
                                                             NetConf,
                                                             _parse_layers)
from odin.bay.vi.autoencoder.variational_autoencoder import SemiSupervisedVAE
from odin.bay.vi.utils import prepare_ssl_inputs
from odin.utils import as_tuple


class MultitaskVAE(AnnealingVAE, SemiSupervisedVAE):
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
      labels: RVconf = RVconf(10,
                              'onehot',
                              projection=True,
                              name="digits"),
      encoder_y: Optional[Union[LayerCreator, Literal['tie', 'copy']]] = None,
      decoder_y: Optional[Union[LayerCreator, Literal['tie', 'copy']]] = None,
      alpha: float = 10.,
      n_semi_iw: int = 10,
      skip_decoder: bool = False,
      name: str = 'MultitaskVAE',
      **kwargs,
  ):
    super().__init__(name=name, **kwargs)
    self.labels = _parse_layers(labels)
    self.labels: DistributionDense
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name='alpha')
    self.n_semi_iw = int(n_semi_iw)
    self.skip_decoder = bool(skip_decoder)
    ## prepare encoder for Y
    if encoder_y is not None:
      units_z = sum(
        np.prod(
          z.event_shape if hasattr(z, 'event_shape') else z.output_shape)
        for z in as_tuple(self.latents))
      if isinstance(encoder_y, string_types):  # copy
        if encoder_y == 'tie':
          layers = []
        elif encoder_y == 'copy':
          layers = [
            keras.models.clone_model(self.encoder),
            keras.layers.Flatten()
          ]
        else:
          raise ValueError(f'No support for encoder_y={encoder_y}')
      else:  # different network
        layers = [_parse_layers(encoder_y)]
      layers.append(
        RVconf(units_z, 'mvndiag', projection=True,
               name='qzy_x').create_posterior())
      encoder_y = keras.Sequential(layers, name='encoder_y')
    self.encoder_y = encoder_y
    ## prepare decoder for Y
    if decoder_y is not None:
      decoder_y = _parse_layers(decoder_y)
    self.decoder_y = decoder_y

  def predict_labels(self,
                     inputs=None,
                     latents=None,
                     training=None,
                     mask=None,
                     n_mcmc=(),
                     mean=False,
                     **kwargs):
    if latents is None:
      latents = self.encode(inputs, training=training, mask=mask, **kwargs)
    # === 0. preprocessing latents
    latents = [(z.mean() if mean else z.sample(n_mcmc))
               if isinstance(z, Distribution) else z
               for z in as_tuple(latents)]
    if len(latents) > 1:
      latents = tf.concat(latents, axis=-1)
    else:
      latents = latents[0]
    # === 1. decode p(y|x,z)
    if not self.skip_decoder:
      h = self.decode(tf.reshape(latents, (-1, latents.shape[-1])),
                      training=training,
                      mask=mask,
                      only_decoding=True,
                      **kwargs)
      if not mean:
        h = tf.reshape(h, (n_mcmc, -1, h.shape[-1]))
    else:
      h = latents
    # === 2. decoder Y
    if self.decoder_y is not None:
      h = self.decoder_y(h, training=training, mask=mask)
    return self.labels(h, training=training, mask=mask)

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    # don't condition on the labels, only accept inputs
    X = X[0]
    h_e = super().encode(X, training=training, mask=None, only_encoding=True,
                         **kwargs)
    qz_x = self.latents(h_e, training=training, mask=None,
                        sample_shape=self.sample_shape)
    if self.encoder_y is not None:
      # tied encoder
      h_y = h_e if len(self.encoder_y.layers) == 1 else X
      qzy_x = self.encoder_y(h_y, training=training, mask=None)
      return as_tuple(qz_x) + (qzy_x,)
    return qz_x

  def decode(self, latents, training=None, mask=None, **kwargs):
    ## decode py_zx
    py_z = self.predict_labels(latents=latents, training=training, mask=mask)
    ## decoder px_z
    if self.encoder_y is not None:
      latents = tf.concat(latents, axis=1)
    h_d = super().decode(latents,
                         training=training,
                         mask=mask,
                         only_decoding=True,
                         **kwargs)
    px_z = self.observation(h_d, training=training, mask=mask)
    return as_tuple(px_z) + (py_z,)

  def _kl_qzy_x(self, Q):
    qzy_x = Q[-1]
    return qzy_x.KL_divergence(analytic=self.analytic,
                               free_bits=self.free_bits,
                               reverse=self.reverse)

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    # unsupervised ELBO
    inputs = as_tuple(inputs)
    X_uns = inputs[0]
    # === 1. unsupervised
    llk_uns, kl_uns = super().elbo_components(X_uns, training=training)
    P, Q = self.last_outputs
    if self.encoder_y is not None:
      kl_uns[f'kl_{self.latents.name}_y'] = self._kl_qzy_x(Q)
    # === 2. supervised
    if len(inputs) > 1:
      X_sup, y_sup = inputs[1:]
      is_empty = tf.size(X_sup) == 0
      llk_sup, kl_sup = super().elbo_components(X_sup, training=training)
      P, Q = self.last_outputs
      if self.encoder_y is not None:
        kl_sup[f'kl_{self.latents.name}_y'] = self._kl_qzy_x(Q)

      def zeros_loss():
        return 0.

      def supervised_llk():
        py = self.predict_labels(latents=Q, training=training, mask=mask,
                                 n_mcmc=self.n_semi_iw)
        llk_y = py.log_prob(
          tf.tile(tf.expand_dims(y_sup, 0), (self.n_semi_iw, 1, 1)))
        # this is important, if loss=0 when using one-hot log_prob,
        # the gradient is NaN
        llk_y = tf.reduce_mean(self.alpha * llk_y)
        llk_y = tf.cond(tf.abs(llk_y) < 1e-8,
                        true_fn=lambda: tf.stop_gradient(llk_y),
                        false_fn=lambda: llk_y)
        return llk_y

      llk_sup = self.ignore_empty(is_empty, llk_sup)
      kl_sup = self.ignore_empty(is_empty, kl_sup)
      llk_sup[f"llk_{self.labels.name}"] = tf.cond(
        is_empty, zeros_loss, supervised_llk)
    else:
      llk_sup, kl_sup = {}, {}
    # === 3. merge all objectives
    return self.merge_objectives(llk_uns, kl_uns, llk_sup, kl_sup)

  def __str__(self):
    text = super().__str__()
    text += "\nLabels:\n "
    text += f"{self.labels}"
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
