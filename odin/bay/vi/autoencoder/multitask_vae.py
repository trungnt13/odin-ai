from __future__ import annotations

from typing import List, Union

import numpy as np
import tensorflow as tf
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import _parse_layers
from odin.bay.vi.utils import prepare_ssl_inputs
from odin.utils import as_tuple
from tensorflow.python import keras
from tensorflow.python.ops import array_ops


class multitaskVAE(betaVAE):
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
  name : str, optional
      by default 'MultitaskVAE'

  Reference
  -----------
  Trong, T. N. et al. Semisupervised Generative Autoencoder for Single-Cell Data.
      Journal of Computational Biology 27, 1190–1203 (2019).
  """

  def __init__(self,
               labels: Union[RVmeta, List[RVmeta]] = RVmeta(10,
                                                            'onehot',
                                                            projection=True,
                                                            name="digits"),
               skip_decoder: bool = False,
               alpha: float = 10.,
               name: str = 'MultitaskVAE',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.labels = [_parse_layers(y) for y in as_tuple(labels)]
    self.alpha = alpha
    self.skip_decoder = bool(skip_decoder)

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
    return super().encode(X, training=training, mask=None, **kwargs)

  def decode(self, latents, training=None, mask=None, **kwargs):
    h_d = super().decode(latents,
                         training=training,
                         mask=mask,
                         only_decoding=True,
                         **kwargs)
    px_z = self.observation(h_d, training=training, mask=mask)
    if isinstance(latents, (tuple, list)):
      latents = tf.concat(latents, axis=-1)
    py_z = [
        fy(latents if self.skip_decoder else h_d, training=training, mask=mask)
        for fy in self.labels
    ]
    return (px_z,) + tuple(py_z)

  def elbo_components(self, inputs, training=None, mask=None):
    # unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super().elbo_components(X[0], mask=mask, training=training)
    P, Q = self.last_outputs
    # supervised log-likelihood
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
    return llk, kl

  @classmethod
  def is_semi_supervised(self) -> bool:
    return True


class skiptaskVAE(multitaskVAE):

  def __init__(self, name: str = 'SkiptaskVAE', **kwargs):
    kwargs.pop('skip_decoder', None)
    super().__init__(skip_decoder=True, name=name, **kwargs)
