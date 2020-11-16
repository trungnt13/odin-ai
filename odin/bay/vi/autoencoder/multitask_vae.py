from __future__ import annotations

from typing import List, Union

import numpy as np
import tensorflow as tf
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.utils import prepare_ssl_inputs
from odin.utils import as_tuple
from tensorflow.python import keras
from tensorflow.python.ops import array_ops


class multitaskVAE(betaVAE):
  r""" Multi-tasks VAE for semi-supervised learning

  Example:
  ```
  from odin.fuel import MNIST
  from odin.bay.vi.autoencoder import multitaskVAE

  # load the dataset, include 50% of the labels for semi-supervised objective
  ds = MNIST()
  train = ds.create_dataset(partition='train', inc_labels=0.5)

  # create and train the model
  vae = multitaskVAE(encoder='mnist',
                     outputs=RVmeta((28, 28, 1),
                                'bern',
                                projection=False,
                                name="Image"),
                     labels=RVmeta(10, 'onehot', projection=True, name="Digit"))
  vae.fit(train, epochs=-1, max_iter=8000, compile_graph=True, sample_shape=1)
  ```

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
               skip_connect: bool = True,
               alpha: float = 10.,
               name: str = 'MultitaskVAE',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.labels = [y.create_posterior() for y in as_tuple(labels)]
    self.alpha = alpha
    self.skip_connect = bool(skip_connect)
    self._projector = None

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
    if self.skip_connect:
      if self._projector is None:
        self._projector = keras.layers.Dense(units=h_d.shape[-1],
                                             activation=None,
                                             name='skip_connect')
      if isinstance(latents, (tuple, list)):
        latents = tf.concat(latents, axis=-1)
      h_d = h_d + self._projector(latents)
    px_z = self.observation(h_d, training=training, mask=mask)
    py_z = [fy(h_d, training=training, mask=mask) for fy in self.labels]
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
