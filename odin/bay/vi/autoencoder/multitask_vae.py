import numpy as np
import tensorflow as tf

from odin.bay.random_variable import RandomVariable as RV
from odin.bay.vi.autoencoder.beta_vae import BetaVAE


class MultitaskVAE(BetaVAE):
  r""" Multi-tasks VAE for semi-supervised learning

  Example:
  ```
  ds = MNIST()
  train = ds.create_dataset(partition='train', inc_labels=0.5)
  vae = MultitaskVAE(encoder='mnist',
                     outputs=RV((28, 28, 1),
                                'bern',
                                projection=False,
                                name="Image"),
                     labels=RV(10, 'onehot', projection=True, name="Digit"))
  vae.fit(train, epochs=-1, max_iter=8000, compile_graph=True, sample_shape=1)
  ```
  """

  def __init__(self,
               outputs=RV(64, 'gaussian', projection=True, name="Input"),
               labels=RV(10, 'onehot', projection=True, name="Label"),
               alpha=10.,
               beta=1.,
               **kwargs):
    labels = tf.nest.flatten(labels)
    outputs = tf.nest.flatten(outputs)
    outputs += labels
    super().__init__(beta=beta, outputs=outputs, **kwargs)
    self.labels = labels
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name='alpha')

  def encode(self, inputs, training=None, mask=None, sample_shape=(), **kwargs):
    n_outputs = len(self.output_layers)
    n_semi = len(self.labels)
    inputs = tf.nest.flatten(inputs)[:(n_outputs - n_semi)]
    if len(inputs) == 1:
      inputs = inputs[0]
    return super().encode(inputs,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape,
                          **kwargs)

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training):
    n_semi = len(self.labels)
    # unsupervised ELBO
    llk, div = super()._elbo(X,
                             pX_Z[:-n_semi],
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training)
    # supervised log-likelihood
    if len(X) > len(self.output_layers) - n_semi:
      Y = X[-n_semi:]
      pY_Z = pX_Z[-n_semi:]
      mask = tf.nest.flatten(mask)
      if len(mask) == 1:
        mask = mask * n_semi
      for layer, y, py, m in zip(self.output_layers[-n_semi:], Y, pY_Z, mask):
        name = layer.name
        lk_y = py.log_prob(y)
        if m is not None:
          m = tf.reshape(m, (-1,))
          # take into account the sample_shape by transpose the batch dim to
          # the first dimension
          lk_y = tf.transpose(tf.boolean_mask(tf.transpose(lk_y), m, axis=0))
        llk["llk_%s" % name] = tf.reduce_mean(self.alpha * lk_y)
    return llk, div

  @property
  def is_semi_supervised(self):
    return True


class MultiheadVAE(BetaVAE):
  r""" Multi-head decoder for multiple output """

  def __init__(self, alpha=10., beta=1., linear_head=True, **kwargs):
    super().__init__(beta=beta, **kwargs)

  @property
  def is_semi_supervised(self):
    return True
