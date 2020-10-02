import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.ops import array_ops

from odin.bay.random_variable import RandomVariable as RV
from odin.bay.vi.autoencoder.beta_vae import betaVAE


class MultitaskVAE(betaVAE):
  r""" Multi-tasks VAE for semi-supervised learning

  Example:

  ```
  from odin.fuel import MNIST
  from odin.bay.vi.autoencoder import MultitaskVAE

  # load the dataset, include 50% of the labels for semi-supervised objective
  ds = MNIST()
  train = ds.create_dataset(partition='train', inc_labels=0.5)

  # create and train the model
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
    self.alpha = alpha

  @property
  def alpha(self):
    return self._alpha

  @alpha.setter
  def alpha(self, a):
    self._alpha = tf.convert_to_tensor(a, dtype=self.dtype, name='alpha')

  def encode(self, inputs, training=None, mask=None, sample_shape=(), **kwargs):
    n_outputs = len(self.observation)
    n_semi = len(self.labels)
    inputs = tf.nest.flatten(inputs)[:(n_outputs - n_semi)]
    if len(inputs) == 1:
      inputs = inputs[0]
    return super().encode(inputs,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape,
                          **kwargs)

  def _elbo(self, inputs, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training, **kwargs):
    n_semi = len(self.labels)
    # unsupervised ELBO
    llk, div = super()._elbo(inputs,
                             pX_Z[:-n_semi],
                             qZ_X,
                             analytic=analytic,
                             reverse=reverse,
                             sample_shape=sample_shape,
                             mask=mask,
                             training=training,
                             **kwargs)
    # supervised log-likelihood
    if len(inputs) > len(self.observation) - n_semi:
      Y = inputs[-n_semi:]
      pY_Z = pX_Z[-n_semi:]
      mask = tf.nest.flatten(mask)
      if len(mask) == 1:
        mask = mask * n_semi
      # iterate over each pair
      for layer, y, py, m in zip(self.observation[-n_semi:], Y, pY_Z, mask):
        name = layer.name
        lk_y = py.log_prob(y)
        if m is not None:
          m = tf.reshape(m, (-1,))
          # take into account the sample_shape by transpose the batch dim to
          # the first dimension
          # need to check the mask here, otherwise the loss can be NaN
          lk_y = tf.cond(
              tf.reduce_all(tf.logical_not(m)),
              lambda: 0.,
              lambda: tf.transpose(
                  tf.boolean_mask(tf.transpose(lk_y), m, axis=0)),
          )
        # this is important, if loss=0 when using one-hot log_prob,
        # the gradient is NaN
        loss = tf.reduce_mean(self.alpha * lk_y)
        loss = tf.cond(
            tf.abs(loss) < 1e-8, lambda: tf.stop_gradient(loss), lambda: loss)
        llk["llk_%s" % name] = loss
    # print(llk, div)
    return llk, div

  @property
  def is_semi_supervised(self):
    return True


class MultiheadVAE(MultitaskVAE):
  r""" A same multi-outputs design as `MultitaskVAE`, however, the
  semi-supervised heads are directly connected to the latent layers to
  exert influences. """

  def __init__(self,
               outputs=RV(64, 'gaussian', projection=True, name="Input"),
               labels=RV(10, 'onehot', projection=True, name="Label"),
               alpha=10.,
               beta=1.,
               **kwargs):
    super().__init__(alpha=alpha,
                     beta=beta,
                     outputs=outputs,
                     labels=[],
                     **kwargs)
    # create and build the semi-supervised output layers
    self.labels = tf.nest.flatten(labels)
    z = keras.Input(shape=self.latent_shape[1:], batch_size=None)
    semi_layers = [
        l.create_posterior(self.latent_shape[1:]) for l in self.labels
    ]
    for layer in semi_layers:
      layer(z)
    # add to the main output layers
    self.observation += semi_layers

  @property
  def is_semi_supervised(self):
    return True

  def decode(self,
             latents,
             training=None,
             mask=None,
             sample_shape=(),
             **kwargs):
    n_semi = len(self.labels)
    semi_layers = self.observation[-n_semi:]
    self.observation = self.observation[:-n_semi]
    # unsupervised outputs
    pX = super().decode(latents, training, mask, sample_shape, **kwargs)
    # semi outputs
    pY = [layer(latents, training=training, mask=mask) for layer in semi_layers]
    for p in pY:  # remember to store the keras mask in outputs
      p._keras_mask = mask
    # recover and return
    self.observation = self.observation + semi_layers
    return tf.nest.flatten(pX) + pY
