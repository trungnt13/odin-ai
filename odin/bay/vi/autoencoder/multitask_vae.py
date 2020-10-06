import numpy as np
import tensorflow as tf
from odin.bay.random_variable import RandomVariable
from odin.bay.vi.autoencoder.beta_vae import betaVAE
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
                     outputs=RandomVariable((28, 28, 1),
                                'bern',
                                projection=False,
                                name="Image"),
                     labels=RandomVariable(10, 'onehot', projection=True, name="Digit"))
  vae.fit(train, epochs=-1, max_iter=8000, compile_graph=True, sample_shape=1)
  ```
  """

  def __init__(self,
               observation=RandomVariable(64,
                                          'gaussian',
                                          projection=True,
                                          name="Observation"),
               labels=RandomVariable(10,
                                     'onehot',
                                     projection=True,
                                     name="Labels"),
               alpha=10.,
               beta=1.,
               **kwargs):
    labels = tf.nest.flatten(labels)
    observation = tf.nest.flatten(observation)
    observation += labels
    super().__init__(beta=beta, observation=observation, **kwargs)
    self.labels = labels
    self.alpha = alpha

  @property
  def alpha(self):
    return self._alpha

  @alpha.setter
  def alpha(self, a):
    self._alpha = tf.convert_to_tensor(a, dtype=self.dtype, name='alpha')

  def encode(self, inputs, training=None, mask=None, **kwargs):
    # don't condition on the labels, only accept inputs
    n_outputs = len(self.observation)
    n_semi = len(self.labels)
    inputs = tf.nest.flatten(inputs)[:(n_outputs - n_semi)]
    if len(inputs) == 1:
      inputs = inputs[0]
    return super().encode(inputs, training=training, mask=mask, **kwargs)

  def elbo_components(self,
                      inputs,
                      training=None,
                      pX_Z=None,
                      qZ_X=None,
                      mask=None,
                      **kwargs):
    n_semi = len(self.labels)
    # unsupervised ELBO
    llk, kl = super().elbo_components(inputs,
                                      pX_Z=pX_Z[:-n_semi],
                                      qZ_X=qZ_X,
                                      mask=mask,
                                      training=training,
                                      **kwargs)
    inputs = tf.nest.flatten(inputs)
    # supervised log-likelihood
    if len(inputs) > len(self.observation) - n_semi:
      obs = self.observation[-n_semi:]
      Y = inputs[-n_semi:]
      pY_Z = pX_Z[-n_semi:]
      # iterate over each pair
      for layer, y, py, m in zip(obs, Y, pY_Z, as_tuple(mask, N=n_semi)):
        name = layer.name
        llk_y = py.log_prob(y)
        if m is not None:
          m = tf.reshape(m, (-1,))
          # take into account the sample_shape by transpose the batch dim to
          # the first dimension
          # need to check the mask here, otherwise the loss can be NaN
          llk_y = tf.cond(
              tf.reduce_all(tf.logical_not(m)),
              lambda: 0.,
              lambda: tf.transpose(
                  tf.boolean_mask(tf.transpose(llk_y), m, axis=0)),
          )
        # this is important, if loss=0 when using one-hot log_prob,
        # the gradient is NaN
        loss = tf.reduce_mean(self.alpha * llk_y)
        loss = tf.cond(
            tf.abs(loss) < 1e-8, lambda: tf.stop_gradient(loss), lambda: loss)
        llk[f"llk_{name}"] = loss
    return llk, kl

  @property
  def is_semi_supervised(self):
    return True


class MultiheadVAE(multitaskVAE):
  r""" A same multi-outputs design as `multitaskVAE`, however, the
  semi-supervised heads are directly connected to the latent layers to
  exert influences. """

  def __init__(self,
               outputs=RandomVariable(64,
                                      'gaussian',
                                      projection=True,
                                      name="Input"),
               labels=RandomVariable(10,
                                     'onehot',
                                     projection=True,
                                     name="Label"),
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
