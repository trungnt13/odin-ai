from functools import partial
from numbers import Number

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.utils import layer_utils
from tensorflow_probability.python.distributions import Distribution

from odin.backend.keras_helpers import layer2text
from odin.bay.vi.utils import permute_dims
from odin.networks import NetworkConfig, SequentialNetwork, SkipConnection
from odin.utils import as_tuple

__all__ = [
    'create_image_autoencoder',
    'ImageNet',
    'FactorDiscriminator',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _nparams(distribution, distribution_kw):
  from odin.bay.distribution_alias import parse_distribution
  distribution, _ = parse_distribution(distribution)
  return int(
      tf.reduce_prod(distribution.params_size(1, **distribution_kw)).numpy())


_CONV = partial(keras.layers.Conv2D, padding="SAME")
_DECONV = partial(keras.layers.Conv2DTranspose, padding="SAME")
_DENSE = partial(keras.layers.Dense, use_bias=True)


class Center0Image(keras.layers.Layer):
  r"""Normalize the image pixel from [0, 1] to [-1, 1]"""

  def call(self, inputs, **kwargs):
    return 2. * inputs - 1.


# ===========================================================================
# Basic Network
# ===========================================================================
def create_image_autoencoder(image_shape=(64, 64, 1),
                             latent_shape=(10,),
                             projection_dim=256,
                             activation='relu',
                             center0=True,
                             distribution='bernoulli',
                             distribution_kw=dict(),
                             skip_connect=False,
                             convolution=True,
                             input_shape=None):
  r""" Initialized the Convolutional encoder and decoder often used in
  Disentangled VAE literatures.

  By default, the image_shape and channels are configurated for binarized MNIST

  Arguments:
    image_shape : tuple of Integer. The shape of input and output image
    input_shape : tuple of Integer (optional). The `input_shape` to the encoder
      is different from the `image_shape` (in case of conditional VAE).
  """
  kw = dict(locals())
  encoder = ImageNet(**kw, decoding=False)
  decoder = ImageNet(**kw, decoding=True)
  return encoder, decoder


# ===========================================================================
# Decoder
# ===========================================================================
class ImageNet(keras.Model):
  r"""
  Arguments:
    image_shape : tuple of Integer. The shape of input and output image
    input_shape : tuple of Integer (optional). The `input_shape` to the encoder
      is different from the `image_shape` (in case of conditional VAE).

  Reference:
    Dieng, A.B., Kim, Y., Rush, A.M., Blei, D.M., 2018. "Avoiding Latent
      Variable Collapse With Generative Skip Models".
      arXiv:1807.04863 [cs, stat].
  """

  def __init__(self,
               image_shape=(28, 28, 1),
               latent_shape=(10,),
               projection_dim=256,
               activation='relu',
               center0=True,
               distribution='bernoulli',
               distribution_kw=dict(),
               skip_connect=False,
               convolution=True,
               decoding=False,
               input_shape=None,
               name=None):
    if name is None:
      name = "Decoder" if decoding else "Encoder"
    super().__init__(name=name)
    if isinstance(image_shape, Number):
      image_shape = (image_shape,)
    if isinstance(latent_shape, Number):
      latent_shape = (latent_shape,)
    ## check multi-inputs
    self.latent_shape = latent_shape
    self.image_shape = [image_shape]
    # input_shape to the encoder is the same as output_shape in the decoder
    if input_shape is None:
      input_shape = image_shape
    # others
    self.skip_connect = bool(skip_connect)
    self.convolution = bool(convolution)
    self.is_mnist = False
    self.pool_size = []
    self.decoding = decoding
    ## prepare layers
    layers = []
    if center0 and not decoding:
      layers.append(Center0Image())
    n_params = _nparams(distribution, distribution_kw)
    ## Dense
    if not convolution:
      if decoding:
        layers += [_DENSE(1000, activation=activation) for i in range(5)] + \
          [_DENSE(int(np.prod(image_shape) * n_params), activation='linear'),
           keras.layers.Reshape(image_shape)]
      else:
        layers += [keras.layers.Flatten()] + \
          [_DENSE(1000, activation=activation) for i in range(5)] + \
          [keras.layers.Dense(projection_dim, use_bias=True, activation='linear')]
    ## MNIST
    elif image_shape[:2] == (28, 28):
      base_depth = 32
      self.is_mnist = True
      if decoding:
        layers = [
            _DENSE(projection_dim, activation=activation),
            keras.layers.Lambda(
                lambda codes: tf.reshape(codes, (-1, 1, 1, projection_dim))),
            _DECONV(2 * base_depth, 7, padding="VALID", activation=activation),
            _DECONV(2 * base_depth, 5, activation=activation),
            _DECONV(2 * base_depth, 5, 2, activation=activation),
            _DECONV(base_depth, 5, activation=activation),
            _DECONV(base_depth, 5, 2, activation=activation),
            _DECONV(base_depth, 5, activation=activation),
            _CONV(image_shape[-1] * n_params, 5, activation='linear'),
            keras.layers.Flatten(),
        ]
      else:
        layers += [
            _CONV(base_depth, 5, 1, activation=activation),
            _CONV(base_depth, 5, 2, activation=activation),
            _CONV(2 * base_depth, 5, 1, activation=activation),
            _CONV(2 * base_depth, 5, 2, activation=activation),
            _CONV(4 * base_depth, 7, padding="VALID", activation=activation),
            keras.layers.Flatten(),
            keras.layers.Dense(projection_dim,
                               use_bias=True,
                               activation='linear')
        ]
        self.pool_size = [1, 2, 2, 4, 28]
    ## Other, must be power of 2
    else:
      assert all(int(np.log2(s)) == np.log2(s) for s in image_shape[:2]), \
        "Image sizes must be power of 2"
      if decoding:
        size = image_shape[1] // 16
        encoder_shape = (size, size, 64)
        layers = [
            _DENSE(projection_dim, activation=activation),
            _DENSE(int(np.prod(encoder_shape)), activation=activation),
            keras.layers.Reshape(encoder_shape),
            _DECONV(64, 4, 2, activation=activation),
            _DECONV(32, 4, 2, activation=activation),
            _DECONV(32, 4, 2, activation=activation),
            _DECONV(image_shape[-1] * n_params, 4, 2, activation='linear'),
            keras.layers.Flatten(),
        ]
      else:
        layers += [
            _CONV(32, 4, 2, activation=activation),
            _CONV(32, 4, 2, activation=activation),
            _CONV(64, 4, 2, activation=activation),
            _CONV(64, 4, 2, activation=activation),
            keras.layers.Flatten(),
            keras.layers.Dense(projection_dim,
                               use_bias=True,
                               activation='linear')
        ]
        self.pool_size = [2, 4, 8, 16]
    ## save the layers
    self._layers = layers
    ## build the network
    if decoding:
      x = keras.layers.Input(shape=latent_shape)
    else:
      x = keras.layers.Input(shape=input_shape)
    self(x)

  def __repr__(self):
    return layer2text(self)

  def __str__(self):
    return layer2text(self)

  def call(self, inputs, training=None, mask=None):
    first_inputs = inputs
    if not self.convolution:  # dense
      if not self.decoding:
        first_inputs = tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
    else:  # convolution
      if self.decoding:
        first_inputs = tf.expand_dims(inputs, axis=-2)
        first_inputs = tf.expand_dims(first_inputs, axis=-2)
    # iterate each layer
    pool_idx = 0
    for layer_idx, layer in enumerate(self._layers):
      outputs = layer(inputs, training=training)
      inputs = outputs
      ### skip connection
      if (self.skip_connect and
          isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D))):
        ## convolution, downsampling
        if self.convolution and isinstance(layer, keras.layers.Conv2D):
          if self.decoding:  # decoder
            if layer_idx == len(self._layers) - 2:  # skip last layer
              continue
            h, w = inputs.shape[-3:-1]
            batch_shape = (1,) * (len(inputs.shape) - 3)
            inputs = tf.concat(
                [inputs, tf.tile(first_inputs, batch_shape + (h, w, 1))],
                axis=-1,
            )
          else:  # encoder
            p = self.pool_size[pool_idx]
            pool_idx += 1
            inputs = tf.concat(
                [
                    inputs,
                    tf.nn.avg_pool2d(first_inputs, (p, p), (p, p), "SAME")
                ],
                axis=-1,
            )
        ## dense layers
        elif not self.convolution:
          if self.decoding and layer_idx == len(self._layers) - 2:
            continue
          inputs = tf.concat([inputs, first_inputs], axis=-1)
    return outputs


# ===========================================================================
# Factor discriminator
# ===========================================================================
class FactorDiscriminator(SequentialNetwork):
  r""" The main goal is minimizing the total correlation (the mutual information
    which quantifies the redundancy or dependency among latent variables).

    We use a discriminator to estimate TC

  Arguments:
    latent_dim : an Integer, the number of latent units used in VAE.
    hdim : an Integer, the number of hidden units for the discriminator.
    n_layer : an Integer, the number of hidden layers.
    activation : Callable or String, activation function of each layer.

  Reference:
    Kim, H., Mnih, A., 2018. Disentangling by Factorising.
      arXiv:1802.05983 [cs, stat].
  """

  def __init__(self,
               input_shape,
               batchnorm=False,
               units=1000,
               n_hidden_layers=5,
               n_outputs=2,
               activation=tf.nn.leaky_relu,
               name="FactorDiscriminator"):
    # 1: real sample for q(z) (or last unit in case n_outputs > 2) and
    # 0: fake sample from q(z-)
    layers = [
        keras.layers.InputLayer(input_shape=tf.nest.flatten(input_shape)),
        keras.layers.Flatten()
    ]
    for idx, (units, activation) in enumerate(
        zip(as_tuple(units, N=n_hidden_layers),
            as_tuple(activation, N=n_hidden_layers))):
      sublayers = [
          keras.layers.Dense(units, use_bias=not batchnorm, activation='linear')
      ]
      if batchnorm:
        sublayers.append(keras.layers.BatchNormalization())
      sublayers.append(keras.layers.Activation(activation))
      layers += sublayers
    layers.append(keras.layers.Dense(int(n_outputs), activation='linear'))
    super().__init__(layers, name=name)
    self.input_ndim = len(self.input_shape) - 1

  def _to_samples(self, qZ_X):
    qZ_X = tf.nest.flatten(qZ_X)
    z = tf.concat([tf.convert_to_tensor(q) for q in qZ_X], axis=-1)
    z = tf.reshape(z, tf.concat([(-1,), z.shape[-self.input_ndim:]], axis=0))
    return z

  def total_correlation(self, qZ_X, training=None):
    r""" Total correlation Eq(3)
      `TC(z) = KL(q(z)||q(z-)) = E_q(z)[log(q(z) / q(z-))]`

    Note:
      In many implementation, `log(q(z-)) - log(q(z))` is referred as `total
      correlation loss`, here, we return `log(q(z)) - log(q(z-))` as the total
      correlation for the construction of the ELBO in Eq(2)

    Arguments:
      qZ_X : a Tensor, [batch_dim, latent_dim] or Distribution

    Return:
      TC(z) : a scalar, approximation of the density-ratio that arises in the
        KL-term.
    """
    # adding log_softmax here could provide more stable loss than minimizing
    # the logit directly
    # tf.nn.log_softmax
    d_z = self(self._to_samples(qZ_X), training=training)
    return tf.reduce_mean(d_z[..., -1] - d_z[..., 0])

  def dtc_loss(self, qZ_X, qZ_Xprime=None, training=None):
    r""" Discriminated total correlation loss Algorithm(2)

    Minimize the probability of:
     - `q(z)` misclassified as `D(z)[:, 0]`
     - `q(z')` misclassified as `D(z')[:, 1]`

    Arguments:
      qZ_X : `Tensor` or `Distribution`.
        Samples of the latents from first batch
      qZ_Xprime : `Tensor` or `Distribution` (optional).
        Samples of the latents from second batch, this will be permuted.
        If not given, then reuse `qZ_X`.

    Return:
      scalar - loss value for training the discriminator
    """
    # we don't want the gradient to be propagated to the encoder
    z = self._to_samples(qZ_X)
    z = tf.stop_gradient(z)
    z_logits = self(z, training=training)
    d_z = -tf.nn.log_softmax(z_logits, axis=-1)  # must be negative here
    #
    if qZ_Xprime is not None:
      z = self._to_samples(qZ_Xprime)
      z = tf.stop_gradient(z)
    z_perm = permute_dims(z)
    zperm_logits = self(z_perm, training=training)
    d_zperm = -tf.nn.log_softmax(zperm_logits, axis=-1)
    # reduce the negative of d_z, and the positive of d_zperm
    # this equal to cross_entropy(d_z, zeros) + cross_entropy(d_zperm, ones)
    loss = 0.5 * (tf.reduce_mean(d_z[..., 0]) +
                  tf.reduce_mean(d_zperm[..., -1]))
    return loss
