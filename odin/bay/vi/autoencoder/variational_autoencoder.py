from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow_probability.python import distributions as tfd

from odin.bay.random_variable import RandomVariable
from odin.bay.vi.utils import permute_dims
from odin.networks import AutoencoderConfig


# ===========================================================================
# Helpers
# ===========================================================================
def _check_rv(rv):
  assert isinstance(rv, (RandomVariable, layers.Layer)), \
    "Variable must be instance of odin.bay.RandomVariable or keras.layers.Layer, " + \
      "but given: %s" % str(type(rv))
  if isinstance(rv, RandomVariable):
    rv = rv.create_posterior()
  ### make sure it is probabilistic layer
  spec = inspect.getfullargspec(rv.call)
  args = spec.args + spec.kwonlyargs
  requires = ['n_mcmc', 'training']
  if not all(i in args for i in requires):
    raise ValueError("Invalid latent layer of type: %s, require probabilistic "
                     "layer with call method arguments: %s, but given: %s" %
                     (str(type(rv)), requires, str(args)))
  assert hasattr(rv, 'sample') and callable(rv.sample), \
    "latent layer of type: %s, must has sample method for sampling from prior."\
      % str(type(rv))
  ### get the event_shape
  shape = rv.event_shape if hasattr(rv, 'event_shape') else \
    rv.output_shape
  return rv, shape


# ===========================================================================
# Model
# ===========================================================================
class VariationalAutoencoder(keras.Model):
  r"""
  Attributes:
    encoder : `keras.layers.Layer`.
    decoder : `keras.layers.Layer`.
    latent : `keras.layers.Layer`.
    output : `keras.layers.Layer`.
  """

  def __init__(self,
               encoder: layers.Layer = None,
               decoder: layers.Layer = None,
               config: AutoencoderConfig = AutoencoderConfig(),
               outputs=RandomVariable(event_shape=64,
                                      posterior='gaus',
                                      name="InputVariable"),
               latents=RandomVariable(event_shape=10,
                                      posterior='diag',
                                      name="LatentVariable"),
               **kwargs):
    super().__init__(**kwargs)
    ### check latent variable
    latents, latent_shape = _check_rv(latents)
    self.latents = latents
    ### check the input distribution
    outputs, input_shape = _check_rv(outputs)
    self.outputs = outputs
    ### already got the encoder and decoder
    if config is None:
      if encoder is None and decoder is None:
        raise ValueError(
            "Either provide both encoder, decoder or only the config.")
    ### create the network from config
    else:
      assert isinstance(config, AutoencoderConfig), \
        "config must be instance of AutoencoderConfig but given: %s" % \
          str(type(config))
      assert input_shape is not None, \
        "Input shape must be provided in case AutoencoderConfig is specified."
      encoder, decoder = config.create_network(input_shape=input_shape,
                                               latent_shape=latent_shape)
    ### check type
    assert isinstance(encoder, layers.Layer), \
      "encoder must be instance of keras.layers.Layer, but given: %s" % \
        str(type(encoder))
    assert isinstance(decoder, layers.Layer), \
      "decoder must be instance of keras.layers.Layer, but given: %s" % \
        str(type(decoder))
    self.encoder = encoder
    self.decoder = decoder

  def sample_prior(self, sample_shape=(), seed=1):
    r""" Sampling from prior distribution """
    s = self.latents.sample(sample_shape=sample_shape, seed=seed)
    if tf.rank(s) == 1:
      s = tf.expand_dims(s, axis=0)
    return s

  def encode(self, inputs, training=None, n_mcmc=(), **kwargs):
    r""" Encoding inputs to latent codes """
    e = self.encoder(inputs, training=training, **kwargs)
    qZ_X = self.latents(e, training=training, n_mcmc=n_mcmc, **kwargs)
    return qZ_X

  def decode(self, latents, training=None, **kwargs):
    r""" Decoding latent codes, this does not guarantee output the
    reconstructed distribution """
    # convert all latents to Tensor
    list_latents = True
    if isinstance(latents, tfd.Distribution):
      list_latents = False
    latents = tf.nest.flatten(latents)
    shapes = [
        tf.concat([z.batch_shape, z.event_shape], axis=0) for z in latents
    ]
    latents = [tf.convert_to_tensor(z) for z in latents]
    # all sample shape must be the same
    sample_shape = [z.shape[:-len(shape)] for z, shape in zip(latents, shapes)]
    s0 = sample_shape[0]
    for s in sample_shape:
      tf.assert_equal(
          s0, s, "Sample shape from all latent variable must be the same.")
    sample_shape = sample_shape[0]
    # remove sample_shape from latent
    latents = [
        z if len(sample_shape) == 0 else \
          tf.reshape(z, tf.concat([(-1,), s[1:]], axis=0))
        for z, s in zip(latents, shapes)
    ]
    # decoding
    d = self.decoder(latents if list_latents else latents[0],
                     training=training,
                     **kwargs)
    list_outputs = False
    if not tf.is_tensor(d):
      list_outputs = True
    # get back the sample shape
    outputs = []
    for x in tf.nest.flatten(d):
      if tf.is_tensor(x):
        shape = tf.concat([sample_shape, (-1,), x.shape[1:]], axis=0)
        x = tf.reshape(x, shape)
      outputs.append(x)
    # create the output distribution
    if not list_outputs:
      outputs = outputs[0]
    outputs = self.outputs(outputs, training=training)
    return outputs

  def importance_weighted(self, elbo, axis=0):
    elbo = tf.reduce_logsumexp(input_tensor=elbo, axis=axis) -\
        tf.math.log(tf.cast(elbo.shape[axis], dtype=tf.float32))
    return elbo

  def call(self, inputs, training=None, n_mcmc=()):
    qZ_X = self.encode(inputs, training=training, n_mcmc=n_mcmc)
    pX = self.decode(qZ_X, training=training)
    return pX, qZ_X
