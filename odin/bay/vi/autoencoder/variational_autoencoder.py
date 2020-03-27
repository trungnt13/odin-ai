from __future__ import absolute_import, division, print_function

import inspect

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow_probability.python import distributions as tfd

from odin.bay.helpers import KLdivergence
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
  if not (hasattr(rv, 'sample') and callable(rv.sample)):
    raise ValueError(
        "latent layer of type: %s, must has sample method for sampling from prior."
        % str(type(rv)))
  ### get the event_shape
  shape = rv.event_shape if hasattr(rv, 'event_shape') else rv.output_shape
  return rv, shape


# ===========================================================================
# Model
# ===========================================================================
class VariationalAutoencoder(keras.Model):
  r""" Base class for all variational autoencoder

  Arguments:
    encoder : `Layer`.
    decoder : `Layer`.
    config : `AutoencoderConfig`.
    outputs : `RandomVariable` or `Layer`.
    latents : `RandomVariable` or `Layer`.

  Call return:
    pX_Z : Distribution
    qZ_X : Distribution

  Layers:
    encoder : `keras.layers.Layer`. Encoding inputs to latents
    decoder : `keras.layers.Layer`. Decoding latents to intermediate states
    latent_layer : `keras.layers.Layer`. The latent variable (random variable)
    output_layer : `keras.layers.Layer`. The output variable (random or
      deterministic variable)
  """

  def __init__(self,
               encoder: layers.Layer = None,
               decoder: layers.Layer = None,
               config: AutoencoderConfig = AutoencoderConfig(),
               output=RandomVariable(event_shape=64,
                                     posterior='gaus',
                                     name="InputVariable"),
               latent=RandomVariable(event_shape=10,
                                     posterior='diag',
                                     name="LatentVariable"),
               **kwargs):
    ### check latent and input distribution
    latent, latent_shape = _check_rv(latent)
    output, input_shape = _check_rv(output)
    super().__init__(**kwargs)
    self.latent_layer = latent
    self.output_layer = output
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

  @property
  def input_shape(self):
    return self.encoder.input_shape

  def sample_prior(self, sample_shape=(), seed=1):
    r""" Sampling from prior distribution """
    s = self.latent_layer.sample(sample_shape=sample_shape, seed=seed)
    if tf.rank(s) == 1:
      s = tf.expand_dims(s, axis=0)
    return s

  def encode(self, inputs, training=None, n_mcmc=(), **kwargs):
    r""" Encoding inputs to latent codes """
    e = self.encoder(inputs, training=training, **kwargs)
    qZ_X = self.latent_layer(e, training=training, n_mcmc=n_mcmc, **kwargs)
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
    sample_shape = [z.shape[:-s.shape[0]] for z, s in zip(latents, shapes)]
    s0 = sample_shape[0]
    for s in sample_shape:
      tf.assert_equal(
          s0, s, "Sample shape from all latent variable must be the same.")
    sample_shape = sample_shape[0]
    # remove sample_shape from latent
    latents = [
        z if len(sample_shape) == 0 else tf.reshape(
            z, tf.concat([(-1,), s[1:]], axis=0))
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
    outputs = self.output_layer(outputs, training=training)
    return outputs

  def elbo(self, X, pX_Z, qZ_X, analytic=False, reverse=True, n_mcmc=1):
    r""" Calculate the distortion (log-likelihood) and rate (KL-divergence)
    for contruction the Evident Lower Bound (ELBO).

    The final ELBO is:
      `ELBO = E_{z~q(Z|X)}[log(p(X|Z))] - KL_{x~p(X)}[q(Z|X)||p(Z)]`

    Arguments:
      analytic : bool (default: False)
        if True, use the close-form solution  for
      n_mcmc : {Tensor, Number}
        number of MCMC samples for MCMC estimation of KL-divergence
      reverse : `bool`. If `True`, calculating `KL(q||p)` which optimizes `q`
        (or p_model) by greedily filling in the highest modes of data (or, in
        other word, placing low probability to where data does not occur).
        Otherwise, `KL(p||q)` a.k.a maximum likelihood, or expectation
        propagation place high probability at anywhere data occur
        (i.e. averagely fitting the data).

    Return:
      log-likelihood : a `Tensor` of shape [sample_shape, batch_size].
        The sample shape could be ommited in case `n_mcmc=()`.
        The log-likelihood or distortion
      divergence : a `Tensor` of shape [n_mcmc, batch_size].
        The reversed KL-divergence or rate
    """
    X = [tf.convert_to_tensor(x) for x in tf.nest.flatten(X)]
    pX_Z = tf.nest.flatten(pX_Z)
    qZ_X = tf.nest.flatten(qZ_X)
    dtype = X[0].dtype
    ### llk
    llk = tf.convert_to_tensor(0., dtype=dtype)
    for x, pX in zip(X, pX_Z):
      llk += pX.log_prob(x)
    ### kl
    div = tf.convert_to_tensor(0., dtype=dtype)
    for qZ in qZ_X:
      div += qZ.KL_divergence(analytic=analytic,
                              reverse=reverse,
                              n_mcmc=n_mcmc,
                              keepdims=True)
    return llk, div

  def importance_weighted(self, elbo, axis=0):
    r""" VAE objective can lead to overly simplified representations which
    fail to use the network’s entire modeling capacity.

    Importance weighted autoencoder (IWAE) uses a strictly tighter
    log-likelihood lower bound derived from importance weighting.

    Using more samples can only improve the tightness of the bound, and
    as our estimator is based on the log of the average importance weights,
    it does not suffer from high variance.

    Reference:
      Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. Importance Weighted
        Autoencoders. In ICLR, 2015. https://arxiv.org/abs/1509.00519
    """
    dtype = elbo.dtype
    iw_dim = tf.cast(elbo.shape[axis], dtype=dtype)
    elbo = tf.reduce_logsumexp(elbo, axis=axis) - tf.math.log(iw_dim)
    return elbo

  def call(self, inputs, training=None, n_mcmc=()):
    qZ_X = self.encode(inputs, training=training, n_mcmc=n_mcmc)
    pX_Z = self.decode(qZ_X, training=training)
    return pX_Z, qZ_X
