from __future__ import absolute_import, division, print_function

import inspect
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python import distributions as tfd

from odin.bay.random_variable import RandomVariable
from odin.networks import NetworkConfig


# ===========================================================================
# Helpers
# ===========================================================================
def _check_rv(rv):
  assert isinstance(rv, (RandomVariable, Layer)), \
    "Variable must be instance of odin.bay.RandomVariable or keras.layers.Layer, " + \
      "but given: %s" % str(type(rv))
  if isinstance(rv, RandomVariable):
    rv = rv.create_posterior()
  ### get the event_shape
  shape = rv.event_shape if hasattr(rv, 'event_shape') else rv.output_shape
  return rv, shape


def _get_args(layer):
  spec = inspect.getfullargspec(layer.call)
  return set(spec.args + spec.kwonlyargs)


def _latent_shape(z):
  if isinstance(z, tfd.Distribution):
    shape = tf.concat([z.batch_shape, z.event_shape], axis=0)
  else:
    shape = tf.convert_to_tensor(z.shape)
  return shape


def _reduce_latents(latents, mode):
  if mode is None:
    return latents
  if isinstance(mode, string_types):
    if mode == 'concat':
      return tf.concat(latents, axis=-1)
    if mode == 'mean':
      return tf.reduce_mean(tf.stack(latents), axis=0)
    if mode == 'sum':
      return tf.reduce_sum(tf.stack(latents), axis=0)
    if mode == 'min':
      return tf.reduce_min(tf.stack(latents), axis=0)
    if mode == 'max':
      return tf.reduce_max(tf.stack(latents), axis=0)
  return mode(latents)


# ===========================================================================
# Model
# ===========================================================================
class VariationalAutoencoder(keras.Model):
  r""" Base class for all variational autoencoder

  Arguments:
    encoder : `Layer`.
    decoder : `Layer`.
    config : `NetworkConfig`.
    outputs : `RandomVariable` or `Layer`. List of output distribution
    latents : `RandomVariable` or `Layer`. List of latent distribution

  Call return:
    pX_Z : Distribution
    qZ_X : Distribution

  Layers:
    encoder : `keras.layers.Layer`. Encoding inputs to latents
    decoder : `keras.layers.Layer`. Decoding latents to intermediate states
    latent_layers : `keras.layers.Layer`. The latent variable (random variable)
    output_layers : `keras.layers.Layer`. The output variable (random or
      deterministic variable)
  """

  def __init__(
      self,
      encoder: Union[Layer, NetworkConfig] = None,
      decoder: Union[Layer, NetworkConfig] = None,
      config: Optional[NetworkConfig] = NetworkConfig(),
      outputs: Union[Layer,
                     RandomVariable] = RandomVariable(event_shape=64,
                                                      posterior='gaus',
                                                      name="InputVariable"),
      latents: Union[Layer,
                     RandomVariable] = RandomVariable(event_shape=10,
                                                      posterior='diag',
                                                      name="LatentVariable"),
      reduce_latent='concat',
      input_shape=None,
      **kwargs):
    name = kwargs.pop('name', None)
    if name is None:
      name = type(self).__name__
    ### check latent and input distribution
    all_latents = [_check_rv(z) for z in tf.nest.flatten(latents)]
    all_outputs = [_check_rv(x) for x in tf.nest.flatten(outputs)]
    super().__init__(**kwargs)
    self.latent_layers = [z[0] for z in all_latents]
    self.output_layers = [x[0] for x in all_outputs]
    # arguments for call
    self.latent_args = [_get_args(i) for i in self.latent_layers]
    self.output_args = [_get_args(i) for i in self.output_layers]
    # input shape
    if input_shape is None:
      input_shape = [shape for _, shape in all_outputs]
      if len(all_outputs) == 1:
        input_shape = input_shape[0]
    # validate method for latent reduction
    assert isinstance(reduce_latent, string_types) or \
      callable(reduce_latent) or reduce_latent is None,\
      "reduce_latent must be None, string or callable, but given: %s" % \
        str(type(reduce_latent))
    latent_shape = [shape for _, shape in all_latents]
    if reduce_latent is None:
      pass
    elif isinstance(reduce_latent, string_types):
      reduce_latent = reduce_latent.strip().lower()
      if reduce_latent == 'concat':
        latent_shape = sum(np.array(s) for s in latent_shape).tolist()
      elif reduce_latent in ('mean', 'min', 'max', 'sum'):
        latent_shape = latent_shape[0]
      else:
        raise ValueError("No support for reduce_latent='%s'" % reduce_latent)
    else:
      zs = [
          tf.zeros(shape=(1,) + tuple(s), dtype=self.dtype)
          for s in latent_shape
      ]
      latent_shape = list(reduce_latent(zs).shape[1:])
    self.reduce_latent = reduce_latent
    ### already got the encoder and decoder
    if config is None:
      if encoder is None and decoder is None:
        raise ValueError(
            "Either provide both encoder, decoder or only the config.")
      if isinstance(encoder, NetworkConfig):
        encoder = encoder.create_network(input_shape=input_shape,
                                         name="Encoder")
      if isinstance(decoder, NetworkConfig):
        decoder = decoder.create_network(input_shape=latent_shape,
                                         name="Decoder")
    ### create the network from config
    else:
      assert isinstance(config, NetworkConfig), \
        "config must be instance of NetworkConfig but given: %s" % \
          str(type(config))
      assert input_shape is not None, \
        "Input shape must be provided in case NetworkConfig is specified."
      encoder, decoder = config.create_network(input_shape=input_shape,
                                               latent_shape=latent_shape)
    ### check type
    assert isinstance(encoder, Layer), \
      "encoder must be instance of keras.Layer, but given: %s" % \
        str(type(encoder))
    assert isinstance(decoder, Layer), \
      "decoder must be instance of keras.Layer, but given: %s" % \
        str(type(decoder))
    self.encoder = encoder
    self.decoder = decoder

  @property
  def input_shape(self):
    return self.encoder.input_shape

  @property
  def latent_shape(self):
    return self.decoder.input_shape

  def sample_prior(self, sample_shape=(), seed=1):
    r""" Sampling from prior distribution """
    samples = []
    for latent in self.latent_layers:
      s = latent.sample(sample_shape=sample_shape, seed=seed)
      if tf.rank(s) == 1:  # at-least 2D
        s = tf.expand_dims(s, axis=0)
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def encode(self, inputs, training=None, n_mcmc=(), **kwargs):
    r""" Encoding inputs to latent codes """
    e = self.encoder(inputs, training=training, **kwargs)
    qZ_X = []
    for latent, args in zip(self.latent_layers, self.latent_args):
      kw = dict(kwargs)
      if 'training' in args:
        kw['training'] = training
      if 'n_mcmc' in args:
        kw['n_mcmc'] = n_mcmc
      qZ_X.append(latent(e, **kw))
    return qZ_X if len(qZ_X) == 1 else tuple(qZ_X)

  def decode(self, latents, training=None, **kwargs):
    r""" Decoding latent codes, this does not guarantee output the
    reconstructed distribution """
    # convert all latents to Tensor
    list_latents = True
    if isinstance(latents, tfd.Distribution) or tf.is_tensor(latents):
      list_latents = False
    latents = tf.nest.flatten(latents)
    shapes = [_latent_shape(z) for z in latents]
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
    d = self.decoder(
        _reduce_latents(latents, self.reduce_latent)
        if list_latents else latents[0],
        training=training,
        **kwargs,
    )
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
    dist = []
    for layer, args in zip(self.output_layers, self.output_args):
      if 'training' in args:
        o = layer(outputs, training=training)
      else:
        o = layer(outputs)
      dist.append(o)
    return dist[0] if len(self.output_layers) == 1 else tuple(dist)

  def call(self, inputs, training=None, n_mcmc=()):
    qZ_X = self.encode(inputs, training=training, n_mcmc=n_mcmc)
    pX_Z = self.decode(qZ_X, training=training)
    return pX_Z, qZ_X

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, n_mcmc, **kwargs):
    r""" The basic components of all ELBO """
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

  def elbo(self,
           X,
           pX_Z,
           qZ_X,
           analytic=False,
           reverse=True,
           n_mcmc=1,
           iw=False,
           return_components=False,
           **kwargs):
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
      iw : a Boolean. If True, the final ELBO is importance weighted sampled.
        This won't be applied if `return_components=True` or `rank(elbo)` <= 1.
      return_components : a Boolean. If True return the log-likelihood and the
        KL-divergence instead of final ELBO.

    Return:
      log-likelihood : a `Tensor` of shape [sample_shape, batch_size].
        The sample shape could be ommited in case `n_mcmc=()`.
        The log-likelihood or distortion
      divergence : a `Tensor` of shape [n_mcmc, batch_size].
        The reversed KL-divergence or rate
    """
    llk, div = self._elbo(X, pX_Z, qZ_X, analytic, reverse, n_mcmc, **kwargs)
    if return_components:
      return llk, div
    elbo = llk - div
    if iw and tf.rank(elbo) > 1:
      elbo = self.importance_weighted(elbo, axis=0)
    return elbo

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

  def __str__(self):
    clses = [
        i for i in type.mro(type(self)) if issubclass(i, VariationalAutoencoder)
    ]
    text = "%s" % "->".join([i.__name__ for i in clses[::-1]])
    ## encoder
    text += "\n Encoder:\n  "
    text += "\n  ".join(str(self.encoder).split('\n'))
    ## Decoder
    text += "\n Decoder:\n  "
    text += "\n  ".join(str(self.decoder).split('\n'))
    ## Latent
    for i, latent in enumerate(self.latent_layers):
      text += "\n Latent#%d:\n  " % i
      text += "\n  ".join(str(latent).split('\n'))
    ## Ouput
    for i, output in enumerate(self.output_layers):
      text += "\n Output#%d:\n  " % i
      text += "\n  ".join(str(output).split('\n'))
    return text
