from __future__ import absolute_import, division, print_function

import glob
import inspect
import os
import pickle
import warnings
from functools import partial
from typing import Callable, List, Optional, Union

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl

from odin import backend as bk
from odin.backend.keras_helpers import layer2text
from odin.bay.random_variable import RandomVariable as RV
from odin.networks import NetworkConfig, SequentialNetwork


# ===========================================================================
# Helpers
# ===========================================================================
def _check_rv(rv, input_shape):
  assert isinstance(rv, (RV, Layer)), \
    "Variable must be instance of odin.bay.RandomVariable or keras.layers.Layer, " + \
      "but given: %s" % str(type(rv))
  if isinstance(rv, RV):
    rv = rv.create_posterior(input_shape=input_shape)
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


def _net2str(net):
  if isinstance(net, keras.Sequential):
    return layer2text(net)
  elif isinstance(net, tfl.DistributionLambda):
    return layer2text(net)
  return str(net)


def _to_optimizer(optimizer, learning_rate, clipnorm):
  optimizer = tf.nest.flatten(optimizer)
  learning_rate = tf.nest.flatten(learning_rate)
  clipnorm = tf.nest.flatten(clipnorm)
  if len(learning_rate) == 1:
    learning_rate = learning_rate * len(optimizer)
  if len(clipnorm) == 1:
    clipnorm = clipnorm * len(clipnorm)
  ## create the optimizer
  all_optimizers = []
  for opt, lr, clip in zip(optimizer, learning_rate, clipnorm):
    # string
    if isinstance(opt, string_types):
      config = dict(learning_rate=float(lr))
      if clip is not None:
        config['clipnorm'] = clip
      opt = tf.optimizers.get({'class_name': opt, 'config': config})
    # the instance
    elif isinstance(opt, tf.optimizers.Optimizer):
      pass
    # type
    elif inspect.isclass(opt) and issubclass(opt, tf.optimizers.Optimizer):
      opt = opt(learning_rate=float(learning_rate)) \
        if clipnorm is None else \
        opt(learning_rate=float(learning_rate), clipnorm=clipnorm)
    # no support
    else:
      raise ValueError("No support for optimizer: %s" % str(opt))
    all_optimizers.append(opt)
  return all_optimizers


def _parse_network_alias(encoder, decoder):
  if isinstance(encoder, string_types):
    encoder = str(encoder).lower().strip()
    from odin.bay.vi.autoencoder.networks import ImageNet
    if encoder in ('mnist', 'fashion_mnist'):
      kw = dict(image_shape=(28, 28, 1),
                projection_dim=128,
                activation='relu',
                center0=True,
                distribution='bernoulli',
                distribution_kw=dict(),
                skip_connect=False,
                convolution=True,
                input_shape=None)
      encoder = ImageNet(**kw)
      decoder = partial(ImageNet, decoding=True, **kw)
    elif encoder in ('shapes3d', 'dsprites', 'celeba', 'stl10', 'legofaces',
                     'cifar10', 'cifar20', 'cifar100'):
      n_channels = 1 if encoder == 'dsprites' else 3
      if encoder in ('cifar10', 'cifar100', 'cifar20'):
        image_shape = (32, 32, 3)
      else:
        image_shape = (64, 64, n_channels)
      kw = dict(image_shape=image_shape,
                projection_dim=256,
                activation='relu',
                center0=True,
                distribution='bernoulli',
                distribution_kw=dict(),
                skip_connect=False,
                convolution=True,
                input_shape=None)
      encoder = ImageNet(**kw)
      decoder = partial(ImageNet, decoding=True, **kw)
    else:
      raise NotImplementedError(
          "No support for predefined network for dataset with name: '%s'" %
          encoder)
  return encoder, decoder


# ===========================================================================
# Training step
# ===========================================================================
class TrainStep:
  r""" A single train step (iteration) for Variational Autoencoder,
  when called will return:

    - a scalar for loss
    - a dictionary of Tensor for monitoring metrics

  Arguments:
    vae : `VariationalAutoencoder`
    inputs : a list of input `Tensor`
    sample_shape : MCMC sample shape
    iw : a Boolean. If True, enable importance weight sampling
    elbo_kw : a Dictionary. Keyword arguments for elbo function
    parameters : optimizing parameters, if None, all parameters of VAE are
      optimized
  """

  def __init__(self,
               vae,
               inputs,
               training=None,
               mask=None,
               sample_shape=(),
               iw=False,
               elbo_kw=dict(),
               parameters=None):
    self.vae = vae
    assert isinstance(vae, VariationalAutoencoder)
    self.parameters = (vae.trainable_variables
                       if parameters is None else parameters)
    self.inputs = inputs
    self.mask = mask
    self.sample_shape = sample_shape
    self.iw = iw
    self.elbo_kw = elbo_kw
    self.training = training

  def __call__(self):
    pX_Z, qZ_X = self.vae(self.inputs,
                          training=self.training,
                          mask=self.mask,
                          sample_shape=self.sample_shape)
    # store so it could be reused
    self.pX_Z = pX_Z
    self.qZ_X = qZ_X
    llk, div = self.vae.elbo(self.inputs,
                             pX_Z,
                             qZ_X,
                             training=self.training,
                             mask=self.mask,
                             return_components=True,
                             **self.elbo_kw)
    # sum all the components log-likelihood and divergence
    llk_sum = tf.constant(0., dtype=self.vae.dtype)
    div_sum = tf.constant(0., dtype=self.vae.dtype)
    for x in llk.values():
      llk_sum += x
    for x in div.values():
      div_sum += x
    elbo = llk_sum - div_sum
    if self.iw and tf.rank(elbo) > 1:
      elbo = self.vae.importance_weighted(elbo, axis=0)
    loss = -tf.reduce_mean(elbo)
    # metrics
    metrics = llk
    metrics.update(div)
    return loss, metrics


# ===========================================================================
# Model
# ===========================================================================
class VariationalAutoencoder(keras.Model):
  r""" Base class for all variational autoencoder

  Arguments:
    encoder : `keras.layers.Layer` or `odin.networks.NetworkConfig`.
    decoder : `keras.layers.Layer` or `odin.networks.NetworkConfig`.
    outputs : `RandomVariable` or `Layer`. List of output distribution
    latents : `RandomVariable` or `Layer`. List of latent distribution

  Call return:
    pX_Z : a single or a list of `tensorflow_probability.Distribution`
    qZ_X : a single or a list of `tensorflow_probability.Distribution`

  Layers:
    encoder : `keras.layers.Layer`. Encoding inputs to latents
    decoder : `keras.layers.Layer`. Decoding latents to intermediate states
    latent_layers : `keras.layers.Layer`. A list of the Dense layer that create
      the latent variable (random variable)
    output_layers : `keras.layers.Layer`. A list of the Dense layer that create
      the output variable (random or deterministic variable)
  """

  def __init__(self,
               encoder: Union[Layer, NetworkConfig] = NetworkConfig(),
               decoder: Union[Layer, NetworkConfig] = None,
               outputs: Union[Layer, RV] = RV(event_shape=64,
                                              posterior='gaus',
                                              projection=True,
                                              name="Input"),
               latents: Union[Layer, RV] = RV(event_shape=10,
                                              posterior='diag',
                                              projection=True,
                                              name="Latent"),
               reduce_latent='concat',
               input_shape=None,
               step=0.,
               **kwargs):
    name = kwargs.pop('name', None)
    path = kwargs.pop('path', None)
    optimizer = kwargs.pop('optimizer', None)
    learning_rate = kwargs.pop('learning_rate', 1e-4)
    clipnorm = kwargs.pop('clipnorm', None)
    if name is None:
      name = type(self).__name__
    ### keras want this supports_masking on to enable support masking
    self.supports_masking = True
    super().__init__(**kwargs)
    ### First, infer the right input_shape
    outputs = tf.nest.flatten(outputs)
    if input_shape is None:
      input_shape = [
          o.event_shape if hasattr(o, 'event_shape') else o.output_shape
          for o in outputs
      ]
      if len(outputs) == 1:
        input_shape = input_shape[0]
    ### Then, create the encoder, so we know the input_shape to latent layers
    config = None
    encoder, decoder = _parse_network_alias(encoder, decoder)
    if isinstance(encoder, NetworkConfig):
      config = encoder
      encoder = encoder.create_network(input_shape, name="Encoder")
    elif hasattr(encoder, 'input_shape') and \
      list(encoder.input_shape[1:]) != input_shape:
      warnings.warn("encoder has input_shape=%s but VAE output_shape=%s" %
                    (str(encoder.input_shape[1:]), str(input_shape)))
    ### check latent and input distribution
    all_latents = [
        _check_rv(z, input_shape=encoder.output_shape[1:])
        for z in tf.nest.flatten(latents)
    ]
    self.latent_layers = [z[0] for z in all_latents]
    self.latent_args = [_get_args(i) for i in self.latent_layers]
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
    ### Create the decoder
    n_parameterization = 1
    if isinstance(outputs[0], RV):
      n_parameterization = outputs[0].n_parameterization
    if isinstance(decoder, partial):
      decoder = decoder(latent_shape=latent_shape)
    if decoder is not None:
      if isinstance(decoder, NetworkConfig):
        decoder = decoder.create_network(latent_shape, name="Decoder")
      elif hasattr(decoder, 'input_shape') and \
        list(decoder.input_shape[-1:]) != latent_shape:
        warnings.warn("decoder has input_shape=%s but latent_shape=%s" %
                      (str(decoder.input_shape[-1:]), str(latent_shape)))
    else:
      decoder = config.create_decoder(encoder=encoder,
                                      latent_shape=latent_shape,
                                      n_parameterization=n_parameterization)
    ### Finally the output distributions
    all_outputs = [_check_rv(x, decoder.output_shape[1:]) for x in outputs]
    self.output_layers = [x[0] for x in all_outputs]
    self.output_args = [_get_args(i) for i in self.output_layers]
    ### check type
    assert isinstance(encoder, Layer), \
      "encoder must be instance of keras.Layer, but given: %s" % \
        str(type(encoder))
    assert isinstance(decoder, Layer), \
      "decoder must be instance of keras.Layer, but given: %s" % \
        str(type(decoder))
    self.encoder = encoder
    self.decoder = decoder
    ### build the latent and output layers
    for layer in self.latent_layers + self.output_layers:
      if hasattr(layer, '_batch_input_shape') and not layer.built:
        shape = layer._batch_input_shape
        # call this dummy input to build the layer
        layer(keras.Input(shape=shape[1:], batch_size=shape[0]))
    ### the training step
    self.step = tf.Variable(step,
                            dtype=self.dtype,
                            trainable=False,
                            name="Step")
    self.trainer = None
    self._trainstep_kw = dict()
    self.latent_names = [i.name for i in self.latent_layers]
    # keras already use output_names, cannot override it
    self.variable_names = [i.name for i in self.output_layers]
    self._compiled_call = None
    ### load saved weights if available
    if optimizer is not None:
      self.optimizer = _to_optimizer(optimizer, learning_rate, clipnorm)
    else:
      self.optimizer = None
    self.load_weights(path, raise_notfound=False)

  @property
  def save_path(self):
    return self._save_path

  def load_weights(self, filepath, raise_notfound=False):
    if isinstance(filepath, string_types):
      files = glob.glob(filepath + '*')
      # load weights
      if len(files) > 0 and all(os.path.isfile(f) for f in files):
        super().load_weights(filepath, by_name=False, skip_mismatch=False)
      elif raise_notfound:
        raise FileNotFoundError(f"Cannot find saved weights at path: {filepath}")
      # load trainer
      trainer_path = filepath + '.trainer'
      if os.path.exists(trainer_path):
        with open(trainer_path, 'rb') as f:
          self.trainer = pickle.load(f)
    self._save_path = filepath
    return self

  def save_weights(self, filepath, overwrite=True):
    r""" Just copy this function here to fix the `save_format` to 'tf'

    Since saving 'h5' will drop certain variables.
    """
    with open(filepath + '.trainer', 'wb') as f:
      pickle.dump(self.trainer, f)
    return super().save_weights(filepath=filepath,
                                overwrite=overwrite,
                                save_format='tf')

  @property
  def is_semi_supervised(self):
    return False

  @property
  def is_self_supervised(self):
    return False

  @property
  def is_weak_supervised(self):
    return False

  @property
  def is_fitted(self):
    return self.step.numpy() > 0

  @property
  def compiled_call(self) -> Callable:
    if self._compiled_call is None:
      self._compiled_call = tf.function(self.call, autograph=False)
    return self._compiled_call

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
      s = bk.atleast_2d(latent.sample(sample_shape=sample_shape, seed=seed))
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def sample_data(self, sample_shape=(), seed=1):
    r""" Sample from p(X) given that the prior of X is known, this could be
    wrong since `RandomVariable` often has a default prior. """
    samples = []
    for output in self.output_layers:
      s = bk.atleast_2d(output.sample(sample_shape=sample_shape, seed=seed))
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def generate(self, sample_shape=(), seed=1, training=None, **kwargs):
    r""" Randomly generate outputs by sampling from prior distribution then
    decode it. """
    z = self.sample_prior(sample_shape, seed)
    return self.decode(z, training=training, **kwargs)

  def encode(self, inputs, training=None, mask=None, sample_shape=(), **kwargs):
    r""" Encoding inputs to latent codes """
    e = self.encoder(inputs, training=training, mask=mask, **kwargs)
    qZ_X = [
        latent(e, training=training, sample_shape=sample_shape)
        for latent in self.latent_layers
    ]
    for q in qZ_X:  # remember to store the keras mask in outputs
      q._keras_mask = mask
    return qZ_X[0] if len(qZ_X) == 1 else tuple(qZ_X)

  def _prepare_decode_latents(self, latents, sample_shape):
    # convert all latents to Tensor
    list_latents = True
    if isinstance(latents, tfd.Distribution) or tf.is_tensor(latents):
      list_latents = False
    latents = tf.nest.flatten(latents)
    # remove sample_shape
    if len(sample_shape) > 0:
      # if we call tf.convert_to_tensor or tf.reshape directly here the llk
      # could go worse for some unknown reason, but using keras layers is ok!
      ndim = len(sample_shape) + 1
      reshape = keras.layers.Lambda(lambda x: tf.reshape(
          x, tf.concat([(-1,), tf.shape(x)[ndim:]], axis=0)))
      latents = [reshape(z) for z in latents]
    # decoding
    latents = _reduce_latents(
        latents, self.reduce_latent) if list_latents else latents[0]
    return latents

  def decode(self,
             latents,
             training=None,
             mask=None,
             sample_shape=(),
             **kwargs):
    r""" Decoding latent codes, this does not guarantee output the
    reconstructed distribution """
    sample_shape = tf.nest.flatten(sample_shape)
    latents = self._prepare_decode_latents(latents, sample_shape)
    outputs = self.decoder(
        latents,
        training=training,
        mask=mask,
        **kwargs,
    )
    # get back the sample shape
    if len(sample_shape) > 0:
      list_outputs = False
      if not tf.is_tensor(outputs):
        list_outputs = True
      outputs = [
          tf.reshape(o, tf.concat([sample_shape, (-1,), o.shape[1:]], axis=0))
          for o in tf.nest.flatten(outputs)
      ]
      if not list_outputs:
        outputs = outputs[0]
    # create the output distribution
    dist = [layer(outputs, training=training) for layer in self.output_layers]
    for p in dist:  # remember to store the keras mask in outputs
      p._keras_mask = mask
    return dist[0] if len(self.output_layers) == 1 else tuple(dist)

  def call(self, inputs, training=None, mask=None, sample_shape=()):
    qZ_X = self.encode(inputs,
                       training=training,
                       mask=mask,
                       sample_shape=sample_shape)
    # transfer the mask from encoder to decoder here
    for q in tf.nest.flatten(qZ_X):
      if hasattr(q, '_keras_mask') and q._keras_mask is not None:
        mask = q._keras_mask
        break
    pX_Z = self.decode(qZ_X,
                       training=training,
                       mask=mask,
                       sample_shape=sample_shape)
    return pX_Z, qZ_X

  @tf.function(autograph=False)
  def marginal_log_prob(self,
                        inputs,
                        training=False,
                        mask=None,
                        sample_shape=100):
    r"""
    Return:
      a Tensor of shape [batch_size]
        marginal log-likelihood
    """
    sample_shape = tf.cast(tf.reduce_prod(sample_shape), tf.int32)
    iw_const = tf.math.log(tf.cast(tf.reduce_prod(sample_shape), self.dtype))
    pX_Z, qZ_X = self.call(inputs,
                           training=training,
                           mask=mask,
                           sample_shape=sample_shape)
    llk = []
    for i, (p,
            x) in enumerate(zip(tf.nest.flatten(pX_Z),
                                tf.nest.flatten(inputs))):
      batch_llk = p.log_prob(x)
      batch_llk = tf.reduce_logsumexp(batch_llk, axis=0) - iw_const
      llk.append(batch_llk)
    return llk[0] if len(llk) == 1 else llk

  def _elbo(self,
            X,
            pX_Z,
            qZ_X,
            analytic,
            reverse,
            sample_shape=None,
            mask=None,
            training=None,
            **kwargs):
    r""" The basic components of all ELBO """
    ### llk
    llk = {}
    for name, x, pX in zip(self.variable_names, X, pX_Z):
      llk['llk_%s' % name] = pX.log_prob(x)
    ### kl
    div = {}
    for name, qZ in zip(self.latent_names, qZ_X):
      div['kl_%s' % name] = qZ.KL_divergence(analytic=analytic,
                                             reverse=reverse,
                                             sample_shape=sample_shape,
                                             keepdims=True)
    return llk, div

  def elbo(self,
           X,
           pX_Z=None,
           qZ_X=None,
           analytic=False,
           reverse=True,
           sample_shape=None,
           mask=None,
           training=None,
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
      sample_shape : {Tensor, Number}
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
      return_elbo : a Boolean. If True, gather the components (log-likelihood
        and KL-divergence) to form and return ELBO.

    Return:
      log-likelihood : dictionary of `Tensor` of shape [sample_shape, batch_size].
        The sample shape could be ommited in case `sample_shape=()`.
        The log-likelihood or distortion
      divergence : dictionary `Tensor` of shape [sample_shape, batch_size].
        The reversed KL-divergence or rate
    """
    if qZ_X is None:
      qZ_X = self.encode(X,
                         training=training,
                         mask=mask,
                         sample_shape=sample_shape)
    if pX_Z is None:
      pX_Z = self.decode(qZ_X,
                         training=training,
                         mask=mask,
                         sample_shape=sample_shape)
    # organize all inputs to list
    X = [tf.convert_to_tensor(x, dtype=self.dtype) for x in tf.nest.flatten(X)]
    pX_Z = tf.nest.flatten(pX_Z)
    qZ_X = tf.nest.flatten(qZ_X)
    # override the default mask
    # if the processed mask from decoder is available
    # but, it still unclear should we use the original mask or the processed
    # mask here
    # if hasattr(pX_Z[0], '_keras_mask') and pX_Z[0]._keras_mask is not None:
    # mask = pX_Z[0]._keras_mask
    llk, div = self._elbo(X,
                          pX_Z,
                          qZ_X,
                          analytic,
                          reverse,
                          sample_shape=sample_shape,
                          mask=mask,
                          training=training,
                          **kwargs)
    if not (isinstance(llk, dict) and isinstance(div, dict)):
      raise RuntimeError(
          "When overriding VariationalAutoencoder _elbo method must return "
          "dictionaries for log-likelihood and KL-divergence.")
    ## only return the components, no need else here but it is clearer
    if return_components:
      return llk, div
    ## calculate the ELBO
    # sum all the components log-likelihood and divergence
    llk_sum = tf.constant(0., dtype=self.dtype)
    div_sum = tf.constant(0., dtype=self.dtype)
    for x in llk.values():
      llk_sum += x
    for x in div.values():
      div_sum += x
    elbo = llk_sum - div_sum
    if iw and tf.rank(elbo) > 1:
      elbo = self.importance_weighted(elbo, axis=0)
    return elbo, llk_sum, div_sum

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

  ################## For training
  def train_steps(self,
                  inputs,
                  training=True,
                  mask=None,
                  sample_shape=(),
                  iw=False,
                  elbo_kw=dict()) -> TrainStep:
    r""" Facilitate multiple steps training for each iteration (smilar to GAN)

    Example:
    ```
    vae = FactorVAE()
    x = vae.sample_data()
    vae_step, discriminator_step = list(vae.train_steps(x))

    # optimizer VAE with total correlation loss
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(vae_step.parameters)
      loss, metrics = vae_step()
      tape.gradient(loss, vae_step.parameters)

    # optimizer the discriminator
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(discriminator_step.parameters)
      loss, metrics = discriminator_step()
      tape.gradient(loss, discriminator_step.parameters)
    ```
    """
    self.step.assign_add(1)
    yield TrainStep(vae=self,
                    inputs=inputs,
                    training=training,
                    mask=mask,
                    sample_shape=sample_shape,
                    iw=iw,
                    elbo_kw=elbo_kw)

  def optimize(self, inputs, training=True, mask=None, optimizer=None):
    if optimizer is None:
      optimizer = tf.nest.flatten(self.optimizer)
    all_metrics = {}
    total_loss = 0.
    optimizer = tf.nest.flatten(optimizer)
    n_optimizer = len(optimizer)
    for i, step in enumerate(
        self.train_steps(inputs=inputs,
                         training=training,
                         mask=mask,
                         **self._trainstep_kw)):
      opt = optimizer[i % n_optimizer]
      parameters = step.parameters
      # this GradientTape somehow more inconvenient than pytorch
      if training:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
          tape.watch(parameters)
          loss, metrics = step()
        # applying the gradients
        gradients = tape.gradient(loss, parameters)
        if self._check_gradients:  # for debugging
          grad_param = zip(gradients, parameters)
        else:
          grad_param = [
              (g, p) for g, p in zip(gradients, parameters) if g is not None
          ]
        opt.apply_gradients(grad_param)
      else:
        tape = None
        loss, metrics = step()
      # update metrics and loss
      all_metrics.update(metrics)
      total_loss += loss
    return total_loss, {i: tf.reduce_mean(j) for i, j in all_metrics.items()}

  def fit(
      self,
      train: tf.data.Dataset,
      valid: Optional[tf.data.Dataset] = None,
      valid_freq=500,
      valid_interval=0,
      optimizer='adam',
      learning_rate=1e-3,
      clipnorm=None,
      epochs=-1,
      max_iter=1000,
      sample_shape=(),  # for ELBO
      analytic=False,  # for ELBO
      iw=False,  # for ELBO
      callback=None,
      compile_graph=True,
      autograph=False,
      logging_interval=2,
      skip_fitted=False,
      log_tag='',
      log_path=None,
      earlystop_threshold=0.001,
      earlystop_progress_length=0,
      earlystop_patience=-1,
      earlystop_min_epoch=-np.inf,
      terminate_on_nan=True,
      check_gradients=False):
    r""" Override the original fit method of keras to provide simplified
    procedure with `VariationalAutoencoder.optimize` and
    `VariationalAutoencoder.train_steps`

    Arguments:
      optimizer : Text, instance of `tf.optimizers.Optimizer`
        or `None`. A list of optimizers is accepted in case of multiple
        steps training.
        - If `None`, re-use stored optimizer, raise `RuntimeError` if no
          predefined optimizer found.
      callback : a Callable, called every `valid_freq` steps or
        `valid_interval` seconds
      compile_graph : a Boolean. If True, using tensorflow autograph for
        optimize function (about 2 times better speed), otherwise, run the
        function in Eager mode (better for debugging).

    """
    if self.is_fitted and skip_fitted:
      return self
    from odin.exp.trainer import Trainer
    if self.trainer is None:
      trainer = Trainer()
      trainer.early_stop
      self.trainer = trainer
    else:
      trainer = self.trainer
    if log_tag is None or len(log_tag) == 0:
      log_tag = self.__class__.__name__
    # create the optimizer
    if optimizer is not None and self.optimizer is None:
      self.optimizer = _to_optimizer(optimizer, learning_rate, clipnorm)
    if self.optimizer is None:
      raise RuntimeError("No optimizer found!")
    self._trainstep_kw = dict(sample_shape=sample_shape,
                              iw=iw,
                              elbo_kw=dict(analytic=analytic))
    self._check_gradients = bool(check_gradients)
    callback_functions = [i for i in tf.nest.flatten(callback) if callable(i)]
    saved_weights = [0]
    earlystop_patience = int(earlystop_patience)
    patience = [earlystop_patience]

    # run early stop and callback
    def _callback():
      for f in callback_functions:
        f()
      if earlystop_patience > 0:
        if valid is not None:
          losses = trainer.valid_loss_epoch
        else:
          losses = trainer.train_loss
          ids = list(range(0, len(losses), valid_freq)) + [len(losses)]
          losses = [np.mean(losses[s:e]) for s, e in zip(ids, ids[1:])]
        signal = Trainer.early_stop(losses=losses,
                                    threshold=earlystop_threshold,
                                    progress_length=earlystop_progress_length,
                                    min_epoch=earlystop_min_epoch,
                                    terminate_on_nan=terminate_on_nan,
                                    verbose=True)
        if signal == Trainer.SIGNAL_BEST:
          saved_weights[0] = self.step.numpy()
          patience[0] = min(patience[0] + 1. / earlystop_patience,
                            earlystop_patience)
          Trainer.save_weights(self)
        elif signal == Trainer.SIGNAL_TERMINATE:
          patience[0] -= 1
          tf.print(
              f"[EarlyStop] Patience decreased: {patience[0]:.2f}/{earlystop_patience}"
          )
          if patience[0] >= 0:
            signal = None
        return signal

    # if already called repeat, then no need to repeat more
    if hasattr(train, 'repeat'):
      train = train.repeat(int(epochs))
    self.trainer.fit(train_ds=train,
                     optimize=self.optimize,
                     valid_ds=valid,
                     valid_freq=valid_freq,
                     valid_interval=valid_interval,
                     compile_graph=compile_graph,
                     autograph=autograph,
                     logging_interval=logging_interval,
                     log_tag=log_tag,
                     log_path=log_path,
                     max_iter=max_iter,
                     callback=_callback)
    self._trainstep_kw = dict()
    # restore best weights
    if saved_weights[0] > 0:
      tf.print("[EarlyStop] Restore best weights from step %d" %
               saved_weights[0])
      Trainer.restore_weights(self)
    return self

  def plot_learning_curves(self,
                           path="/tmp/tmp.png",
                           summary_steps=[100, 10],
                           show_validation=True,
                           dpi=100,
                           title=None):
    assert self.trainer is not None, \
      "fit method must be called before plotting learning curves"
    self.trainer.plot_learning_curves(path=path,
                                      summary_steps=summary_steps,
                                      show_validation=show_validation,
                                      dpi=dpi,
                                      title=title)
    return self

  def __str__(self):
    cls = [
        i for i in type.mro(type(self)) if issubclass(i, VariationalAutoencoder)
    ]
    text = "%s supervising(semi:%s self:%s weak:%s)" % (
        "->".join([i.__name__ for i in cls[::-1]]), self.is_semi_supervised,
        self.is_self_supervised, self.is_weak_supervised)
    ## encoder
    text += "\n Encoder:\n  "
    text += "\n  ".join(_net2str(self.encoder).split('\n'))
    ## Decoder
    text += "\n Decoder:\n  "
    text += "\n  ".join(_net2str(self.decoder).split('\n'))
    ## Latent
    for i, latent in enumerate(self.latent_layers):
      text += "\n Latent#%d:\n  " % i
      text += "\n  ".join(_net2str(latent).split('\n'))
    ## Ouput
    for i, output in enumerate(self.output_layers):
      text += "\n Output#%d:\n  " % i
      text += "\n  ".join(_net2str(output).split('\n'))
    ## Optimizer
    if hasattr(self, 'optimizer'):
      for i, opt in enumerate(tf.nest.flatten(self.optimizer)):
        if isinstance(opt, tf.optimizers.Optimizer):
          text += "\n Optimizer#%d:\n  " % i
          text += "\n  ".join(
              ["%s:%s" % (k, str(v)) for k, v in opt.get_config().items()])
    return text
