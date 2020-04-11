from __future__ import absolute_import, division, print_function

import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl

from odin.backend.keras_helpers import layer2text
from odin.bay.random_variable import RandomVariable
from odin.exp import Trainer
from odin.networks import NetworkConfig, SequentialNetwork


# ===========================================================================
# Helpers
# ===========================================================================
def _check_rv(rv, input_shape):
  assert isinstance(rv, (RandomVariable, Layer)), \
    "Variable must be instance of odin.bay.RandomVariable or keras.layers.Layer, " + \
      "but given: %s" % str(type(rv))
  if isinstance(rv, RandomVariable):
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


# ===========================================================================
# Training step
# ===========================================================================
class TrainStep:
  r""" A single train step (iteration) for Variational Autoencoder,
  when called will return:

    - a scalar for loss
    - a list of Tensor or scale of metrics
  """

  def __init__(self,
               vae,
               inputs,
               sample_shape=(),
               iw=False,
               elbo_kw=dict(),
               parameters=None):
    self.vae = vae
    self.parameters = (vae.trainable_variables
                       if parameters is None else parameters)
    self.inputs = inputs
    self.sample_shape = sample_shape
    self.iw = iw
    self.elbo_kw = elbo_kw

  def __call__(self, training=True):
    pX_Z, qZ_X = self.vae(self.inputs,
                          training=training,
                          sample_shape=self.sample_shape)
    # store so it could be reused
    self.pX_Z = pX_Z
    self.qZ_X = qZ_X
    llk, div = self.vae.elbo(self.inputs,
                             pX_Z,
                             qZ_X,
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

  def __init__(self,
               encoder: Union[Layer, NetworkConfig] = None,
               decoder: Union[Layer, NetworkConfig] = None,
               config: Optional[NetworkConfig] = NetworkConfig(),
               outputs: Union[Layer,
                              RandomVariable] = RandomVariable(event_shape=64,
                                                               posterior='gaus',
                                                               name="Input"),
               latents: Union[Layer,
                              RandomVariable] = RandomVariable(event_shape=10,
                                                               posterior='diag',
                                                               name="Latent"),
               reduce_latent='concat',
               input_shape=None,
               step=0.,
               **kwargs):
    name = kwargs.pop('name', None)
    if name is None:
      name = type(self).__name__
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
    if encoder is not None:
      if isinstance(encoder, NetworkConfig):
        encoder = encoder.create_network(input_shape, name="Encoder")
      elif hasattr(encoder, 'input_shape'):
        assert list(encoder.input_shape[1:]) == input_shape, \
          "encoder has input_shape=%s but VAE input_shape=%s" % \
            (str(encoder.input_shape[1:]), str(input_shape))
    else:
      assert isinstance(config, NetworkConfig), \
        "config must be instance of NetworkConfig but given: %s" % \
          str(type(config))
      encoder = config.create_network(input_shape)
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
    if decoder is not None:
      if isinstance(decoder, NetworkConfig):
        decoder = decoder.create_network(latent_shape, name="Decoder")
      elif hasattr(decoder, 'input_shape'):
        assert list(decoder.input_shape[-1:]) == latent_shape, \
          "decoder has input_shape=%s but latent_shape=%s" % \
            (str(decoder.input_shape[-1:]), str(latent_shape))
    else:
      decoder = config.create_decoder(encoder=encoder,
                                      latent_shape=latent_shape)
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
        shape = [1 if i is None else i for i in layer._batch_input_shape]
        x = tf.ones(shape=shape, dtype=layer.dtype)
        layer(x)  # call this dummy input to build the layer
    ### the training step
    self.step = tf.Variable(step,
                            dtype=self.dtype,
                            trainable=False,
                            name="Step")
    self._trainstep_kw = dict()
    self.latent_names = [i.name for i in self.latent_layers]
    # keras already use output_names, cannot override it
    self.variable_names = [i.name for i in self.output_layers]
    self._compiled_call = None

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
      s = latent.sample(sample_shape=sample_shape, seed=seed)
      if tf.rank(s) == 1:  # at-least 2D
        s = tf.expand_dims(s, axis=0)
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def sample_data(self, sample_shape=(), seed=1):
    r""" Sample from p(X) given that the prior of X is known, this could be
    wrong since `RandomVariable` often has a default prior. """
    samples = []
    for output in self.output_layers:
      s = output.sample(sample_shape=sample_shape, seed=seed)
      if tf.rank(s) == 1:  # at-least 2D
        s = tf.expand_dims(s, axis=0)
      samples.append(s)
    return samples[0] if len(samples) == 1 else tuple(samples)

  def generate(self, sample_shape=(), seed=1, training=None, **kwargs):
    z = self.sample_prior(sample_shape, seed)
    return self.decode(z, training=training, **kwargs)

  def encode(self, inputs, training=None, sample_shape=(), **kwargs):
    r""" Encoding inputs to latent codes """
    e = self.encoder(inputs, training=training, **kwargs)
    qZ_X = [
        latent(e, training=training, sample_shape=sample_shape)
        for latent in self.latent_layers
    ]
    return qZ_X[0] if len(qZ_X) == 1 else tuple(qZ_X)

  def decode(self, latents, training=None, sample_shape=(), **kwargs):
    r""" Decoding latent codes, this does not guarantee output the
    reconstructed distribution """
    sample_shape = tf.nest.flatten(sample_shape)
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
    outputs = self.decoder(
        latents,
        training=training,
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
    return dist[0] if len(self.output_layers) == 1 else tuple(dist)

  def call(self, inputs, training=None, sample_shape=()):
    qZ_X = self.encode(inputs, training=training, sample_shape=sample_shape)
    pX_Z = self.decode(qZ_X, training=training, sample_shape=sample_shape)
    return pX_Z, qZ_X

  @tf.function(autograph=False)
  def marginal_log_prob(self, inputs, training=False, sample_shape=100):
    r""" marginal log-likelihood of shape [batch_size]
    """
    sample_shape = tf.cast(tf.reduce_prod(sample_shape), tf.int32)
    iw_const = tf.math.log(tf.cast(sample_shape, self.dtype))
    pX_Z, qZ_X = self.call(inputs, training=training, sample_shape=sample_shape)
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
           pX_Z,
           qZ_X,
           analytic=False,
           reverse=True,
           sample_shape=None,
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
    X = [tf.convert_to_tensor(x, dtype=self.dtype) for x in tf.nest.flatten(X)]
    pX_Z = tf.nest.flatten(pX_Z)
    qZ_X = tf.nest.flatten(qZ_X)
    llk, div = self._elbo(X, pX_Z, qZ_X, analytic, reverse, sample_shape,
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

  def train_steps(self,
                  inputs,
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
                    sample_shape=sample_shape,
                    iw=iw,
                    elbo_kw=elbo_kw)

  def optimize(self, inputs, tape=None, training=True, optimizer=None):
    if optimizer is None:
      optimizer = tf.nest.flatten(self.optimizer)
    all_metrics = {}
    total_loss = 0.
    for opt, step in zip(optimizer,
                         self.train_steps(inputs, **self._trainstep_kw)):
      loss, metrics = step(training=training)
      # update metrics and loss
      all_metrics.update(metrics)
      total_loss += loss
      if tape is not None:
        Trainer.apply_gradients(tape, opt, loss, step.parameters)
        # tape need to be reseted for next
        tape.reset()
    return total_loss, {i: tf.reduce_mean(j) for i, j in all_metrics.items()}

  def fit(
      self,
      train: tf.data.Dataset,
      valid: Optional[tf.data.Dataset] = None,
      valid_freq=1000,
      valid_interval=0,
      optimizer='adam',
      learning_rate=1e-3,
      clipnorm=None,
      epochs=2,
      max_iter=-1,
      sample_shape=(),  # for ELBO
      analytic=False,  # for ELBO
      iw=False,  # for ELBO
      callback=lambda: None,
      compile_graph=True,
      autograph=False,
      logging_interval=2,
      log_tag='',
      log_path=None):
    trainer = Trainer()
    self.trainer = trainer
    if optimizer is not None:
      self.optimizer = _to_optimizer(optimizer, learning_rate, clipnorm)
    if self.optimizer is None:
      raise RuntimeError("No optimizer found!")
    self._trainstep_kw = dict(sample_shape=sample_shape,
                              iw=iw,
                              elbo_kw=dict(analytic=analytic))
    trainer.fit(train_ds=train.repeat(int(epochs)) if epochs > 1 else train,
                optimize=self.optimize,
                valid_ds=valid,
                valid_freq=valid_freq,
                valid_interval=valid_interval,
                persistent_tape=True,
                compile_graph=compile_graph,
                autograph=autograph,
                logging_interval=logging_interval,
                log_tag=log_tag,
                log_path=log_path,
                max_iter=max_iter,
                callback=callback)
    self._trainstep_kw = dict()
    return self

  def __str__(self):
    clses = [
        i for i in type.mro(type(self)) if issubclass(i, VariationalAutoencoder)
    ]
    text = "%s" % "->".join([i.__name__ for i in clses[::-1]])
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
    return text
