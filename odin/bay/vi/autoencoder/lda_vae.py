from __future__ import absolute_import, division, print_function

from typing import Optional
from warnings import warn

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import (Input, Model, Sequential, constraints,
                                     initializers, regularizers)
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow_probability.python.distributions import Dirichlet
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.math import softplus_inverse

from odin.bay.distributions import Dirichlet, Distribution, OneHotCategorical
from odin.bay.helpers import coercible_tensor, kl_divergence
from odin.bay.random_variable import RandomVariable
from odin.bay.vi.autoencoder import TrainStep, VariationalAutoencoder
from odin.networks.sequential_networks import NetworkConfig


class LDAdecoder(Layer):

  def __init__(self,
               n_topics: int,
               n_words: int,
               topics_words_logits: Optional[tf.Variable] = None):
    super().__init__(name="LDA_decoder")
    self.n_topics = n_topics
    self.n_words = n_words
    if isinstance(topics_words_logits, tf.Variable):
      raise NotImplementedError
    else:
      self.topics_words_logits = self.add_weight(
          'topics_words_logits',
          shape=[n_topics, n_words],
          initializer=initializers.get('glorot_normal'),
          regularizer=regularizers.get(None),
          constraint=constraints.get(None),
          dtype=self.dtype,
          trainable=True)
    # initialize
    self(Input(shape=(n_topics,)))

  def call(self, topics, *args, **kwargs):
    topics_words = tf.nn.softmax(self.topics_words_logits, axis=-1)
    word_probs = tf.matmul(topics, topics_words)
    return word_probs


class LDAVAE(VariationalAutoencoder):
  r""" Variational Latent Dirichlet Allocation

  To maintain good intuition behind the algorithm, we name the
  attributes as for topics discovery task in natural language
  processing.

  Arguments:
    n_topics : int, optional (default=10)
      Number of topics in LDA.
    components_prior : float (default=0.7)
      the topic prior concentration for Dirichlet distribution
    prior_warmup : int (default: 10000)
      The number of training steps with fixed prior.

  References
    David M. Blei, Andrew Y. Ng, Michael I. Jordan. Latent Dirichlet
      Allocation. In JMLR, 2003.
    Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization
      Gradients, 2018.  https://arxiv.org/abs/1805.08498
    Akash Srivastava, Charles Sutton. Autoencoding Variational Inference For
      Topic Models. In ICLR, 2017.
    Matthew D. Hoffman, David M. Blei, and Francis Bach. 2010. Online learning
      for Latent Dirichlet Allocation. In NIPS, 2010

  """

  def __init__(self,
               n_words,
               n_topics=10,
               alpha_activation='softplus',
               alpha_clip=True,
               temperature=0,
               prior_init=0.7,
               prior_warmup=10000,
               encoder=NetworkConfig(name="TopicsEncoder"),
               **kwargs):
    ### topic latents distribution
    latents = kwargs.pop("latents", None)
    if latents is not None:
      warn(message=f"Ignore provided latents variable {latents}",
           category=UserWarning)
    n_topics = int(n_topics)
    latents = RandomVariable(event_shape=(n_topics,),
                             posterior="dirichlet",
                             projection=True,
                             kwargs=dict(alpha_activation=alpha_activation,
                                         alpha_clip=alpha_clip),
                             name="Topics")
    ### input shape
    n_words = int(n_words)
    input_shape = kwargs.pop('input_shape', None)
    if input_shape is not None:
      warn(message=f"Ignore provided input_shape={input_shape}",
           category=UserWarning)
    input_shape = (n_words,)
    ### LDA decoder
    decoder = kwargs.pop('decoder', None)
    if decoder is not None:
      warn(message=f"Ignore provided decoder={decoder}")
    decoder = LDAdecoder(n_topics=n_topics, n_words=n_words)
    ### output layer
    outputs = kwargs.pop('outputs', None)
    if outputs is not None:
      warn(message=f"Ignore provided outputs={outputs}")
    kw = dict(probs_input=True)
    if temperature > 0:
      posterior = 'relaxedonehot'
      kw['temperature'] = temperature
    else:
      posterior = 'onehot'
    outputs = RandomVariable(event_shape=(n_words,),
                             posterior=posterior,
                             projection=False,
                             kwargs=kw,
                             name="Words")
    ### analytic
    if 'analytic' not in kwargs:
      kwargs['analytic'] = True
    super().__init__(latents=latents,
                     input_shape=input_shape,
                     encoder=encoder,
                     decoder=decoder,
                     outputs=outputs,
                     **kwargs)
    ### store attributes
    self.n_topics = int(n_topics)
    self.n_words = int(n_words)
    self.alpha_clip = bool(alpha_clip)
    ### create the prior
    self._topics_prior_logits = self.add_weight(
        initializer=initializers.constant(
            value=softplus_inverse(prior_init).numpy()),
        shape=[1, n_topics],
        dtype=self.dtype,
        trainable=True,
        name="topics_prior_logits",
    )
    self.latent_layers[0].prior = self.topics_prior_distribution
    self.prior_warmup = int(prior_warmup)

  @property
  def topics_words_logits(self) -> tf.Variable:
    r""" Logits of the topics-words distribution, the return matrix shape
    `(n_topics, n_words)` """
    return self.decoder.topics_words_logits

  @property
  def topics_words_probs(self) -> tf.Tensor:
    r""" Probabilities of the topics-words distribution, the return matrix
    shape `(n_topics, n_words)`
    """
    return tf.nn.softmax(self.decoder.topics_words_logits, axis=-1)

  @property
  def topics_prior_logits(self) -> tf.Variable:
    r""" Logits of the Dirichlet topics distribution, shape `(1, n_topics)` """
    return self._topics_prior_logits

  def topics_prior_distribution(self) -> Dirichlet:
    r""" Create the prior distribution (i.e. the Dirichlet topics distribution),
    `batch_shape=(1,)` and `event_shape=(n_topics,)` """
    concentration = tf.nn.softplus(self.topics_prior_logits)
    if self.alpha_clip:
      concentration = tf.clip_by_value(concentration, 1e-3, 1e3)
    # prior warm-up: stop gradients update for prior parameters
    concentration = tf.cond(self.step <= self.prior_warmup,
                            true_fn=lambda: tf.stop_gradient(concentration),
                            false_fn=lambda: concentration)
    return Dirichlet(concentration=concentration, name="topics_prior")
