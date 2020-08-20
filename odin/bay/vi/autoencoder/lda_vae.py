from __future__ import absolute_import, division, print_function

from functools import partial
from typing import Callable, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras.layers import BatchNormalization, Dense, Layer
from tensorflow_probability.python.distributions import (Dirichlet,
                                                         MultivariateNormalDiag)
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.math import softplus_inverse

from odin.backend.tensor import dropout
from odin.bay.distributions import Dirichlet, Distribution, OneHotCategorical
from odin.bay.helpers import coercible_tensor, kl_divergence
from odin.bay.random_variable import RandomVariable
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import TrainStep
from odin.networks.sequential_networks import NetworkConfig


# ===========================================================================
# Helpers
# ===========================================================================
class LDAdecoder(Layer):

  def __init__(self,
               n_topics: int,
               n_words: int,
               lda_posterior: str,
               dropout_topics: float = 0.0,
               dropout_words: float = 0.0,
               batch_norm: bool = False):
    super().__init__(name="LDA_decoder")
    assert lda_posterior in ("dirichlet", "gaussian"), \
      ("Only support 'dirichlet'-Dirichlet or 'gaussian'-Logistic "
       f"Gaussian distribution, but given: {lda_posterior}")
    self.n_topics = n_topics
    self.n_words = n_words
    self.dropout_topics = float(dropout_topics)
    self.dropout_words = float(dropout_words)
    self.lda_posterior = lda_posterior
    self.batch_norm = bool(batch_norm)
    self.topics_words_logits = self.add_weight(
        'topics_words_logits',
        shape=[n_topics, n_words],
        initializer=tf.initializers.glorot_normal(),
        trainable=True)
    # initialize
    self(Input(shape=(n_topics,)))

  def call(self, docs_topics, training=None, *args, **kwargs):
    topics_words_probs = tf.nn.softmax(self.topics_words_logits, axis=-1)
    docs_words_probs = tf.matmul(docs_topics, topics_words_probs)
    return docs_words_probs

  def __str__(self):
    return (
        f"<LDAdecoder topics:{self.n_topics} vocab:{self.n_words} "
        f"posterior:{self.lda_posterior} "
        f"dropout:(topics:{self.dropout_topics}, words:{self.dropout_words}) "
        f"batchnorm:{self.batch_norm}>")


# ===========================================================================
# Main class
# ===========================================================================
class LDAVAE(BetaVAE):
  r""" Variational Latent Dirichlet Allocation

  Two models are implemented:
    - Prod LDA: Latent Dirichlet Allocation with Products of Experts
      (Srivastava et al. 2017). ProdLDA replaces this word-level mixture with
      a weighted product of experts, resulting in a drastic improvement in
      topic coherence (explicit reparameterization using logistic-normal
      distribution).
    - Implicit LDA (Figurnov et al. 2018) : amortized variational inference in
      LDA using implicit reparameterization.

  To maintain good intuition behind the algorithm, we name the attributes
  as for topics discovery task in natural language processing.

  Arguments:
    n_words : int
      Dictionary size
    n_topics : int, optional (default=10)
      Number of topics in LDA.
    lda_posterior : {"gaussian", "dirichlet"},
      "gaussian" - logistic gaussian (explicit) reparameterization
      "dirichlet" - latent Dirichlet (implicit) reparameterization
    activation : str or Callable.
      Activation for the concentration of Dirichlet distribution
    clipping: bool.
      If True, clipping the concentration to range `[1e-3, 1e3]` for numerical
      stability.
    dropout_topics: float (default: 0.0).
      Dropout value for the topics samples from Dirichlet latents posterior.
      This is recommended to batter the topics collapse.
    dropout_words: float (default: 0.0).
      Dropout value for the topics-words probabilities matrix.
    batch_norm: bool (default: False).
      Batch normalization for
    prior_init : float (default=0.7)
      the initial topic prior concentration for Dirichlet distribution
    prior_warmup : int (default: 10000)
      The number of training steps with fixed prior, only applied for Dirichlet
      LDA's posterior


  Note:
    The algorithm is trained for 180000 iteration on Newsgroup20 dataset,
      of which 120000 iteration for "warmup".

  References:
    David M. Blei, Andrew Y. Ng, Michael I. Jordan. Latent Dirichlet
      Allocation. In JMLR, 2003.
    Salakhutdinov, R., Hinton, G. Replicated Softmax: an Undirected Topic Model.
      In NIPS, 2009.
    Matthew D. Hoffman, David M. Blei, and Francis Bach. Online learning
      for Latent Dirichlet Allocation. In NIPS, 2010
    Miao, Y., Yu, L. and Blunsom, P. Neural Variational Inference for Text
      Processing. arXiv:1511.06038 [cs, stat], 2016.
    Akash Srivastava, Charles Sutton. Autoencoding Variational Inference For
      Topic Models. In ICLR, 2017.
    Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization
      Gradients, 2018.  https://arxiv.org/abs/1805.08498
  """

  def __init__(
      self,
      n_words: int,
      n_topics: int = 20,
      lda_posterior="dirichlet",
      words_distribution="categorical",
      activation: Union[str, Callable] = 'softplus',
      clipping: bool = True,
      dropout_topics: float = 0.0,
      dropout_words: float = 0.0,
      batch_norm: bool = False,
      prior_init: float = 0.7,
      prior_warmup: int = 10000,
      encoder: Union[Layer, NetworkConfig] = NetworkConfig(name="Encoder"),
      beta: float = 1.0,
      **kwargs,
  ):
    ### topic latents distribution
    lda_posterior = str(lda_posterior).lower().strip()
    latents = kwargs.pop("latents", None)
    if latents is not None:
      warn(message=f"Ignore provided latents variable {latents}",
           category=UserWarning)
    n_topics = int(n_topics)
    if lda_posterior == 'dirichlet':
      posterior = "dirichlet"
      post_kwargs = dict(alpha_activation=activation, alpha_clip=clipping)
    elif lda_posterior == "gaussian":
      posterior = "gaussiandiag"
      post_kwargs = dict(loc_activation='identity', scale_activation=activation)
    else:
      raise NotImplementedError(
          "Support one of the following latent distribution: "
          "'gaussian', 'logistic', 'dirichlet'")
    latents = RandomVariable(event_shape=(n_topics,),
                             posterior=posterior,
                             projection=True,
                             kwargs=post_kwargs,
                             name="DocsTopics")
    self.lda_posterior = lda_posterior
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
    decoder = partial(LDAdecoder,
                      n_topics=n_topics,
                      n_words=n_words,
                      lda_posterior=lda_posterior,
                      dropout_topics=dropout_topics,
                      dropout_words=dropout_words,
                      batch_norm=batch_norm)
    ### output layer
    # The observations are bag of words and therefore not one-hot. However,
    # log_prob of OneHotCategorical computes the probability correctly in
    # this case.
    outputs = kwargs.pop('outputs', None)
    if outputs is not None:
      warn(message=f"Ignore provided outputs={outputs}")
    kw = dict()
    if words_distribution in ('onehot', 'categorical'):
      kw['probs_input'] = True
      posterior = 'onehot'
    elif words_distribution in ('negativebinomial', 'nb'):
      posterior = 'negativebinomial'
    elif words_distribution in ('zinb',):
      posterior = 'zinb'
    outputs = RandomVariable(event_shape=(n_words,),
                             posterior=posterior,
                             projection=False,
                             kwargs=kw,
                             name="DocsWords")
    ### analytic
    if 'analytic' not in kwargs:
      kwargs['analytic'] = True
    super().__init__(latents=latents,
                     input_shape=input_shape,
                     encoder=encoder,
                     decoder=decoder,
                     outputs=outputs,
                     beta=beta,
                     **kwargs)
    ### store attributes
    self.n_topics = int(n_topics)
    self.n_words = int(n_words)
    self.clipping = bool(clipping)
    ### create the prior
    self.prior_warmup = int(prior_warmup)
    if lda_posterior == "dirichlet":
      self._topics_prior_logits = self.add_weight(
          initializer=tf.initializers.constant(
              value=softplus_inverse(prior_init).numpy()),
          shape=[1, n_topics],
          trainable=True,
          name="topics_prior_logits",
      )
      self.latent_layers[0].prior = self.topics_prior_distribution
    else:
      self.latent_layers[0].prior = MultivariateNormalDiag(
          loc=tf.zeros(shape=[1, n_topics], dtype=self.dtype),
          scale_identity_multiplier=1.,
          name="TopicsPrior")

  @property
  def topics_words_logits(self) -> tf.Variable:
    r""" Logits of the topics-words distribution, the return matrix shape
    `(n_topics, n_words)` """
    return self.decoder.topics_words_logits

  @property
  def topics_words_probs(self) -> tf.Tensor:
    r""" Probabilities values of the topics-words distribution, the return
    matrix shape `(n_topics, n_words)` """
    return tf.nn.softmax(self.decoder.topics_words_logits, axis=-1)

  @property
  def topics_prior_logits(self) -> Union[tf.Variable, tf.Tensor]:
    r""" Logits of the Dirichlet topics distribution, shape `(1, n_topics)` """
    if self.lda_posterior == "dirichlet":
      return self._topics_prior_logits
    return tf.transpose(
        tf.reduce_sum(self.topics_words_logits, axis=1, keepdims=True))

  @property
  def topics_prior_alpha(self) -> tf.Tensor:
    r""" Dirichlet concentrations for the topics, a `[1, n_topics]` tensor """
    if self.lda_posterior == "dirichlet":
      return tf.nn.softplus(self.topics_prior_logits)
    # logistic-normal
    alpha_logits = self.topics_prior_logits
    return alpha_logits / tf.reduce_sum(alpha_logits)

  def topics_prior_distribution(self) -> Dirichlet:
    r""" Create the prior distribution (i.e. the Dirichlet topics distribution),
    `batch_shape=(1,)` and `event_shape=(n_topics,)` """
    if self.lda_posterior == "dirichlet":
      concentration = tf.nn.softplus(self.topics_prior_logits)
      if self.clipping:
        concentration = tf.clip_by_value(concentration, 1e-3, 1e3)
      # prior warm-up: stop gradients update for prior parameters
      concentration = tf.cond(self.step <= self.prior_warmup,
                              true_fn=lambda: tf.stop_gradient(concentration),
                              false_fn=lambda: concentration)
      return Dirichlet(concentration=concentration, name="TopicsPrior")
    # logistic-normal
    return self.latent_layers[0].prior

  ######## Utilities methods
  def get_topics_string(self,
                        vocabulary: Dict[int, str],
                        n_words: int = 10,
                        n_topics: int = 10,
                        show_word_prob: bool = False) -> List[str]:
    r""" Print most relevant topics and its most representative words
    distribution """
    topics = self.topics_words_probs
    alpha = np.squeeze(self.topics_prior_alpha, axis=0)
    # Use a stable sorting algorithm so that when alpha is fixed
    # we always get the same topics.
    text = []
    for idx, topic_idx in enumerate(
        np.argsort(-alpha, kind="mergesort")[:int(n_topics)]):
      words = topics[topic_idx]
      desc = " ".join(f"{vocabulary[i]}_{words[i]:.2f}"
                      if show_word_prob else f"{vocabulary[i]}"
                      for i in np.argsort(-words)[:int(n_words)])
      text.append(
          f"[#{idx}]index:{topic_idx:3d} alpha={alpha[topic_idx]:.2f} {desc}")
    return np.array(text)

  def perplexity(self, inputs, elbo=None) -> float:
    r""" The perplexity is an exponent of the average negative ELBO per word. """
    if isinstance(inputs, sp.sparse.spmatrix):
      inputs = inputs.toarray()
    if elbo is None:
      pX, qZ = self(inputs, training=False, sample_shape=())
      elbo = self.elbo(inputs,
                       pX_Z=pX,
                       qZ_X=qZ,
                       sample_shape=(),
                       training=False,
                       analytic=True)
    words_per_doc = tf.reduce_sum(inputs, axis=1)
    log_perplexity = -elbo / words_per_doc
    perplexity = tf.exp(tf.reduce_mean(log_perplexity))
    return perplexity
