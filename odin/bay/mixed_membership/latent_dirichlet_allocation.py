from __future__ import print_function, division, absolute_import

import numpy as np

import tensorflow as tf

from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, Layer

from tensorflow_probability.python.distributions import (
  softplus_inverse, Dirichlet)

from odin.bay.distribution_layers import DirichletLayer, OneHotCategoricalLayer
from odin.bay.helpers import kl_divergence

__all__ = [
    'LatentDirichletAllocation'
]


class LatentDirichletAllocation(Model):
  """ Variational Latent Dirichlet Allocation

  To maintain good intuition behind the algorithm, we name the
  attributes as for topics discovery task in natural language
  processing.

  Parameters
  ----------
  n_components : int, optional (default=10)
    Number of topics in LDA.

  components_prior : float (default=0.7)
    the topic prior concentration for Dirichlet distribution

  References
  ----------
  [1]: David M. Blei, Andrew Y. Ng, Michael I. Jordan. Latent Dirichlet
       Allocation. In _Journal of Machine Learning Research_, 2003.
       http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
  [2]: Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization
       Gradients, 2018
       https://arxiv.org/abs/1805.08498
  [3]: Akash Srivastava, Charles Sutton. Autoencoding Variational Inference For
       Topic Models. In _International Conference on Learning Representations_,
       2017.
       https://arxiv.org/abs/1703.01488
  """

  def __init__(self, n_components=10, components_prior=0.7,
               encoder_layers=[64, 64], activation='relu',
               n_mcmc_samples=1, analytic_kl=True,
               random_state=None):
    super(LatentDirichletAllocation, self).__init__()
    self._random_state = np.random.RandomState(seed=random_state) \
        if not isinstance(random_state, np.random.RandomState) \
          else random_state
    self._initializer = tf.initializers.GlorotNormal(
        seed=self._random_state.randint(1e8))

    self.n_components = int(n_components)
    self.components_prior = np.array(softplus_inverse(components_prior))

    self.n_mcmc_samples = n_mcmc_samples
    self.analytic_kl = analytic_kl
    # ====== encoder ====== #
    encoder = Sequential(name="Encoder")
    for num_hidden_units in encoder_layers:
      encoder.add(
          Dense(num_hidden_units,
                activation=activation,
                kernel_initializer=self._initializer))
    encoder.add(
        Dense(n_components,
              activation=tf.nn.softplus,
              kernel_initializer=self._initializer,
              name="DenseConcentration"))
    encoder.add(DirichletLayer(clip_for_stable=True,
                               pre_softplus=False,
                               name="topics_posterior"))
    self.encoder = encoder
    # ====== decoder ====== #
    # The observations are bag of words and therefore not one-hot. However,
    # log_prob of OneHotCategorical computes the probability correctly in
    # this case.
    self.decoder = OneHotCategoricalLayer(
      probs_input=True, name="bag_of_words")

  def build(self, input_shape):
    n_features = input_shape[1]
    # decoder
    self.topics_words_logits = self.add_weight(
        name="topics_words_logits",
        shape=[self.n_components, n_features],
        initializer=self._initializer)
    # prior
    self.prior_logit = self.add_weight(
        name="prior_logit",
        shape=[1, self.n_components],
        trainable=False,
        initializer=tf.initializers.Constant(self.components_prior))
    # call this to set built flag to True
    super(LatentDirichletAllocation, self).build(input_shape)

  def call(self, inputs):
    docs_topics_posterior = self.encoder(inputs)
    docs_topics_samples = docs_topics_posterior.sample(self.n_mcmc_samples)

    # [n_topics, n_words]
    topics_words_probs = tf.nn.softmax(self.topics_words_logits, axis=1)
    # [n_docs, n_words]
    docs_words_probs = tf.matmul(docs_topics_samples, topics_words_probs)
    output_dist = self.decoder(
      tf.clip_by_value(docs_words_probs, 1e-4, 1 - 1e-4))

    # initiate prior, concentration is clipped to stable range
    # for Dirichlet
    concentration = tf.clip_by_value(
        tf.nn.softplus(self.prior_logit), 1e-3, 1e3)
    topics_prior = Dirichlet(
        concentration=concentration, name="topics_prior")

    # ELBO
    kl = kl_divergence(q=docs_topics_posterior, p=topics_prior,
                       use_analytic_kl=self.analytic_kl,
                       q_sample=self.n_mcmc_samples,
                       auto_remove_independent=True)
    if self.analytic_kl:
      kl = tf.expand_dims(kl, axis=0)
    llk = output_dist.log_prob(inputs)
    ELBO = llk - kl

    # maximizing ELBO, hence, minizing following loss
    self.add_loss(tf.reduce_mean(-ELBO))
    self.add_metric(tf.reduce_mean(kl), aggregation='mean', name="MeanKL")
    self.add_metric(tf.reduce_mean(-llk), aggregation='mean', name="MeanNLLK")

    return output_dist
