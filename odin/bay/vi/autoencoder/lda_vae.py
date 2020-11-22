from __future__ import absolute_import, annotations, division, print_function

from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import scipy as sp
import tensorflow as tf
from odin.backend.tensor import dropout as apply_dropout
from odin.bay.distributions import (Dirichlet, Distribution, LogitNormal,
                                    MultivariateNormalDiag, OneHotCategorical)
from odin.bay.layers import (BinomialLayer, DistributionDense, DirichletLayer,
                             DistributionLambda, MultinomialLayer,
                             MultivariateNormalLayer, NegativeBinomialLayer,
                             OneHotCategoricalLayer, PoissonLayer,
                             ZINegativeBinomialLayer)
from odin.bay.random_variable import RVmeta
from odin.bay.vi._base import VariationalModel
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import (LayerCreator,
                                                             TensorTypes,
                                                             VAEStep)
from odin.networks import NetworkConfig
from scipy import sparse
from tensorflow import Tensor, Variable
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.keras import Input, Model, Sequential, activations
from tensorflow.python.keras.layers import (Activation, BatchNormalization,
                                            Dense, InputLayer, Lambda, Layer)
from tensorflow.python.training.tracking import base as trackable
from tensorflow_probability.python.math import softplus_inverse
from tqdm import tqdm
from typing_extensions import Literal

# __all__ = ['LatentDirichletDecoder', 'AmortizedLDA', 'TwoStageLDA', 'VDA', ]


# ===========================================================================
# Helpers
# ===========================================================================
class LatentDirichletDecoder(Model):
  r""" Parameterized word distribution layer for latent dirichlet allocation
  algorithm

  To maintain good intuition behind the algorithm, we name the attributes
  as for topics discovery task in natural language processing.

  Two models are implemented:
    - Prod LDA: Latent Dirichlet Allocation with Products of Experts
      (Srivastava et al. 2017). ProdLDA replaces this word-level mixture with
      a weighted product of experts, resulting in a drastic improvement in
      topic coherence (explicit reparameterization using logistic-normal
      distribution).
    - Implicit LDA (Figurnov et al. 2018) : amortized variational inference in
      LDA using implicit reparameterization.


  Arguments:
    n_words : int
      number of input features or size of word dictionary for bag-of-words
    n_topics : int, optional (default=10)
      Number of topics in LDA.
    posterior : {"gaussian", "dirichlet"},
      "gaussian" - logistic gaussian (explicit) reparameterization
      "dirichlet" - latent Dirichlet (implicit) reparameterization
    distribution : {'onehot', 'negativebinomial', 'binomial', 'poisson',
      'zinb'}
      the output distribution for documents-words matrix
    warmup: int = 10000,
      The number of training steps with fixed prior
    dropout: float (default: 0.0).
      Dropout value for the docs-words logits matrix.
    dropout_strategy: {'all', 'warmup', 'finetune'}
      decide when applying dropout on docs-words logits matrix:
        - 'warmup' : only applying dropout during warmup phase.
        - 'finetune' : only applying dropout during topic priors finetuning.
        - 'all' : always applying dropout.
    batch_norm: bool (default: False).
      Batch normalization for docs-words logits matrix

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
      posterior: Literal['gaussian', 'dirichlet'] = 'dirichlet',
      posterior_activation: Union[str, Callable[[], Tensor]] = 'softplus',
      concentration_clip: bool = True,
      distribution: Literal['onehot', 'negativebinomial', 'binomial', 'poisson',
                            'zinb'] = 'onehot',
      dropout: float = 0.0,
      dropout_strategy: Literal['all', 'warmup', 'finetune'] = 'warmup',
      batch_norm: bool = False,
      trainable_prior: bool = True,
      warmup: int = 10000,
      step: Union[int, Variable] = 0,
      input_shape: Optional[List[int]] = None,
      name: str = "Topics",
  ):
    super().__init__(name=name)
    self.n_words = int(n_words)
    self.n_topics = int(n_topics)
    self.batch_norm = bool(batch_norm)
    self.warmup = int(warmup)
    self.posterior = str(posterior).lower()
    self.distribution = str(distribution).lower()
    self.dropout = float(dropout)
    self.warmup = int(warmup)
    assert dropout_strategy in ('all', 'warmup', 'finetune'), \
      ("Support dropout strategy: all, warmup, finetune; "
       f"but given:{dropout_strategy}")
    self.dropout_strategy = str(dropout_strategy)
    if isinstance(step, Variable):
      self.step = step
    else:
      self.step = Variable(int(step),
                           dtype=tf.float32,
                           trainable=False,
                           name="Step")
    ### batch norm
    if self.batch_norm:
      self._batch_norm_layer = BatchNormalization(trainable=True)
    ### posterior
    kw = dict(event_shape=(n_topics,), name="TopicsPosterior")
    if posterior == 'dirichlet':
      kw['posterior'] = DirichletLayer
      init_value = softplus_inverse(0.7).numpy()
      post_kw = dict(concentration_activation=posterior_activation,
                     concentration_clip=concentration_clip)
    elif posterior == "gaussian":
      kw['posterior'] = MultivariateNormalLayer
      init_value = 0.
      post_kw = dict(covariance='diag',
                     loc_activation='identity',
                     scale_activation=posterior_activation)
    else:
      raise NotImplementedError(
          "Support one of the following latent distribution: "
          "'gaussian', 'dirichlet'")
    self.topics_prior_logits = self.add_weight(
        initializer=tf.initializers.constant(value=init_value),
        shape=[1, n_topics],
        trainable=bool(trainable_prior),
        name="topics_prior_logits")
    self.posterior_layer = DistributionDense(
        posterior_kwargs=post_kw,
        prior=self.topics_prior_distribution,
        projection=True,
        **kw)
    ### output distribution
    kw = dict(event_shape=(self.n_words,), name="WordsDistribution")
    count_activation = 'softplus'
    if self.distribution in ('onehot',):
      self.distribution_layer = OneHotCategoricalLayer(probs_input=True, **kw)
      self.n_parameterization = 1
    elif self.distribution in ('poisson',):
      self.distribution_layer = PoissonLayer(**kw)
      self.n_parameterization = 1
    elif self.distribution in ('negativebinomial', 'nb'):
      self.distribution_layer = NegativeBinomialLayer(
          count_activation=count_activation, **kw)
      self.n_parameterization = 2
    elif self.distribution in ('zinb',):
      self.distribution_layer = ZINegativeBinomialLayer(
          count_activation=count_activation, **kw)
      self.n_parameterization = 3
    elif self.distribution in ('binomial',):
      self.distribution_layer = BinomialLayer(count_activation=count_activation,
                                              **kw)
      self.n_parameterization = 2
    else:
      raise ValueError(f"No support for word distribution: {self.distribution}")
    # topics words parameterization
    self.topics_words_params = self.add_weight(
        'topics_words_params',
        shape=[self.n_topics, self.n_words * self.n_parameterization],
        initializer=tf.initializers.glorot_normal(),
        trainable=True)
    # initialize the Model if input_shape given
    if input_shape is not None:
      self.build((None,) + tuple(input_shape))

  @property
  def topics_words_logits(self) -> Union[Tensor, Variable]:
    r""" Logits of the topics-words distribution, the return matrix shape
    `(n_topics, n_words)` """
    logits = self.topics_words_params
    if isinstance(self.distribution_layer, OneHotCategoricalLayer):
      pass
    elif isinstance(self.distribution_layer, PoissonLayer):
      pass
    elif isinstance(self.distribution_layer,
                    (NegativeBinomialLayer, MultinomialLayer, BinomialLayer)):
      # total_count, success_logits
      logits = logits[..., -self.n_words:]
    elif isinstance(self.distribution_layer, ZINegativeBinomialLayer):
      # total_count, success_logits, zeros_inflation_rate
      logits = logits[..., -2 * self.n_words:-self.n_words]
    return logits

  @property
  def topics_words_probs(self) -> tf.Tensor:
    r""" Probabilities values of the topics-words distribution, the return
    matrix shape `(n_topics, n_words)` """
    return tf.nn.softmax(self.topics_words_logits, axis=-1)

  @property
  def topics_concentration(self) -> tf.Tensor:
    r""" Dirichlet concentration for the topics, a `[1, n_topics]` tensor """
    # dirichlet
    if self.posterior == "dirichlet":
      return tf.nn.softplus(self.topics_prior_logits)
    # logistic-normal
    elif self.posterior == "gaussian":
      logits = tf.transpose(
          tf.reduce_sum(self.topics_words_logits, axis=1, keepdims=True))
      return logits / tf.reduce_sum(logits)

  def topics_prior_distribution(
      self) -> Union[Dirichlet, MultivariateNormalDiag]:
    r""" Create the prior distribution (i.e. the Dirichlet topics distribution),
    `batch_shape=(1,)` and `event_shape=(n_topics,)` """
    # warm-up: stop gradients update for prior parameters
    logits = tf.cond(self.step < self.warmup,
                     true_fn=lambda: tf.stop_gradient(self.topics_prior_logits),
                     false_fn=lambda: self.topics_prior_logits)
    if self.posterior == "dirichlet":
      concentration = tf.nn.softplus(logits)
      concentration = tf.clip_by_value(concentration, 1e-3, 1e3)
      prior = Dirichlet(concentration=concentration, name="TopicsPrior")
    # logistic-normal
    elif self.posterior == "gaussian":
      prior = MultivariateNormalDiag(loc=logits,
                                     scale_identity_multiplier=1.,
                                     name="TopicsPrior")
    return prior

  def encode(
      self,
      inputs: Union[TensorTypes, List[TensorTypes]],
      training: Optional[bool] = None,
      mask: Optional[TensorTypes] = None,
      sample_shape: List[int] = ()
  ) -> Distribution:
    docs_topics_dist = self.posterior_layer(inputs,
                                            training=training,
                                            sample_shape=sample_shape)
    return docs_topics_dist

  def decode(self,
             docs_topics_dist: Union[Distribution, TensorTypes],
             training: Optional[bool] = None) -> Distribution:
    if self.posterior == 'dirichlet':
      docs_topics_probs = docs_topics_dist
    elif self.posterior == 'gaussian':
      docs_topics_probs = tf.nn.softmax(docs_topics_dist, axis=-1)
    docs_words_logits = tf.matmul(docs_topics_probs, self.topics_words_params)
    # perform dropout
    if self.dropout > 0:
      if self.dropout_strategy == 'all':
        docs_words_logits = apply_dropout(docs_words_logits,
                                          p_drop=self.dropout,
                                          training=training)
      else:
        if self.dropout_strategy == 'finetune':
          condition = self.step <= self.warmup
        elif self.dropout_strategy == 'warmup':
          condition = self.step > self.warmup
        docs_words_logits = tf.cond(
            condition,
            true_fn=lambda: docs_words_logits,
            false_fn=lambda: apply_dropout(
                docs_words_logits, p_drop=self.dropout, training=training))
    # perform batch normalization
    if self.batch_norm:
      docs_words_logits = self._batch_norm_layer(docs_words_logits,
                                                 training=training)
    # something wrong, using logits value for OneHotCategorical make the model
    # does not converge
    if self.distribution == 'onehot':
      docs_words_probs = tf.nn.softmax(docs_words_logits, axis=-1)
      docs_words_dist = self.distribution_layer(docs_words_probs,
                                                training=training)
    else:
      docs_words_dist = self.distribution_layer(docs_words_logits,
                                                training=training)
    return docs_words_dist

  def call(self,
           inputs: Union[TensorTypes, List[TensorTypes]],
           training: Optional[bool] = None,
           mask: Optional[TensorTypes] = None,
           sample_shape: List[int] = (),
           **kwargs) -> Tuple[Distribution, Distribution]:
    r"""
    Return:
      The documents words distribution `p(x|z)`
      The topics posterior distribution `q(z|x)`
    """
    docs_topics_dist = self.encode(inputs,
                                   training=training,
                                   sample_shape=sample_shape,
                                   mask=mask)
    docs_words_dist = self.decode(docs_topics_dist, training=training)
    return docs_words_dist, docs_topics_dist

  def get_topics_string(self,
                        vocabulary: Dict[int, str],
                        n_topics: int = 10,
                        n_words: int = 10,
                        show_word_prob: bool = False) -> List[str]:
    """Present the topics in readable format

    Parameters
    ----------
    vocabulary : Dict[int, str]
        mapping from features indices to word in dictionary
    n_topics : int, optional
        number of topics with the highest weights (alpha) be printed, by default 10
    n_words : int, optional
        number of words to be printed for each topic, by default 10
    show_word_prob : bool, optional
        show the probability value for each word, by default False

    Returns
    -------
    List[str]
        List of topics string
    """
    n_topics = min(int(n_topics), self.n_topics)
    n_words = min(int(n_words), self.n_words)
    topics = self.topics_words_probs
    alpha = np.squeeze(self.topics_concentration, axis=0)
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

  def __str__(self):
    if hasattr(self, 'input_shape'):
      shape = self.input_shape[1:]
    else:
      shape = None
    return (f"<LatentDirichletDecoder inputs:{shape} "
            f"step:{int(self.step.numpy())} warmup:{self.warmup} "
            f"posterior:{self.posterior} distribution:{self.distribution} "
            f"topics:{self.n_topics} vocab:{self.n_words} "
            f"dropout:({self.dropout_strategy}{self.dropout}) "
            f"batchnorm:{self.batch_norm}>")


# ===========================================================================
# Main class
# ===========================================================================
class amortizedLDA(betaVAE):
  """Amortized Latent Dirichlet Autoencoding"""

  def __init__(
      self,
      ldd: LatentDirichletDecoder,
      encoder: LayerCreator = NetworkConfig([300, 300, 300],
                                            flatten_inputs=True,
                                            name="Encoder"),
      decoder: LayerCreator = 'identity',
      latents: LayerCreator = 'identity',
      warmup: Optional[int] = None,
      beta: float = 1.0,
      **kwargs,
  ):
    if warmup is not None:
      ldd.warmup = int(warmup)
    super().__init__(latents=latents,
                     encoder=encoder,
                     decoder=decoder,
                     observation=ldd,
                     beta=beta,
                     analytic=True,
                     **kwargs)
    ldd.step = self.step
    self._ldd_layer = ldd

  @property
  def ldd(self) -> LatentDirichletDecoder:
    return self._ldd_layer

  def elbo_components(self, inputs, training=None, mask=None, **kwargs):
    llk, kl = super().elbo_components(inputs=inputs,
                                      mask=mask,
                                      training=training)
    px_z, qz_x = self.last_outputs
    p_topics = tf.nest.flatten(px_z)[-1]
    kl[f'kl_{self.ldd.name}'] = p_topics.KL_divergence(analytic=self.analytic)
    return llk, kl

  ######## Utilities methods
  def predict_topics(
      self,
      inputs: Union[TensorTypes, List[TensorTypes], DatasetV2],
      hard_topics: bool = False,
      verbose: bool = False
  ) -> Union[Dirichlet, MultivariateNormalDiag, tf.Tensor]:
    if not isinstance(inputs, DatasetV2):
      inputs = [inputs]
    if verbose:
      inputs = tqdm(inputs, desc="Predicting topics")
    concentration = []
    loc, scale_diag = [], []
    for x in inputs:
      (_, qZ_X), _ = self(x, training=False)
      if self.ldd.posterior == 'dirichlet':
        concentration.append(qZ_X.concentration)
      elif self.ldd.posterior == 'gaussian':
        loc.append(qZ_X.loc)
        scale_diag.append(qZ_X.scale._diag)
    # final distribution
    if self.ldd.posterior == 'dirichlet':
      concentration = tf.concat(concentration, axis=0)
      dist = Dirichlet(concentration=concentration, name="TopicsDistribution")
      if hard_topics:
        return tf.argmax(dist.mean(), axis=-1)
      return dist
    elif self.ldd.posterior == 'gaussian':
      loc = tf.concat(loc, axis=0)
      scale_diag = tf.concat(scale_diag, axis=0)
      dist = MultivariateNormalDiag(loc=loc,
                                    scale_diag=scale_diag,
                                    name="TopicsDistribution")
      if hard_topics:
        probs = tf.nn.softmax(dist.mean(), axis=-1)
        return tf.argmax(probs, axis=-1)
      return dist

  def get_topics_string(self,
                        vocabulary: Dict[int, str],
                        n_topics: int = 10,
                        n_words: int = 10,
                        show_word_prob: bool = False) -> List[str]:
    r""" Print most relevant topics and its most representative words
    distribution """
    return self.ldd.get_topics_string(vocabulary=vocabulary,
                                      n_words=n_words,
                                      n_topics=n_topics,
                                      show_word_prob=show_word_prob)


class auxiliaryLDA(amortizedLDA):
  """Amortized LDA as auxiliary loss"""

  def __init__(
      self,
      ldd: LatentDirichletDecoder,
      encoder: LayerCreator = NetworkConfig([300, 300, 300],
                                            flatten_inputs=True,
                                            name="Encoder"),
      decoder: LayerCreator = NetworkConfig([300, 300, 300],
                                            flatten_inputs=True,
                                            name="Decoder"),
      latents: LayerCreator = RVmeta(10, 'mvndiag', True, name='Latents'),
      warmup: Optional[int] = None,
      beta: float = 1.0,
      alpha: float = 1.0,
      **kwargs,
  ):
    ...


# ===========================================================================
# Two-stage VAE
# ===========================================================================
class nonlinearLDA(amortizedLDA):
  r""" Two-stage latent dirichlet allocation """

  def __init__(
      self,
      ldd: LatentDirichletDecoder,
      encoder: LayerCreator = NetworkConfig(flatten_inputs=True,
                                            name="Encoder"),
      decoder: LayerCreator = NetworkConfig(flatten_inputs=True,
                                            name="Decoder"),
      latents: LayerCreator = RVmeta(10,
                                     posterior='mvndiag',
                                     projection=True,
                                     name="Latents"),
      **kwargs,
  ):
    super().__init__(ldd=ldd,
                     latents=latents,
                     encoder=encoder,
                     decoder=decoder,
                     **kwargs)
    # this layer won't train the KL divergence or the encoder
    self.encoder.trainable = False
    for l in self.latent_layers:
      l.trainable = False

  def _elbo(
      self,
      inputs: Union[Tensor, List[Tensor]],
      pX_Z: Union[Distribution, List[Distribution]],
      qZ_X: Union[Distribution, List[Distribution]],
      **kwargs,
  ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    llk, kl = super(betaVAE, self)._elbo(inputs, pX_Z, qZ_X, **kwargs)
    # stop all the kl gradients just for sure
    kl = {k: tf.stop_gradient(v) for k, v in kl.items()}
    # add the topics KL
    topics = pX_Z[-1]
    kl_topics = topics.KL_divergence(
        analytic=kwargs.get('analytic', self.analytic))
    kl[f'kl_{self.ldd.name}'] = kl_topics
    return llk, kl


# ===========================================================================
# VDA
# ===========================================================================
class ALDA(VariationalModel):
  """Amortized Latent Dirichlet Allocation"""

  def __init__(
      self,
      ldd: LatentDirichletDecoder,
      encoder: Layer,
      warmup: Optional[int] = None,
      beta: float = 1.0,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.ldd = ldd
    ldd.step = self.step
    if warmup is not None:
      ldd.warmup = int(warmup)
    self.encoder = encoder
    self.beta = beta

  def encode(self, inputs, training=None, **kwargs):
    e = self.encoder(inputs, training=training)
    return self.ldd.encode(e, training=training, sample_shape=self.sample_shape)

  def decode(self, latents, training=None, **kwargs):
    return self.ldd.decode(latents, training=training)

  def call(self, inputs, training=None, **kwargs):
    e = self.encoder(inputs, training=training)
    px, qz = self.ldd(e, training=training, sample_shape=self.sample_shape)
    return px, qz

  def elbo_components(self, inputs, training, **kwargs):
    px, qz = self(inputs, training=training)
    llk = px.log_prob(inputs)
    kl = self.beta * qz.KL_divergence(analytic=self.analytic)
    return dict(llk=llk), dict(kl=kl)
