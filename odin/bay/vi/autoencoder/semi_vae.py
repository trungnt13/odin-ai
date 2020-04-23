from __future__ import absolute_import, division, print_function

import os
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow_probability.python.distributions import OneHotCategorical

from odin import backend as bk
from odin.backend.keras_helpers import layer2text
from odin.bay.helpers import kl_divergence
from odin.bay.random_variable import RandomVariable as RV
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.networks import FactorDiscriminator, ImageNet
from odin.bay.vi.utils import marginalize_categorical_labels
from odin.networks.conditional_embedding import get_conditional_embedding

__all__ = ['M2VAE']


class M2VAE(BetaVAE):
  r""" Implementation of M2 model (Kingma et al. 2014). The default
  configuration of this layer is optimized for MNIST.

  ```
  q(z|y,x) = N(z|f_mu(y,x),f_sig(x)))
  q(y|x) = Cat(y|pi)
  q(pi|x) = g(x)
  ```

  Arguments:
    conditional_embedding : {'repeat', 'embed', 'project'}. Strategy for
      concatenating one-hot encoded labels to inputs.
    sample_label_prior : a Boolean. If True, directly sample from known
      labels prior in case of unlablled data, otherwise, marginalize the
      labels.
    alpha : a Scalar. The weight of discriminative objective added to the
      labelled data objective. In the paper, it is recommended:
      `alpha = 0.1 * (n_total_samples / n_labelled_samples)`

  Example:
  ```
  from odin.fuel import MNIST
  from odin.bay.vi.autoencoder import M2VAE
  ds = MNIST()
  train = ds.create_dataset(partition='train', inc_labels=True)
  test = ds.create_dataset(partition='test', inc_labels=True)
  encoder, decoder = create_image_autoencoder(image_shape=(28, 28, 1),
                                              input_shape=(28, 28, 2),
                                              center0=True,
                                              latent_shape=20)
  vae = M2VAE(encoder=encoder, decoder=decoder, conditional_embedding='embed')
  vae.fit(train, compile_graph=False)
  ```

  Reference:
    Kingma, D.P., Rezende, D.J., Mohamed, S., Welling, M., 2014.
      "Semi-Supervised Learning with Deep Generative Models".
      arXiv:1406.5298 [cs, stat].
  """

  def __init__(self,
               latents=RV(10, 'diag', projection=True, name='Latent'),
               outputs=RV((28, 28, 1),
                          'bernoulli',
                          projection=False,
                          name='Image'),
               labels=RV(10, 'onehot', projection=False, name="Label"),
               classifier=dict(units=1000, n_hidden_layers=5),
               conditional_embedding="embed",
               sample_label_prior=False,
               alpha=0.1,
               **kwargs):
    assert isinstance(labels, RV), "labels must be instance of %s" % str(RV)
    self.n_labels = int(np.prod(labels.event_shape))
    super().__init__(latents=latents, outputs=outputs, **kwargs)
    # the distribution of labels
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name="alpha")
    self.labels = labels.create_posterior()
    self.sample_label_prior = bool(sample_label_prior)
    if self.labels.prior is None:
      self.labels.prior = OneHotCategorical(
          logits=np.log([1. / self.n_labels] * self.n_labels),
          name="LabelPrior",
      )
    # create the classifier
    if not isinstance(classifier, keras.layers.Layer):
      classifier = dict(classifier)
      classifier['n_outputs'] = self.n_labels
      if 'input_shape' not in classifier:
        input_shape = [i.event_shape for i in self.output_layers]
        if len(input_shape) == 1:
          input_shape = input_shape[0]
        classifier['input_shape'] = input_shape
      classifier = FactorDiscriminator(**classifier)
    self.classifier = classifier
    # resolve the conditional embedding
    embedder = get_conditional_embedding(conditional_embedding)
    self.embedder = embedder(num_classes=self.n_labels,
                             output_shape=outputs.event_shape)
    self.embedder.build(self.labels.event_shape)
    # validate the shape
    assert tuple(classifier.output_shape[1:]) == tuple(self.labels.event_shape), \
      "Classifier output shape is: %s but labels event shape is: %s" % \
        (classifier.output_shape[1:], self.labels.event_shape)
    # conditioned encoder
    s1 = tuple(self.output_layers[0].event_shape)
    s2 = tuple(self.embedder.embedding_shape[1:])
    s = s1[:-1] + (s1[-1] + s2[-1],)
    s3 = tuple(self.encoder.input_shape[1:])
    assert s == s3, \
      ("Encoder input shape is %s, must equal to the concatenation of "
        "inputs %s and labels %s" % (s3, s1, s2))
    # conditioned decoder
    s1 = self.latent_layers[0].event_shape
    s3 = self.decoder.input_shape[1:]
    assert s1[-1] + self.n_labels == s3[-1], \
      ("Decoder input shape is %s, must equal to the concatenation of "
       "latents %s and labels (%d,)" % (s3, s1, self.n_labels))

  def classify(self, X, proba=False):
    y = self.classifier(X)
    if proba:
      return tf.nn.softmax(y, axis=-1)
    return self.labels(y)

  def sample_labels(self, sample_shape=(), seed=1):
    return bk.atleast_2d(
        self.labels.prior.sample(sample_shape=sample_shape, seed=seed))

  def encode(self, inputs, training=None, sample_shape=(), **kwargs):
    inputs = tf.nest.flatten(inputs)
    # Given the label for semi-supervised learning
    if len(inputs) > len(self.output_layers):
      X = inputs[:-1]
      y = inputs[-1]
    # unlablled, marginalize by y, by sampling from y prior
    else:
      if self.sample_label_prior:
        y = self.labels.prior.sample(sample_shape=inputs[0].shape[0] *
                                     self.n_labels)
      else:
        y = marginalize_categorical_labels(batch_size=inputs[0].shape[0],
                                           num_classes=self.n_labels,
                                           dtype=self.dtype)
      X = [tf.repeat(i, self.n_labels, axis=0) for i in inputs]
    # conditional embedding y
    y_embedded = self.embedder(y, training=training)
    X = [tf.concat([i, y_embedded], axis=-1) for i in X]
    if len(self.output_layers) == 1:
      X = X[0]
    # encode normally
    qZ_X = super().encode(X,
                          training=training,
                          sample_shape=sample_shape,
                          **kwargs)
    return qZ_X, y

  def decode(self, latents, training=None, sample_shape=(), **kwargs):
    qZ_X, y = latents
    # again we need to repeat y to match qZ_X
    n_samples = tf.nest.flatten(sample_shape)
    if len(n_samples) > 0:
      for _ in range(len(n_samples)):
        y = tf.expand_dims(y, axis=0)
      for i, n in enumerate(n_samples):
        y = tf.repeat(y, n, axis=i)
    Z = tf.concat([qZ_X, y], axis=-1)
    return super().decode(latents=Z,
                          training=training,
                          sample_shape=sample_shape,
                          **kwargs)

  def call(self, inputs, training=None, sample_shape=(), return_labels=False):
    qZ_X = self.encode(inputs, training=training, sample_shape=sample_shape)
    pX_Z = self.decode(qZ_X, training=training, sample_shape=sample_shape)
    y = qZ_X[-1]
    qZ_X = qZ_X[:-1]
    if len(self.latent_layers) == 1:
      qZ_X = qZ_X[0]
    if return_labels:
      return pX_Z, qZ_X, y
    return pX_Z, qZ_X

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, sample_shape):
    ## check if data is unlablled and be marginalized along Y
    is_unlablled = False
    if X[0].shape[0] != pX_Z[0].batch_shape[-1]:
      is_unlablled = True
    ## Normal ELBO
    llk, div = super()._elbo(
        [tf.repeat(x, self.n_labels, axis=0) for x in X] if is_unlablled else X,
        pX_Z,
        qZ_X,
        analytic,
        reverse,
        sample_shape,
    )
    ## special case of unlablled data
    if is_unlablled:
      pY_X = self.classify(X[0])
      probs = pY_X.probs_parameter()
      new_llk = {}
      for name, x in llk.items():
        shape = tf.concat([x.shape[:-1], (-1, self.n_labels)], axis=0)
        x = tf.reshape(x, shape)
        x = tf.reduce_sum(x * probs, axis=-1)
        new_llk[name] = x
      new_div = {}
      for name, x in div.items():
        shape = tf.concat([x.shape[:-1], (-1, self.n_labels)], axis=0)
        x = tf.reshape(x, shape)
        x = tf.reduce_sum(x * probs, axis=-1)
        new_div[name] = x
      new_div['kl_label'] = kl_divergence(pY_X,
                                          self.labels.prior,
                                          analytic=True)
      llk = new_llk
      div = new_div
    ## labelled data, add the discriminative objective
    elif len(X) > len(self.output_layers):
      y = X[-1]
      pY_X = self.classify(X[0])
      llk['llk_labels'] = self.alpha * pY_X.log_prob(y)
    return llk, div

  def __str__(self):
    text = super().__str__()
    text += "\n Classifier:\n  "
    text += "\n  ".join(str(self.classifier).split('\n'))
    text += "\n Conditional Embedder:\n  "
    text += "\n  ".join(layer2text(self.embedder).split('\n'))
    return text


class StructuredSemiVAE(BetaVAE):
  r"""
  Reference:
    Siddharth, N., Paige, B., et al., 2017. "Learning Disentangled
      Representations with Semi-Supervised Deep Generative Models".
      arXiv:1706.00400 [cs, stat].
  """
