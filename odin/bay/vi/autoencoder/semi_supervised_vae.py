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
from odin.bay.layers.distribution_util_layers import ConditionalTensorLayer
from odin.bay.random_variable import RandomVariable as RV
from odin.bay.vi.autoencoder.beta_vae import BetaVAE
from odin.bay.vi.autoencoder.networks import FactorDiscriminator, ImageNet
from odin.bay.vi.utils import marginalize_categorical_labels
from odin.networks.conditional_embedding import get_conditional_embedding

__all__ = ['ConditionalM2VAE']


def _batch_size(x):
  batch_size = x.shape[0]
  if batch_size is None:
    batch_size = tf.shape(x)[0]
  return batch_size


class ConditionalM2VAE(BetaVAE):
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
    alpha : a Scalar. The weight of discriminative objective added to the
      labelled data objective. In the paper, it is recommended:
      `alpha = 0.1 * (n_total_samples / n_labelled_samples)`

  Example:
  ```
  from odin.fuel import MNIST
  from odin.bay.vi.autoencoder import ConditionalM2VAE

  ds = MNIST()
  train = ds.create_dataset(partition='train', inc_labels=0.1)
  test = ds.create_dataset(partition='test', inc_labels=True)

  encoder, decoder = create_image_autoencoder(image_shape=(28, 28, 1),
                                              input_shape=(28, 28, 2),
                                              center0=True,
                                              latent_shape=20)

  vae = ConditionalM2VAE(encoder=encoder,
                         decoder=decoder,
                         conditional_embedding='embed',
                         alpha=0.1 * 10)
  vae.fit(train, compile_graph=True, epochs=-1, max_iter=8000, sample_shape=5)
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
               alpha=0.1,
               **kwargs):
    assert isinstance(labels, RV), "labels must be instance of %s" % str(RV)
    self.n_labels = int(np.prod(labels.event_shape))
    super().__init__(latents=latents, outputs=outputs, **kwargs)
    # the distribution of labels
    self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype, name="alpha")
    self.labels = labels.create_posterior()
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

  def prepare_inputs(self, inputs, mask):
    n_labels = self.n_labels
    n_outputs = len(self.output_layers)
    inputs = tf.nest.flatten(inputs)
    batch_size = _batch_size(inputs[0])
    # no labels provided:
    if len(inputs) == n_outputs:
      X = inputs
      y = None
      mask = tf.cast(tf.zeros(shape=(batch_size, 1)), tf.bool)
    else:
      X = inputs[:-1]
      y = inputs[-1]
      if mask is None:
        mask = tf.cast(tf.ones(shape=(batch_size, 1)), tf.bool)
    # split into unlabelled and labelled data
    mask = tf.reshape(mask, (-1,))
    X_unlabelled = [tf.boolean_mask(i, tf.logical_not(mask), axis=0) for i in X]
    X_labelled = [tf.boolean_mask(i, mask, axis=0) for i in X]
    # for unlabelled data
    y_unlabelled = marginalize_categorical_labels(
        batch_size=_batch_size(X_unlabelled[0]),
        num_classes=n_labels,
        dtype=inputs[0].dtype,
    )
    X_unlabelled = [tf.repeat(i, n_labels, axis=0) for i in X_unlabelled]
    # for labelled data
    if y is not None:
      y_labelled = tf.boolean_mask(y, mask, axis=0)
      y = tf.concat([y_unlabelled, y_labelled], axis=0)
      mask = tf.cast(
          tf.concat(
              [
                  tf.zeros(shape=(_batch_size(y_unlabelled), 1)),
                  tf.ones(shape=(_batch_size(y_labelled), 1))
              ],
              axis=0,
          ), tf.bool)
    # for only unlabelled data
    else:
      y = y_unlabelled
      mask = tf.repeat(mask, n_labels, axis=0)
    X = [
        tf.concat([unlab, lab], axis=0)
        for unlab, lab in zip(X_unlabelled, X_labelled)
    ]
    return X, y, mask

  def encode(self, inputs, training=None, mask=None, sample_shape=(), **kwargs):
    X, y, mask = self.prepare_inputs(inputs, mask=mask)
    # conditional embedding y
    y_embedded = self.embedder(y, training=training)
    X = [tf.concat([i, y_embedded], axis=-1) for i in X]
    if len(self.output_layers) == 1:
      X = X[0]
    # encode normally
    qZ_X = super().encode(X,
                          training=training,
                          mask=mask,
                          sample_shape=sample_shape,
                          **kwargs)
    qZ_X = [
        ConditionalTensorLayer(sample_shape=sample_shape)([qZ_X, y])
        for q in tf.nest.flatten(qZ_X)
    ]
    # remember to store the new mask
    for q in qZ_X:
      q._keras_mask = mask
    return qZ_X[0] if len(qZ_X) == 1 else tuple(qZ_X)

  def _elbo(self, X, pX_Z, qZ_X, analytic, reverse, sample_shape, mask,
            training):
    inputs = X[:len(self.output_layers)]
    if mask is None:
      if len(X) == len(self.output_layers):  # no labelled
        X_unlabelled = inputs
      else:  # all data is labelled
        X_unlabelled = [tf.zeros(shape=(0,) + i.shape[1:]) for i in inputs]
    else:
      m = tf.logical_not(tf.reshape(mask, (-1,)))
      X_unlabelled = [tf.boolean_mask(i, m, axis=0) for i in inputs]
    ## prepare inputs as usual
    X, y, mask = self.prepare_inputs(X, mask)
    ## Normal ELBO
    llk, div = super()._elbo(X,
                             pX_Z,
                             qZ_X,
                             analytic,
                             reverse,
                             sample_shape,
                             mask=mask,
                             training=training)
    mask = tf.reshape(mask, (-1,))
    ### for unlablled data
    mask_unlablled = tf.logical_not(mask)
    pY_X = self.classify(X_unlabelled)
    probs = pY_X.probs_parameter()
    # log-likehood
    llk_unlabelled = {}
    for name, lk in llk.items():
      lk = tf.transpose(lk)
      lk = tf.boolean_mask(lk, mask_unlablled, axis=0)
      lk = tf.transpose(tf.reshape(lk, (self.n_labels, tf.shape(probs)[0], -1)))
      lk = tf.reduce_sum(lk * probs, axis=-1)
      llk_unlabelled[name + '_unlabelled'] = lk
    # kl-divergence
    div_unlabelled = {}
    for name, dv in div.items():
      dv = tf.transpose(dv)
      dv = tf.boolean_mask(dv, mask_unlablled, axis=0)
      dv = tf.transpose(tf.reshape(dv, (self.n_labels, tf.shape(probs)[0], -1)))
      dv = tf.reduce_sum(dv * probs, axis=-1)
      div_unlabelled[name + '_unlablled'] = dv
    div_unlabelled['kl_classifier'] = kl_divergence(pY_X,
                                                    self.labels.prior,
                                                    analytic=True)
    ### for labelled data, add the discriminative objective
    X_labelled = [tf.boolean_mask(i, mask, axis=0) for i in X]
    # log-likehood
    llk_labelled = {
        name + '_labelled':
        tf.transpose(tf.boolean_mask(tf.transpose(lk), mask, axis=0))
        for name, lk in llk.items()
    }
    # add the classification (discrimination) loss
    y_labelled = tf.boolean_mask(y, mask, axis=0)
    pY_X = self.classify(X_labelled)
    llk_labelled['llk_classifier'] = self.alpha * pY_X.log_prob(y_labelled)
    # kl-divergence
    div_labelled = {
        name + '_labelled':
        tf.transpose(tf.boolean_mask(tf.transpose(dv), mask, axis=0))
        for name, dv in div.items()
    }
    ### merge everything
    llk = {
        k: tf.reduce_mean(v)
        for k, v in dict(**llk_unlabelled, **llk_labelled).items()
    }
    div = {
        k: tf.reduce_mean(v)
        for k, v in dict(**div_unlabelled, **div_labelled).items()
    }
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
