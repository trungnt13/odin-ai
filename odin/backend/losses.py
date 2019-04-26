import numpy as np
import tensorflow as tf

from odin.autoconfig import EPS
from odin.utils import is_number
from odin.backend.role import (return_roles, DifferentialLoss, add_roles)
from odin.backend.tensor import is_tensor, to_nonzeros, dimshuffle

# ===========================================================================
# Similarity measurement
# ===========================================================================
@return_roles(roles=DifferentialLoss)
def contrastive_loss(y_true, y_pred, margin=1, name=None):
  """
  Reference
  ---------
  [1] http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
  """
  # ====== tensorflow Tensor ====== #
  if is_tensor(y_true) and is_tensor(y_pred):
    with tf.name_scope(name, 'contrastive_loss', [y_true, y_pred]):
      loss = tf.reduce_mean(
          y_true * tf.square(y_pred) +
          (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))
  # ====== ndarray or similar ====== #
  else:
    loss = np.mean(y_true * np.square(y_pred) +
                  (1 - y_true) * np.square(np.maximum(margin - y_pred, 0)))
  return loss

@return_roles(roles=DifferentialLoss)
def triplet_loss(y_true, y_pred):
  pass

def contrastive_loss_andre(left_feature, right_feature, label, margin):
  """
  Compute the contrastive loss as in
  https://gitlab.idiap.ch/biometric/xfacereclib.cnn/blob/master/xfacereclib/cnn/scripts/experiment.py#L156
  With Y = [-1 +1] --> [POSITIVE_PAIR NEGATIVE_PAIR]
  L = log( m + exp( Y * d^2)) / N
  **Parameters**
   left_feature: First element of the pair
   right_feature: Second element of the pair
   label: Label of the pair (0 or 1)
   margin: Contrastive margin
  **Returns**
   Return the loss operation
  """

  with tf.name_scope("contrastive_loss_andre"):
    label = tf.to_float(label)
    d = compute_euclidean_distance(left_feature, right_feature)

    loss = tf.log(tf.exp(tf.mul(label, d)))
    loss = tf.reduce_mean(loss)

    # Within class part
    genuine_factor = tf.mul(label - 1, 0.5)
    within_class = tf.reduce_mean(tf.log(tf.exp(tf.mul(genuine_factor, d))))

    # Between class part
    impostor_factor = tf.mul(label + 1, 0.5)
    between_class = tf.reduce_mean(tf.log(tf.exp(tf.mul(impostor_factor, d))))

    # first_part = tf.mul(one - label, tf.square(d))  # (Y-1)*(d^2)
    return loss, between_class, within_class

@return_roles(DifferentialLoss)
def cosine_similarity(y_true, y_pred, weights=1.0,
                      unit_norm=True, one_vs_all=True,
                      name=None):
  """
  Parameters
  ----------
  y_true : {Tensor, ndarray}
      enrollment vectors, one sample per row
      (i.e. shape=(nb_samples, nb_features))
  y_pred : {Tensor, ndarray}
      test vectors, one sample per row
      (i.e. shape=(nb_samples, nb_features))
  unit_norm : bool (default: True)
      if True, normalize length of each vector to 1
  one_vs_all : bool (default: True)
      if True, calculate the similarity of one to all other
      samples, otherwise, it is `one_vs_one`

  Return
  ------
  scores : score matrix,
      comparing all models against all tests

  """
  # ====== tensorflow Tensor ====== #
  if is_tensor(y_true) and is_tensor(y_pred):
    with tf.name_scope(name, "cosine_similarity", (y_true, y_pred, weights)):
      y_pred.shape.assert_is_compatible_with(y_true.shape)
      if unit_norm:
        y_true /= to_nonzeros(tf.linalg.norm(y_true, ord=2, axis=-1, keep_dims=True), 1.)
        y_pred /= to_nonzeros(tf.linalg.norm(y_pred, ord=2, axis=-1, keep_dims=True), 1.)
      if one_vs_all:
        scores = tf.matmul(tf.transpose(y_true), y_pred)
      else:
        scores = 1 - tf.reduce_sum(tf.multiply(y_true, y_pred),
                                   axis=(-1,), keep_dims=True)
  # ====== ndarray or similar ====== #
  else:
    assert y_true.shape == y_pred.shape
    if unit_norm:
      y_true /= to_nonzeros(np.linalg.norm(y_true, ord=2, axis=-1, keepdims=True), 1.)
      y_pred /= to_nonzeros(np.linalg.norm(y_pred, ord=2, axis=-1, keepdims=True), 1.)
    if one_vs_all:
      scores = np.dot(y_true.T, y_pred)
    else:
      scores = np.sum(y_true * y_pred,
                      axis=(-1,), keepdims=True)
  return scores

# ===========================================================================
# Cross-entropy variations
# ===========================================================================
@return_roles(DifferentialLoss)
def bayes_crossentropy(y_true, y_pred, nb_classes=None, reduction=tf.reduce_mean,
                       name=None):
  with tf.name_scope(name, "bayes_crossentropy", [y_true, y_pred]):
    y_pred_shape = y_pred.shape
    if y_pred_shape.ndims == 1 or y_pred_shape[-1].value == 1:
      if y_pred_shape.ndims == 1:
        y_pred = tf.expand_dims(y_pred, -1)
      y_pred0 = 1. - y_pred
      y_pred = tf.concat([y_pred0, y_pred], axis=-1)
    # get number of classes
    if y_true.shape.ndims == 1:
      if nb_classes is None:
        raise Exception('y_pred and y_true must be one_hot encoded, '
                        'otherwise you have to provide nb_classes.')
      y_true = tf.one_hot(y_true, depth=nb_classes)
    elif nb_classes is None:
      nb_classes = y_true.shape[1].value
    # avoid numerical instability with _EPSILON clipping
    y_pred = tf.clip_by_value(y_pred, EPS, 1.0 - EPS)
    # ====== check distribution ====== #
    distribution = tf.reduce_sum(y_true, axis=0)
    # probability distribution of each class
    prob_distribution = dimshuffle(distribution / tf.reduce_sum(distribution),
                                   ('x', 0))
    # we need to clip the prior probability distribution also
    prob_distribution = tf.clip_by_value(prob_distribution, EPS, 1.0 - EPS)
    # ====== init confusion info loss ====== #
    # weighted by y_true
    loss = y_true * tf.log(y_pred)
    loss = - 1 / nb_classes * tf.reduce_sum(loss / prob_distribution, axis=1)
    return reduction(loss)

@return_roles(DifferentialLoss)
def bayes_binary_crossentropy(y_true, y_pred):
  y_pred = tf.concat([1 - y_pred, y_pred], axis=-1)
  y_true = tf.one_hot(tf.cast(y_true, 'int32'), depth=2)
  return bayes_crossentropy(y_pred, y_true, nb_classes=2)


# ===========================================================================
# Variational
# ===========================================================================
def jacobian_regularize(hidden, params):
  """ Computes the jacobian of the hidden layer with respect to
  the input, reshapes are necessary for broadcasting the
  element-wise product on the right axis
  """
  hidden = hidden * (1 - hidden)
  L = tf.expand_dims(hidden, 1) * tf.expand_dims(params, 0)
  # Compute the jacobian and average over the number of samples/minibatch
  L = tf.reduce_sum(tf.pow(L, 2)) / hidden.shape[0]
  return tf.reduce_mean(L)


def correntropy_regularize(x, sigma=1.):
  """
  Note
  ----
  origin implementation from seya:
  https://github.com/EderSantana/seya/blob/master/seya/regularizers.py
  Copyright (c) EderSantana
  """
  return -tf.reduce_sum(tf.reduce_mean(tf.exp(x**2 / sigma), axis=0)) / tf.sqrt(2 * np.pi * sigma)


def kl_gaussian(mu, logsigma,
                prior_mu=0., prior_logsigma=0.):
  """ KL-divergence between two gaussians.
  Useful for Variational AutoEncoders. Use this as an activation regularizer

  For taking kl_gaussian as variational regularization, you can take mean of
  the return matrix

  Parameters:
  -----------
  mean, logsigma: parameters of the input distributions
  prior_mean, prior_logsigma: paramaters of the desired distribution (note the
      log on logsigma)


  Return
  ------
  matrix: (n_samples, n_features)

  Note
  ----
  origin implementation from:
  https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
  Copyright (c) Philip Bachman
  """
  if is_number(prior_mu):
    prior_mu = tf.convert_to_tensor(prior_mu, name='prior_mu',
        dtype=mu.dtype.base_dtype)
  if is_number(prior_logsigma):
    prior_logsigma = tf.convert_to_tensor(
        prior_logsigma, name='prior_logsigma',
        dtype=logsigma.dtype.base_dtype)
  gauss_klds = 0.5 * (2 * (prior_logsigma - logsigma) +
          (tf.exp(2 * logsigma) / tf.exp(2 * prior_logsigma)) +
          (tf.pow((mu - prior_mu), 2.0) / tf.exp(2 * prior_logsigma)) - 1.0)
  return gauss_klds
