from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import torch
from scipy.special import logsumexp

from odin.backend import tensor as ts
from odin.backend.tensor import _normalize_axis


# ===========================================================================
# Linear Algebra
# ===========================================================================
def matmul(x, y):
  """ Matrix product of two tensors
  This function support broadcasting.

  Example:

    (2, 3).(4, 3, 5) => (4, 2, 5)
    (2, 3, 4).(4, 5) => (2, 3, 5)
    (5, 3, 4).(5, 4, 6) => (5, 3, 6)

  """
  if tf.is_tensor(x) or tf.is_tensor(y):
    return tf.matmul(x, y)
  if torch.is_tensor(x) or torch.is_tensor(y):
    return torch.matmul(x, y)
  return np.matmul(x, y)


# ===========================================================================
# Normalization
# ===========================================================================
def length_norm(x, axis=-1, epsilon=1e-12, ord=2):
  """ L2-normalization (or vector unit length normalization)

  Parameters
  ----------
  x : array
  axis : int
  ord : int
    order of norm (1 for L1-norm, 2 for Frobenius or Euclidean)
  """
  ord = int(ord)
  if ord not in (1, 2):
    raise ValueError(
        "only support `ord`: 1 for L1-norm; 2 for Frobenius or Euclidean")
  if ord == 2:
    x_norm = tf.sqrt(
        tf.maximum(tf.reduce_sum(x**2, axis=axis, keepdims=True), epsilon))
  else:
    x_norm = tf.maximum(tf.reduce_sum(tf.abs(x), axis=axis, keepdims=True),
                        epsilon)
  return x / x_norm


def calc_white_mat(X):
  """ calculates the whitening transformation for cov matrix X
  """
  return tf.linalg.cholesky(tf.linalg.inv(X))


def log_norm(x, axis=1, scale_factor=10000, eps=1e-8):
  """ Seurat log-normalize
  y = log(X / (sum(X, axis) + epsilon) * scale_factor)

  where `log` is natural logarithm
  """
  eps = tf.cast(eps, x.dtype)
  return tf.math.log1p(x / (tf.reduce_sum(x, axis=axis, keepdims=True) + eps) *
                       scale_factor)


def delog_norm(x, x_sum=1, scale_factor=10000):
  """ This perform de-log normalization of `log_norm` values
  if `x_sum` is not given (i.e. default value 1), then all the
  """
  return (tf.exp(x) - 1) / scale_factor * (x_sum + EPS)


# ===========================================================================
# Activation function
# ===========================================================================
def softmax(x, axis=-1):
  """ `f(x) = exp(x_i) / sum_j(exp(x_j))` """
  if tf.is_tensor(x):
    return tf.nn.softmax(x, axis=axis)
  if torch.is_tensor(x):
    return torch.nn.functional.softmax(x, dim=axis)
  return tf.nn.softmax(x, axis=axis).numpy()


def softmin(x, axis=None):
  """ `f(x) = exp(-x_i) / sum_j(exp(-x_j))` """
  if torch.is_tensor(x):
    return torch.softmin(x, dim=axis)
  return softmax(-x, axis=axis)


def relu(x):
  """ `f(x) = max(x, 0)` """
  if tf.is_tensor(x):
    return tf.nn.relu(x)
  if torch.is_tensor(x):
    return torch.relu(x)
  return np.max(x, 0)


def selu(x):
  """ `f(x) = scale * [max(0, x) + min(0, alpha*(exp(x)-1))]`
  where:
    scale = ~1.0507
    alpha = ~1.6733
  chose by solving Eq.(4) and Eq.(5), with the fixed point
  `(mean, variance) = (0, 1)`, which is typical for activation normalization.

  Reference
  ---------
  [1] Klambauer, G., Unterthiner, T., Mayr, A., Hochreiter, S., 2017.
      Self-Normalizing Neural Networks. arXiv:1706.02515 [cs, stat].

  """
  if tf.is_tensor(x):
    return tf.nn.selu(x)
  if torch.is_tensor(x):
    return torch.nn.functional.selu(x)
  scale = 1.0507009873554804934193349852946
  alpha = 1.6732632423543772848170429916717
  return scale * (np.maximum(x, 0) + np.minimum(alpha * (np.exp(x) - 1), 0))


def tanh(x):
  """ `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))` """
  if tf.is_tensor(x):
    return tf.math.tanh(x)
  if torch.is_tensor(x):
    return torch.tanh(x)
  return np.tanh(x)


def softsign(x):
  """ `f(x) = x / (1 + |x|)`"""
  if tf.is_tensor(x):
    return tf.math.softsign(x)
  if torch.is_tensor(x):
    return torch.nn.functional.softsign(x)
  return x / (1 + np.abs(x))


def softplus(x, beta=1, threshold=20):
  """ `f(x) = 1/beta * log(exp(beta * x) + 1)`
  threshold : values above this revert to a linear function
  """
  if tf.is_tensor(x):
    mask = (x * beta) > threshold
    beta = tf.cast(beta, dtype=x.dtype)
    return tf.where(mask, x, 1 / beta * tf.nn.softplus(x * beta))
  if torch.is_tensor(x):
    return torch.nn.functional.softplus(x, beta=beta, threshold=threshold)
  return torch.nn.functional.softplus(torch.from_numpy(x),
                                      beta=beta,
                                      threshold=threshold).numpy()


def sigmoid(x):
  if tf.is_tensor(x):
    return tf.math.sigmoid(x)
  if torch.is_tensor(x):
    return torch.sigmoid(x)
  return 1 / (1 + np.exp(-x))


def mish(x, beta=1, threshold=20):
  """ Mish: A Self Regularized Non-Monotonic Neural Activation Function
  `f(x) = x * tanh(softplus(x))`

  Reference
  ---------
  [1] Misra, D., 2019. Mish: A Self Regularized Non-Monotonic Neural Activation
      Function. arXiv:1908.08681 [cs, stat].

  """
  return x * tanh(softplus(x, beta=beta, threshold=threshold))


def swish(x):
  """ Swish: smooth, non-monotonic function
  `f(x) = x * sigmoid(x)`
  """
  return x * sigmoid(x)


# ===========================================================================
# Math
# ===========================================================================
def sqrt(x):
  if tf.is_tensor(x):
    return tf.math.sqrt(x)
  if torch.is_tensor(x):
    return torch.sqrt(x)
  return np.sqrt(x)


def power(x, exponent):
  if tf.is_tensor(x):
    return tf.math.pow(x, exponent)
  if torch.is_tensor(x):
    return x.pow(exponent)
  return np.power(x, exponent)


def square(x):
  if tf.is_tensor(x):
    return tf.math.square(x)
  if torch.is_tensor(x):
    return torch.mul(x, x)
  return np.square(x)


def renorm_rms(X, axis=1, target_rms=1.0):
  """ Scales the data such that RMS of the features dimension is 1.0
  scale = sqrt(x^t x / (D * target_rms^2)).

  NOTE
  ----
  by defaults, assume the features dimension is `1`
  """
  D = sqrt(X.shape[axis])
  D = ts.cast(D, X.dtype)
  l2norm = sqrt(ts.reduce_sum(X**2, axis=axis, keepdims=True))
  X_rms = l2norm / D
  X_rms = ts.where(ts.equal(X_rms, 0.),
                   x=ts.ones_like(X_rms, dtype=X_rms.dtype),
                   y=X_rms)
  return target_rms * X / X_rms


# ===========================================================================
# Statistics and reduction
# ===========================================================================
def _torch_axis(x, axis):
  if axis is None:
    axis = list(range(x.ndim))
  return axis


def moments(x, axis=None, keepdims=False):
  """ Calculates the mean and variance of `x`.

  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.
  """
  if tf.is_tensor(x):
    mean, variance = tf.nn.moments(x, axes=axis, keepdims=keepdims)
  elif torch.is_tensor(x):
    mean = reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = (x - mean)**2
    variance = reduce_mean(devs_squared, axis=axis, keepdims=keepdims)
    if not keepdims:
      mean = mean.squeeze(axis)
  else:
    mean = np.mean(x, axis=axis, keepdims=keepdims)
    variance = np.var(x, axis=axis, keepdims=keepdims)
  return mean, variance


def reduce_var(x, axis=None, keepdims=False, mean=None):
  """ Calculate the variance of `x` along given `axis`
  if `mean` is given,
  """
  if isinstance(x, np.ndarray):
    return np.var(x, axis=axis, keepdims=keepdims)
  ndim = x.ndim
  axis = _normalize_axis(axis, ndim)
  m = reduce_mean(x, axis=axis, keepdims=True) if mean is None else mean
  devs_squared = (x - m)**2
  return reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
  return sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_min(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_min(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.min(dim=_torch_axis(x, axis), keepdim=keepdims)[0]
  return np.min(x, axis=axis, keepdims=keepdims)


def reduce_max(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_max(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.max(dim=_torch_axis(x, axis), keepdim=keepdims)[0]
  return np.max(x, axis=axis, keepdims=keepdims)


def reduce_mean(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.mean(dim=_torch_axis(x, axis), keepdim=keepdims)
  return np.mean(x, axis=axis, keepdims=keepdims)


def reduce_sum(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.sum(dim=_torch_axis(x, axis), keepdim=keepdims)
  return np.sum(x, axis=axis, keepdims=keepdims)


def reduce_prod(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_prod(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.prod(dim=_torch_axis(x, axis), keepdim=keepdims)
  return np.prod(x, axis=axis, keepdims=keepdims)


def reduce_all(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_all(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.bool().all(dim=_torch_axis(x, axis), keepdim=keepdims)
  return np.all(x, axis=axis, keepdims=keepdims)


def reduce_any(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_any(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.bool().any(dim=_torch_axis(x, axis), keepdim=keepdims)
  return np.any(x, axis=axis, keepdims=keepdims)


def reduce_logsumexp(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_logsumexp(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.logsumexp(dim=_torch_axis(x, axis), keepdim=keepdims)
  return logsumexp(x, axis=axis, keepdims=keepdims)


def reduce_logexp(x, reduction_function=tf.reduce_mean, axis=None, name=None):
  """ log-reduction-exp over axis to avoid overflow and underflow

  Parameters
  ----------
  `x` : [nb_sample, feat_dim]
  `axis` should be features dimension
  """
  with tf.name_scope(name, "logreduceexp"):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    y = tf.log(reduction_function(tf.exp(x - x_max), axis=axis,
                                  keepdims=True)) + x_max
    return tf.squeeze(y)


def cumsum(x, axis):
  if tf.is_tensor(x):
    return tf.math.cumsum(x, axis=axis)
  if torch.is_tensor(x):
    return torch.cumsum(x, dim=_torch_axis(x, axis))
  return np.cumsum(x, axis=axis)


# ===========================================================================
# Conversion
# ===========================================================================
def to_llh(x, name=None):
  ''' Convert a matrix of probabilities into log-likelihood
  :math:`LLH = log(prob(data|target))`
  '''
  with tf.name_scope(name, "log_likelihood", [x]):
    x /= tf.reduce_sum(x, axis=-1, keepdims=True)
    x = tf.clip_by_value(x, EPS, 1 - EPS)
    return tf.log(x)


def to_llr(x, name=None):
  ''' Convert a matrix of probabilities into log-likelihood ratio
  :math:`LLR = log(\\frac{prob(data|target)}{prob(data|non-target)})`
  '''
  with tf.name_scope(name, "log_likelihood_ratio", [x]):
    nb_classes = x.shape.as_list()[-1]
    new_arr = []
    for j in range(nb_classes):
      scores_copy = tf.transpose(
          tf.gather(tf.transpose(x), [i for i in range(nb_classes) if i != j]))
      scores_copy -= tf.expand_dims(x[:, j], axis=-1)
      new_arr.append(-logsumexp(scores_copy, 1))
    return tf.concat(new_arr, axis=-1) + np.log(13)


def to_sample_weights(indices, weights, name=None):
  """ Convert indices or one-hot matrix and
  give weights to sample weights for training """
  with tf.name_scope(name, "to_sample_weights", [indices]):
    # ====== preprocess indices ====== #
    ndim = len(indices.shape)
    if ndim <= 1:  # indices vector
      indices = tf.cast(indices, dtype=tf.int64)
    else:
      indices = tf.argmax(indices, axis=-1)
    # ====== prior weights ====== #
    if isinstance(weights, (tuple, list, np.ndarray)):
      prior_weights = tf.constant(weights, dtype=floatX, name="prior_weights")
    # ====== sample weights ====== #
    weights = tf.gather(prior_weights, indices)
  return weights
