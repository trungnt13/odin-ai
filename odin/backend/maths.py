from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import torch

from odin.backend import tensor as ts



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
def softmax(x, axis=None):
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
