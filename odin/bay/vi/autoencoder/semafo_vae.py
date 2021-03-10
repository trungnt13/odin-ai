from typing import List, Union, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import NOT_REPARAMETERIZED, Distribution

from odin.backend.interpolation import Interpolation, linear
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import annealingVAE
from odin.bay.vi.autoencoder.variational_autoencoder import _parse_layers
from odin.bay.vi.utils import prepare_ssl_inputs
from odin.bay.layers import DistributionDense
from odin.utils import as_tuple


# ===========================================================================
# helpers
# ===========================================================================
def _get_llk_y(py, y, mask, alpha):
  llk_y = 0.
  # if labels is provided,
  # p(y|z) is q(y|z) here since we use the same network for both
  if len(y) > 0:
    # support only 1 labels set provided
    if isinstance(y, (tuple, list)):
      y = y[0]
    llk_y = py.log_prob(y)
    if mask is not None:
      llk_y = tf.cond(
          tf.reduce_any(mask),
          true_fn=lambda: tf.transpose(
              tf.boolean_mask(tf.transpose(llk_y), mask, axis=0)),
          false_fn=lambda: 0.,
      )
    llk_y = tf.reduce_mean(alpha * llk_y)
    llk_y = tf.cond(tf.abs(llk_y) < 1e-8,
                    true_fn=lambda: 0.,
                    false_fn=lambda: llk_y)
  return llk_y


class _semafo(annealingVAE):

  def __init__(
      self,
      labels: RVmeta = RVmeta(10, 'onehot', projection=True, name="digits"),
      alpha: float = 10.0,
      mi_coef: Union[float, Interpolation] = linear(vmin=0.1,
                                                    vmax=0.05,
                                                    length=20000),
      reverse_mi: bool = False,
      steps_without_mi: int = 1000,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self.labels = _parse_layers(labels)
    self._mi_coef = mi_coef
    self.alpha = alpha
    self.steps_without_mi = int(steps_without_mi)
    self.reverse_mi = bool(reverse_mi)

  def predict_factors(self,
                      inputs=None,
                      latents=None,
                      training=None,
                      mask=None,
                      **kwargs) -> Distribution:
    """Return the predictive distribution of the factors (a.k.a labels) `p(y|z)`"""
    if inputs is not None:
      latents = self.encode(inputs, training=training, mask=mask, **kwargs)
    elif latents is None:
      raise ValueError("Either 'inputs' or 'latents' must be provided")
    py_z = self.labels(tf.concat(as_tuple(latents), axis=-1),
                       training=training,
                       mask=mask)
    return py_z

  @property
  def mi_coef(self):
    if isinstance(self._mi_coef, Interpolation):
      step = tf.maximum(0.,
                        tf.cast(self.step - self.steps_without_mi, tf.float32))
      return self._mi_coef(step)
    return tf.constant(self._mi_coef, dtype=self.dtype)

  @classmethod
  def is_hierarchical(self) -> bool:
    return False

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True


# ===========================================================================
# SemafoVAE
# ===========================================================================
class semafoCVAE(_semafo):
  """ Semafo Conditional VAE
  """

  def __init__(
      self,
      mi_coef: Union[float, Interpolation] = linear(vmin=1.0,
                                                    vmax=0.5,
                                                    length=20000),
      name: str = 'SemafoCVAE',
      **kwargs,
  ):
    super().__init__(mi_coef=mi_coef, name=name, **kwargs)

  def sample_prior(self,
                   sample_shape: Union[int, List[int]] = (),
                   seed: int = 1) -> tf.Tensor:
    r""" Sampling from prior distribution """
    z1 = super().sample_prior(sample_shape=sample_shape, seed=seed)
    z2 = self.mutual_codes.prior.sample(sample_shape, seed=seed)
    return (z1, z2)

  def encode(self, inputs, **kwargs):
    h_e = self.encoder(inputs, **kwargs)
    # create the latents distribution
    qz_x = self.latents(h_e, **kwargs)
    qy_x = self.labels(h_e, **kwargs)
    # need to keep the keras mask
    mask = kwargs.get('mask', None)
    qz_x._keras_mask = mask
    qy_x._keras_mask = mask
    return (qz_x, qy_x)

  def decode(self, latents, **kwargs):
    latents = tf.concat(latents, axis=-1)
    return super().decode(latents, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super().elbo_components(X[0], mask=mask, training=training)
    px_z, (qz_x, py_x) = self.last_outputs
    ## supervised loss
    llk[f"llk_{self.labels.name}"] = _get_llk_y(py_x, y, mask, self.alpha)
    ## mutual information estimation
    batch_shape = px_z.batch_shape
    z_samples = qz_x.KL_divergence.prior.sample(batch_shape)
    y_samples = py_x.KL_divergence.prior.sample(batch_shape)
    ## decoding
    px = self.decode([z_samples, y_samples], training=training)
    if px.reparameterization_type == NOT_REPARAMETERIZED:
      x = px.mean()
    else:
      x = tf.convert_to_tensor(px)
    qz_xprime, qy_x = self.encode(x, training=training)
    #' mutual information (we want to maximize this, hence, add it to the llk)
    llk[f'mi_{self.labels.name}'] = self.mi_coef * tf.cond(
        self.step > self.steps_without_mi,
        true_fn=lambda: qy_x.log_prob(y_samples),
        false_fn=lambda: 0.)
    ## this value is just for monitoring
    mi_z = tf.reduce_mean(tf.stop_gradient(qz_xprime.log_prob(z_samples)))
    mi_z = tf.cond(tf.math.is_nan(mi_z),
                   true_fn=lambda: 0.,
                   false_fn=lambda: tf.clip_by_value(mi_z, -1e8, 1e8))
    llk['mi_latents'] = mi_z
    return llk, kl


class semafoVAE(_semafo):
  """A semaphore is a variable or abstract data type used to control access to
  a common resource by multiple processes and avoid critical section problems in
  a concurrent system

  For MNIST, `mi_coef` could be from 0.1 to 0.5.
  For dSprites, CelebA and Shapes3D, `mi_coef=0.1` has been tested.

  It is also seems that the choice of `mi_coef` is crucial, regardless
  the percentage of labels, or the networks design.

  It is also crucial to get the right value at the beginning of training
  (i.e. small enough)

  Parameters
  ----------
  reverse_mi : bool
      if True, minimize `D_kl(p(y|z)||q(y|z))`, otherwise, `D_kl(q(y|z)||p(y|z))`
  steps_without_mi : int
      number of step without mutual information maximization which allows
      the network to fit better encoder. Gradients backpropagated to encoder
      often NaNs if optimize all objective from beginning, default: 1000

  SemafoVAE  [Callback#50001]:
  llk_x:-73.33171081542969
  llk_y:-0.9238954782485962
  acc_y:0.7268000245094299

  Without autoregressive
  llk_x:-72.9976577758789
  llk_y:-0.7141319513320923
  acc_y:0.8095999956130981

  Idea: set mi_coef = 1. / alpha
  """

  def __init__(self, name: str = 'SemafoVAE', **kwargs):
    super().__init__(name=name, **kwargs)

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    # don't condition on the labels, only accept inputs
    X = X[0]
    qz_x = super().encode(X, training=training, mask=None, **kwargs)
    return qz_x

  def decode(self, latents, training=None, mask=None, **kwargs):
    px_z = super().decode(latents, training, mask, **kwargs)
    py_z = self.predict_factors(latents=latents, training=training, mask=mask)
    return as_tuple(px_z) + (py_z,)

  def _mi_loss(self,
               Q,
               py_z,
               training=None,
               mask=None) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    ## sample the prior
    batch_shape = Q[0].batch_shape
    z_prime = [q.KL_divergence.prior.sample(batch_shape) for q in Q]
    if len(z_prime) == 1:
      z_prime = z_prime[0]
    ## decoding
    px = self.decode(z_prime, training=training)[0]
    if px.reparameterization_type == NOT_REPARAMETERIZED:
      x = px.mean()
    else:
      x = tf.convert_to_tensor(px)
    # should not stop gradient here, generator need to be updated
    # x = tf.stop_gradient(x)
    Q_prime = self.encode(x, training=training)
    qy_z = self.predict_factors(latents=Q_prime, training=training)
    ## y ~ p(y|z), stop gradient here is important to prevent the encoder updated twice
    # this significantly increase the stability, otherwise, encoder and latents often
    # get NaNs gradients
    if self.reverse_mi:  # D_kl(p(y|z)||q(y|z))
      y_samples = tf.stop_gradient(py_z.sample())
      Dkl = py_z.log_prob(y_samples) - qy_z.log_prob(y_samples)
    else:  # D_kl(q(y|z)||p(y|z))
      y_samples = tf.stop_gradient(qy_z.sample())
      Dkl = qy_z.log_prob(y_samples) - py_z.log_prob(y_samples)
    ## only calculate MI for unsupervised data
    mi_mask = tf.logical_not(mask)  # TODO: tf.reduce_any(mi_mask)
    mi_y = tf.reduce_mean(tf.boolean_mask(Dkl, mask=mi_mask, axis=0))
    ## mutual information (we want to maximize this, hence, add it to the llk)
    if training:
      mi_y = tf.cond(
          self.step >= self.steps_without_mi,
          true_fn=lambda: self.mi_coef * mi_y,
          false_fn=lambda: tf.stop_gradient(mi_y),
      )
    else:
      mi_y = tf.stop_gradient(mi_y)
    ## this value is just for monitoring
    mi_z = []
    for q, z in zip(as_tuple(Q_prime), as_tuple(z_prime)):
      mi = tf.reduce_mean(tf.stop_gradient(q.log_prob(z)))
      mi = tf.cond(tf.math.is_nan(mi),
                   true_fn=lambda: 0.,
                   false_fn=lambda: tf.clip_by_value(mi, -1e8, 1e8))
      mi_z.append(mi)
    return mi_y, mi_z

  def elbo_components(self, inputs, training=None, mask=None):
    ## unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super().elbo_components(X[0], mask=mask, training=training)
    P, Q = self.last_outputs
    px_z = P[:-1]
    py_z = P[-1]
    Q = as_tuple(Q)  # q(z|x)
    ## supervised loss
    llk[f"llk_{self.labels.name}"] = _get_llk_y(py_z, y, mask, self.alpha)
    ## MI objective
    mi_y, mi_z = self._mi_loss(Q, py_z, training=training, mask=mask)
    llk[f'mi_{self.labels.name}'] = mi_y
    for z, mi in zip(as_tuple(self.latents), mi_z):
      llk[f'mi_{z.name}'] = mi
    return llk, kl


class remafoVAE(semafoVAE):
  """ semafoVAE with reversed KL for the mutual information

  (i.e. minimize `D_kl(p(y|z)||q(y|z))`)
  """

  def __init__(self, name: str = 'RemafoVAE', **kwargs):
    super().__init__(reverse_mi=True, name=name, **kwargs)


class semafoDVAE(semafoVAE):
  """Semafo VAE using generative method for density-ratio estimation of
  `D(q(y|z)||p(y|z))`
  """

  def __init__(self, name: str = 'SemafoDVAE', **kwargs):
    super().__init__(name=name, **kwargs)
    labels_kw = self.labels.get_config()
    labels_kw['name'] += '_q'
    self.labels_p = self.labels
    self.labels_q = DistributionDense(**labels_kw)

  def build(self, input_shape):
    super().build(input_shape)
    self.labels_q(keras.Input(self.latents.event_shape))
    return self

  def elbo_components(self, inputs, training=None, mask=None):
    ## unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super(annealingVAE, self).elbo_components(X[0],
                                                        mask=mask,
                                                        training=training)
    P, Q = self.last_outputs
    px_z = P[:-1]
    py_z = P[-1]
    Q = as_tuple(Q)  # q(z|x)
    ## supervised loss
    llk[f"llk_{self.labels.name}"] = _get_llk_y(py_z, y, mask, self.alpha)
    ## MI objective
    self.labels = self.labels_q
    mi_y, mi_z = self._mi_loss(Q, py_z, training=training, mask=mask)
    self.labels = self.labels_p
    ## maximizing the MI
    llk[f'mi_{self.labels.name}'] = mi_y
    for z, mi in zip(as_tuple(self.latents), mi_z):
      llk[f'mi_{z.name}'] = mi
    return llk, kl
