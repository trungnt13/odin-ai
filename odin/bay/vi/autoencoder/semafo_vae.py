from typing import Union
import tensorflow as tf
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.random_variable import RVmeta
from odin.backend.interpolation import Interpolation, linear
from odin.bay.vi.autoencoder.variational_autoencoder import _parse_layers
from odin.bay.vi.utils import prepare_ssl_inputs


class semafoVAE(betaVAE):
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
  warmup_step : int
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

  def __init__(
      self,
      labels: RVmeta = RVmeta(10, 'onehot', projection=True, name="digits"),
      alpha: float = 10.0,
      mi_coef: Union[float, Interpolation] = linear(vmin=0.1,
                                                    vmax=0.01,
                                                    length=20000),
      reverse_mi: bool = False,
      warmup_step: int = 1000,
      beta: Union[float, Interpolation] = linear(vmin=1e-6,
                                                 vmax=1.,
                                                 length=2000),
      name='SemafoVAE',
      **kwargs,
  ):
    super().__init__(beta=beta, name=name, **kwargs)
    self.labels = _parse_layers(labels)
    self._mi_coef = mi_coef
    self.alpha = alpha
    self.warmup_step = int(warmup_step)
    self.reverse_mi = bool(reverse_mi)

  def build(self, input_shape):
    return super().build(input_shape)

  @property
  def mi_coef(self):
    if isinstance(self._mi_coef, Interpolation):
      step = tf.maximum(0., tf.cast(self.step - self.warmup_step, tf.float32))
      return self._mi_coef(step)
    return tf.constant(self._mi_coef, dtype=self.dtype)

  @classmethod
  def is_semi_supervised(cls) -> bool:
    return True

  def encode(self, inputs, training=None, mask=None, **kwargs):
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    # don't condition on the labels, only accept inputs
    X = X[0]
    qz_x = super().encode(X, training=training, mask=None, **kwargs)
    py_z = self.labels(tf.convert_to_tensor(qz_x), training=training, mask=mask)
    ## keep the mask
    mask = kwargs.get('mask', None)
    qz_x._keras_mask = mask
    py_z._keras_mask = mask
    return (qz_x, py_z)

  def decode(self, latents, training=None, mask=None, **kwargs):
    if isinstance(latents, (tuple, list)):
      latents = latents[0]
    return super().decode(latents, training, mask, **kwargs)

  def elbo_components(self, inputs, training=None, mask=None):
    ## unsupervised ELBO
    X, y, mask = prepare_ssl_inputs(inputs, mask=mask, n_unsupervised_inputs=1)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
    llk, kl = super().elbo_components(X[0], mask=mask, training=training)
    px_z, (qz_x, qy_z) = self.last_outputs
    ## supervised loss
    llk_y = 0.
    # if labels is provided, p(y|z) is q(y|z) here since we use the same network for both
    if len(y) > 0:
      llk_y = qy_z.log_prob(y[0])  # support only 1 labels set provided
      if mask is not None:
        llk_y = tf.cond(
            tf.reduce_any(mask),
            true_fn=lambda: tf.transpose(
                tf.boolean_mask(tf.transpose(llk_y), mask, axis=0)),
            false_fn=lambda: 0.,
        )
      llk_y = tf.reduce_mean(self.alpha * llk_y)
      llk_y = tf.cond(tf.abs(llk_y) < 1e-8,
                      true_fn=lambda: 0.,
                      false_fn=lambda: llk_y)
    llk[f"llk_{self.labels.name}"] = llk_y
    ## sample the prior
    batch_shape = qz_x.batch_shape
    z_prime = qz_x.KL_divergence.prior.sample(batch_shape)
    ## decoding
    px = self.decode(z_prime, training=training)
    if px.reparameterization_type == NOT_REPARAMETERIZED:
      x = px.mean()
    else:
      x = tf.convert_to_tensor(px)
    # should not stop gradient here, generator need to be updated
    # x = tf.stop_gradient(x)
    qz_xprime, py_z = self.encode(x, training=training)
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
    llk[f'mi_{self.labels.name}'] = tf.cond(
        tf.logical_and(self.step >= self.warmup_step, training),
        true_fn=lambda: self.mi_coef * mi_y,
        false_fn=lambda: tf.stop_gradient(mi_y),
    )
    ## this value is just for monitoring
    mi_z = tf.reduce_mean(tf.stop_gradient(qz_xprime.log_prob(z_prime)))
    mi_z = tf.cond(tf.math.is_nan(mi_z),
                   true_fn=lambda: 0.,
                   false_fn=lambda: tf.clip_by_value(mi_z, -1e8, 1e8))
    llk[f'mi_{self.latents.name}'] = mi_z
    return llk, kl


class remafoVAE(semafoVAE):
  """ semafoVAE with reversed KL for the mutual information

  (i.e. minimize `D_kl(p(y|z)||q(y|z))`)
  """

  def __init__(self, name: str = 'RemafoVAE', **kwargs):
    super().__init__(reverse_mi=True, name=name, **kwargs)
