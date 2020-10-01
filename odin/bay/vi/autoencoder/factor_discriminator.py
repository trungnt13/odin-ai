from typing import Any, Callable, List, Optional, Union

import numpy as np
import tensorflow as tf
from odin.bay.random_variable import RandomVariable
from odin.bay.vi.utils import permute_dims
from odin.networks import SequentialNetwork
from tensorflow_probability.python.distributions import (Distribution,
                                                         Independent)
from typing_extensions import Literal


class FactorDiscriminator(SequentialNetwork):
  r""" The main goal is minimizing the total correlation (the mutual information
  which quantifies the redundancy or dependency among latent variables).

  We use a discriminator to estimate total-correlation

  This class also support Semi-supervised factor discriminator, a combination
  of supervised objective and total correlation estimation using density-ratio.

    - 0: real sample for q(z) (or last unit in case n_outputs > 2) and
    - 1: fake sample from q(z-)

  If `n_outputs` > 2, suppose the number of classes is `K` then:

    - 0 to K: is the classes' logits for real sample from q(z)
    - K + 1: fake sample from q(z-)

  Paramters
  --------
  units : a list of Integer, the number of hidden units for each hidden layer.
  n_outputs : an Integer or instance of `RandomVariable`,
    the number of output units and its distribution
  ss_strategy : {'sum', 'logsumexp', 'mean', 'max', 'min'}.
    Strategy for combining the outputs semi-supervised learning into the
    logit for real sample from q(z):
    - 'logsumexp' : used for semi-supervised GAN in (Salimans T. 2016)

  References
  ----------
  Kim, H., Mnih, A., 2018. "Disentangling by Factorising".
    arXiv:1802.05983 [cs, stat].
  Salimans, T., Goodfellow, I., Zaremba, W., et al 2016.
    "Improved Techniques for Training GANs". arXiv:1606.03498 [cs.LG].
  """

  def __init__(self,
               input_shape: List[int],
               batchnorm: bool = False,
               input_dropout: float = 0.,
               dropout: float = 0.,
               units: List[int] = [1000, 1000, 1000, 1000, 1000],
               outputs: RandomVariable = RandomVariable(1,
                                                        'bernoulli',
                                                        name="Discriminator"),
               activation: Union[str, Callable[[], Any]] = tf.nn.leaky_relu,
               ss_strategy: Literal['sum', 'logsumexp', 'mean', 'max',
                                    'min'] = 'logsumexp',
               name: str = "FactorDiscriminator"):
    if not isinstance(outputs, (tuple, list)):
      outputs = [outputs]
    outputs = tf.nest.flatten(outputs)
    assert len(outputs) > 0, "No output is given for FactorDiscriminator"
    assert all(isinstance(o, RandomVariable) for o in outputs), \
      (f"outputs must be instance of RandomVariable, but given:{outputs}")
    n_outputs = 0
    for o in outputs:
      o.projection = True
      o.event_shape = (int(np.prod(o.event_shape)),)
      n_outputs += o.event_shape[0]
    layers = dense_network(units=units,
                           batchnorm=batchnorm,
                           dropout=dropout,
                           flatten_inputs=True,
                           input_dropout=input_dropout,
                           activation=activation,
                           input_shape=tf.nest.flatten(input_shape))
    super().__init__(layers, name=name)
    shape = self.output_shape[1:]
    self.distributions = [o.create_posterior(shape) for o in outputs]
    self.n_outputs = sum(d.output_shape[-1] for d in self.distributions)
    self.input_ndim = len(self.input_shape) - 1
    self.ss_strategy = str(ss_strategy)
    assert self.ss_strategy in {'sum', 'logsumexp', 'mean', 'max', 'min'}

  def call(self, inputs, **kwargs):
    outputs = super().call(inputs, **kwargs)
    # project into different output distributions
    distributions = [d(outputs, **kwargs) for d in self.distributions]
    return distributions[0] if len(distributions) == 1 else tuple(distributions)

  def _to_samples(self, qZ_X, mean=False, stop_grad=False):
    qZ_X = tf.nest.flatten(qZ_X)
    if mean:
      z = tf.concat([q.mean() for q in qZ_X], axis=-1)
    else:
      z = tf.concat([tf.convert_to_tensor(q) for q in qZ_X], axis=-1)
    z = tf.reshape(z, tf.concat([(-1,), z.shape[-self.input_ndim:]], axis=0))
    if stop_grad:
      z = tf.stop_gradient(z)
    return z

  def _tc_logits(self, logits):
    # use ss_strategy to infer appropriate logits value for
    # total-correlation estimator (logits for q(z)) in case of n_outputs > 1
    Xs = []
    for x in tf.nest.flatten(logits):
      if isinstance(x, Distribution):
        if isinstance(x, Independent):
          x = x.distribution
        if hasattr(x, 'logits'):
          x = x.logits
        elif hasattr(x, 'concentration'):
          x = x.concentration
        else:
          raise RuntimeError(
              f"Distribution {x} doesn't has 'logits' or 'concentration' "
              "attributes, cannot not be used for estimating total correlation."
          )
      Xs.append(x)
    # concatenate the outputs
    if len(Xs) == 0:
      raise RuntimeError(
          f"No logits values found for total correlation: {logits}")
    elif len(Xs) == 1:
      Xs = Xs[0]
    else:
      Xs = tf.concat(Xs, axis=-1)
    # only 1 unit, only estimate TC
    if self.n_outputs == 1:
      return Xs[..., 0]
    # multiple units, reduce
    return getattr(tf, 'reduce_%s' % self.ss_strategy)(Xs, axis=-1)

  def total_correlation(self,
                        qZ_X: Distribution,
                        training: Optional[bool] = None) -> tf.Tensor:
    r""" Total correlation Eq(3)
    ```
    TC(z) = KL(q(z)||q(z-)) = E_q(z)[log(q(z) / q(z-))]
          ~ E_q(z)[ log(D(z)) - log(1 - D(z)) ]
    ```

    We want to minimize the total correlation to achieve factorized latent units

    Note:
      In many implementation, `log(q(z-)) - log(q(z))` is referred as `total
      correlation loss`, here, we return `log(q(z)) - log(q(z-))` as the total
      correlation for the construction of the ELBO in Eq(2)

    Arguments:
      qZ_X : a Tensor, [batch_dim, latent_dim] or Distribution

    Return:
      TC(z) : a scalar, approximation of the density-ratio that arises in the
        KL-term.
    """
    z = self._to_samples(qZ_X, stop_grad=False)
    logits = self(z, training=training)
    logits = self._tc_logits(logits)
    # in case using sigmoid, other implementation use -logits here but it
    # should be logits.
    # if it is negative here, TC is reduce, but reasonably, it must be positive (?)
    return tf.reduce_mean(logits)

  def dtc_loss(self,
               qZ_X: Distribution,
               qZ_Xprime: Optional[Distribution] = None,
               training: Optional[bool] = None) -> tf.Tensor:
    r""" Discriminated total correlation loss Algorithm(2)

    Minimize the probability of:
     - `q(z)` misclassified as `D(z)[:, 0]`
     - `q(z')` misclassified as `D(z')[:, 1]`

    Arguments:
      qZ_X : `Tensor` or `Distribution`.
        Samples of the latents from first batch
      qZ_Xprime : `Tensor` or `Distribution` (optional).
        Samples of the latents from second batch, this will be permuted.
        If not given, then reuse `qZ_X`.

    Return:
      scalar - loss value for training the discriminator
    """
    # we don't want the gradient to be propagated to the encoder
    z = self._to_samples(qZ_X, stop_grad=True)
    z_logits = self._tc_logits(self(z, training=training))
    # using log_softmax function give more numerical stabalized results than
    # logsumexp yourself.
    d_z = -tf.math.log_sigmoid(z_logits)  # must be negative here
    # for X_prime
    if qZ_Xprime is not None:
      z = self._to_samples(qZ_Xprime, stop_grad=True)
    z_perm = permute_dims(z)
    zperm_logits = self._tc_logits(self(z_perm, training=training))
    d_zperm = -tf.math.log_sigmoid(zperm_logits)  # also negative here
    # reduce the negative of d_z, and the positive of d_zperm
    # this equal to cross_entropy(d_z, zeros) + cross_entropy(d_zperm, ones)
    loss = 0.5 * (tf.reduce_mean(d_z) + tf.reduce_mean(zperm_logits + d_zperm))
    return loss

  def supervised_loss(self,
                      labels: tf.Tensor,
                      qZ_X: Distribution,
                      mean: bool = False,
                      mask: Optional[tf.Tensor] = None,
                      training: Optional[bool] = None) -> tf.Tensor:
    labels = tf.nest.flatten(labels)
    z = self._to_samples(qZ_X, mean=mean, stop_grad=True)
    distributions = tf.nest.flatten(self(z, training=training))
    ## applying the mask (1-labelled, 0-unlabelled)
    if mask is not None:
      mask = tf.reshape(mask, (-1,))
      # labels = [tf.boolean_mask(y, mask, axis=0) for y in labels]
      # z_logits = tf.boolean_mask(z_logits, mask, axis=0)
    ## calculate the loss
    loss = 0.
    for dist, y_true in zip(distributions, labels):
      tf.assert_rank(y_true, 2)
      llk = dist.log_prob(y_true)
      # check the mask careful here
      # if no data for labels, just return 0
      if mask is not None:
        llk = tf.cond(tf.reduce_all(tf.logical_not(mask)), lambda: 0.,
                      lambda: tf.boolean_mask(llk, mask, axis=0))
      # negative log-likelihood here
      loss += -llk
    # check non-zero, if zero the gradient must be stop or NaN gradient happen
    loss = tf.reduce_mean(loss)
    loss = tf.cond(
        tf.abs(loss) < 1e-8, lambda: tf.stop_gradient(loss), lambda: loss)
    return loss
