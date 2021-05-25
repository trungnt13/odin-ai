from typing import Dict, List, Optional, Tuple, Union, Sequence

import numpy as np
import tensorflow as tf
from typing_extensions import Literal

from odin.networks import Networks, TrainStep
from odin.backend import TensorType
from odin.utils import as_tuple
from scipy import sparse
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow_probability.python.distributions import Distribution
from tqdm import tqdm

__all__ = ['VariationalModel', 'TrainStep']


class VariationalModel(Networks):
  """Variational networks with configurable ELBO"""

  def __init__(
      self,
      analytic: bool = False,
      reverse: bool = True,
      free_bits: Optional[float] = None,
      sample_shape: Union[int, Sequence[int]] = (),
      allow_negative_kl: bool = True,
      **kwargs,
  ):
    self.supports_masking = True
    super().__init__(**kwargs)
    self._sample_shape = sample_shape
    self.analytic = analytic
    self.reverse = reverse
    self.free_bits = free_bits
    self.allow_negative_kl = bool(allow_negative_kl)

  @classmethod
  def is_hierarchical(cls) -> bool:
    """Hierarchical models have multiple latents"""
    return False

  @property
  def sample_shape(self) -> Sequence[int]:
    return as_tuple(self._sample_shape, t=int)

  @property
  def sample_ndim(self) -> int:
    return len(as_tuple(self.sample_shape))

  def set_elbo_configs(
      self,
      analytic: Optional[bool] = None,
      reverse: Optional[bool] = None,
      sample_shape: Optional[Union[int, Sequence[int]]] = None,
      free_bits: Optional[float] = None,
  ) -> 'VariationalModel':
    """Set the configuration for ELBO

    Parameters
    ----------
    analytic : Optional[bool], optional
        if True use close-form solution for KL, by default None
    reverse : Optional[bool], optional
        If `True`, calculating `KL(q||p)` which optimizes `q`
        (or p_model) by greedily filling in the highest modes of data (or, in
        other word, placing low probability to where data does not occur).
        Otherwise, `KL(p||q)` a.k.a maximum likelihood, or expectation
        propagation place high probability at anywhere data occur
        (i.e. averagely fitting the data)., by default None
    sample_shape : Optional[Union[int, List[int]]], optional
        number of MCMC samples for MCMC estimation of KL-divergence,
        by default None
    free_bits : float, optional
        free bits, recommended in range [0.125, 0.25, 0.5]

    Returns
    -------
    VariationalAutoencoder
        the object itself for method chaining
    """
    if analytic is not None:
      self.analytic = bool(analytic)
    if reverse is not None:
      self.reverse = bool(reverse)
    if sample_shape is not None:
      self._sample_shape = sample_shape
    self.free_bits = free_bits
    return self

  def importance_weighted(self, elbo: TensorType, axis: int = 0) -> tf.Tensor:
    """VAE objective can lead to overly simplified representations which
    fail to use the network’s entire modeling capacity.

    Importance weighted autoencoder (IWAE) uses a strictly tighter
    log-likelihood lower bound derived from importance weighting.

    Using more samples can only improve the tightness of the bound, and
    as our estimator is based on the log of the average importance weights,
    it does not suffer from high variance.

    Parameters
    ----------
    elbo : TensorTypes
        the ELBO
    axis : int, optional
        axis for calculating importance-weights, by default 0

    Returns
    -------
    tf.Tensor
        importance-weighted ELBO

    Reference
    ---------
      Yuri Burda, Roger Grosse, Ruslan Salakhutdinov. Importance Weighted
        Autoencoders. In ICLR, 2015. https://arxiv.org/abs/1509.00519
    """
    dtype = elbo.dtype
    iw_dim = tf.cast(elbo.shape[axis], dtype=dtype)
    elbo = tf.reduce_logsumexp(elbo, axis=axis) - tf.math.log(iw_dim)
    return elbo

  def elbo_components(
      self,
      inputs: Union[TensorType, Sequence[TensorType]],
      training: Optional[bool] = None,
      *args,
      **kwargs) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Calculate the distortion (log-likelihood) and rate (KL-divergence)
    for the calculation of the "Evident Lower Bound" (ELBO).

    Parameters
    ----------
    inputs : TensorTypes
        inputs
    training : Optional[bool], optional
        training or evaluation mode, by default None

    Returns
    -------
    Dict[str, tf.Tensor]
        mapping from observation name to its log-likelihood values,
        shape `[n_samples]`
    Dict[str, tf.Tensor]]
        mapping from latents name to its KL-divergence values,
        shape `[sample_shape, n_samples]`
    """
    raise NotImplementedError

  def elbo(
      self,
      llk: Optional[Dict[str, tf.Tensor]] = None,
      kl: Optional[Dict[str, tf.Tensor]] = None,
  ) -> tf.Tensor:
    """Calculate the distortion (log-likelihood) and rate (KL-divergence)
    for contruction the Evident Lower Bound (ELBO).

    The final ELBO is:
      `ELBO = E_{z~q(Z|X)}[log(p(X|Z))] - KL_{x~p(X)}[q(Z|X)||p(Z)]`

    Parameters
    ----------
    llk : Dict[str, Tensor], optional
        log-likelihood components, by default `{}`
    kl : Dict[str, Tensor], optional
        KL-divergence components, by default `{}`

    Returns
    -------
    tf.Tensor
        the ELBO - a Tensor shape `[sample_shape, batch_size]`.
        dictionary mapping the components of the ELBO (e.g. rate, distortion) to
        their values.
    """
    if llk is None:
      llk = dict()
    if kl is None:
      kl = dict()
    # sum all the components log-likelihood and KL-divergence
    llk_sum = tf.constant(0., dtype=self.dtype)
    kl_sum = tf.constant(0., dtype=self.dtype)
    for x in llk.values():  # log-likelihood
      llk_sum += x
    for name, x in kl.items():  # kl-divergence
      if not self.allow_negative_kl:
        tf.debugging.assert_greater(
          x,
          -1e-3,
          message=(f"Negative KL-divergence values for '{name}', "
                   "probably because of numerical instability."))
      kl_sum += x
    elbo = llk_sum - kl_sum
    return elbo

  def marginal_log_prob(self,
                        inputs: Union[TensorType, List[TensorType]],
                        training: Optional[bool] = None,
                        *args,
                        **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Marginal log likelihood `log(p(X))`, an biased estimation.
    With sufficient amount of MCMC samples (-> inf), the value will converges
    to `log(p(X))`

    With large amount of sample, recommending reduce the batch size to very
    small number or use CPU for the calculation `with tf.device("/CPU:0"):`

    Note: this function will need further modification for more complicated
    prior and latent space, only work for:

      - vanilla-VAE or
      - with proper prior injected into qZ_X and pZ_X using
        `qZ_X.KL_divergence.prior = ...` during `encode` or `decode` methods


    Parameters
    ----------
    inputs : TensorTypes
        inputs
    training : Optional[bool], optional
        training or evaluation mode, by default None

    Returns
    -------
    Tuple[tf.Tensor, Dict[str, tf.Tensor]]
      marginal log-likelihood - a Tensor of shape `[batch_size]` marginal
      log-likelihood of p(X), and
      a Dictionary mapping from distribution name to Tensor of shape `[batch_size]`,
      the negative reconstruction cost (a.k.a distortion or reconstruction)
    """
    raise NotImplementedError

  def perplexity(self,
                 inputs: Union[TensorType, DatasetV2],
                 training: Optional[bool] = None,
                 elbo: Optional[TensorType] = None,
                 verbose: bool = False,
                 *args,
                 **kwargs) -> tf.Tensor:
    """The perplexity is an exponent of the average negative ELBO per word
    (or feature).

    Parameters
    ----------
    inputs : Union[TensorTypes, DatasetV2]
        inputs
    elbo : Optional[TensorTypes], optional
        pre-calculated ELBO if available, by default None
    verbose : bool, optional
        show the progress incase of `tensorflow.data.Dataset` is given,
        by default False

    Returns
    -------
    tf.Tensor
        the perplexity, a scalar value
    """
    ### given an tensorflow interable dataset
    if isinstance(inputs, DatasetV2):
      log_perplexity = []
      if verbose:
        inputs = tqdm(inputs, desc="Calculating perplexity")
      for x in inputs:
        llk, kl = self.elbo_components(x, training=training, *args, **kwargs)
        elbo = self.elbo(llk=llk, kl=kl)
        words_per_doc = tf.reduce_sum(x, axis=-1)
        log_perplexity.append(-elbo / words_per_doc)
      log_perplexity = tf.concat(log_perplexity, axis=-1)
    ### just single calculation
    else:
      if isinstance(inputs, sparse.spmatrix):
        inputs = inputs.toarray()
      inputs = tf.convert_to_tensor(inputs, dtype_hint=self.dtype)
      if elbo is None:
        llk, kl = self.elbo_components(inputs,
                                       training=training,
                                       *args,
                                       **kwargs)
        elbo = self.elbo(llk=llk, kl=kl)
      # calculate the perplexity
      words_per_doc = tf.reduce_sum(inputs, axis=-1)
      log_perplexity = -elbo / words_per_doc
    ### final average
    log_perplexity_tensor = tf.reduce_mean(log_perplexity)
    perplexity_tensor = tf.exp(log_perplexity_tensor)
    return perplexity_tensor

  def train_steps(self, inputs, training=True, name='', *args, **kwargs):

    def loss():
      llk, kl = self.elbo_components(inputs, training, *args, **kwargs)
      elbo = self.elbo(llk, kl)
      metrics = {k: tf.reduce_mean(v) for k, v in dict(**llk, **kl).items()}
      return -tf.reduce_mean(elbo), metrics

    yield loss

  def get_latents(
      self,
      inputs: Optional[Union[TensorType, Sequence[TensorType]]] = None,
      training: Optional[bool] = None,
      mask: Optional[TensorType] = None,
      return_prior: bool = False,
      **kwargs) -> Union[Distribution, Sequence[Distribution]]:
    """Get the posterior (and prior) latents distribution `q(z|x)` and `p(z)`"""
    if inputs is None:
      if self.last_outputs is None:
        raise RuntimeError(f'Cannot get_latents of {self.name} '
                           'both inputs and last_outputs are None.')
      P, Q = self.last_outputs
    else:
      Q = self.encode(inputs, training=training, mask=mask, **kwargs)
    if return_prior:
      if isinstance(Q, (tuple, list)):
        return Q, tuple([q.KL_divergence.prior for q in Q])
      return Q, Q.KL_divergence.prior
    return Q

  def sample_observation(self, n: int = 1, seed: int = 1,
                         training: bool = False, **kwargs) -> Distribution:
    raise NotImplementedError

  def sample_prior(self, n: int = 1, seed: int = 1, **kwargs) -> tf.Tensor:
    """Sampling from prior distribution"""
    raise NotImplementedError

  def sample_traverse(
      self,
      inputs: Union[TensorType, List[TensorType]],
      n_traverse_points: int = 11,
      n_best_latents: int = 5,
      min_val: Union[float, Sequence[float]] = -2.0,
      max_val: Union[float, Sequence[float]] = 2.0,
      mode: Literal['linear', 'quantile', 'gaussian'] = 'linear',
      smallest_stddev: bool = True,
      training: bool = False,
      mask: Optional[TensorType] = None) -> Tuple[Distribution, Sequence[int]]:
    """Traversing a dimension of a matrix between given range

    Parameters
    ----------
    min_val : int, optional
        minimum value of the traverse, by default -2.0
    max_val : int, optional
        maximum value of the traverse, by default 2.0
    n_traverse_points : int, optional
        number of points in the traverse, must be odd number, by default 11
    mode : {'linear', 'quantile', 'gaussian'}, optional
        'linear' mode take linear interpolation between the `min_val` and
        `max_val`.
        'quantile' mode return `num` quantiles based on min and max values
        inferred from the data.
        'gaussian' mode takes `num` Gaussian quantiles, by default 'linear'

    Returns
    -------
    np.ndarray
        the ndarray with traversed axes

    Example
    --------
    For `n_traverse_points=3`, and `feature_indices=[0]`,
    the return latents are:
    ```
    [[-2., 0.47],
     [ 0., 0.47],
     [ 2., 0.47]]
    ```
    """
    from odin.bay.vi import traverse_dims
    latents = self.encode(inputs, training=training, mask=mask)
    stddev = np.sum(latents.stddev(), axis=0)
    # smaller stddev is better
    if smallest_stddev:
      top_latents = np.argsort(stddev)[:int(n_best_latents)]
    else:
      top_latents = np.argsort(stddev)[::-1][:int(n_best_latents)]
    latents = traverse_dims(latents,
                            feature_indices=top_latents,
                            min_val=min_val, max_val=max_val,
                            n_traverse_points=n_traverse_points,
                            mode=mode)
    return self.decode(latents, training=training, mask=mask), top_latents

  def encode(self,
             inputs: Union[TensorType, Sequence[TensorType]],
             training: Optional[bool] = None,
             *args,
             **kwargs) -> Union[Distribution, Sequence[Distribution]]:
    """Project data points into latents space

    Parameters
    ----------
    inputs : Union[TensorTypes, List[TensorTypes]]
        the inputs
    training : Optional[bool], optional
        training or evaluation mode, by default None

    Returns
    -------
    Union[Distribution, List[Distribution]]
        a single or list of latents distribution
    """
    raise NotImplementedError

  def decode(
      self,
      latents: Union[TensorType, Distribution, Sequence[TensorType],
                     Sequence[Distribution]],
      training: Optional[bool] = None,
      *args,
      **kwargs) -> Union[Distribution, Sequence[Distribution]]:
    """Project the latents vector back into the input space

    Parameters
    ----------
    latents : Union[TensorTypes, Distribution, List[TensorTypes], List[Distribution]]
        the distribution or single point in latents space
    training : Optional[bool], optional
        training or evaluation mode, by default None

    Returns
    -------
    Union[Distribution, List[Distribution]]
        a single or list of output distribution
    """
    raise NotImplementedError
