from __future__ import absolute_import, division, print_function

import inspect
import os

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow_probability.python.bijectors import FillScaleTriL
from tensorflow_probability.python.distributions import (Categorical,
                                                         Distribution,
                                                         Independent,
                                                         MixtureSameFamily,
                                                         MultivariateNormalDiag,
                                                         MultivariateNormalTriL,
                                                         Normal)

__all__ = ['GaussianMixture']


class GaussianMixture(MixtureSameFamily):
  r""" Gaussian mixture distribution

  The inner distributions are:
    - mixture : batch_shape=[...], event_shape=[n_components]
    - components : batch_shape=[..., n_components], event_shape=[n_dims]

  Number of Parameters required for different covariance type:
    - 'tril'/'full' : [n_dims * (n_dims + 1) // 2]
    - 'diag' : [n_dims]
    - 'none' : [n_dims]

  The output distribution shape is: batch_shape=[...], event_shape=[n_dims]

  Arguments:
    loc : a Tensor or Variable.
    scale : a Tensor or Variable.
    logits : a Tensor or Variable (optional).
    probs : a Tensor or Variable (optional).
    trainable : a Boolean. If True, convert all tensor to variable which
      make the GaussianMixture trainable via `fit` method.
    covariance_type : {'tril', 'diag' (default), 'spherical'/'none'}
      String describing the type of covariance parameters to use.
      Must be one of:
      'tril'/'full' - each component has its own general covariance matrix
      'diag' - each component has its own diagonal covariance matrix
      'none' - independent gaussian each component has its own variance
  """

  def __init__(self,
               loc,
               scale,
               logits=None,
               probs=None,
               covariance_type='diag',
               trainable=False,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    kw = dict(validate_args=validate_args, allow_nan_stats=allow_nan_stats)
    self._trainable = bool(trainable)
    self._llk_history = []
    if trainable:
      loc = tf.Variable(loc, trainable=True, name='loc')
      scale = tf.Variable(scale, trainable=True, name='scale')
      if logits is not None:
        logits = tf.Variable(logits, trainable=True, name='logits')
      if probs is not None:
        probs = tf.Variable(probs, trainable=True, name='probs')
    ### initialize mixture Categorical
    mixture = Categorical(logits=logits,
                          probs=probs,
                          name="MixtureWeights",
                          **kw)
    n_components = mixture._num_categories()
    ### initialize Gaussian components
    covariance_type = str(covariance_type).lower().strip()
    if name is None:
      name = 'Mixture%sGaussian' % \
        (covariance_type.capitalize() if covariance_type != 'none' else
         'Independent')
    ## create the components
    if covariance_type == 'diag':
      if tf.rank(scale) == 0:  # scalar
        extra_kw = dict(scale_identity_multiplier=scale)
      else:  # a tensor
        extra_kw = dict(scale_diag=scale)
      components = MultivariateNormalDiag(loc=loc, name=name, **kw, **extra_kw)
    elif covariance_type in ('tril', 'full'):
      if tf.rank(scale) == 1 or \
        (scale.shape[-1] != scale.shape[-2]):
        scale_tril = FillScaleTriL(diag_shift=np.array(
            1e-5,
            tf.convert_to_tensor(scale).dtype.as_numpy_dtype()))
        scale = scale_tril(scale)
      components = MultivariateNormalTriL(loc=loc,
                                          scale_tril=scale,
                                          name=name,
                                          **kw)
    elif covariance_type == 'none':
      components = Independent(distribution=Normal(loc=loc, scale=scale, **kw),
                               reinterpreted_batch_ndims=1,
                               name=name)
    else:
      raise ValueError("No support for covariance_type: '%s'" % covariance_type)
    ### validate the n_components
    assert (components.batch_shape[-1] == int(n_components)), \
      "Number of components mismatch, given:%d, mixture:%d, components:%d" % \
        (mixture.event_shape[-1], components.batch_shape[-1], int(n_components))
    super().__init__(mixture_distribution=mixture,
                     components_distribution=components,
                     name=name,
                     **kw)

  @property
  def trainable(self):
    return self._trainable

  @property
  def llk_history(self):
    return list(self._llk_history)

  @property
  def loc(self):
    return self.components_distribution.loc

  @property
  def scale(self):
    return self.components_distribution.scale

  @property
  def weights(self):
    r""" The mixture probability """
    return self.mixture_distribution.probs_parameter()

  @property
  def logits(self):
    r""" The mixture logits """
    return self.mixture_distribution.logits_parameter()

  @tf.function
  def _fit(self, X, optimizer, params, max_iter, history, verbose):
    curr_time = tf.timestamp()
    nsamples = 0
    for niter, x in X:
      nsamples += tf.shape(x)[0]
      with tf.GradientTape(persistent=False,
                           watch_accessed_variables=False) as tape:
        tape.watch(params)
        nllk = -tf.reduce_mean(self.log_prob(x))
      gs = tape.gradient(target=nllk, sources=params)
      optimizer.apply_gradients(grads_and_vars=zip(gs, params))
      ## print log and store the llk
      if verbose > 0 and tf.timestamp() - curr_time > verbose:
        tf.print("#iter", niter, "#samples", nsamples, "LLK:", -nllk)
        curr_time = tf.timestamp()
      ## maximum iteration
      if niter >= max_iter:
        if verbose:
          tf.print("#iter", niter, "#samples", nsamples, "LLK:", -nllk)
        break
      history[niter].assign(-nllk)

  def fit(self,
          X,
          optimizer='adam',
          opt_kwargs={},
          max_iter=100,
          batch_size=1024,
          strategy='mle',
          seed=1,
          verbose=False):
    r""" Fit the Gaussian mixture

    Arguments:
      X : a training Tensor of shape `[nsamples, event_shape[0]]`
      optimizer : a String
      strategy : {'mle', 'mcmc'}. Method for estimating the distribution
        parameters
          'mle' - maximum likelihood estimation
          'mcmc' - Hamilton Monte-Carlos sampling
    """
    if not self.trainable:
      raise RuntimeError("This GaussianMixture is not trainable.")
    strategy = str(strategy).lower().strip()
    batch_size = int(batch_size)
    max_iter = int(max_iter)
    assert strategy in ('mle',), \
      "Not support for fitting strategy: %s" % strategy
    ### check input X
    assert X.shape[1] == self.event_shape[0], \
      "The distribution has batch_shape=%s event_shape=%s, " \
        % (self.batch_shape, self.event_shape) + \
        "but wrong fitting data: %s" % (X.shape)
    ### check optimizer
    if isinstance(optimizer, string_types):
      name = optimizer.lower()
      optimizer = None
      for k, v in inspect.getmembers(tf.optimizers):
        if isinstance(v, type) and k.lower() == name:
          optimizer = v(**opt_kwargs)
          break
      if optimizer is None:
        raise ValueError("Cannot find optimizer with name: %s" % optimizer)
    else:
      optimizer = tf.optimizers.get(optimizer)
    assert isinstance(optimizer, tf.optimizers.Optimizer)
    ### prepare data
    X = tf.data.Dataset.from_tensor_slices(X).map(
        lambda x: tf.cast(x, self.dtype)).shuffle(
            buffer_size=int(1e5),
            seed=seed).batch(batch_size).repeat(None).enumerate()
    ### prepare parameters
    history = tf.Variable(tf.zeros(shape=(max_iter,), dtype=self.dtype))
    self._fit(X,
              optimizer,
              params=self.trainable_variables,
              max_iter=max_iter,
              history=history,
              verbose=float(verbose))
    self._llk_history += history.numpy().tolist()
    return self

  @staticmethod
  def init(X,
           n_components=1,
           covariance_type='full',
           tol=1e-3,
           reg_covar=1e-6,
           max_iter=100,
           n_init=1,
           init_params='kmeans',
           weights_init=None,
           means_init=None,
           precisions_init=None,
           random_state=1,
           warm_start=False,
           verbose=0,
           verbose_interval=10,
           max_samples=None,
           batch_shape=None,
           trainable=False,
           return_sklearn=False):
    r""" This method fit a sklearn GaussianMixture, then convert it to
    tensorflow_probability GaussianMixture. Hence, the method could be use
    to initialize a trainable GaussianMixture.

    Arguments:
      covariance_type : {'full' , 'tied', 'diag', 'spherical'}
      max_samples : an Integer (optional).
      batch_shape : a Shape tuple (optional).
    """
    if covariance_type == 'tril':
      covariance_type = 'full'
    kwargs = dict(locals())
    ## remove non-GMM kwargs
    X = kwargs.pop('X')
    return_sklearn = kwargs.pop('return_sklearn')
    max_samples = kwargs.pop('max_samples')
    batch_shape = kwargs.pop('batch_shape')
    trainable = bool(kwargs.pop('trainable'))
    ## downsample X
    if max_samples is not None and max_samples < X.shape[0]:
      ids = np.random.choice(np.arange(X.shape[0]),
                             size=int(max_samples),
                             replace=False)
      X = X[ids]
    ## fit sklearn GMM
    from sklearn.mixture import GaussianMixture as _GMM
    gmm = _GMM(**kwargs)
    gmm.fit(X)
    ## convert sklearn GMM to tensorflow_probability
    loc = tf.convert_to_tensor(gmm.means_)
    logits = tf.convert_to_tensor(np.log(gmm.weights_ + 1e-8))
    cov = gmm.covariance_type
    covariance = gmm.covariances_
    ## processing the covariance
    if cov == 'tied':  # [ndims, ndims]
      cov = 'tril'
      scale = np.linalg.cholesky(covariance)
      scale = np.repeat(scale[np.newaxis, :], n_components, axis=0)
    elif cov == 'full':  # [ncomponents, ndims, ndims]
      cov = 'tril'
      scale = np.linalg.cholesky(covariance)
    elif cov == 'spherical':
      cov = 'none'
      scale = np.sqrt(covariance)
      scale = np.repeat(scale[:, np.newaxis], loc.shape[1], axis=1)
    elif cov == 'diag':
      scale = np.sqrt(covariance)
    scale = tf.convert_to_tensor(scale)
    params = [loc, scale, logits]
    ## create the GMM with given batch_shape
    if batch_shape is not None:
      batch_shape = tf.nest.flatten(batch_shape)
      for _ in batch_shape:
        params = [tf.expand_dims(p, 0) for p in params]
      params = [
          tf.tile(\
            p,
            multiples=batch_shape + [1 for _ in range(len(p.shape) - len(batch_shape))])
          for p in params
      ]
    tfp_gmm = GaussianMixture(*params,
                              covariance_type=cov,
                              trainable=trainable,
                              name='Mixture%sGaussian' %
                              gmm.covariance_type.capitalize())
    if return_sklearn:
      return tfp_gmm, gmm
    return tfp_gmm


class NegativeBinomialMixture(Distribution):

  def __init__(self):
    raise NotImplementedError
