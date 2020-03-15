from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.bijectors import FillScaleTriL, ScaleTriL
from tensorflow_probability.python.distributions import (
    Categorical, Distribution, Independent, MixtureSameFamily,
    MultivariateNormalDiag, MultivariateNormalTriL, Normal)

__all__ = ['GaussianMixture']


class GaussianMixture(MixtureSameFamily):
  r""" Gaussian mixture distribution

  The inner distributions are:
    - mixture : batch_shape=[...], event_shape=[n_components]
    - components : batch_shape=[..., n_components], event_shape=[n_dims]

  Number of Parameters required for different covariance type:
    - 'tril' : [n_dims * (n_dims + 1) // 2]
    - 'diag' : [n_dims]
    - 'spherical' : [n_dims]

  The output distribution shape is: batch_shape=[...], event_shape=[n_dims]

  Arguments:
    covariance_type : {'tril', 'diag' (default), 'spherical'/'none'}
      String describing the type of covariance parameters to use.
      Must be one of:
      'tril' - each component has its own general covariance matrix
      'diag' - each component has its own diagonal covariance matrix
      'spherical'/'none' - each component has its own single variance
  """

  @staticmethod
  def fit(X,
          n_components=1,
          covariance_type='diag',
          tol=1e-3,
          reg_covar=1e-6,
          max_iter=100,
          n_init=1,
          init_params='kmeans',
          weights_init=None,
          means_init=None,
          precisions_init=None,
          random_state=None,
          warm_start=False,
          verbose=0,
          verbose_interval=10):
    if covariance_type in ('full', 'tied'):
      raise ValueError("No support for covariance type: '%s'" % covariance_type)
    if covariance_type == 'tril':
      covariance_type = 'full'
    kwargs = dict(locals())
    X = kwargs.pop('X')
    ## fit sklearn GMM
    from sklearn.mixture import GaussianMixture as _GMM
    gmm = _GMM(**kwargs)
    gmm.fit(X)
    ## convert sklearn GMM to tensorflow_probability
    loc = tf.convert_to_tensor(gmm.means_)
    logits = tf.convert_to_tensor(np.log(gmm.weights_ + 1e-8))
    cov = gmm.covariance_type
    if cov == 'full':
      cov = 'tril'
      for i in np.sqrt(gmm.covariances_):
        print(np.tril_indices(i))
      exit()
      scale = tf.convert_to_tensor(
          np.concatenate([np.tril(i) for i in np.sqrt(gmm.covariances_)],
                         axis=0))
      print(scale.shape)
      exit()
    else:
      scale = tf.convert_to_tensor(np.sqrt(gmm.covariances_))

    return GaussianMixture(loc=loc,
                           scale=scale,
                           logits=logits,
                           covariance_type=cov)

  @staticmethod
  def params_size(ndims, covariance_type):
    covariance_type = str(covariance_type).lower().strip()
    if covariance_type == 'none':
      covariance_type = 'spherical'
    ndims = int(ndims)
    # calculate
    if covariance_type == 'diag':
      return ndims
    elif covariance_type == 'tril':
      return int(ndims * (ndims + 1) / 2)
    elif covariance_type == 'spherical':
      return int(ndims)
    raise ValueError("No support covariance_type='%s'" % covariance_type)

  def __init__(self,
               loc,
               scale,
               logits=None,
               probs=None,
               covariance_type='diag',
               validate_args=False,
               allow_nan_stats=True,
               name="GaussianMixture"):
    kw = dict(validate_args=validate_args, allow_nan_stats=allow_nan_stats)
    ### initialize mixture Categorical
    mixture = Categorical(logits=logits,
                          probs=probs,
                          name="MixtureWeights",
                          **kw)
    n_components = mixture._num_categories()
    ### initialize Gaussian components
    covariance_type = str(covariance_type).lower().strip()
    if covariance_type == 'none':
      covariance_type = 'spherical'
    name = 'Mixture%sGaussian' % covariance_type.capitalize()
    ## create the components
    if covariance_type == 'diag':
      if tf.rank(scale) == 0:  # scalar
        extra_kw = dict(scale_identity_multiplier=scale)
      else:  # a tensor
        extra_kw = dict(scale_diag=scale)
      components = MultivariateNormalDiag(loc=loc, name=name, **kw, **extra_kw)
    elif covariance_type == 'tril':
      scale_tril = FillScaleTriL(diag_shift=np.array(
          1e-5,
          tf.convert_to_tensor(scale).dtype.as_numpy_dtype()))
      components = MultivariateNormalTriL(loc=loc,
                                          scale_tril=scale_tril(scale),
                                          name=name,
                                          **kw)
    elif covariance_type in ('spherical', 'none'):
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


class NegativeBinomialMixture(Distribution):

  def __init__(self):
    raise NotImplementedError
