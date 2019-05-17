# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import math
import random

from numbers import Number
from itertools import chain
from collections import defaultdict, Iterator, OrderedDict, Mapping

import numpy as np

from odin.autoconfig import get_rng
from odin.maths import interp
from odin.utils import as_tuple, flatten_list, ctext, batching

def prior2weights(prior, exponential=False,
                  min_value=0.1, max_value=None,
                  norm=False):
  """
  Parameters
  ----------
  prior: numpy.ndarray [nb_classes,]
      probabilty values of each classes prior,
      sum of all prior must be equal to 1.
  exponential: bool
  min_value: bool
      minimum value for the class with highest prior
  max_value: bool
      maximum value for the class with smalles prior
  norm: bool
      if True, normalize output weights to sum up to 1.
  """
  # idea is the one with highest prior equal to 1.
  # and all other classes is the ratio to this prior
  prior = np.array(prior).ravel()
  # make sure everything sum to 1 (probability values)
  prior = prior / np.sum(prior)
  zero_ids = [i for i, j in enumerate(prior) if j == 0]
  nonzero_prior = np.array([j for i, j in enumerate(prior) if j != 0])
  prior = 1. / nonzero_prior * np.max(nonzero_prior)
  if exponential:
    prior = sorted([(i, p) for i, p in enumerate(prior)],
                   key=lambda x: x[-1], reverse=False)
    alpha = interp.expIn(n=len(prior), power=10)
    prior = {i: a * p for a, (i, p) in zip(alpha, prior)}
    prior = np.array([prior[i] for i in range(len(prior))]) + 1
  # ====== rescale everything within max_value ====== #
  if min_value is not None and max_value is not None:
    min_value = float(min_value)
    max_value = float(max_value)
    prior = (max_value - min_value) * (prior - np.min(prior)) \
        / (np.max(prior) - np.min(prior)) + min_value
  # ====== normaize by ====== #
  if norm:
    prior = prior / np.sum(prior)
  # ====== set zero indices ====== #
  prior = prior.tolist()
  for i in zero_ids:
    prior.insert(i, 0)
  return np.array(prior)

# ===========================================================================
# Diagnose
# ===========================================================================
def _split_list(x, rng, train=0.6, idfunc=None, inc_test=True):
  # ====== shuffle input ====== #
  if idfunc is not None:
    x_id = defaultdict(list)
    for i in x:
      x_id[idfunc(i)].append(i)
  else:
    x_id = {i: [j] for i, j in enumerate(x)}
  # shuffle ID(s)
  id_list = list(x_id.keys())
  rng.shuffle(id_list)
  # ====== split ====== #
  N = len(id_list)
  if N == 1:
    raise ValueError("Only find 1 sample, cannot split")
  train = int(np.floor(float(train) * N))
  if train >= N:
    raise ValueError("train proportion must larger than 0 and smaller than 1.")
  valid = (N - train) // (2 if inc_test else 1)
  # ====== return splitted ====== #
  rets = (flatten_list((x_id[i] for i in id_list[:train]), level=1),
          flatten_list((x_id[i] for i in id_list[train: train + valid]), level=1))
  if inc_test:
    rets += (flatten_list((x_id[i] for i in id_list[train + valid:]), level=1),)
  else:
    rets += ([],)
  assert sum(len(r) for r in rets) == len(x), \
      "Number of returned data inconsitent from original data, %d != %d" % \
      (sum(len(r) for r in rets), len(x))
  return rets

def train_valid_test_split(x, train=0.6, cluster_func=None, idfunc=None,
                           inc_test=True, seed=None):
  """ Split given list into 3 dataset: for training, validating,
  and testing.

  Parameters
  ----------
  x: list, tuple, numpy.ndarray
      the dataset.
  train: float (0.0 - 1.0)
      proportion used for training, validating and testing will be
      half of the remain.
  cluster_func: None or call-able
      organize data into cluster, then applying the same
      train_valid_test split strategy for each cluster.
  idfunc: None or call-able
      a function transform the task list into unique identity for splitting,
      `idfunc` has to return comparable keys.
  inc_test: bool
      if split a proportion of data for testing also.
  seed: int
      random seed to produce a re-producible results.
  """
  if seed is not None:
    rng = np.random.RandomState(seed)
  else:
    rng = get_rng()
  # ====== check input ====== #
  if isinstance(x, Mapping):
    x = x.items()
  elif isinstance(x, np.ndarray):
    x = x.tolist()
  # ====== check idfunc ====== #
  if not hasattr(idfunc, '__call__'):
    idfunc = None
  # ====== clustering ====== #
  if cluster_func is None:
    cluster_func = lambda x: 8 # lucky number
  if not hasattr(cluster_func, '__call__'):
    raise ValueError("'cluster_func' must be call-able or None.")
  clusters = defaultdict(list)
  for i in x:
    clusters[cluster_func(i)].append(i)
  # ====== applying data split for each cluster separately ====== #
  train_list, valid_list, test_list = [], [], []
  for name, clus in clusters.items():
    _1, _2, _3 = _split_list(clus, rng, train=train, idfunc=idfunc,
                             inc_test=inc_test)
    train_list += _1
    valid_list += _2
    test_list += _3
  # ====== return the results ====== #
  if inc_test:
    return train_list, valid_list, test_list
  return train_list, valid_list

def freqcount(x, key=None, count=1, normalize=False, sort=False,
              pretty_return=False):
  """ x: list, iterable

  Parameters
  ----------
  key: call-able
      extract the key from each item in the list
  count: call-able, int
      extract the count from each item in the list
  normalize: bool
      if normalize, all the values are normalized from 0. to 1. (
      which sum up to 1. in total).
  sort: boolean
      if True, the list will be sorted in ascent order.
  pretty_return: boolean
      if True, return pretty formatted text.

  Return
  ------
  dict: x(obj) -> freq(int)
  if `pretty_return` is `True`, return pretty formatted string.
  """
  freq = defaultdict(int)
  if key is None:
    key = lambda x: x
  if count is None:
    count = 1
  if isinstance(count, Number):
    _ = int(count)
    count = lambda x: _
  for i in x:
    c = count(i)
    i = key(i)
    freq[i] += c
  # always return the same order
  s = float(sum(v for v in freq.values()))
  freq = OrderedDict([(k, freq[k] / s if normalize else freq[k])
                      for k in sorted(freq.keys())])
  # check sort
  if sort:
    freq = OrderedDict(sorted(freq.items(), key=lambda x: x[1]))
  # check pretty return
  if pretty_return:
    s = ''
    for name, value in freq.items():
      s += ' %s: %d\n' % (ctext(name, 'yellow'), value)
    return s
  return freq

# ===========================================================================
# Bayesian
# ===========================================================================
def KL_divergence(P, Q):
  """ KL(P||Q) = ∑_i • p_i • log(p_i/q_i)
  The smaller this number, the better P match Q distribution
  """
  if isinstance(P, Mapping) and isinstance(Q, Mapping):
    keys = sorted(P.keys())
    P = [P[k] for k in keys]
    Q = [Q[k] for k in keys]
  # ====== normalize to probability 0-1 ====== #
  P = np.array(P)
  P = P / np.sum(P, axis=-1)
  Q = np.array(Q)
  Q = Q / np.sum(Q, axis=-1)
  # ====== calcuate the KL-div ====== #
  D = 0
  for pi, qi in zip(P, Q):
    D += pi * np.log(pi / qi)
  return D

# ===========================================================================
# Sampler
# ===========================================================================
def sampling_iter(it, k, p=None, return_iter=True, seed=5218,
                  progress_bar=None):
  """ Reservoir sampling, randomly choosing a sample of k items from a
  list S containing n items, where n is either a very large or unknown number.
  Typically n is large enough that the list doesn't fit into main memory.

  Parameters
  ----------
  it : iteration
    any instance of of iteration (i.e. `hasattr` '__iter__' )
  k : int
    number of sample for sampling
  p : {None, float (0.-1.)} (default: None)
    if `p` is None, perform reservoir sampling that guarantee every
    elements of the iteration is equally selected.
    if `p` is scalar, perform decayed sampling with `p` is the start
    probability (recommended in case `k` does not fit in the memory)
  return_iter : bool (default: True)
    if True, return an iteration of results instead of extracted list
  seed : int (default: 5218)
    random seed for reproducibility
  progress_bar : {None, odin.utils.Progbar.ProgBar}
  """
  k = int(k); assert k > 0
  if p is not None:
    p = float(p); assert 0. < p < 1.
  assert hasattr(it, '__iter__')
  # ====== check progress bar ====== #
  if progress_bar is not None:
    from odin.utils import Progbar
    assert isinstance(progress_bar, Progbar),\
    '`progress_bar` must be instance of odin.utils.progbar.Progbar, but given: %s' \
    % str(type(progress_bar))
  # ====== reservoir sampling ====== #
  if p is None:
    random.seed(seed)
    ret = []
    for i, x in enumerate(it):
      if i < k:
        ret.append(x)
      else:
        # as the iteration move forward,
        # the chance of picking new sample decrease
        r = random.randint(0, i)
        if r < k:
          ret[r] = x
      # update progress bar
      if progress_bar is not None:
        progress_bar.add(x)
    return tuple(ret)

  # ====== simulating the probability decay ====== #
  def _sampling():
    n = 0
    prob = p
    ret = []
    # this is compromise of randomness for speed
    n_buffer = 12000
    rand = np.random.RandomState(seed=seed)
    buffered_random = rand.rand(n_buffer)
    buffered_index = rand.randint(0, k, size=n_buffer, dtype=int)
    for i, x in enumerate(it):
      r = buffered_random[i % n_buffer]
      # initialize the reservoir
      if len(ret) < k:
        ret.append(x)
      # sample selected
      elif r < prob:
        yield x
        n += 1
      # update the reservoir
      else:
        ret[buffered_index[i % n_buffer]] = x
      # check break condition
      if n >= k:
        break
      # update the probability
      prob = 0.8 * prob + 0.2 / (i + 1)
      # update progress bar
      if progress_bar is not None:
        progress_bar.add(x)
    # return the rest of the samples to have enough k sample
    for i in range(k - n):
      yield ret[i]
  return _sampling() if return_iter else list(_sampling)

# ===========================================================================
# Statistics
# ===========================================================================
def sparsity_percentage(x, batch_size=5218):
  n_zeros = 0
  n_total = np.prod(x.shape)
  for start, end in batching(batch_size=batch_size, n=x.shape[0],
                             seed=None):
    y = x[start:end]
    n_nonzeros = np.count_nonzero(y)
    n_zeros += np.prod(y.shape) - n_nonzeros
  return n_zeros / n_total

def logVMR(x, axis=None, logged_values=False):
  """ Calculate the variance to mean ratio (VMR) in non-logspace
  (return answer in log-space)

  VMR (variance-to-mean ratio = index of dispersion) is equal to zero
  in the case of a constant random variable (not dispersed).

  It is equal to one in a Poisson distribution, higher than one in a
  negative binomial distribution (over dispersed) and between zero
  and one in a binomial distribution (under dispersed).

  Reference
  ---------
  https://www.quantshare.com/item-1029-index-of-dispersion-variance-to-mean-ratio-vmr

  """
  if logged_values:
    x = np.expm1(x)
  return np.log1p(np.var(x, axis=axis) / np.mean(x, axis=axis))

# ===========================================================================
# Diagnosis
# ===========================================================================
def classification_diagnose(X, y_true, y_pred,
                            num_samples=8, return_list=False, top_n=None,
                            seed=5218):
  """
  Return
  ------
  OrderedDict: (true_class, pred_class) -> [list of samples from `X`]
  sorted by most frequence to less frequence
  """
  # Mapping from classesID -> [Sample ID]
  # ====== check argument ====== #
  if y_true.ndim == 2:
    y_true = np.argmax(y_true, axis=-1)
  elif y_true.ndim != 1:
    raise ValueError("Only support 1-D or 2-D one-hot `y_true`, "
                     "given `y_true` with shape: %s" % str(y_true.shape))
  if y_pred.ndim == 2:
    y_pred = np.argmax(y_pred, axis=-1)
  elif y_pred.ndim != 1:
    raise ValueError("Only support 1-D or 2-D one-hot `y_pred`, "
                     "given `y_pred` with shape: %s" % str(y_pred.shape))
  assert len(y_true) == len(y_pred) == len(X), "Inconsistent number of samples"
  # ====== initialize ====== #
  rand = np.random.RandomState(seed)
  miss = defaultdict(list)
  # ====== sampling ====== #
  for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
    if true != pred:
      miss[(true, pred)].append(idx)
  # sort by the most freq mistake classes
  outputs = OrderedDict()
  for (true, pred), samples in sorted(miss.items(),
                                      key=lambda x: len(x[-1]),
                                      reverse=True):
    rand.shuffle(samples)
    outputs[(true, pred)] = (X[samples[:num_samples]]
                             if isinstance(X, np.ndarray) else
                             [X[i] for i in samples])
  # select top frequence
  if top_n is not None and isinstance(top_n, Number):
    top_n = int(top_n)
    assert top_n >= 1
    outputs = OrderedDict(list(outputs.items())[:top_n])
  return list(outputs.items()) if return_list else outputs

def classification_report(y_pred, y_true, labels):
  """
  Parameters
  ----------
  pass

  Return
  ------
  Classification report in form of string
  """
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  # ====== validate labels ====== #
  labels = as_tuple(labels)
  target_names = [str(i) for i in labels]
  labels = list(range(0, len(labels)))
  # ====== create report ====== #
  s = ""
  s += "Accuracy: %f\n" % accuracy_score(y_true, y_pred, normalize=True)
  s += "Confusion matrix:\n"
  s += str(confusion_matrix(y_true, y_pred, labels=labels)) + '\n'
  s += "Report:\n"
  s += str(classification_report(y_true, y_pred, labels=labels, digits=3,
                                 target_names=target_names))
  return s

def summary(x, axis=None, shorten=False):
  """ Return string of statistical summary given series `x`
    {#:%s|mi:%s|q1:%s|md:%s|mn:%s|q3:%s|ma:%s|sd:%s}
  """
  if isinstance(x, Iterator):
    x = list(x)
  if isinstance(x, (tuple, list, set)):
    x = np.array(x)
  mean, std = np.mean(x, axis=axis), np.std(x, axis=axis)
  median = np.median(x, axis=axis)
  qu1, qu3 = np.percentile(x, [25, 75], axis=axis)
  min_, max_ = np.min(x, axis=axis), np.max(x, axis=axis)
  s = ""
  if not shorten:
    x = x.ravel()
    samples = ', '.join([str(i)
           for i in np.random.choice(x, size=min(8, len(x)), replace=False).tolist()])
    s += "***** Summary *****\n"
    s += "    Min : %s\n" % str(min_)
    s += "1st Qu. : %s\n" % str(qu1)
    s += " Median : %s\n" % str(median)
    s += "   Mean : %g\n" % mean
    s += "3rd Qu. : %s\n" % str(qu3)
    s += "    Max : %s\n" % str(max_)
    s += "-------------------\n"
    s += "    Std : %g\n" % std
    s += "#Samples: %d\n" % len(x)
    s += "Samples : %s\n" % samples
    s += "Sparsity: %.4f\n" % sparsity_percentage(x)
  else:
    s += "{#:%s|mi:%s|q1:%s|md:%s|mn:%s|q3:%s|ma:%s|sd:%s}" %\
    (ctext(len(x), 'cyan'),
     ctext('%g' % min_, 'cyan'),
     ctext('%g' % qu1, 'cyan'),
     ctext('%g' % median, 'cyan'),
     ctext('%g' % mean, 'cyan'),
     ctext('%g' % qu3, 'cyan'),
     ctext('%g' % max_, 'cyan'),
     ctext('%g' % std, 'cyan'))
  return s

describe = summary
