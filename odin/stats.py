# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numbers import Number
from itertools import chain
from collections import defaultdict, Iterator, OrderedDict, Mapping

import numpy as np

from odin.config import get_rng
from odin.utils.math_utils import interp
from odin.utils import as_tuple, flatten_list, ctext

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


def summary(x, axis=None, shorten=False):
  if isinstance(x, Iterator):
    x = list(x)
  if isinstance(x, (tuple, list)):
    x = np.array(x)
  mean, std = np.mean(x, axis=axis), np.std(x, axis=axis)
  median = np.median(x, axis=axis)
  qu1, qu3 = np.percentile(x, [25, 75], axis=axis)
  min_, max_ = np.min(x, axis=axis), np.max(x, axis=axis)
  samples = ', '.join([str(i)
             for i in np.random.choice(x.ravel(), size=8, replace=False).tolist()])
  s = ""
  if not shorten:
    s += "***** Summary *****\n"
    s += "    Min : %s\n" % str(min_)
    s += "1st Qu. : %s\n" % str(qu1)
    s += " Median : %s\n" % str(median)
    s += "   Mean : %.8f\n" % mean
    s += "3rd Qu. : %s\n" % str(qu3)
    s += "    Max : %s\n" % str(max_)
    s += "-------------------\n"
    s += "    Std : %.8f\n" % std
    s += "#Samples : %d\n" % len(x)
    s += "Samples : %s\n" % samples
  else:
    s += "{#:%d|min:%s|qu1:%s|med:%s|mea:%.8f|qu3:%s|max:%s|std:%.8f}" %\
    (len(x), str(min_), str(qu1), str(median), mean, str(qu3), str(max_), std)
  return s


def KLdivergence(P, Q):
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
