from __future__ import print_function, division, absolute_import

import sys
import inspect
from six import string_types
from collections import OrderedDict

import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from odin.nnet.base import Dense
from odin.utils import string_normalize, ctext
from odin.backend.activations import softplus_inverse
from odin.backend.helpers import is_tensor, get_epsilon
from odin.bay.zero_inflated import ZeroInflated
# ===========================================================================
# Helpers
# ===========================================================================
def _support(y, info, print_log, padding, **kwargs):
  eps = kwargs.get('eps', None)
  if eps is None:
    eps = get_epsilon(y)
  support = info.get('support', None)
  if support is not None:
    y_min, y_max = support
    # print out log
    if print_log:
      print(padding + "   support: [%s, %s, eps=%s]" %
        (str(y_min), str(y_max), str(eps)))
    return tf.clip_by_value(y, y_min + eps, y_max - eps)
  return y

def _network(x, out_dim, name, info,
             print_log, padding):
  network = info.get('network', Dense)
  # create the network first
  if isinstance(network, type):
    params = inspect.signature(network.__init__).parameters
    if 'name' in params or any(p.kind == inspect.Parameter.VAR_KEYWORD
                               for p in params.values()):
      network = network(out_dim, name=name)
    else:
      network = network(out_dim)
  # applying on the input
  if callable(network):
    # print out log
    if print_log:
      print(padding + "   network: %s" % network.__class__.__name__)
    y = network(x)
  else:
    raise RuntimeError("Type %s is not callable" % type(network))
  return y

def _activate(y, info, print_log, padding, **kwargs):
  fn = info.get('fn', lambda x: x)
  args = []
  args_name = []
  for p_name, p_val in inspect.signature(fn).parameters.items():
    if p_name in kwargs:
      p_val = kwargs[p_name]
    else:
      p_val = p_val.default
    args.append(p_val)
    args_name.append(p_name)
  args[0] = y

  for a_name, a_val in zip(args_name, args):
    if a_val == inspect.Parameter.empty:
      raise RuntimeError("Cannot extract value for argument name: '%s'" % a_name)
  # print out log
  if print_log:
    print(padding + "   activation: <%s>(%s)" % (
        fn.__name__,
        '; '.join([ctext(i, 'cyan') + ':' + str(j)
                 for i, j in zip(args_name, args)])))
  y = fn(*args)
  return y

def _parse_parameter(x, out_dim,
                     name, info,
                     print_log, padding,
                     **kwargs):
  # ====== parsing distribution ====== #
  if isinstance(info, DistributionDescription):
    if print_log:
      print(padding + "   Info:", 'DistributionDescription')
    y = info.set_print_log(print_log
      ).set_padding_log('    '
      )(x, out_dim, **kwargs)
  # ====== parsing network ====== #
  elif isinstance(info, dict):
    if print_log:
      print(padding + "   Info:", str(info))
    y = _network(x, out_dim, name, info, print_log, padding)
    y = _support(y, info, print_log, padding, **kwargs)
    y = _activate(y, info, print_log, padding, **kwargs)
  # ====== just tensor ====== #
  else:
    if print_log:
      print(padding + "   Info:", 'Tensor')
    y = tf.convert_to_tensor(info)
  if print_log:
    print(padding + "   Output:", ctext(y, 'cyan'))
  return y

# ===========================================================================
# Main
# ===========================================================================
def get_described_distributions():
  all_distributions = []
  for name, member in inspect.getmembers(sys.modules[__name__]):
    if isinstance(member, DistributionDescription):
      all_distributions.append(member)
  return sorted(all_distributions, key=lambda x: x.normalized_name)

def get_distribution_description(dist_name):
  dist_name = string_normalize(dist_name, lower = True,
                               remove_non_alphanumeric=True,
                               remove_whitespace='')
  dist_name = dist_name.replace('_', '')
  dists = get_described_distributions()
  matches = []
  # match exact
  for dist in dists:
    if dist.normalized_name == dist_name:
      matches.append(dist)
  # match contain
  if len(matches) == 0:
    for dist in dists:
      if dist_name == dist.normalized_name[:len(dist_name)]:
        matches.append(dist)
  # check
  if len(matches) == 0:
    raise RuntimeError("Cannot find distribution with name: '%s', all distributions are: %s" %
      (dist_name, ', '.join([i.normalized_name for i in dists])))
  elif len(matches) > 1:
    raise RuntimeError("Found multiple distribution for name: '%s' which are: %s" %
      (dist_name, ', '.join([i.normalized_name for i in matches])))
  # final
  match = matches[0]
  return match

# ===========================================================================
# Base
# ===========================================================================
class DistributionDescription(object):

  def __init__(self, distribution, **kwargs):
    if not issubclass(distribution, tfd.Distribution):
      raise ValueError("'distribution' attribute must be subclass of %s but given %s"
        % (str(tfd.Distribution), str(distribution)))
    self.distribution = distribution
    self.kwargs = kwargs
    self._print_log = False
    self._padding = ''

  def set_print_log(self, print_log):
    self._print_log = bool(print_log)
    return self

  def set_padding_log(self, padding):
    self._padding = str(padding)
    return self

  def __call__(self, x=None, out_dim=None, n_eventdim=0, **kwargs):
    n_eventdim = int(n_eventdim)
    if x is not None:
      x = tf.convert_to_tensor(x)
    if out_dim is not None:
      out_dim = int(out_dim)
    print_log = self._print_log
    padding = self._padding

    if print_log:
      print(padding + ctext("Parsing distribution:", 'lightyellow'),
        '%s/%s' % (ctext(self.normalized_name, 'lightcyan'),
                   ctext(self.distribution.__name__, 'cyan')))

    args = {}
    for p_name, p_val in self.get_ordered_arguments():
      if print_log:
        print(padding + " Parsing parameter:", ctext(p_name, 'cyan'))
      p_val = _parse_parameter(x, out_dim,
                               p_name, p_val,
                               print_log=print_log,
                               padding=padding,
                               **kwargs)
      args[p_name] = p_val

    dist = self.distribution(**args)
    if n_eventdim > 0:
      dist = tfd.Independent(distribution=dist,
                             reinterpreted_batch_ndims=n_eventdim)
    if print_log:
      print(' Distribution:', ctext(dist, 'cyan'))
    self._print_log = False
    self._padding = ''
    return dist

  @property
  def normalized_name(self):
    for name, _ in globals().items():
      if isinstance(_, DistributionDescription) and \
      id(_) == id(self):
        name = string_normalize(name, lower=True,
                                remove_whitespace='',
                                remove_non_alphanumeric=True)
        return name

  def get_ordered_arguments(self):
    args = []
    spec = inspect.signature(self.distribution.__init__)
    # without `self` argument
    for i, (p_name, p_val) in enumerate(spec.parameters.items()):
      if i == 0:
        continue
      if (p_val.default == inspect.Parameter.empty or
      p_val.default is None) and p_name in self.kwargs:
        p_val = self.kwargs[p_name]
        args.append((p_name, p_val))
    return args

  def copy(self, distribution=None, **kwargs):
    new_kwargs = dict(self.kwargs)
    new_kwargs.update(kwargs)
    return DistributionDescription(
        distribution=self.distribution if distribution is None else distribution,
        **new_kwargs)

  def __str__(self):
    s = ["Distribution: %s/%s" % (ctext(self.normalized_name, 'lightcyan'),
                                  ctext(self.distribution.__name__, 'cyan'))]
    for name, val in self.get_ordered_arguments():
      if isinstance(val, DistributionDescription):
        s.append('    %s: ' % name)
        s.append('\n'.join(['      ' + line for line in str(val).split('\n')]))
      else:
        s.append('    %s: %s' % (name, ctext(val, 'cyan')))
    return '\n'.join(s)

# ===========================================================================
# Normal distributions
# ===========================================================================
Normal = DistributionDescription(
    tfd.Normal,
    loc=dict(),
    scale=dict(support=None,
               fn=lambda x: tf.nn.softplus(x + softplus_inverse(1.0)))
)
Normal1 = Normal.copy(scale=1)
Normal01 = Normal.copy(loc=tf.zeros(shape=[1]),
                       scale=tf.ones(shape=[1]))
# ===========================================================================
# Bernoulli
# ===========================================================================
Bernoulli = DistributionDescription(
    tfd.Bernoulli,
    logits=dict()
)
ZeroInflatedBernoulli = Bernoulli.copy(
    ZeroInflated,
    dist=Bernoulli,
    pi=dict(support=[0, 1],
            fn=tf.nn.sigmoid)
)
Categorical = DistributionDescription(
    tfd.Categorical,
    logits=dict(support=None, fn=tf.identity)
)
# ===========================================================================
# Relaxed discrete distribution
# ===========================================================================
GumbelSoftmax = DistributionDescription(
    tfd.RelaxedOneHotCategorical,
    temperature=0.5,
    logits=dict(support=None, fn=tf.identity)
)
GumbelSigmoid = DistributionDescription(
    tfd.RelaxedBernoulli,
    temperature=0.5,
    logits=dict(support=None, fn=tf.identity)
)
# ===========================================================================
# Poisson
# ===========================================================================
Poisson = DistributionDescription(
    tfd.Poisson,
    log_rate=dict(support=None,
                  fn=tf.identity)
)
ConstrainedPoisson = DistributionDescription(
    tfd.Poisson,
    rate=dict(support=[0, 1],
              fn=lambda logit, N: tf.nn.softmax(logit) * N)
)
ZeroInflatedPoisson = Poisson.copy(
    ZeroInflated,
    dist=Poisson,
    pi=dict(support=[0, 1],
            fn=tf.nn.sigmoid)
)
# ===========================================================================
# Negative Binomial
# ===========================================================================
NegativeBinomial = DistributionDescription(
    tfd.NegativeBinomial,
    total_count=dict(support=[-10, 10],
                     fn=lambda x: tf.exp(x)),
    logits=dict(support=None,
                fn=tf.identity),
)
ZeroInflatedNegativeBinomial = NegativeBinomial.copy(
    ZeroInflated,
    dist=NegativeBinomial,
    pi=dict(support=[0, 1],
            fn=tf.nn.sigmoid)
)
zinb = ZeroInflatedNegativeBinomial.copy()
