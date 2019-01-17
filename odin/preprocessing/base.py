from __future__ import print_function, division, absolute_import

import os
import re
import inspect
from enum import Enum

from six import add_metaclass, string_types
from abc import ABCMeta, abstractmethod
from collections import Mapping

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline as _make_pipeline

from odin.fuel import Dataset
from odin.preprocessing.signal import delta, mvn, stack_frames
from odin.utils import (get_all_files, is_string, as_tuple, is_pickleable,
                        ctext, flatten_list, dummy_formatter,
                        get_formatted_datetime)

class ExtractorSignal(object):
  """ ExtractorSignal """

  def __init__(self):
    super(ExtractorSignal, self).__init__()
    self._timestamp = get_formatted_datetime(only_number=False)
    self._extractor = None
    self._msg = ''
    self._action = 'ignore'
    self._last_input = {}

  @property
  def message(self):
    return self._msg

  @property
  def action(self):
    return self._action

  def set_message(self, extractor, msg, last_input):
    if self._extractor is not None:
      raise RuntimeError(
          "This signal has stored message, cannot set message twice.")
    assert isinstance(extractor, Extractor), \
    '`extractor` must be instance of odin.preprocessing.base.Extractor, ' +\
    'but given type: %s' % str(type(extractor))
    self._extractor = extractor
    self._msg = str(msg)
    self._last_input = last_input
    return self

  def set_action(self, action):
    action = str(action).lower()
    assert action in ('warn', 'error', 'ignore'), \
    "`action` can be one of the following values: 'warn', 'error', 'ignore'; " + \
    "but given: %s" % action
    self._action = action
    return self

  def __str__(self):
    if self._extractor is None:
      raise RuntimeError("The Signal has not been configured by the Extractor")
    s = '[%s]' % self._timestamp
    s += '%s' % ctext(self._extractor.__class__.__name__, 'cyan') + '\n'
    s += 'Error message: "%s"' % ctext(self._msg, 'yellow') + '\n'
    s += 'Action: "%s"' % ctext(self._action, 'yellow') + '\n'
    # last input
    s += 'Last input: \n'
    if isinstance(self._last_input, Mapping):
      for k, v in sorted(self._last_input.items(),
                         key=lambda x: x[0]):
        s += '  %s: %s\n' % (ctext(str(k), 'yellow'), dummy_formatter(v))
    else:
      s += '  Type: %s\n' % ctext(type(self._last_input), 'yellow')
      s += '  Object: %s\n' % ctext(str(self._last_input), 'yellow')
    # parameters
    s += 'Attributes: \n'
    s += '  ' + ctext('InputLayer', 'yellow') + ': ' + str(self._extractor.is_input_layer) + '\n'
    s += '  ' + ctext('RobustLevel', 'yellow') + ': ' + self._extractor.robust_level + '\n'
    s += '  ' + ctext('InputName', 'yellow') + ': ' + str(self._extractor.input_name) + '\n'
    s += '  ' + ctext('OutputName', 'yellow') + ': ' + str(self._extractor.output_name) + '\n'
    for name, param in self._extractor.get_params().items():
      if name not in ('_input_name', '_output_name'):
        s += '  ' + ctext(name, 'yellow') + ': ' + dummy_formatter(param) + '\n'
    return s

# ===========================================================================
# Helper
# ===========================================================================
def make_pipeline(steps, debug=False):
  """ NOTE: this method automatically revmove None entries

   - Flatten list or dictionary found in steps.
   - Remove any object that not is instance of `Extractor`

  during creation of `Pipeline`.
  """
  ID = [0]

  def item2step(x):
    if isinstance(x, (tuple, list)):
      if len(x) == 1 and isinstance(x[0], Extractor):
        x = x[0]
        ID[0] += 1
        return (x.__class__.__name__ + str(ID[0]), x)
      elif len(x) == 2:
        if is_string(x[0]) and isinstance(x[1], Extractor):
          return x
        elif is_string(x[1]) and isinstance(x[0], Extractor):
          return (x[1], x[0])
    elif isinstance(x, Extractor):
      ID[0] += 1
      return (x.__class__.__name__ + str(ID[0]), x)
    return None

  if isinstance(steps, Mapping):
    steps = steps.items()
  elif not isinstance(steps, (tuple, list)):
    steps = [steps]
  steps = [item2step(i) for i in steps]
  # remove None
  steps = [s for s in steps if s is not None]
  if len(steps) == 0:
    raise ValueError("No instance of odin.preprocessing.base.Extractor found in `steps`.")
  # ====== set debug mode ====== #
  set_extractor_debug([i[1] for i in steps],
                      debug=bool(debug))
  # ====== return pipeline ====== #
  ret = Pipeline(steps=steps)
  return ret

def set_extractor_debug(extractors, debug):
  # ====== prepare ====== #
  if isinstance(extractors, (tuple, list)):
    extractors = [i for i in flatten_list(extractors)
                  if isinstance(i, Extractor)]
  elif isinstance(extractors, Pipeline):
    extractors = [i[-1] for i in extractors.steps]
  elif isinstance(extractors, Mapping):
    extractors = [i[-1] for i in extractors.items()]
  else:
    raise ValueError("No support for `extractors` type: %s" % type(extractors))
  # ====== set the value ====== #
  for i in extractors:
    i._debug = bool(debug)
  return extractors

def _equal_inputs_outputs(x, y):
  try:
    if x != y:
      return False
  except Exception:
    pass
  return True

def _preprocess(x):
  if isinstance(x, np.str_):
    x = str(x)
  return x

# ===========================================================================
# Basic extractors
# ===========================================================================
@add_metaclass(ABCMeta)
class Extractor(BaseEstimator, TransformerMixin):
  """ Extractor

  The developer must override the `_transform` method:
   - If the return is instance of `collections.Mapping`, the new features
     will be merged into existed features dictionary (i.e. the input dictionary)
   - If the return is not instance of `Mapping`, the name of given classes will
     be used to name the returned features.
   - If `None` is returned, no `_transform` is called, just return None for
     the whole pipeline (i.e. None act as terminal signal)

  Parameters
  ----------
  input_name : {None, string, list of string}
    list of string represent the name of feature
  output_name : {None, string, list of string}
    default name for the output feature (in case the return is not
    instance of dictionary);
    If `input_name` is None and `output_name` is None, use lower
    case of class name as default
  is_input_layer : bool (default: False)
    An input layer accept any type of input to `transform`,
    otherwise, only accept a dictionary type as input.
  robust_level : {'ignore', 'warn', 'error'}
    'ignore' - ignore error files
    'warn' - warn about error file during processing
    'error' - raise Exception and stop processing
  """

  def __init__(self, input_name=None, output_name=None,
               is_input_layer=False, robust_level='ignore'):
    super(Extractor, self).__init__()
    self._debug = False
    self._is_input_layer = bool(is_input_layer)
    self._last_debugging_text = ''
    # ====== robust level ====== #
    robust_level = str(robust_level).lower()
    assert robust_level in ('ignore', 'warn', 'error'),\
    "`robust_level` can be one of the following values: " + \
    "'warn', 'error', 'ignore'; but given: %s" % robust_level
    self._robust_level = robust_level
    # ====== check input_name ====== #
    if input_name is None:
      pass
    elif isinstance(input_name, string_types):
      pass
    elif hasattr(input_name, '__iter__'):
      input_name = tuple([str(i).lower() for i in input_name])
    else:
      raise ValueError("No support for `input_name` type: %s" % str(type(input_name)))
    self._input_name = input_name
    # ====== check output_name ====== #
    if output_name is None:
      if input_name is None:
        output_name = self.__class__.__name__.lower()
      else:
        output_name = input_name
    elif isinstance(output_name, string_types):
      pass
    elif hasattr(output_name, '__iter__'):
      output_name = tuple([str(i).lower() for i in output_name])
    else:
      raise ValueError("No support for `output_name` type: %s" % str(type(output_name)))
    self._output_name = output_name

  @property
  def last_debugging_text(self):
    """ Return the last debugging information recorded during
    calling the `transform` method with `debug=True` """
    if not hasattr(self, '_last_debugging_text'):
      self._last_debugging_text = ''
    return self._last_debugging_text

  @property
  def input_name(self):
    return self._input_name

  @property
  def output_name(self):
    return self._output_name

  @property
  def is_input_layer(self):
    return self._is_input_layer

  @property
  def robust_level(self):
    return self._robust_level

  def set_debug(self, debug):
    self._debug = bool(debug)
    return self

  def fit(self, X, y=None):
    # Do nothing here
    return self

  @abstractmethod
  def _transform(self, X):
    raise NotImplementedError

  def transform(self, X):
    # NOTE: do not override this method
    if isinstance(X, ExtractorSignal):
      return X
    # ====== interpret different signal ====== #
    if X is None:
      return ExtractorSignal(
      ).set_message(extractor=self,
                    msg="`None` value is returned by extractor",
                    last_input=X
      ).set_action(self.robust_level)
    # ====== input layer ====== #
    if not self.is_input_layer and not isinstance(X, Mapping):
      err_msg = "the input to `Extractor.transform` must be instance of dictionary, " + \
      "but given type: %s" % str(type(X))
      return ExtractorSignal(
      ).set_message(extractor=self,
                    msg=err_msg,
                    last_input=X
      ).set_action(self.robust_level)
    # ====== the transformation ====== #
    if self.input_name is not None and isinstance(X, Mapping):
      for name in as_tuple(self.input_name, t=string_types):
        if name not in X:
          return ExtractorSignal(
          ).set_message(extractor=self,
                        msg="Cannot find features with name: %s" % name,
                        last_input=X
          ).set_action('error')
    y = self._transform(X)
    # if return Signal or None, no post-processing
    if isinstance(y, ExtractorSignal):
      return y
    if y is None:
      return ExtractorSignal(
      ).set_message(extractor=self,
                    msg="`None` value is returned by the extractor: %s" % self.__class__.__name__,
                    last_input=X
      ).set_action(self.robust_level)
    # ====== return type must always be a dictionary ====== #
    if not isinstance(y, Mapping):
      if isinstance(y, (tuple, list)):
        y = {i: j
             for i, j in zip(as_tuple(self.output_name, t=string_types),
                             y)}
      else:
        y = {self.output_name: y}
    # ====== Merge previous results ====== #
    # remove None values
    tmp = {}
    for name, feat in y.items():
      if any(c.isupper() for c in name):
        return ExtractorSignal(
        ).set_message(extractor=self,
                      msg="Name for features cannot contain upper case",
                      last_input=X
        ).set_action('error')
      if feat is None:
        continue
      tmp[name] = feat
    y = tmp
    # add old features extracted in X, but do NOT override new features in y
    if isinstance(X, Mapping):
      for name, feat in X.items():
        if any(c.isupper() for c in name):
          return ExtractorSignal(
          ).set_message(extractor=self,
                        msg="Name for features cannot contain upper case",
                        last_input=X
          ).set_action('error')
        if name not in y:
          y[name] = _preprocess(feat)
    # ====== print debug text ====== #
    # maybe someone implement __getstate__ and forget _debug
    if not hasattr(self, '_debug'):
      self._debug = False
    if self._debug:
      debug_text = ''
      debug_text += '%s %s\n' % (ctext("[Extractor]", 'cyan'),
                                 ctext(self.__class__.__name__, 'magenta'))
      # inputs
      if not _equal_inputs_outputs(X, y):
        debug_text += '  %s\n' % ctext("Inputs:", 'yellow')
        debug_text += '  %s\n' % ctext("-------", 'yellow')
        if isinstance(X, Mapping):
          for k, v in X.items():
            debug_text += '    %s : %s\n' % (ctext(k, 'blue'), dummy_formatter(v))
        else:
          debug_text += '    %s\n' % dummy_formatter(X)
      # outputs
      debug_text += '  %s\n' % ctext("Outputs:", 'yellow')
      debug_text += '  %s\n' % ctext("-------", 'yellow')
      if isinstance(y, Mapping):
        for k, v in y.items():
          debug_text += '    %s : %s\n' % (ctext(k, 'blue'), dummy_formatter(v))
      else:
        debug_text += '    %s\n' % dummy_formatter(y)
      # parameters
      for name, param in self.get_params().items():
        if name not in ('_input_name',
                        '_output_name'):
          debug_text += '  %s : %s\n' % (ctext(name, 'yellow'), dummy_formatter(param))
      self._last_debugging_text = debug_text
      print(debug_text)
    return y

# ===========================================================================
# General extractor
# ===========================================================================
class Converter(Extractor):

  """ Convert the value under `input_name` to a new value
  using `converter` function, and save the new value to
  the `output_name`.

  This could be mapping 1 -> 1 or many -> 1; in case of
  many to 1 mapping, the `converter` function will be
  call as `converter(*args)`

  Parameters
  ----------
  converter: {Mapping, call-able}
      convert `inputs['name'] = converter(inputs[keys])`

  """

  def __init__(self, converter, input_name='name', output_name='name'):
    super(Converter, self).__init__(input_name=as_tuple(input_name, t=string_types),
                                    output_name=str(output_name))
    # ====== check converter ====== #
    if not hasattr(converter, '__call__') and \
    not isinstance(converter, Mapping):
      raise ValueError("`converter` must be call-able.")
    # converter can be function or dictionary
    self.converter = converter

  def _transform(self, feat):
    X = [feat[name] for name in self.input_name]
    if hasattr(self.converter, '__call__'):
      name = self.converter(*X)
    else:
      name = self.converter[X[0] if len(X) == 1 else X]
    return {self.output_name: name}

class DeltaExtractor(Extractor):

  """ Extracting the delta coefficients given the axis

  Parameters
  ----------
  input_name : list of str
      list of all features name for calculating the delta
  width : int
      amount of frames taken into account for 1 delta
  order : list of int
      list of all delta order will be concatenate (NOTE: keep `0` in
      the list if you want to keep original features)
  axis : int
      which dimension calculating the delta
      (suggest time-dimension for acoustic features, i.e. axis=0)
  """

  def __init__(self, input_name, output_name=None,
               width=9, order=(0, 1), axis=0):
    super(DeltaExtractor, self).__init__(
        input_name=as_tuple(input_name, t=string_types),
        output_name=output_name)
    # ====== check width ====== #
    width = int(width)
    if width % 2 == 0 or width < 3:
      raise ValueError("`width` must be odd integer >= 3, give value: %d" % width)
    self.width = width
    # ====== check order ====== #
    self.order = as_tuple(order, t=int)
    # ====== axis ====== #
    self.axis = axis

  def _calc_deltas(self, X):
    all_deltas = delta(data=X, width=self.width, order=max(self.order), axis=self.axis)
    if not isinstance(all_deltas, (tuple, list)):
      all_deltas = (all_deltas,)
    else:
      all_deltas = tuple(all_deltas)
    all_deltas = (X,) + all_deltas
    all_deltas = tuple([d for i, d in enumerate(all_deltas)
                        if i in self.order])
    return np.concatenate(all_deltas, axis=-1)

  def _transform(self, feat):
    return [self._calc_deltas(feat[name])
            for name in self.input_name]

class EqualizeShape0(Extractor):
  """ EqualizeShape0
  The final length of all features is the `minimum length`.

  This Extractor shrink the shape of all given features in `feat_name`
  to the same length.

  Raise Error if given files is shorted than desire length

  Parameters
  ----------
  input_name: {None, list of string}
      list of features name will be used for calculating the
      running statistics.
      If None, calculate the statistics for all `numpy.ndarray`
  shrink_mode: 'center', 'left', 'right'
       center: remove data points from both left and right
       left: remove data points at the beginning (left)
       right: remove data points at the end (right)
  """

  def __init__(self, input_name=None, shrink_mode='right'):
    super(EqualizeShape0, self).__init__(
        input_name=as_tuple(input_name, t=string_types) if input_name is not None else None)
    shrink_mode = str(shrink_mode).lower()
    if shrink_mode not in ('center', 'left', 'right'):
      raise ValueError("shrink mode support include: center, left, right")
    self.shrink_mode = shrink_mode

  def _transform(self, feat):
    if self.input_name is None:
      X = []
      output_name = []
      for key, val in feat.items():
        if isinstance(val, np.ndarray) and val.ndim > 0:
          X.append(val)
          output_name.append(key)
    else:
      X = [feat[name] for name in self.input_name]
      output_name = self.input_name
    # ====== searching for desire length ====== #
    n = min(i.shape[0] for i in X)
    # ====== equalize ====== #
    equalized = {}
    for name, y in zip(output_name, X):
      # cut the features in left and right
      # if the shape[0] is longer
      if y.shape[0] != n:
        diff = y.shape[0] - n
        if diff < 0:
          print("Feature length: %d which is smaller "
                "than desire length: %d, feature name is '%s'" %
                (y.shape[0], n, X['name']))
          return None
        elif diff > 0:
          if self.shrink_mode == 'center':
            diff_left = diff // 2
            diff_right = diff - diff_left
            y = y[diff_left:-diff_right]
          elif self.shrink_mode == 'right':
            y = y[:-diff]
          elif self.shrink_mode == 'left':
            y = y[diff:]
      equalized[name] = y
    return equalized

class RunningStatistics(Extractor):
  """ Running statistics

  Parameters
  ----------
  input_name: {string, list of string}
      list of features name will be used for calculating the
      running statistics.
      If None, calculate the statistics for all `numpy.ndarray`
      with `ndim` > 0
  axis : int (default: 0)
      the axis for calculating the statistics
  prefix : ''
      the prefix append to 'sum1' and 'sum2'
  """

  def __init__(self, input_name=None, axis=0, prefix=''):
    super(RunningStatistics, self).__init__(
        input_name=as_tuple(input_name, t=string_types) if input_name is not None else None)
    self.axis = axis
    self.prefix = str(prefix)

  def get_sum1_name(self, feat_name):
    return '%s_%ssum1' % (feat_name, self.prefix)

  def get_sum2_name(self, feat_name):
    return '%s_%ssum2' % (feat_name, self.prefix)

  def _transform(self, feat):
    if self.input_name is None:
      X = []
      output_name = []
      for key, val in feat.items():
        if isinstance(val, np.ndarray) and val.ndim > 0:
          X.append(val)
          output_name.append(key)
    else:
      X = [feat[name] for name in self.input_name]
      output_name = self.input_name
    # ====== calculate the statistics ====== #
    for name, y in zip(output_name, X):
      # ====== SUM of x^1 ====== #
      sum1 = np.sum(y, axis=self.axis, dtype='float64')
      s1_name = self.get_sum1_name(name)
      if s1_name not in feat:
        feat[s1_name] = sum1
      else:
        feat[s1_name] += sum1
      # ====== SUM of x^2 ====== #
      sum2 = np.sum(np.power(y, 2), axis=self.axis, dtype='float64')
      s2_name = self.get_sum2_name(name)
      if s2_name not in feat:
        feat[s2_name] = sum2
      else:
        feat[s2_name] += sum2
    return feat

class AsType(Extractor):
  """ An extractor convert given features to given types

  Parameters
  ----------
  dtype : {string, numpy.dtype}
      desire type
  input_name: {string, list of string}
      list of features name will be used for calculating the
      running statistics.
      If None, calculate the statistics for all object `hasattr`
      'astype'
  exclude_pattern : {string, None}
      regular expression pattern to exclude all features with given
      name pattern, only used when `input_name=None`.
      By default, exclude all running statistics '*_sum1' and '*_sum2'
  """

  def __init__(self, dtype, input_name=None, exclude_pattern=".+\_sum[1|2]"):
    super(AsType, self).__init__(
        input_name=as_tuple(input_name, t=string_types) if input_name is not None else None)
    self.dtype = np.dtype(dtype)
    if isinstance(exclude_pattern, string_types):
      exclude_pattern = re.compile(exclude_pattern)
    else:
      exclude_pattern = None
    self.exclude_pattern = exclude_pattern

  def _transform(self, feat):
    # ====== preprocessing ====== #
    if self.input_name is None:
      X = []
      output_name = []
      for key, val in feat.items():
        if hasattr(val, 'astype'):
          if self.exclude_pattern is not None and \
          self.exclude_pattern.search(key):
            continue
          X.append(val)
          output_name.append(key)
    else:
      X = [feat[name] for name in self.input_name]
      output_name = self.input_name
    # ====== astype ====== #
    updates = {}
    for name, y in zip(output_name, X):
      updates[name] = y.astype(self.dtype)
    return updates

class DuplicateFeatures(Extractor):

  def __init__(self, input_name, output_name):
    super(DuplicateFeatures, self).__init__(
        input_name=as_tuple(input_name, t=string_types),
        output_name=as_tuple(output_name, t=string_types))

  def _transform(self, feat):
    return {out_name: feat[in_name]
            for in_name, out_name in zip(self.input_name, self.output_name)}

class RenameFeatures(Extractor):

  def __init__(self, input_name, output_name):
    super(RenameFeatures, self).__init__(
        input_name=as_tuple(input_name, t=string_types),
        output_name=as_tuple(output_name, t=string_types))

  def _transform(self, X):
    return X

  def transform(self, feat):
    if isinstance(feat, Mapping):
      for old_name, new_name in zip(self.input_name, self.output_name):
        if old_name in feat: # only remove if it exist
          X = feat[old_name]
          del feat[old_name]
          feat[new_name] = X
    return feat

class DeleteFeatures(Extractor):
  """ Remove features by name from extracted features dictionary """

  def __init__(self, input_name):
    super(DeleteFeatures, self).__init__()
    self._name = as_tuple(input_name, t=string_types)

  def _transform(self, X):
    return X

  def transform(self, feat):
    if isinstance(feat, Mapping):
      for name in self._name:
        if name in feat: # only remove if it exist
          del feat[name]
    return feat

# ===========================================================================
# Shape
# ===========================================================================
class StackFeatures(Extractor):
  """ Stack context (or splice multiple frames) into
  single vector.

  Parameters
  ----------
  n_context : int
      number of context frame on the left and right, the final
      number of stacked frame is `context * 2 + 1`
      NOTE: the stacking process, ignore `context` frames at the
      beginning on the left, and at the end on the right.
  input_name : {None, list of string}
      list of features name will be used for calculating the
      running statistics.
      If None, calculate the statistics for all `numpy.ndarray`
  mvn: bool
      if True, preform mean-variance normalization on input features.
  """

  def __init__(self, n_context, input_name=None):
    super(StackFeatures, self).__init__(
        input_name=as_tuple(input_name, t=string_types) if input_name is not None else None)
    self.n_context = int(n_context)
    assert self.n_context > 0

  def _transform(self, feat):
    if self.input_name is None:
      X = []
      output_name = []
      for key, val in feat.items():
        if isinstance(val, np.ndarray) and val.ndim > 0:
          X.append(val)
          output_name.append(key)
    else:
      X = [feat[name] for name in self.input_name]
      output_name = self.input_name
    # ====== this ====== #
    for name, y in zip(output_name, X):
      # stacking the context frames
      y = stack_frames(y, frame_length=self.n_context * 2 + 1,
                       step_length=1, keep_length=True,
                       make_contigous=True)
      feat[name] = y
    return feat
