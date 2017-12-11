from __future__ import print_function, division, absolute_import

import os
import inspect

from six import add_metaclass
from abc import ABCMeta, abstractmethod
from collections import Mapping

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline as _make_pipeline

from odin.fuel import Dataset
from odin.utils import (get_all_files, is_string, as_tuple, is_pickleable,
                        ctext, flatten_list, dummy_formatter)
from .signal import delta, mvn, stack_frames


# ===========================================================================
# Helper
# ===========================================================================
def make_pipeline(steps, debug=False):
    """ NOTE: this method automatically:

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
    steps = [s for s in steps if s is not None]
    # ====== set debug mode ====== #
    if debug:
        for name, extractor in steps:
            extractor.set_debug(True)
    # ====== return pipeline ====== #
    return Pipeline(steps=steps)


def set_extractor_debug(debug, *extractors):
    extractors = [i for i in flatten_list(extractors)
                  if isinstance(i, Extractor)]
    for i in extractors:
        i.debug = bool(debug)
    return extractors


def _equal_inputs_outputs(x, y):
    try:
        if x != y:
            return False
    except Exception:
        pass
    return True


# ===========================================================================
# Basic extractors
# ===========================================================================
@add_metaclass(ABCMeta)
class Extractor(BaseEstimator, TransformerMixin):
    """ Extractor

    The developer must override the `_transform` method.
     - Any returned features in form of `Mapping` (e.g. dictionary) will
       be stored with the new extracted features.
     - If the returned new features is not `Mapping`, all previous extracted
       features will be ignored, and only return the new features.

    """

    def __init__(self, debug=False):
        super(Extractor, self).__init__()
        self._debug = bool(debug)

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
        # nothing to transform from None results
        if X is None:
            return None
        # NOTE: do not override this method
        try:
            y = self._transform(X)
        except Exception as e:
            print('\n')
            import traceback; traceback.print_exc()
            raise e
        # ====== check returned types ====== #
        if not isinstance(y, (Mapping, type(None))):
            raise RuntimeError("Extractor can only return Mapping or None, but "
                               "the returned type is: %s" % str(type(y)))
        # ====== Merge previous results ====== #
        if isinstance(y, Mapping):
            # remove None values
            tmp = {}
            for name, feat in y.items():
                if any(c.isupper() for c in name):
                    raise RuntimeError("name for features cannot contain "
                                       "upper case.")
                if feat is None:
                    continue
                tmp[name] = feat
            y = tmp
            # add old features extracted in X,
            # but do NOT override new features in y
            if isinstance(X, Mapping):
                for name, feat in X.items():
                    if any(c.isupper() for c in name):
                        raise RuntimeError("name for features cannot contain "
                                           "upper case.")
                    if name not in y:
                        y[name] = feat
        # maybe someone implement __getstate__ and forget _debug
        if not hasattr(self, '_debug'):
            self._debug = False
        # ====== print debug text ====== #
        if self._debug:
            print(ctext("[Extractor]", 'cyan'),
                  ctext(self.__class__.__name__, 'magenta'))
            # inputs
            if not _equal_inputs_outputs(X, y):
                print('  ', ctext("Inputs:", 'yellow'))
                if isinstance(X, Mapping):
                    for k, v in X.items():
                        print('    ', ctext(k, 'blue'), ':', dummy_formatter(v))
                else:
                    print('    ', dummy_formatter(X))
            # outputs
            print('  ', ctext("Outputs:", 'yellow'))
            if isinstance(y, Mapping):
                for k, v in y.items():
                    print('    ', ctext(k, 'blue'), ':', dummy_formatter(v))
            else:
                print('    ', dummy_formatter(y))
            # parameters
            for name, param in self.get_params().items():
                print('  ', ctext(name, 'yellow'), ':', dummy_formatter(param))
        return y


# ===========================================================================
# General extractor
# ===========================================================================
def _match_feat_name(feat_name, name):
    if feat_name is None:
        return True
    elif hasattr(feat_name, '__call__'):
        return bool(feat_name(name))
    return name in feat_name


class NameConverter(Extractor):

    """
    Parameters
    ----------
    converter: Mapping, function
        convert `inputs['name'] = converter(inputs[keys])`
    keys: str, or list of str
        the order in the list is priority of each key, searching
        through the inputs for given key and convert it to
        new `name`
    """

    def __init__(self, converter, keys=None):
        super(NameConverter, self).__init__()
        # ====== check converter ====== #
        from odin.utils.decorators import functionable
        if not hasattr(converter, '__call__') and \
        not isinstance(converter, Mapping):
            raise ValueError("`converter` must be call-able.")
        # converter can be function or dictionary
        if inspect.isfunction(converter):
            self.converter = functionable(converter)
        else:
            self.converter = converter
        # ====== check keys ====== #
        self.keys = ('name', 'path') if keys is None else \
            as_tuple(keys, t=str)

    def _transform(self, X):
        if isinstance(X, Mapping):
            for key in self.keys:
                name = X.get(key, None)
                if is_string(name):
                    break
            name = self.converter(name) if hasattr(self.converter, '__call__') \
                else self.converter[name]
            X['name'] = str(name)
        return X


class DeltaExtractor(Extractor):

    """
    Parameters
    ----------
    width: int
        amount of frames taken into account for 1 delta
    order: list of int
        list of all delta order will be concatenate (NOTE: keep `0` in
        the list if you want to keep original features)
    axis: int
        which dimension calculating the data
        (suggest time-dimension for acoustic features)
    feat_name: list of str
        list of all features name for calculating the delta
    """

    def __init__(self, width=9, order=(0, 1), axis=0, feat_name=None):
        super(DeltaExtractor, self).__init__()
        # ====== check width ====== #
        width = int(width)
        if width % 2 == 0 or width < 3:
            raise ValueError("`width` must be odd integer >= 3, give value: %d" % width)
        self.width = width
        # ====== check order ====== #
        self.order = as_tuple(order, t=int)
        # ====== axis ====== #
        self.axis = axis
        self.feat_name = feat_name

    def _transform(self, X):
        if isinstance(X, Mapping):
            max_order = max(self.order)
            for name, feat in X.items():
                if _match_feat_name(self.feat_name, name):
                    all_deltas = delta(data=feat, width=self.width,
                                       order=max_order, axis=self.axis)
                    if not isinstance(all_deltas, (tuple, list)):
                        all_deltas = (all_deltas,)
                    else:
                        all_deltas = tuple(all_deltas)
                    all_deltas = (feat,) + all_deltas
                    all_deltas = tuple([d for i, d in enumerate(all_deltas)
                                        if i in self.order])
                    feat = np.concatenate(all_deltas, axis=-1)
                X[name] = feat
        return X


class EqualizeShape0(Extractor):
    """ EqualizeShape0
    The final length of all features is the `minimum length`.

    This Extractor shrink the shape of all given features in `feat_name`
    to the same length.

    Raise Error if given files is shorted than desire length

    Parameters
    ----------
    feat_name: None or list of string
        list of features name will be used for calculating the
        running statistics.
        If None, calculate the statistics for all `numpy.ndarray`
   shrink_mode: 'center', 'left', 'right'
        center: remove data points from both left and right
        left: remove data points at the beginning (left)
        right: remove data points at the end (right)
    length_dict: dict
        dictionary mapping name of file to desire length.
    keys: string, or list of string
        if `length_dict` if provided, searching for the key in
        this feature names.
    """

    def __init__(self, feat_name, shrink_mode='right',
                 length_dict=None, keys=('name', 'path')):
        super(EqualizeShape0, self).__init__()
        if feat_name is None:
            pass
        elif hasattr(feat_name, '__call__'):
            if not is_pickleable(feat_name):
                raise ValueError("`feat_name` must be a pickle-able call-able.")
        else:
            feat_name = tuple([f.lower() for f in as_tuple(feat_name, t=str)])
        self.feat_name = feat_name
        shrink_mode = str(shrink_mode).lower()
        if shrink_mode not in ('center', 'left', 'right'):
            raise ValueError("shrink mode support include: center, left, right")
        self.shrink_mode = shrink_mode
        # ====== check length dict ====== #
        if length_dict is not None and not isinstance(length_dict, Mapping):
            raise ValueError("`length_dict` can be None or Mapping.")
        self.length_dict = length_dict
        self.keys = as_tuple(keys, t=str)

    def _transform(self, X):
        if isinstance(X, Mapping):
            equalized = {}
            # ====== searching for desire length ====== #
            n = None
            if self.length_dict is None:
                n = min(feat.shape[0]
                        for name, feat in X.items()
                        if _match_feat_name(self.feat_name, name))
            else:
                for k in self.keys:
                    if k in X and X[k] in self.length_dict:
                        n = self.length_dict[X[k]]
                        break
            if n is None:
                raise RuntimeError("Cannot find desire length.")
            # ====== equalize ====== #
            for name, feat in X.items():
                # cut the features in left and right
                # if the shape[0] is longer
                if _match_feat_name(self.feat_name, name) and feat.shape[0] != n:
                    diff = feat.shape[0] - n
                    if diff < 0:
                        print("Feature length: %d which is smaller "
                            "than desire length: %d, feature name is '%s'" %
                            (feat.shape[0], n, X['name']))
                        return None
                    elif diff > 0:
                        if self.shrink_mode == 'center':
                            diff_left = diff // 2
                            diff_right = diff - diff_left
                            feat = feat[diff_left:-diff_right]
                        elif self.shrink_mode == 'right':
                            feat = feat[:-diff]
                        elif self.shrink_mode == 'left':
                            feat = feat[diff:]
                equalized[name] = feat
            X = equalized
        return X


class RunningStatistics(Extractor):
    """ Running statistics

    Parameters
    ----------
    feat_name: None or list of string
        list of features name will be used for calculating the
        running statistics.
        If None, calculate the statistics for all `numpy.ndarray`
    """

    def __init__(self, feat_name=None, axis=0, prefix=''):
        super(RunningStatistics, self).__init__()
        self.feat_name = None if feat_name is None else \
            as_tuple(feat_name, t=str)
        self.axis = axis
        self.prefix = str(prefix)

    def get_sum1_name(self, feat_name):
        return '%s_%ssum1' % (feat_name, self.prefix)

    def get_sum2_name(self, feat_name):
        return '%s_%ssum2' % (feat_name, self.prefix)

    def _transform(self, X):
        if isinstance(X, Mapping):
            # ====== preprocessing feat_name ====== #
            feat_name = self.feat_name
            if feat_name is None:
                feat_name = [name for name, feat in X.items()
                             if isinstance(feat, np.ndarray) and feat.ndim >= 1]
            # ====== calculate the statistics ====== #
            for feat_name in feat_name:
                feat = X[feat_name]
                # ====== SUM of x^1 ====== #
                sum1 = np.sum(feat, axis=self.axis,
                              dtype='float64')
                s1_name = self.get_sum1_name(feat_name)
                if s1_name not in X:
                    X[s1_name] = sum1
                else:
                    X[s1_name] += sum1
                # ====== SUM of x^2 ====== #
                sum2 = np.sum(np.power(feat, 2), axis=self.axis,
                              dtype='float64')
                s2_name = self.get_sum2_name(feat_name)
                if s2_name not in X:
                    X[s2_name] = sum2
                else:
                    X[s2_name] += sum2
        return X


class AsType(Extractor):
    """ An extractor convert given features to given types

    Parameters
    ----------
    type_map: Mapping, or list, tuple of (name, dtype)
        mapping from feature name -> desire numpy dtype of the features.
        This is only applied for features which is `numpy.ndarray`
    """

    def __init__(self, type_map={}):
        super(AsType, self).__init__()
        if isinstance(type_map, Mapping):
            type_map = type_map.items()
        self.type_map = {str(feat_name): np.dtype(feat_name)
                         for feat_name, feat_name in type_map}

    def _transform(self, X):
        if isinstance(X, Mapping):
            for feat_name, feat_name in self.type_map.items():
                if feat_name in X:
                    feat = X[feat_name]
                    if isinstance(feat, np.ndarray) and feat_name != feat.dtype:
                        X[feat_name] = feat.astype(feat_name)
        return X


class DuplicateFeatures(Extractor):

    def __init__(self, name, new_name):
        super(DuplicateFeatures, self).__init__()
        self.name = str(name)
        self.new_name = str(new_name)

    def _transform(self, X):
        if self.name not in X:
            raise RuntimeError("Cannot find feature with name: '%s' in processed "
                               "features list." % self.name)
        X[self.new_name] = X[self.name]
        return X


class RemoveFeatures(Extractor):
    """ Remove features by name from extracted features dictionary """

    def __init__(self, feat_name=()):
        super(RemoveFeatures, self).__init__()
        self.feat_name = as_tuple(feat_name, t=str)

    def _transform(self, X):
        return X

    def transform(self, X):
        if isinstance(X, Mapping):
            for f in self.feat_name:
                del X[f]
        return super(RemoveFeatures, self).transform(X)


# ===========================================================================
# Shape
# ===========================================================================
class StackFeatures(Extractor):
    """ Stack context (or splice multiple frames) into
    single vector.

    Parameters
    ----------
    feat_name: None or list of string
        list of features name will be used for calculating the
        running statistics.
        If None, calculate the statistics for all `numpy.ndarray`
    context: int
        number of context frame on the left and right, the final
        number of stacked frame is `context * 2 + 1`
        NOTE: the stacking process, ignore `context` frames at the
        beginning on the left, and at the end on the right.
    mvn: bool
        if True, preform mean-variance normalization on input features.
    """

    def __init__(self, context, feat_name=(), mvn=True):
        super(StackFeatures, self).__init__()
        self.context = int(context)
        self.mvn = bool(mvn)
        self.feat_name = as_tuple(feat_name, t=str)

    def _transform(self, X):
        for name in self.feat_name:
            if name in X:
                y = X[name]
                # normalize
                if self.mvn:
                    y = mvn(y, varnorm=True)
                # stacking the context frames
                if self.context > 0:
                    y = stack_frames(y, frame_length=self.context * 2 + 1,
                                     step_length=1, keepdims=True,
                                     make_contigous=True)
                X[name] = y
        return X
