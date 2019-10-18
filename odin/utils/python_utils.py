from __future__ import absolute_import, division, print_function

import inspect
import io
import numbers
import os
import re
import string
import tarfile
import types
import warnings
from collections import (Iterable, Iterator, Mapping, OrderedDict, defaultdict,
                         deque)
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from six import add_metaclass, string_types
from six.moves import cPickle

# ===========================================================================
# File type check
# ===========================================================================
GZIP_MAGIC_NUMBER = "1f8b"


def is_gzip_file(path):
  """ Credit:
  https://kite.com/python/examples/4945/gzip-check-if-a-file-is-gzip-compressed
  """
  if isinstance(path, string_types) and os.path.isfile(path):
    with open(path, 'rb') as f:
      return f.read(2).encode("hex") == GZIP_MAGIC_NUMBER
  elif hasattr(path, 'read') and hasattr(path, 'tell'):
    last_pos = path.tell()
    path.seek(0)
    indicator = path.read(2)
    indicator = (indicator.encode("hex")
                 if isinstance(indicator, string_types) else indicator.hex())
    path.seek(last_pos)
    return indicator == GZIP_MAGIC_NUMBER
  return False


def is_tar_file(path):
  if not os.path.isfile(path):
    return False
  return tarfile.is_tarfile(path)


# ===========================================================================
# Regular expression
# ===========================================================================
RE_NUMBER = re.compile(r'^[+-]*((\d*\.\d+)|(\d+))$')


# ===========================================================================
# Data structure
# ===========================================================================
class struct(dict):
  '''Flexible object can be assigned any attribtues'''

  def __init__(self, *args, **kwargs):
    super(struct, self).__init__(*args, **kwargs)
    # copy all dict to attr
    for i, j in self.items():
      if is_string(i) and not hasattr(self, i):
        super(struct, self).__setattr__(i, j)

  def __setattr__(self, name, val):
    super(struct, self).__setattr__(name, val)
    super(struct, self).__setitem__(name, val)

  def __setitem__(self, x, y):
    super(struct, self).__setitem__(x, y)
    if is_string(x):
      super(struct, self).__setattr__(x, y)


class bidict(dict):
  """ Bi-directional dictionary (i.e. a <-> b)
  Note
  ----
  When you iterate over this dictionary, it will be a doubled size
  dictionary
  """

  def __init__(self, *args, **kwargs):
    super(bidict, self).__init__(*args, **kwargs)
    # this is duplication
    self._inv = dict()
    for i, j in self.items():
      self._inv[j] = i

  @property
  def inv(self):
    return self._inv

  def __setitem__(self, key, value):
    super(bidict, self).__setitem__(key, value)
    self._inv[value] = key
    return None

  def __getitem__(self, key):
    if key not in self:
      return self._inv[key]
    return super(bidict, self).__getitem__(key)

  def update(self, *args, **kwargs):
    for k, v in dict(*args, **kwargs).items():
      self[k] = v
      self._inv[v] = k

  def __delitem__(self, key):
    del self._inv[super(bidict, self).__getitem__(key)]
    return dict.__delitem__(self, key)


class defaultdictkey(defaultdict):
  """ Enhanced version of `defaultdict`, instead of return a
  default value, return an "improvised" default value based on
  the given key.

  Example
  -------
  >>> from odin.utils.python_utils import defaultdictkey
  >>> d = defaultdictkey(lambda x: str(x))
  >>> print(d['123']) # '123'
  """

  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    else:
      ret = self[key] = self.default_factory(key)
      return ret


def multikeysdict(d):
  assert isinstance(d, dict)
  new_d = d.__class__()
  for i, j in d.items():
    if isinstance(i, tuple):
      for k in i:
        new_d[k] = j
    else:
      new_d[i] = j
  return new_d


# ===========================================================================
# Getter
# ===========================================================================
def get_formatted_datetime(only_number=True):
  if only_number:
    return "{:%H%M%S%d%m%y}".format(datetime.now())
  return "{:%H:%M:%S-%d%b%y}".format(datetime.now())


def get_all_properties(obj):
  """ Return all attributes which are properties of given Object
  """
  properties = []
  clazz = obj if isinstance(obj, type) else obj.__class__
  for key in dir(clazz):
    if '__' in key:
      continue
    val = getattr(clazz, key)
    if isinstance(val, property):
      properties.append(key)
  return properties if isinstance(obj, type) else \
  {p: getattr(obj, p) for p in properties}


def get_string_placeholders(s):
  assert isinstance(s, string_types)
  fmt = []
  for (_, key, spec, _) in string.Formatter().parse(save_path):
    if spec is not None:
      fmt.append(key)
  return tuple(fmt)


# ===========================================================================
# Data converter
# ===========================================================================
def as_tuple(x, N=None, t=None):
  """
  Coerce a value to a tuple of given length (and possibly given type).

  Parameters
  ----------
  x : {value, iterable}
  N : {integer}
      length of the desired tuple
  t : {type, call-able, optional}
      required type for all elements

  Returns
  -------
  tuple
      ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.

  Raises
  ------
  TypeError
      if `type` is given and `x` or any of its elements do not match it
  ValueError
      if `x` is iterable, but does not have exactly `N` elements

  Note
  ----
  This function is adpated from Lasagne
  Original work Copyright (c) 2014-2015 lasagne contributors
  All rights reserved.

  LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE
  """
  # special case numpy array
  if not isinstance(x, tuple):
    if isinstance(x, (types.GeneratorType, list)):
      x = tuple(x)
    else:
      x = (x,)
  # ====== check length ====== #
  if is_number(N):
    N = int(N)
    if len(x) == 1:
      x = x * N
    elif len(x) != N:
      raise ValueError('x has length=%d, but required length N=%d' %
                       (len(x), N))
  # ====== check type ====== #
  if t is None:
    filter_func = lambda o: True
  elif isinstance(t, type) or isinstance(t, (tuple, list)):
    filter_func = lambda o: isinstance(o, t)
  elif hasattr(t, '__call__'):
    filter_func = t
  else:
    raise ValueError("Invalid value for `t`: %s" % str(t))
  if not all(filter_func(v) for v in x):
    raise TypeError("expected a single value or an iterable "
                    "of {0}, got {1} instead".format(t.__name__, x))
  return x


def as_list(x, N=None, t=None):
  return list(as_tuple(x, N, t))


def as_bytes(x, nbytes=None, order='little'):
  """ Convert some python object to bytes array, support type:
  * string, unicode
  * integer
  * numpy.ndarray

  Note
  ----
  This method is SLOW
  """
  if is_string(x):
    return x.encode()
  elif isinstance(x, int):
    return x.to_bytes(nbytes, order, signed=False)
  elif isinstance(x, np.ndarray):
    return x.tobytes()
  else:
    raise ValueError("Not support bytes conversion for type: %s" %
                     type(x).__name__)


# ===========================================================================
# Types check
# ===========================================================================
def is_lambda(v):
  LAMBDA = lambda: 0
  return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def is_pickleable(x):
  try:
    cPickle.dumps(x, protocol=cPickle.HIGHEST_PROTOCOL)
    return True
  except cPickle.PickleError:
    return False


def is_fileobj(f):
  """ Check if an object `f` is intance of FileIO object created
  by `open()`"""
  return isinstance(f, io.TextIOBase) or \
      isinstance(f, io.BufferedIOBase) or \
      isinstance(f, io.RawIOBase) or \
      isinstance(f, io.IOBase)


def is_callable(x):
  return hasattr(x, '__call__')


def is_string(s):
  return isinstance(s, string_types)


def is_path(path):
  if is_string(path):
    try:
      os.path.exists(path)
      return True
    except Exception as e:
      return False
  return False


def is_number(i, string_number=False):
  if isinstance(i, string_types) and string_number:
    return RE_NUMBER.match(i) is not None
  return isinstance(i, numbers.Number)


def is_bool(b):
  return isinstance(b, type(True))


def is_primitives(x, inc_ndarray=True, exception_types=[]):
  """Primitive types include: number, string, boolean, None
  and numpy.ndarray (optional) and numpy.generic (optional)

  Parameters
  ----------
  inc_ndarray: bool
      if True, include `numpy.ndarray` and `numpy.generic` as a primitive types
  """
  # complex list or Mapping
  if isinstance(x, (tuple, list)):
    return all(
        is_primitives(
            i, inc_ndarray=inc_ndarray, exception_types=exception_types)
        for i in x)
  elif isinstance(x, Mapping):
    return all(
        is_primitives(
            i, inc_ndarray=inc_ndarray, exception_types=exception_types) and
        is_primitives(
            j, inc_ndarray=inc_ndarray, exception_types=exception_types)
        for i, j in x.items())
  # check for number, string, bool, and numpy array
  if is_number(x) or is_string(x) or is_bool(x) or x is None or \
  (any(isinstance(x, t) for t in exception_types)) or \
  (inc_ndarray and isinstance(x, (np.ndarray, np.generic))):
    return True
  return False


# ===========================================================================
# IO utilities
# ===========================================================================
def get_all_files(path, filter_func=None):
  ''' Recurrsively get all files in the given path '''
  file_list = []
  if os.access(path, os.R_OK):
    for p in os.listdir(path):
      p = os.path.join(path, p)
      if os.path.isdir(p):
        file_list += get_all_files(p, filter_func)
      else:
        if filter_func is not None and not filter_func(p):
          continue
        # remove dump files of Mac
        if '.DS_Store' in p or '.DS_STORE' in p or \
            '._' == os.path.basename(p)[:2]:
          continue
        file_list.append(p)
  return file_list


def get_all_ext(path):
  """ Recurrsively get all extension of files in the given path

  Parameters
  ----------
  path : str
    input folder

  """
  file_list = []
  if os.access(path, os.R_OK):
    for p in os.listdir(path):
      p = os.path.join(path, p)
      if os.path.isdir(p):
        file_list += get_all_ext(p)
      else:
        # remove dump files of Mac
        if '.DS_Store' in p or '.DS_STORE' in p or \
            '._' == os.path.basename(p)[:2]:
          continue
        ext = p.split('.')
        if len(ext) > 1:
          file_list.append(ext[-1])
  file_list = list(set(file_list))
  return file_list


def folder2bin(path):
  """ This function read all files within a Folder
  in binary mode,
  then, store all the data in a dictionary mapping:
  `relative_path -> binary_data`
  """
  if not os.path.isdir(path):
    raise ValueError('`path`=%s must be a directory.' % path)
  path = os.path.abspath(path)
  files = get_all_files(path)
  data = {}
  for f in files:
    name = f.replace(path + '/', '')
    with open(f, 'rb') as f:
      data[name] = f.read()
  return data


def bin2folder(data, path, override=False):
  """ Convert serialized data from `folder2bin` back
  to a folder at `path`

  Parameters
  ----------
  data: {string, dict}
      if string, `data` can be pickled string, or path to a file.
      if dict, `data` is the output from `folder2bin`
  path: string
      path to a folder
  override: bool
      if True, override exist folder at `path`
  """
  # ====== check input ====== #
  if is_string(data):
    if os.path.isfile(data):
      with open(data, 'rb') as f:
        data = pickle.load(f)
    else:
      data = pickle.loads(data)
  if not isinstance(data, dict):
    raise ValueError(
        "`data` must be dictionary type, or string, or path to file.")
  # ====== check outpath ====== #
  path = os.path.abspath(str(path))
  if not os.path.exists(path):
    os.mkdir(path)
  elif os.path.isfile(path):
    raise ValueError("`path` must be path to a directory.")
  elif os.path.isdir(path):
    if not override:
      raise RuntimeError("Folder at path:%s exist, cannot override." % path)
    shutil.rmtree(path)
    os.mkdir(path)
  # ====== deserialize ====== #
  for name, dat in data.items():
    with open(os.path.join(path, name), 'wb') as f:
      f.write(dat)
  return path


# ===========================================================================
# String processing
# ===========================================================================
_space_char = re.compile(r"\s")
_multiple_spaces = re.compile(r"\s\s+")
_non_alphanumeric_char = re.compile(r"\W")


def string_normalize(text,
                     lower=True,
                     remove_non_alphanumeric=True,
                     remove_duplicated_spaces=True,
                     remove_whitespace=False,
                     escape_pattern=False):
  text = str(text).strip()
  if bool(lower):
    text = text.lower()
  if bool(escape_pattern):
    text = re.escape(text)
  if bool(remove_non_alphanumeric):
    text = _non_alphanumeric_char.sub(' ', text)
    text = text.strip()
  if bool(remove_duplicated_spaces):
    text = _multiple_spaces.sub(' ', text)
  if bool(remove_whitespace):
    if isinstance(remove_whitespace, string_types):
      text = _space_char.sub(remove_whitespace, text)
    else:
      text = _space_char.sub('', text)
  return text


text_normalize = string_normalize


# ===========================================================================
# List utils
# ===========================================================================
def unique(seq, keep_order=False):
  if keep_order:
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]
  else:
    return list(set(seq))


# ===========================================================================
# Async file IO
# ===========================================================================
# TODO


# ===========================================================================
# Object and class attributes
# ===========================================================================
class abstractclassmethod(classmethod):

  __isabstractmethod__ = True

  def __init__(self, method):
    method.__isabstractmethod__ = True
    super(abstractclassmethod, self).__init__(method)


class classproperty(object):

  def __init__(self, fn):
    super(classproperty, self).__init__()
    self.fn = fn

  def __get__(self, obj, owner):
    return self.fn(owner)


# ===========================================================================
# Path utils
# ===========================================================================
def select_path(*paths, default=None, create_new=False):
  """
  Parameters
  ----------
  paths : str
    multiple path are given

  default : str
    default path for return

  create_new : bool (default: False)
    if no path is found, create new folder based on the
    first path found to be `creat-able`
  """
  all_paths = []
  for p in paths:
    if isinstance(p, (tuple, list)):
      all_paths += p
    elif isinstance(p, string_types):
      all_paths.append(p)
    else:
      raise ValueError("Given `path` has type: '%s', which must be string or "
                       "list of string")
  # ====== return the first found exists path ====== #
  for p in all_paths:
    if os.path.exists(p):
      return p
  # ====== check if create_new ====== #
  if default is not None:
    return str(default)
  if create_new:
    for p in paths:
      base_dir = os.path.dirname(p)
      if os.path.exists(base_dir):
        os.mkdir(p)
        return p
    raise ValueError("Cannot create new folder from list: %s" % str(paths))
  # ====== raise exception ====== #
  raise RuntimeError("Cannot find any exists path from list: %s" %
                     '; '.join(all_paths))


# ===========================================================================
# Warnings and Exception
# ===========================================================================
@contextmanager
def catch_warnings_error(w):
  """ This method turn any given warnings into exception

  use: `warnings.Warning` for all warnings

  Example
  -------
  >>> with catch_warnings([RuntimeWarning, UserWarning]):
  >>>   try:
  >>>     warnings.warn('test', category=RuntimeWarning)
  >>>   except RuntimeWarning as w:
  >>>     pass
  """
  with warnings.catch_warnings():
    warnings.filterwarnings(action='error', category=w)
    yield


@contextmanager
def catch_warnings_ignore(w):
  """ This method ignore any given warnings

  use: `warnings.Warning` for all warnings
  """
  with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=w)
    yield
