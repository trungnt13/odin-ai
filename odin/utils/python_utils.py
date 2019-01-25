from __future__ import print_function, division, absolute_import

import os
import re
import inspect
import warnings
from contextlib import contextmanager

from datetime import datetime
from collections import defaultdict
from six import string_types, add_metaclass

import numpy as np


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

# ===========================================================================
# IO utilities
# ===========================================================================
def savetxt(fname, X, fmt='%g', delimiter=' ', newline='\n',
            header='', footer='', index=None,
            comments='# ', encoding=None,
            async_backend='thread'):
  """ Save an array to a text file.

  Parameters
  ----------

  fname : filename or file handle
      If the filename ends in .gz, the file is automatically saved
      in compressed gzip format. loadtxt understands gzipped files
      transparently.

  X : 1D or 2D array_like
      Data to be saved to a text file.

  fmt : str or sequence of strs, optional
      A single format (%10.5f), a sequence of formats, or a multi-format
      string, e.g. ‘Iteration %d – %10.5f’, in which case delimiter is
      ignored. For complex X, the legal options for fmt are:

      a single specifier, fmt=’%.4e’, resulting in numbers formatted like
      ‘ (%s+%sj)’ % (fmt, fmt)
      a full string specifying every real and imaginary part, e.g.
      ‘ %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej’ for 3 columns
      a list of specifiers, one per column - in this case, the real and
      imaginary part must have separate specifiers,
      e.g. [‘%.3e + %.3ej’, ‘(%.15e%+.15ej)’] for 2 columns

  delimiter : str, optional
      String or character separating columns.

  newline : str, optional
      String or character separating lines.

  header : str, optional
      String that will be written at the beginning of the file.

  footer : str, optional
      String that will be written at the end of the file.

  comments : str, optional
      String that will be prepended to the header and footer strings, to mark them as comments. Default: ‘# ‘, as expected by e.g. numpy.loadtxt.

  encoding : {None, str}, optional
      Encoding used to encode the outputfile. Does not apply to output streams. If the encoding is something other than ‘bytes’ or ‘latin1’ you will not be able to load the file in NumPy versions < 1.14. Default is ‘latin1’.

  async_backend : {'thread', 'process', None}
      save the data to disk asynchronously using multi-threading or
      multi-processing
  """
  pass

# ===========================================================================
# String processing
# ===========================================================================
_space_char = re.compile(r"\s")
_multiple_spaces = re.compile(r"\s\s+")
_non_alphanumeric_char = re.compile(r"\W")

def string_normalize(text, lower=True,
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
class defaultdictkey(defaultdict):
  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    else:
      ret = self[key] = self.default_factory(key)
      return ret


class abstractclassmethod(classmethod):

  __isabstractmethod__ = True

  def __init__(self, method):
    method.__isabstractmethod__ = True
    super(abstractclassmethod, self).__init__(method)

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
        print("Created new folder at path:", str(p))
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
