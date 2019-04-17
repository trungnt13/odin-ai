# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import io
import sys
# import uuid
import time
import math
import types
import signal
import shutil
import timeit
import inspect
import tarfile
import numbers
import tempfile
import platform
import argparse
import subprocess
import contextlib
from multiprocessing import cpu_count, Lock, current_process
from collections import OrderedDict, deque, Iterable, Iterator, Mapping
from itertools import islice, tee, chain

from six import string_types
from six.moves import cPickle
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError, HTTPError

try:
  from numba import jit, autojit, vectorize, guvectorize
except:
  pass
import numpy

from odin.utils.progbar import Progbar, add_notification
from odin.utils.path_utils import *
from odin.utils.cache_utils import *
from odin.utils.python_utils import *
from odin.utils.np_utils import *
from odin.utils.mpi import segment_list, SharedCounter, async, async_mpi, MPI
from odin.utils.crypto import md5_checksum

from odin.utils import mpi
from odin.utils import shape_calculation
from odin.utils import decorators
from odin.utils import crypto

def array_size(arr):
  """ Return size of an numpy.ndarray in bytes """
  return np.prod(arr.shape) * arr.dtype.itemsize

# ===========================================================================
# Pretty print
# ===========================================================================
def ctext(s, color='red'):
  """ Colored text support
  * BLACK
  * RED
  * GREEN
  * YELLOW
  * BLUE
  * MAGENTA
  * CYAN
  * WHITE
  * RESET
  * LIGHTBLACK_EX
  * LIGHTRED_EX
  * LIGHTGREEN_EX
  * LIGHTYELLOW_EX
  * LIGHTBLUE_EX
  * LIGHTMAGENTA_EX
  * LIGHTCYAN_EX
  * LIGHTWHITE_EX

  Note
  ----
  * colored text is longer in length and can overlength the screen
  (i.e. make the progress-bar ugly).
  * ctext(x + y + z) is BETTER than ctext(x) + ctext(y) + ctext(z)
  because it save more space.
  """
  s = str(s)
  try:
    from colorama import Fore
    color = str(color).strip()
    color = color.upper()
    # special fix for light color
    if 'LIGHT' == color[:5] and '_EX' != color[-3:]:
      color = color + '_EX'
    color = getattr(Fore, color, '')
    return color + s + Fore.RESET
  except ImportError:
    pass
  return s


def _type_name(x):
  if isinstance(x, np.dtype):
    return x.name
  return type(x).__name__


def dummy_formatter(x):
  s = str(x)
  if len(s) < 120 and '\n' not in s:
    return s
  if isinstance(x, (tuple, list)):
    return "(list)length=%d;type=%s" % \
        (len(x), _type_name(x[0]) if len(x) > 0 else '*empty*')
  if isinstance(x, np.ndarray):
    return "(ndarray)shape=%s;dtype=%s" % (str(x.shape), str(x.dtype))
  if isinstance(x, Mapping):
    return "(map)length=%d;dtype=%s" % (len(x),
        ';'.join([_type_name(i) for i in next(iter(x.items()))]))
  # dataset type
  if 'odin.fuel.dataset.Dataset' in str(type(x)):
    return ("(ds)path:%s" % x.path)
  # NNOp types
  if any('odin.nnet.base.NNOp' in str(i) for i in type.mro(type(x))):
    return "\n" + '\n'.join(['\t' + line for line in str(x).split('\n')])
  if is_string(x):
    return str(x) if len(x) < 250 else '(str)length:%d' % len(x)
  return str(x)


@contextlib.contextmanager
def UnitTimer(factor=1, name=None):
  start = timeit.default_timer()
  yield None
  end = timeit.default_timer()
  # set name for timing task
  if name is None:
    name = 'Task'
  print('"%s"' % ctext(name, 'yellow'),
      "Time:",
      ctext((end - start) / factor, 'cyan'),
      '(sec)')


# ===========================================================================
# Basics
# ===========================================================================
def is_same_shape(shape1, shape2):
  """
  Return
  ------
  True if two objects is the same shape tuple,
  otherwise, False
  """
  # if is string, evaluate to python object
  if is_string(shape1):
    shape1 = eval(shape1)
  if is_string(shape2):
    shape2 = eval(shape2)
  # tensorflow TensorShape
  if hasattr(shape1, 'as_list'):
    shape1 = shape1.as_list()
  if hasattr(shape2, 'as_list'):
    shape2 = shape2.as_list()
  # if is number, convert to shape tuple
  if is_number(shape1) or shape1 is None:
    shape1 = (shape1,)
  shape1 = as_tuple(shape1)
  if is_number(shape2) or shape2 is None:
    shape2 = (shape2,)
  shape2 = as_tuple(shape2)
  # check the number of dimension
  if len(shape1) != len(shape2):
    return False
  for s1, s2 in zip(shape1, shape2):
    if s1 is not None and s2 is not None and s1 != s2:
      return False
  return True

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


def is_number(i):
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
    return all(is_primitives(i, inc_ndarray=inc_ndarray,
                             exception_types=exception_types)
               for i in x)
  elif isinstance(x, Mapping):
    return all(is_primitives(i, inc_ndarray=inc_ndarray,
                             exception_types=exception_types) and
               is_primitives(j, inc_ndarray=inc_ndarray,
                             exception_types=exception_types)
               for i, j in x.items())
  # check for number, string, bool, and numpy array
  if is_number(x) or is_string(x) or is_bool(x) or x is None or \
  (any(isinstance(x, t) for t in exception_types)) or \
  (inc_ndarray and isinstance(x, (numpy.ndarray, numpy.generic))):
    return True
  return False


def type_path(obj):
  clazz = type(obj)
  return clazz.__module__ + "." + clazz.__name__


def is_lambda(v):
  LAMBDA = lambda: 0
  return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def is_pickleable(x):
  try:
    cPickle.dumps(x, protocol=cPickle.HIGHEST_PROTOCOL)
    return True
  except cPickle.PickleError:
    return False


def iter_chunk(it, n):
  """ Chunking an iterator into small chunk of size `n`
  Note: this can be used to slice data into mini batches
  """
  if not isinstance(it, Iterator):
    it = iter(it)
  obj = list(islice(it, n))
  while obj:
    yield obj
    obj = list(islice(it, n))


def to_bytes(x, nbytes=None, order='little'):
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


def batching(batch_size, n=None, start=0, end=None, seed=None):
  """
  Parameters
  ----------
  batch_size: int
      number of samples for 1 single batch.
  n: {int, None}
      number of samples
  start : int (default: 0)
      starting point of the iteration (in number of sample)
  end : {int, None}
      ending point of the iteration (in number of sample),
      end = total_amount_of_sample if None
  seed : {int, None}
      random seed for shuffling the returned batches,
      no shuffling performed if seed is None

  Return
  ------
  iteration: [(start, end), (start, end), ...]
  """
  if end is None and n is None:
    raise ValueError('you must provide either `end` or `n`')
  if end is None:
    end = n
  start = int(start)
  end = int(end)
  assert end > start, "`end` must > `start`"
  batch_size = int(batch_size)
  # ====== no shuffling ====== #
  if seed is None:
    return ((i, min(i + batch_size, end))
            for i in range(start, end + batch_size, batch_size)
            if i < end)
  batches = list(range(start, end + batch_size, batch_size))
  rand = np.random.RandomState(seed)
  rand.shuffle(batches)
  return (((i, min(i + batch_size, end)))
          for i in batches
          if i < end)

def read_lines(file_path):
  if not os.path.exists(file_path):
    raise ValueError('File at path: %s does not exist.' % file_path)
  if os.path.isdir(file_path):
    raise ValueError("Path to %s is a folder" % file_path)
  lines = []
  with open(file_path, 'r') as f:
    for i in f:
      lines.append(i[:-1] if i[-1] == '\n' else i)
  return lines


# ===========================================================================
# Others
# ===========================================================================
def raise_return(e):
  raise e


_CURRENT_STDIO = None


class _LogWrapper():

  def __init__(self, stream, own_buffer):
    self.stream = stream
    self.own_buffer = bool(own_buffer)

  def write(self, message):
    # no backtrack for writing to file
    self.stream.write(message.replace("\b", ''))
    sys.__stdout__.write(message)

  def flush(self):
    self.stream.flush()
    sys.__stdout__.flush()

  def end(self):
    self.stream.flush()
    sys.__stdout__.flush()
    if self.own_buffer:
      try:
        self.stream.close()
      except Exception:
        pass

def get_stdio_path():
  return _CURRENT_STDIO

def stdio(path=None, suppress=False, stderr=True):
  """
  Parameters
  ----------
  path: {None, str, io.StringIO}
      if str or StringIO, specified path for saving all stdout (and stderr)
      if None, reset to the system default stdout and stderr
  suppress: boolean
      totally turn-off all stdout (and stdeer)
  stderr:
      apply output file with stderr also

  NOTE
  ----
  Redirect the system logging can slightly decrease the
  performance in logging intensive application.
  """
  # turn off stdio
  if suppress:
    f = open(os.devnull, "w")
    path = None
  # reset
  elif path is None:
    f = None
  # redirect to a file
  elif is_string(path):
    f = _LogWrapper(open(path, "w"), own_buffer=True)
  # redirect to a buffer
  elif is_fileobj(path):
    f = _LogWrapper(path, own_buffer=False)
    if hasattr(path, 'name'):
      path = f.name
    else:
      path = str(path)
  else:
    raise ValueError("Unsupport for path=`%s` in `stdio`." %
        str(type(path)))
  # ====== set current stdio path ====== #
  global _CURRENT_STDIO
  _CURRENT_STDIO = path
  # ====== assign stdio ====== #
  if f is None: # reset
    if isinstance(sys.stdout, _LogWrapper):
      sys.stdout.end()
    if isinstance(sys.stderr, _LogWrapper):
      sys.stderr.end()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
  else: # redirect to file
    sys.stdout = f
    if stderr:
      sys.stderr = f

def auto_logging(log_dir=None, prefix='', num_max=None):
  """
  Parameters
  ----------
  log_dir : str
    path to directory stored all the logs
  prefix : str
    prefix to all saved log (using script name by default)
  num_max : {None, int}
    maximum number of log will be stored
  """
  if log_dir is None:
    log_dir = get_script_path()
  if not os.path.isdir(log_dir):
    raise ValueError("'%s' is not a directory" % str(log_dir))
  prefix = get_script_name() if prefix is None or len(prefix) == 0 else str(prefix)
  date_time = get_formatted_datetime(only_number=False)
  path = os.path.join(log_dir, prefix + '[%s]' % date_time + '.txt')
  # ====== check maximum number of log ====== #
  if num_max is not None:
    from stat import ST_CTIME
    num_max = int(num_max)
    past_logs = [i for i in os.listdir(log_dir) if prefix in i]
    past_logs = sorted(past_logs,
                       key=lambda x: os.stat(os.path.join(log_dir, x))[ST_CTIME],
                       reverse=True)
    # remove previous logs
    for i, name in enumerate(past_logs):
      if i >= num_max - 1:
        os.remove(os.path.join(log_dir, name))
  return stdio(path=path, suppress=False, stderr=True)

def eprint(text):
  """Print ERROR message to stderr"""
  print(ctext('[Error]', 'red') + str(text), file=sys.stderr)

def wprint(text):
  """Print WARNING message to stderr"""
  print(ctext('[Warning]', 'yellow') + str(text), file=sys.stderr)

# ===========================================================================
# Universal ID
# ===========================================================================
_uuid_chars = list(chain(map(chr, range(65, 91)),  # ABCD
                         map(chr, range(97, 123)),  # abcd
                         map(chr, range(48, 57)))) # 0123
_uuid_random_state = numpy.random.RandomState(int(str(int(time.time() * 100))[3:]))


def uuid(length=8):
  """ Generate random UUID 8 characters with very very low collision """
  # m = time.time()
  # uniqid = '%8x%4x' % (int(m), (m - int(m)) * 1000000)
  # uniqid = str(uuid.uuid4())[:8]
  uniquid = ''.join(
      _uuid_random_state.choice(_uuid_chars,
                                size=length,
                                replace=True))
  return uniquid


@contextlib.contextmanager
def change_recursion_limit(limit):
  """Temporarily changes the recursion limit."""
  old_limit = sys.getrecursionlimit()
  if old_limit < limit:
    sys.setrecursionlimit(limit)
  yield
  sys.setrecursionlimit(old_limit)


@contextlib.contextmanager
def signal_handling(sigint=None, sigtstp=None, sigquit=None):
  # We cannot handle SIGTERM, because it prevent subproces from
  # .terminate()
  orig_int = signal.getsignal(signal.SIGINT)
  orig_tstp = signal.getsignal(signal.SIGTSTP)
  orig_quit = signal.getsignal(signal.SIGQUIT)

  if sigint is not None: signal.signal(signal.SIGINT, sigint)
  if sigtstp is not None: signal.signal(signal.SIGTSTP, sigtstp)
  if sigquit is not None: signal.signal(signal.SIGQUIT, sigquit)

  yield
  # reset
  signal.signal(signal.SIGINT, orig_int)
  signal.signal(signal.SIGTSTP, orig_tstp)
  signal.signal(signal.SIGQUIT, orig_quit)


# ===========================================================================
# UniqueHasher
# ===========================================================================
class UniqueHasher(object):
  """ This hash create strictly unique hash value by using
  its memory to remember which key has been assigned

  Note
  ----
  This function use deterministic hash, which give the same
  id for all labels, whenever you call it
  """

  def __init__(self, nb_labels=None):
    super(UniqueHasher, self).__init__()
    self.nb_labels = nb_labels
    self._memory = {} # map: key -> hash_key
    self._current_hash = {} # map: hash_key -> key

  def hash(self, value):
    key = abs(hash(value))
    if self.nb_labels is not None:
      key = key % self.nb_labels
    # already processed hash
    if value in self._current_hash:
      return self._current_hash[value]
    # not yet processed
    if key in self._memory:
      if self.nb_labels is not None and \
          len(self._memory) >= self.nb_labels:
        raise Exception('All %d labels have been assigned, outbound value:"%s"' %
                        (self.nb_labels, value))
      else:
        while key in self._memory:
          key += 1
          if self.nb_labels is not None and key >= self.nb_labels:
            key = 0
    # key not in memory
    self._current_hash[value] = key
    self._memory[key] = value
    return key

  def __call__(self, value):
    return self.hash(value)

  def map(self, order, array):
    """ Re-order an ndarray to new column order """
    order = as_tuple(order)
    # get current order
    curr_order = self._current_hash.items()
    curr_order.sort(key=lambda x: x[1])
    curr_order = [i[0] for i in curr_order]
    # new column order
    order = [curr_order.index(i) for i in order]
    return array[:, order]


class FuncDesc(object):
  """ This class store the description of arguments given a function.
  Automatically match the argument and keyword-arguments in following
  order:
   - Match all positional arguments.
   - if the function has `varargs`, keep the remains positional arguments.
   - if the function has `keywords`, keep the unknown argument name in `kwargs`
   - Add the missing default kwargs.
   - Return new tuple of (arg, kwargs)
  """

  def __init__(self, func):
    super(FuncDesc, self).__init__()
    # copy information from other FuncDesc
    if isinstance(func, FuncDesc):
      self._args = func._args
      self._defaults = func._defaults
      self._is_include_args = func._inc_args
      self._is_include_kwargs = func._inc_kwargs
      self._func = func._func
    # extract information from a function or method
    elif inspect.isfunction(func) or inspect.ismethod(func) or \
    isinstance(func, decorators.functionable):
      if isinstance(func, decorators.functionable):
        sign = inspect.signature(func.function)
      else:
        sign = inspect.signature(func)
      self._args = [n for n, p in sign.parameters.items()
                    if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                      inspect.Parameter.VAR_KEYWORD)]
      self._is_include_args = any(i.kind == inspect.Parameter.VAR_POSITIONAL
                           for i in sign.parameters.values())
      self._is_include_kwargs = any(i.kind == inspect.Parameter.VAR_KEYWORD
                             for i in sign.parameters.values())
      self._defaults = {n: p.default
                        for n, p in sign.parameters.items()
                        if p.default != inspect.Parameter.empty}
      self._func = func
    else:
      raise ValueError("`func` must be function, method, or FuncDesc, "
                       "but given: %s" % str(type(func)))
    self.__name__ = self._func.__name__

  def __getstate__(self):
    import dill
    # check if need to re-load function module during setstate
    if not is_lambda(self._func):
      module = inspect.getmodule(self._func)
      func_module_key = [i
                         for i, j in list(sys.modules.items())
                         if j == module][0]
      func_module_name = module.__name__
      func_module_path = inspect.getfile(module)
    else:
      func_module_key = None
      func_module_name = None
      func_module_path = None
    # using dill to dump function
    func_str = dill.dumps(self._func)
    return (func_str,
            func_module_key, func_module_name, func_module_path,
            self.__name__,
            self._args, self._is_include_args,
            self._is_include_kwargs, self._defaults)

  def __setstate__(self, states):
    import dill
    (func_str,
     func_module_key, func_module_name, func_module_path,
     self.__name__,
     self._args, self._is_include_args,
     self._is_include_kwargs, self._defaults) = states
    # iterate through all loaded modules to find module contain given function
    if func_module_key is not None and \
    func_module_name is not None and \
    func_module_path is not None:

      found = None
      for name, module in list(sys.modules.items()):
        try:
          if inspect.getfile(module) == func_module_path or \
          (module.__name__.split('.')[-1] == func_module_name and
           self.__name__ in dir(module)):
            found = (name, module)
        except Exception as e:
          pass

      if found is None:
        # TODO: search for module in sys.path
        import imp
        path = func_module_path
        script_path = get_script_path()
        folders = func_module_path.split('/')
        for i in range(1, len(folders)):
          if os.path.exists(path) and os.path.isfile(path):
            break
          path = os.path.join(script_path, '/'.join(folders[-i:]))
        imp.load_source(name=func_module_name, pathname=path)
      elif found[0] != func_module_name:
        sys.modules[func_module_name] = sys.modules[found[0]]

    # load function
    self._func = dill.loads(func_str)

  @property
  def name(self):
    return self.__name__

  @property
  def qualname(self):
    """ The qualified name of the class, function, method, descriptor,
    or generator instance.

    New in version 3.3.
    """
    assert hasattr(self._func, '__qualname__'), \
    "The attribute `__qualname__` only be introduced from python 3.3"
    return self._func.__qualname__

  @property
  def path(self):
    """ Return the absolute path to the script contain the function """
    return inspect.getfile(inspect.getmodule(self._func))

  @property
  def args(self):
    return tuple(self._args)

  @property
  def defaults(self):
    return dict(self._defaults)

  @property
  def is_args(self):
    return self._is_include_args

  @property
  def is_kwargs(self):
    return self._is_include_kwargs

  def _match(self, *args, **kwargs):
    keywords = OrderedDict()
    for name, val in zip(self.args, args):
      keywords[name] = val
    # extra args
    if self._is_include_args and len(self.args) < len(args):
      args = args[len(self.args):]
    else:
      args = ()
    # remove extra kwargs in inc_kwargs=False
    if not self._is_include_kwargs:
      kwargs = {name: kwargs[name] for name in self._args
                if name in kwargs}
    keywords.update(kwargs)
    # ====== update the default ====== #
    for name, val in self._defaults.items():
      if name not in keywords:
        keywords[name] = val
    return args, keywords

  def __call__(self, *args, **kwargs):
    args, kwargs = self._match(*args, **kwargs)
    return self._func(*args, **kwargs)

  def __str__(self):
    s = "<%s>args:%s defaults:%s varargs:%s keywords:%s" % \
        (ctext(self._func.__name__, 'cyan'), self._args, self._defaults,
            self._is_include_args, self._is_include_kwargs)
    return s

# ===========================================================================
# ArgCOntrol
# ===========================================================================
def args_parse(descriptions):
  """ Shortcut for parsing the argument from terminal
  command, by a list of following tuple:
    [(name, help, enum_values, default_value), ...]

  Parameter
  ---------
  descriptions : tuple, list
    a list of tuple contain 1 to 3 of following information
    (name, help, enum_values, default_value), where:
    - name (string) is the name of argument can be `name` positional,
      `-name` optional, or `--name` flag trigger
    - help (string) is text for printing help
    - enum_values (tuple, list, None) list of string (or Number) contain
      acceptable values (set to None if no values given)
    - default_value (Number, string) default value for the argument

  Note
  ----
  The order is different from `ArgController.add`
  """
  args = ArgController()
  for desc in descriptions:
    if len(desc) == 1:
      args.add(name=desc[0], help='')
    elif len(desc) == 2:
      args.add(name=desc[0], help=desc[1])
    elif len(desc) == 3:
      args.add(name=desc[0], help=desc[1], enum=desc[2])
    elif len(desc) == 4:
      args.add(name=desc[0], help=desc[1], enum=desc[2], default=desc[3])
    else:
      raise ValueError("No support for given description: %s" % str(desc))
  return args.parse()

class ArgController(object):
  """ Simple interface to argparse """

  def __init__(self, print_parsed=True):
    super(ArgController, self).__init__()
    self.parser = None
    self._require_input = False
    self.args_preprocessor = {}
    self.args_enum = {}
    self.name = []
    self.print_parsed = print_parsed

  def _process_name(self, name):
    if '--' == name[:2]:
      name = name[2:]
    elif '-' == name[0]:
      name = name[1:]
    name = name.replace('-', '_')
    return name

  def _is_positional(self, name):
    if name[0] != '-' and name[:2] != '--':
      return True
    return False

  def _parse_input(self, key, val):
    key = self._process_name(key)
    # ====== search if manual preprocessing available ====== #
    for i, preprocess in self.args_preprocessor.items():
      if key == i and preprocess is not None:
        return preprocess(val)
    # ====== auto preprocess ====== #
    try:
      val = float(val)
      if int(val) == val:
        val = int(val)
    except Exception as e:
      val = str(val)
    return val

  def add(self, name, help, default=None, enum=None,
          preprocess=None):
    """ NOTE: if the default value is not given, the argument is
    required

    Parameters
    ----------
    name: str
        [-name] for optional parameters
        [name] for positional parameters
    help: str
        description of the argument
    default: {str, Number}
        default value for the argument
    enum : {tuple, list}
        list or tuple of acceptant values, for post-process
        validating
    preprocess: call-able
        a function to perform preprocessing of the parsed argument
    """
    if self.parser is None:
      self.parser = argparse.ArgumentParser(
          description='Automatic argument parser (yes,true > True; no,false > False)',
          add_help=True)
    # ====== NO default value ====== #
    if default is None:
      if self._is_positional(name):
        self.parser.add_argument(name, help=help, type=str, action="store",
            metavar='')
      else:
        self.parser.add_argument(name, help=help, type=str, action="store",
            required=True, metavar='')
      self._require_input = True
    # ====== boolean default value ====== #
    elif isinstance(default, bool):
      help += ' (default: %s)' % str(default)
      self.parser.add_argument(name, help=help,
                               action="store_%s" % str(not default).lower())
      preprocess = lambda x: bool(x)
    # ====== add defaults value ====== #
    else:
      help += ' (default: %s)' % str(default)
      self.parser.add_argument(name, help=help, type=str, action="store",
          default=str(default), metavar='')

    # ====== store preprocess dictionary ====== #
    name = self._process_name(name)
    if not hasattr(preprocess, '__call__'):
      preprocess = None
    self.args_preprocessor[name] = preprocess
    # store the enum values
    if enum is not None and not isinstance(enum, (tuple, list)):
      enum = (enum,)
    self.args_enum[name] = enum
    return self

  def parse(self, config_path=None):
    if self.parser is None:
      raise Exception('Call add to assign at least 1 argument for '
                      'for the function.')
    exit_now = False
    # ====== validate arguments ====== #
    try:
      if len(sys.argv) == 1 and self._require_input:
        self.parser.print_help()
        exit_now = True
      else:
        args = self.parser.parse_args()
    except Exception as e:
      # if specfy version or help, don't need to print anything else
      if all(i not in ['-h', '--help', '-v', '--version']
             for i in sys.argv):
        self.parser.print_help()
      exit_now = True
    if exit_now:
      exit()
    # ====== parse the arguments ====== #
    try:
      args = {i: self._parse_input(i, j)
              for i, j in args._get_kwargs()}
    except Exception as e:
      print('Error parsing argument with name "%s"' % str(e))
      self.parser.print_help()
      exit()
    # ====== checking enumerate values ====== #
    for name, val in args.items():
      enum = self.args_enum[self._process_name(name)]
      if enum is None:
        continue
      if val not in enum:
        raise ValueError("Argument with name '%s' given value: `%s`, but "
                         "only accept one of the following: %s" %
                         (name, val, enum))
    # ====== reset everything ====== #
    self.parser = None
    self.args_preprocessor = {}
    self.args_enum = {}
    self._require_input = False
    # ====== print parsed values ====== #
    if self.print_parsed:
      max_len = max(len(i) for i in args.keys())
      max_len = '%-' + str(max_len) + 's'
      print('\n******** Parsed arguments ********')
      for name, val in sorted(args.items()):
        print(max_len % name, ': ', val)
      print('**********************************\n')
    # convert it to struct
    return struct(args)

# ===========================================================================
# Simple math and processing
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

def as_tuple_of_shape(x):
  if not isinstance(x, (tuple, list)):
    x = (x,)
  if is_number(x[0]):
    x = (x,)
  return x

def as_list(x, N=None, t=None):
  return list(as_tuple(x, N, t))

def axis_normalize(axis, ndim,
                   return_tuple=False):
  """ Normalize the axis
   * -1 => ndim - 1
   * 10 => 10 % ndim
   * None => list(range(ndim))

  Parameters
  ----------
  return_tuple: bool
      if True, always return a tuple of normalized axis
  """
  if isinstance(axis, np.ndarray):
    axis = axis.ravel().tolist()
  if ndim == 0:
    return ()
  if axis is None or \
  (isinstance(axis, (tuple, list)) and len(axis) == 1 and axis[0] is None):
    return list(range(ndim))
  if not hasattr(axis, '__len__'):
    axis = int(axis) % ndim
    return (axis,) if return_tuple else axis
  axis = as_tuple(axis, t=int)
  return tuple([i % ndim for i in axis])


def flatten_list(x, level=None):
  """
  Parameters
  ----------
  level: int, or None
      how deep the function go into element of x to search for list and
      flatten it. If None is given, flatten all list found.

  Example
  -------
  >>> l = [1, 2, 3, [4], [[5], [6]], [[7], [[8], [9]]]]
  >>> print(flatten_list(l, level=1))
  >>> # [1, 2, 3, 4, [5], [6], [7], [[8], [9]]]
  >>> print(flatten_list(l, level=2))
  >>> # [1, 2, 3, 4, 5, 6, 7, [8], [9]]
  >>> print(flatten_list(l, level=None))
  >>> # [1, 2, 3, 4, 5, 6, 7, 8, 9]
  """
  if isinstance(x, Iterator):
    x = list(x)
  if level is None:
    level = 10e8
  if not isinstance(x, (tuple, list)):
    return [x]
  if any(isinstance(i, (tuple, list)) for i in x):
    _ = []
    for i in x:
      if isinstance(i, (tuple, list)) and level > 0:
        _ += flatten_list(i, level - 1)
      else:
        _.append(i)
    return _
  return x


# ===========================================================================
# Python
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


# Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
# urllib module, known to have issues with proxy management
if sys.version_info[0] == 2:
  def urlretrieve(url, filename, reporthook=None, data=None):
    '''
    This function is adpated from: https://github.com/fchollet/keras
    Original work Copyright (c) 2014-2015 keras contributors
    '''
    def chunk_read(response, chunk_size=8192, reporthook=None):
      total_size = response.info().get('Content-Length').strip()
      total_size = int(total_size)
      count = 0
      while 1:
        chunk = response.read(chunk_size)
        if not chunk:
          break
        count += 1
        if reporthook:
          reporthook(count, chunk_size, total_size)
        yield chunk

    response = urlopen(url, data)
    with open(filename, 'wb') as fd:
      for chunk in chunk_read(response, reporthook=reporthook):
        fd.write(chunk)
else:
  from six.moves.urllib.request import urlretrieve

def get_file(fname, origin, outdir):
  '''
  Parameters
  ----------
  fname: output file name
  origin: url, link
  outdir: path to output dir
  '''
  fpath = os.path.join(outdir, fname)
  # ====== remove empty folder ====== #
  if os.path.exists(fpath):
    if os.path.isdir(fpath) and len(os.listdir(fpath)) == 0:
      shutil.rmtree(fpath)
  # ====== download package ====== #
  if not os.path.exists(fpath):
    prog = Progbar(target=-1,
                   name="Downloading: %s" % os.path.basename(origin),
                   print_report=True, print_summary=True)

    def dl_progress(count, block_size, total_size):
      if prog.target < 0:
        prog.target = total_size
      else:
        prog.add(count * block_size - prog.seen_so_far)
    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
      try:
        urlretrieve(origin, fpath, dl_progress)
      except URLError as e:
        raise Exception(error_msg.format(origin, e.errno, e.reason))
      except HTTPError as e:
        raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
      if os.path.exists(fpath):
        os.remove(fpath)
      raise
  return fpath

# ===========================================================================
# Python utilities
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
    raise ValueError("`data` must be dictionary type, or string, or path to file.")
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
# Package utils
# ===========================================================================
def package_installed(name, version=None):
  import pip
  for i in pip.get_installed_distributions():
    if name.lower() == i.key.lower() and \
    (version is None or version == i.version):
      return True
  return False


def package_list(include_version=False):
  """
  Return
  ------
  ['odin', 'lasagne', 'keras', ...] if include_version is False
  else ['odin==8.12', 'lasagne==25.18', ...]
  """

  all_packages = []
  import pip
  for i in pip.get_installed_distributions():
    all_packages.append(i.key +
        (('==' + i.version) if include_version is True else ''))
  return all_packages


def get_module_from_path(identifier, path='.',
                         prefix='', suffix='', exclude='',
                         prefer_compiled=False, return_error=False):
  ''' Algorithms:
   - Search all files in the `path` matched `prefix` and `suffix`
   - Exclude all files contain any str in `exclude`
   - Sorted all files based on alphabet
   - Load all modules based on `prefer_compiled`
   - return list of identifier found in all modules

  Parameters
  ----------
  identifier : str
      identifier of object, function or anything in script files

  prefix : str
      prefix of file to search in the `path`

  suffix : str
      suffix of file to search in the `path`

  path : str
      searching path of script files

  exclude : str, list(str)
      any files contain str in this list will be excluded

  prefer_compiled : bool
      True mean prefer .pyc file, otherwise prefer .py

  return_error : bool (default: False)
      return all the error happened during loading the modules

  Returns
  -------
  list(object, function, ..) :
      any thing match given identifier in all found script file
  errors (optional) :
      error while loading any of the script

  Notes
  -----
  File with multiple . character my procedure wrong results

  If the script run this this function match the searching process, a
  infinite loop may happen!

  * This function try to import each modules and find desire function,
  it may mess up something.

  '''
  import re
  import imp
  # ====== validate input ====== #
  if exclude == '':
    exclude = []
  if type(exclude) not in (list, tuple, numpy.ndarray):
    exclude = [exclude]
  prefer_flag = 1 if prefer_compiled else -1
  # ====== create pattern and load files ====== #
  pattern = re.compile(r"^%s.*%s\.pyc?" % (prefix, suffix)) # py or pyc
  file_name = os.listdir(path)
  file_name = [f for f in file_name
           if pattern.match(f) and
           sum([i in f for i in exclude]) == 0]
  # ====== remove duplicated pyc files ====== #
  file_name = sorted(file_name,
                     key=lambda x: prefer_flag * len(x)) # pyc is longer
  # .pyc go first get overrided by .py
  file_name = sorted({f.split('.')[0]: f for f in file_name}.values())
  # ====== load all modules ====== #
  modules = []
  modules_error = {}
  # NOTE: this will load the module ignore the relative import path
  # For example: for module A.B.C
  # `from A.B import C` will result 'A.B.C' -> C
  # `imp.load_source`, here, will result 'C' -> C
  for fname in file_name:
    try:
      if '.pyc' in fname:
        modules.append(
            imp.load_compiled(fname.split('.')[0],
                              os.path.join(path, fname))
        )
      else:
        modules.append(
            imp.load_source(fname.split('.')[0],
                            os.path.join(path, fname))
        )
    except Exception as e:
      modules_error[fname] = str(e)
      eprint("Cannot loading modules from file: '%s' - %s" %
        (ctext(fname, 'yellow'), ctext(str(e), 'red')))
  # ====== Find all identifier in modules ====== #
  ids = []
  for m in modules:
    for i in inspect.getmembers(m):
      if identifier in i:
        ids.append(i[1])
  # remove duplicate py
  if bool(return_error):
    return ids, modules_error
  return ids


def ordered_set(seq):
  seen = {}
  result = []
  for marker in seq:
    if marker in seen: continue
    seen[marker] = 1
    result.append(marker)
  return result


def dict_union(*dicts, **kwargs):
  r"""Return union of a sequence of disjoint dictionaries.

  Parameters
  ----------
  dicts : dicts
      A set of dictionaries with no keys in common. If the first
      dictionary in the sequence is an instance of `OrderedDict`, the
      result will be OrderedDict.
  \*\*kwargs
      Keywords and values to add to the resulting dictionary.

  Raises
  ------
  ValueError
      If a key appears twice in the dictionaries or keyword arguments.

  """
  dicts = list(dicts)
  if dicts and isinstance(dicts[0], OrderedDict):
    result = OrderedDict()
  else:
    result = {}
  for d in list(dicts) + [kwargs]:
    duplicate_keys = set(result.keys()) & set(d.keys())
    if duplicate_keys:
      raise ValueError("The following keys have duplicate entries: {}"
                       .format(", ".join(str(key) for key in
                                         duplicate_keys)))
    result.update(d)
  return result


# ===========================================================================
# PATH, path manager
# ===========================================================================
@contextlib.contextmanager
def TemporaryDirectory(add_to_path=False):
  """
  add_to_path: bool
      temporary add the directory to system $PATH
  """
  temp_dir = tempfile.mkdtemp()
  if add_to_path:
    os.environ['PATH'] = temp_dir + ':' + os.environ['PATH']
  current_dir = os.getcwd()
  os.chdir(temp_dir)
  yield temp_dir
  os.chdir(current_dir)
  if add_to_path:
    os.environ['PATH'] = os.environ['PATH'].replace(temp_dir + ':', '')
  shutil.rmtree(temp_dir)


def get_tempdir():
  return tempfile.mkdtemp()


def _get_managed_path(folder, name, override, is_folder=False,
                      root='~', odin_base=True):
  if root == '~':
    root = os.path.expanduser('~')
  datadir_base = os.path.join(root, '.odin') if odin_base else root
  if not os.path.exists(datadir_base):
    os.mkdir(datadir_base)
  elif not os.access(datadir_base, os.W_OK):
    raise Exception('Cannot acesss path: ' + datadir_base)
  datadir = os.path.join(datadir_base, folder)
  if not os.path.exists(datadir):
    os.makedirs(datadir)
  # ====== check given path with name ====== #
  if is_string(name):
    datadir = os.path.join(datadir, str(name))
    if os.path.exists(datadir) and override:
      if os.path.isfile(datadir): # remove file
        os.remove(datadir)
      else: # remove and create new folder
        shutil.rmtree(datadir)
    if is_folder and not os.path.exists(datadir):
      os.mkdir(datadir)
  return datadir


def get_datasetpath(name=None, override=False, is_folder=True, root='~'):
  return _get_managed_path('datasets', name, override,
                           is_folder=is_folder, root=root)


def get_figpath(name=None, override=False, root='~'):
  return _get_managed_path('figs', name, override,
                           is_folder=True, root=root)

def get_modelpath(name=None, override=False, root='~'):
  """ Default model path for saving O.D.I.N networks """
  return _get_managed_path('models', name, override,
                           is_folder=True, root=root)


def get_logpath(name=None, override=False, increasing=True,
                odin_base=True, root='~'):
  """
  name : (string) name of log file
  override : (bool) if True, override old log file if exist
  increasing : (bool) append a number (e.g. "log.0.txt"...), if log file exists
  odin_base : (bool) if True, create log file under '.odin/logs' folder
  root : (string) root folder that contain all the log file
  """
  def _check_logpath(log_path):
    main_path, ext = os.path.splitext(log_path)
    main_path = main_path.split('.')
    try:
      current_log_index = int(main_path[-1])
      main_path = main_path[:-1]
    except ValueError:
      current_log_index = 0
    main_path = '.'.join(main_path)
    # ====== increase log index until found a new file ====== #
    while True:
      path = main_path + '.' + str(current_log_index) + ext
      if not os.path.exists(path):
        break
      current_log_index += 1
    return main_path + '.' + str(current_log_index) + ext

  path = _get_managed_path('logs' if odin_base else '', name, override,
                           is_folder=False, root=root,
                           odin_base=odin_base)
  if increasing:
    path = _check_logpath(path)
  return path


def get_exppath(tag, name=None, override=False, prompt=False,
                root='~'):
  """ Specific path for experiments results

  Parameters
  ----------
  tag: string
      specific tag for the task you are working on
  name: string
      name of the folder contains all the results (NOTE: the
      name can has subfolder)
  override: bool
      if True, remove exist folder
  prompt: bool
      if True, display prompt and require (Y) input before
      delete old folder, if (N), the program exit.
  root: string
      root path for the results (default: "~/.odin")
  """
  path = _get_managed_path('exp', tag, False,
                           is_folder=True, root=root,
                           odin_base=False)
  # only return the main folder
  if name is None:
    pass
  # return the sub-folder
  else:
    name = str(name).split('/')
    for i in name:
      path = os.path.join(path, i)
      if not os.path.exists(path):
        os.mkdir(path)
  # ====== check if override ====== #
  if override and len(os.listdir(path)) > 0:
    if prompt:
      user_cmd = raw_input('Do you want to delete "%s" (Y for yes):').lower()
      if user_cmd != 'y':
        exit()
    shutil.rmtree(path)
    os.mkdir(path)
  return path

# ===========================================================================
# Misc
# ===========================================================================
def run_script(s, wait=True, path='/tmp'):
  """
  Parameters
  ----------
  s : string
    python script

  wait : bool (default: True)
    blocking the current process until finishing the script

  path : string (default: '/tmp')
    saving the script to a temporary path before running

  Return
  ------
  status: of executed command, if `wait`=True else return Popen object
  out: (string, utf-8) - stdout message
  err: (string, utf-8) - stderr message
  """
  # ====== path preprocessing ====== #
  if path is None:
    path = '/tmp'
  if os.path.isdir(path):
    path = os.path.join(path, 'tmp_%s' % uuid(length=25) + '.py')
  path = os.path.abspath(path)
  # ====== script preprocessing ====== #
  s = str(s).strip()
  # ====== run the script ====== #
  try:
    with open(path, 'w') as f:
      f.write(s)
    cmd = subprocess.Popen("python %s" % path, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = cmd.communicate()
    if wait:
      status = cmd.wait()
    else:
      status = cmd
  finally:
    if os.path.exists(path):
      os.remove(path)
  return status, str(out, encoding='utf-8'), str(err, encoding='utf-8')

def exec_commands(cmds, print_progress=True):
  ''' Execute a command or list of commands in parallel with multiple process
  (as much as we have CPU)

  Parameters
  ----------
  cmds: str or list of str
      string represent command you want to run

  Return
  ------
  failed: list of failed command

  '''
  if not cmds: return [] # empty list
  if not isinstance(cmds, (list, tuple)):
    cmds = [cmds]
  else:
    cmds = list(cmds)

  def done(p):
    return p.poll() is not None

  def success(p):
    return p.returncode == 0

  max_task = cpu_count()
  processes = []
  processes_map = {}
  failed = []
  if print_progress:
    prog = Progbar(target=len(cmds), name="Execute Commands")
  while True:
    while cmds and len(processes) < max_task:
      task = cmds.pop()
      p = subprocess.Popen(task, shell=True)
      processes.append(p)
      processes_map[p] = task
      # print process
      if print_progress:
        prog.add(1)

    for p in processes:
      if done(p):
        if success(p):
          processes.remove(p)
        else:
          failed.append(processes_map[p])

    if not processes and not cmds:
      break
    else:
      time.sleep(0.005)
  return failed


def save_wav(path, s, fs):
  from scipy.io import wavfile
  wavfile.write(path, fs, s)


def play_audio(data, fs, volumn=1, speed=1):
  ''' Play audio from numpy array.

  Parameters
  ----------
  data : numpy.ndarray
          signal data
  fs : int
          sample rate
  volumn: float
          between 0 and 1
  speed: float
          > 1 mean faster, < 1 mean slower

  Note
  ----
  Only support play audio on MacOS
  '''
  import soundfile as sf
  import os

  data = numpy.asarray(data, dtype=numpy.int16)
  if data.ndim == 1:
    channels = 1
  else:
    channels = data.shape[1]
  with TemporaryDirectory() as temppath:
    path = os.path.join(temppath, 'tmp_play.wav')
    with sf.SoundFile(path, 'w', fs, channels, subtype=None,
        endian=None, format=None, closefd=None) as f:
      f.write(data)
    os.system('afplay -v %f -q 1 -r %f %s &' % (volumn, speed, path))
    raw_input('<enter> to stop audio.')
    os.system("kill -9 `ps aux | grep -v 'grep' | grep afplay | awk '{print $2}'`")


# ===========================================================================
# System query
# ===========================================================================
__process_pid_map = {}


def get_process_status(pid=None, memory_usage=False, memory_shared=False,
                       memory_virtual=False, memory_maps=False,
                       cpu_percent=False, threads=False,
                       status=False, name=False, io_counters=False):
  import psutil
  if pid is None:
    pid = os.getpid()
  if pid in __process_pid_map:
    process = __process_pid_map[pid]
  else:
    process = psutil.Process(pid)
    __process_pid_map[pid] = process

  if status:
    return process.status()
  if name:
    return process.name()
  if io_counters:
    return process.io_counters()
  if memory_usage:
    return process.memory_info().rss / float(2 ** 20)
  if memory_shared:
    return process.memory_info().shared / float(2 ** 20)
  if memory_virtual:
    return process.memory_info().vms / float(2 ** 20)
  if memory_maps:
    return {i[0]: i[1] / float(2 ** 20)
            for i in process.memory_maps()}
  if cpu_percent:
    # first time call always return 0
    process.cpu_percent(None)
    return process.cpu_percent(None)
  if threads:
    return {i.id: (i.user_time, i.system_time) for i in process.threads()}


def get_system_status(memory_total=False, memory_total_actual=False,
                      memory_total_usage=False, memory_total_free=False,
                      all_pids=False, swap_memory=False, pid=False):
  """
  Parameters
  ----------
  threads: bool
      return dict {id: (user_time, system_time)}
  memory_maps: bool
      return dict {path: rss}

  Note
  ----
  All memory is returned in `MiB`
  To calculate memory_percent:
      get_system_status(memory_usage=True) / get_system_status(memory_total=True) * 100
  """
  import psutil
  # ====== general system query ====== #
  if memory_total:
    return psutil.virtual_memory().total / float(2 ** 20)
  if memory_total_actual:
    return psutil.virtual_memory().available / float(2 ** 20)
  if memory_total_usage:
    return psutil.virtual_memory().used / float(2 ** 20)
  if memory_total_free:
    return psutil.virtual_memory().free / float(2 ** 20)
  if swap_memory:
    tmp = psutil.swap_memory()
    tmp.total /= float(2 ** 20)
    tmp.used /= float(2 ** 20)
    tmp.free /= float(2 ** 20)
    tmp.sin /= float(2**20)
    tmp.sout /= float(2**20)
    return tmp
  if all_pids:
    return psutil.pids()
  if pid:
    return os.getpid()
