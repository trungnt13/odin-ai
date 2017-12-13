# ===========================================================================
# Available properties
#  * device
#  * device_info
#  * floatX
#  * epsilon
#  * multigpu
#  * optimizer
#  * cnmem
#  * backend
#  * seed
# ===========================================================================
from __future__ import division, absolute_import

import os
import re
import sys
import pip
import shutil
import tempfile
import subprocess
import warnings
from six import string_types
from multiprocessing import cpu_count
try:
  from functools import lru_cache
except ImportError:
  def lru_cache(maxsize=128):
    def tmp_func(func):
      return func
    return tmp_func

import numpy as np


# ===========================================================================
# Helper
# ===========================================================================
def _ctext(s, color='red'):
  # just a copy of ctext implementation to make
  # the config log look better.
  try:
    from colorama import Fore
    color = color.upper()
    color = getattr(Fore, color, '')
    return color + str(s) + Fore.RESET
  except ImportError:
    pass
  return s


def _warning(text):
  print(_ctext('[WARNING]', 'red'), text)


def _check_package_available(name):
  for i in pip.get_installed_distributions():
    if name.lower() == i.key.lower():
      return True
  return False


def _query_gpu_info():
  """ This function query GPU information:
  ngpu
  [device_name, device_compute_capability, device_total_memory]

  Note
  ----
  this function use deviceQuery command, so you better have it
  in your path
  """
  dev = {'ngpu': 1,
     # deviceName: [cardName, computeCapability, mem(MB)]
     'dev0': ['Unknown', 3.0, 1024]}
  temp_dir = tempfile.mkdtemp()
  p = os.path.join(temp_dir, 'tmp.txt')
  queried = subprocess.call('deviceQuery > ' + p,
          shell=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE) == 0
  dev = {}
  if queried: # found deviceQuery
    info = open(p, 'r').read()
    devNames = re.compile(r'Device \d: ".*"').findall(info)
    devNames = [i.strip().split(':')[-1].replace('"', '') for i in devNames]
    ngpu = len(devNames)
    comCap = re.compile(
        r'CUDA Capability Major\/Minor version number:\s*.*').findall(info)
    comCap = [float(i.strip().split(':')[-1]) for i in comCap]
    totalMems = re.compile(
        r'Total amount of global memory:\s*\d*').findall(info)
    totalMems = [int(i.strip().split(':')[-1]) for i in totalMems]
    # ====== create dev ====== #
    dev['ngpu'] = ngpu
    for i, (name, com, mem) in enumerate(zip(devNames, comCap, totalMems)):
      dev['dev%d' % i] = [name, com, mem]
  else:
    _warning('Cannot use "deviceQuery" to get GPU information for configuration.')
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    dev['ngpu'] = 0
    for i, name in (x.name for x in local_device_protos
        if x.device_type == 'GPU'):
      dev['dev%d' % i] = [name, None, None]
      dev['ngpu'] += 1
  # remove temp-dir
  shutil.rmtree(temp_dir)
  return dev


class AttributeDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__

  def __setstate__(self, states):
    for i, j in states:
      self[i] = j

  def __getstate__(self):
    return self.items()

# ===========================================================================
# Auto config
# ===========================================================================
CONFIG = None
EPS = None
_SESSION = {}
_RNG_GENERATOR = None


def set_session(session):
  global _SESSION
  _SESSION = session


def get_session(graph=None):
  """ Calling this method will make sure you create
  only 1 Session per graph.
  """
  # avoid duplicate Session for the same Graph when
  # graph is None
  if graph is None:
    import tensorflow as tf
    default_graph = tf.get_default_graph()
    if default_graph in _SESSION:
      _SESSION[None] = _SESSION[default_graph]
  # another case
  elif None in _SESSION and _SESSION[None].graph == graph:
    _SESSION[graph] = _SESSION[None]
  # ====== initialize tensorflow session ====== #
  if graph not in _SESSION:
    import tensorflow as tf
    session_args = {
        'intra_op_parallelism_threads': CONFIG['nthread'],
        'inter_op_parallelism_threads': CONFIG['ncpu'],
        'allow_soft_placement': True,
        'log_device_placement': CONFIG['debug'],
    }
    if CONFIG['ngpu'] > 0:
      if CONFIG['cnmem'] > 0:
        session_args['gpu_options'] = tf.GPUOptions(
            per_process_gpu_memory_fraction=CONFIG['cnmem'],
            allow_growth=False)
      else:
        session_args['gpu_options'] = tf.GPUOptions(
            allow_growth=True)
    _SESSION[graph] = tf.Session(config=tf.ConfigProto(**session_args),
             graph=graph)
  return _SESSION[graph]


def auto_config(config=None):
  ''' Auto-configure ODIN using os.environ['ODIN'].

  Parameters
  ----------
  config : object
    predefined config object return from auto_config
  check : boolean
    if True, raise Exception of CONFIG have not initialized

  Returns
  -------
  config : object
    simple object (kind of dictionary) contain all configuraitons

  '''
  global CONFIG
  global _RNG_GENERATOR
  global EPS
  if CONFIG is not None:
    warnings.warn('You should not auto_config twice, old configuration already '
        'existed, and cannot be re-configured.')
    return CONFIG
  # ====== specific pattern ====== #
  valid_cnmem_name = re.compile('(cnmem)[=]?[10]?\.\d*')
  valid_seed = re.compile('seed\D?(\d*)')

  floatX = 'float32'
  epsilon = 1e-8
  cnmem = 0.
  seed = 1208251813
  debug = False
  # number of devices
  ncpu = 0
  ngpu = False
  nthread = 0
  log_level = '3'
  # ====== parsing the config ====== #
  if config is None: # load config from flags
    odin_flags = os.getenv("ODIN", "")
    for c in (';', ':', '.', '*'):
      odin_flags.replace(c, ',')
    s = odin_flags.split(',')
    # ====== processing each tag ====== #
    for i in s:
      i = i.lower().strip()
      # ====== Data type ====== #
      if 'float' in i or 'int' in i:
        floatX = i
      # ====== Devices ====== #
      elif 'cpu' in i:
        if '=' not in i:
          _warning("Found `cpu` tag, but number of CPU is not "
             "specified, auto select all %d CPU-cores " %
             cpu_count())
        else:
          ncpu = min(int(i.split('=')[-1]), ncpu)
      elif 'gpu' in i:
        ngpu = True
      # ====== number thread ====== #
      elif 'thread' in i:
        if '=' not in i:
          _warning("Found `thread` tag, but number of thread is "
             "not specified, only 1 thread per process (CPU-core) is "
             "used by default.")
        else:
          nthread = int(i.split('=')[-1])
      # ====== cnmem ====== #
      elif 'cnmem' in i:
        match = valid_cnmem_name.match(i)
        if match is None:
          raise ValueError('Unsupport CNMEM format: %s. '
               'Valid format must be: cnmem=0.75 or cnmem=.75 '
               ' or cnmem.75' % str(i))
        i = i[match.start():match.end()].replace('cnmem', '').replace('=', '')
        cnmem = float(i)
      # ====== seed ====== #
      elif 'seed' in i:
        match = valid_seed.match(i)
        if match is None:
          raise ValueError('Invalid pattern for specifying seed value, '
               'you can try: [seed][non-digit][digits]')
        seed = int(match.group(1))
      # ====== debug ====== #
      elif 'debug' in i:
        debug = True
      # ====== log-leve ====== #
      elif 'log' in i:
        if '=' in i:
          log_level = i.split('=')[-1]
        else:
          log_level = '0'
  else: # load config from object
    floatX = config['floatX']
    ncpu = config['ncpu']
    ngpu = config['ngpu']
    nthread = config['nthread']
    cnmem = config['cnmem']
    seed = config['seed']
    debug = config['debug']
  # epsilon
  EPS = np.finfo(np.dtype(floatX)).eps
  # devices
  dev = {}
  if ngpu:
    dev.update(_query_gpu_info())
  else:
    dev['ngpu'] = 0
  dev['ncpu'] = ncpu
  dev['nthread'] = nthread

  # ====== Log the configuration ====== #
  def print_log(tag, value, nested=False):
    if nested:
      s = _ctext('  * ', 'MAGENTA')
    else:
      s = '[Auto-Config] '
    s += _ctext(tag, 'yellow') + ' : '
    s += _ctext(value, 'cyan') + '\n'
    sys.stderr.write(s)
  print_log('#CPU', 'auto' if dev['ncpu'] == 0 else dev['ncpu'])
  print_log('#Thread/core',
      'auto' if dev['nthread'] == 0 else dev['nthread'],
      nested=True)
  print_log('#GPU', dev['ngpu'])
  if dev['ngpu'] > 0:
    for i in range(dev['ngpu']):
      print_log('GPU-dev%d' % i, dev['dev%d' % i], nested=True)
  print_log('FloatX', floatX)
  print_log('Epsilon', epsilon)
  print_log('CNMEM', cnmem)
  print_log('SEED', seed)
  print_log('Debug', debug)
  print_log('Log-level', log_level)
  # ==================== create theano flags ==================== #
  if dev['ngpu'] == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
  # ====== Return global objects ====== #
  CONFIG = AttributeDict()
  CONFIG.update({
      'ncpu': dev['ncpu'],
      'ngpu': dev['ngpu'],
      'nthread': dev['nthread'],
      'device_info': dev,
      'floatX': floatX,
      'epsilon': epsilon,
      'cnmem': cnmem,
      'seed': seed,
      'debug': debug
  })
  _RNG_GENERATOR = np.random.RandomState(seed=seed)
  # tensorflow log level
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
  return CONFIG


# ===========================================================================
# Getter
# ===========================================================================
def __validate_config():
  if CONFIG is None:
    raise Exception("auto_config has not been called.")


def get_rng():
  """return the numpy random state as a Randomness Generator"""
  global _RNG_GENERATOR
  if _RNG_GENERATOR is None:
    _RNG_GENERATOR = np.random.RandomState(seed=120825)
  return _RNG_GENERATOR


def randint(low=0, high=10e8, size=None, dtype='int32'):
  """Randomly generate and integer seed for any stochastic function."""
  return _RNG_GENERATOR.randint(low=low, high=high, size=size,
    dtype=dtype)


def get_ncpu():
  """ Return number of inter_op_parallelism_threads """
  __validate_config()
  return CONFIG['ncpu']


def get_ngpu():
  """ Return number of GPU """
  __validate_config()
  return CONFIG['ngpu']


def get_nthread():
  """ Return number of intra_op_parallelism_threads """
  __validate_config()
  return CONFIG['nthread']


def get_nb_processors():
  """ In case using CPU, return number of cores
  If GPU is used, return number of graphics card.
  """
  __validate_config()
  return CONFIG['device_info']['n']


def get_device_info():
  """ Device info contains:
  {"n": ngpu,
   "dev0": [device_name, device_compute_capability, device_total_memory],
   "dev1": [device_name, device_compute_capability, device_total_memory],
   ...
  }
  """
  __validate_config()
  return CONFIG['device_info']


def get_floatX():
  __validate_config()
  return CONFIG['floatX']


def get_optimizer():
  __validate_config()
  return CONFIG['optimizer']


def get_cnmem():
  __validate_config()
  return CONFIG['cnmem']


def get_backend():
  __validate_config()
  return CONFIG['backend']


def get_seed():
  __validate_config()
  return CONFIG['seed']
