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
  return str(s)

def _warning(text):
  print(_ctext('[WARNING]', 'red'), text)

def _check_package_available(name):
  for i in pip.get_installed_distributions():
    if name.lower() == i.key.lower():
      return True
  return False

def get_gpu_info():
  """ Example of return
  [
    {'name': 'Titan TITAN V', 'id': 0,
     'fan': 53,
     'mem_total': 12028, 'mem_used': 4295,
     'graphics_clock': 1200, 'mem_clock': 850
    },
    {'name': 'GeForce TITAN Xp', 'id': 1,
    'fan': 23,
    'mem_total': 12196, 'mem_used': 0,
    'graphics_clock': 1404, 'mem_clock': 5705
    }
  ]
  """
  temp_dir = tempfile.mkdtemp()
  p = os.path.join(temp_dir, 'tmp.txt')
  queried = subprocess.call('nvidia-smi -q -x > ' + p,
          shell=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE) == 0
  devices = []
  if queried: # found nvidia-smi
    from xml.etree import ElementTree as ET
    tree = ET.parse(p)
    root = tree.getroot()
    for child in root:
      if child.tag.lower() != 'gpu':
        continue
      gpu = {}
      brand = ''
      for i in child:
        tag = i.tag.lower()
        if tag == 'product_name':
          gpu['name'] = i.text
        elif tag == 'product_brand':
          brand = i.text
        elif tag == 'minor_number':
          gpu['id'] = int(i.text)
        elif tag == 'fan_speed':
          gpu['fan'] = int(i.text.split(' ')[0])
        elif tag == 'fb_memory_usage':
          gpu['mem_total'] = int(i.findall('total')[0].text.split(' ')[0])
          gpu['mem_used'] = int(i.findall('used')[0].text.split(' ')[0])
        elif tag == 'clocks':
          for j in i:
            gpu[j.tag] = int(j.text.split(' ')[0])
      gpu['name'] = brand + ' ' + gpu['name']
      devices.append(gpu)
    # sort by minor number
    devices = sorted(devices, key=lambda x: x['id'])
  else: # NO GPU devices
    pass
  # remove temp-dir
  shutil.rmtree(temp_dir)
  return devices

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

def get_session_config():
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
      session_args['gpu_options'] = tf.GPUOptions(allow_growth=True)
  return session_args

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
    session_args = get_session_config()
    sess = tf.Session(config=tf.ConfigProto(**session_args), graph=graph)
    _SESSION[graph] = sess
    with sess.graph.as_default():
      tf.Variable(initial_value=False, dtype=tf.bool, name='IsTraining__',
                  trainable=False)
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
  valid_cnmem_name = re.compile(r"(cnmem)[=]?[10]?\.\d*")
  valid_seed = re.compile(r"seed\D?(\d*)")

  floatX = 'float32'
  epsilon = 1e-8
  cnmem = 0.
  seed = 1208251813
  debug = False
  # number of devices
  ncpu = 0
  nthread = 0
  log_level = '3'

  # Handling GPU devices
  #  * Nothing => "0" => use first GPU found
  #  * "gpu" => "-1" => use all available GPU devices
  #  * "gpu=" => "" => remove all GPU devices
  #  * "gpu=0_1" => "0,1" => use 0-th and 1-st GPU
  #  * "gpu=0_1_-1" => "-1" => use all available GPU devices
  ngpu = "0"
  # ====== parsing the config ====== #
  if config is None: # load config from flags
    odin_flags = os.getenv("ODIN", "")
    for c in (';', ':', '.', '*'):
      odin_flags.replace(c, ',')
    for c in ('+', '_'):
      odin_flags.replace(c, '_')
    s = odin_flags.split(',')
    # ====== processing each tag ====== #
    for i in s:
      i = i.lower().strip()
      # ====== Data type ====== #
      if 'float' in i or 'int' in i:
        floatX = i
      # ====== Devices ====== #
      elif 'cpu' in i: # CPU devices
        if '=' not in i:
          _warning("Found `cpu` tag, but number of CPU is not "
             "specified, auto select all %d CPU-cores " %
             cpu_count())
        else:
          ncpu = min(int(i.split('=')[-1]), cpu_count())
      elif 'gpu' in i: # GPU devices
        # given a set of GPU devices, multiple devices specified by 0,1,2,3
        if '=' in i:
          ngpu = i.split('=')[-1]
          if len(ngpu) > 0:
            try:
              [int(_) for _ in ngpu.split('_')]
            except Exception as e:
              raise RuntimeError("Invalid input for GPU configuration")
            if any(int(_) < 0 for _ in ngpu.split('_')):
              ngpu = "-1"
        # use all available GPU devices
        else:
          ngpu = "-1"
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
  # ====== set log level ====== #
  # This only worked if I put the os.environ before tensorflow was imported
  # 0 = all messages are logged (default behavior)
  # 1 = INFO messages are not printed
  # 2 = INFO and WARNING messages are not printed
  # 3 = INFO, WARNING, and ERROR messages are not printed
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level
  # ====== get device info ====== #
  # epsilon
  EPS = np.finfo(np.dtype(floatX)).eps
  # devices
  dev = {}
  dev['ncpu'] = ncpu
  dev['nthread'] = nthread
  # ==================== processing GPU devices ==================== #
  gpu_info = get_gpu_info()
  if len(ngpu) > 0:
    if ngpu == '-1':
      dev['ngpu'] = len(gpu_info)
    else:
      dev['ngpu'] = min(len(gpu_info), len(ngpu.split('_')))
  else:
    dev['ngpu'] = 0
  #
  if dev['ngpu'] == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    gpu_info = []
  elif ngpu == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(
        [str(i) for i in range(len(gpu_info))])
  else:
    all_gpu = sorted([_
                      for _ in ngpu.split('_')
                      if int(_) < len(gpu_info)])
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(all_gpu)
    gpu_info = [gpu_info[int(i)] for i in all_gpu]
  dev['gpu'] = gpu_info
  dev['gpu_indices'] = [
      int(i)
      for i in str(os.environ['CUDA_VISIBLE_DEVICES']).split(',')
      if len(i) > 0 and i.isdigit()]

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
  print_log('#CPU-native', str(cpu_count()))
  print_log('#Thread/core',
      'auto' if dev['nthread'] == 0 else dev['nthread'],
      nested=True)
  print_log('#GPU', dev['ngpu'])
  if dev['ngpu'] > 0:
    for i in dev['gpu']:
      print_log('[%d]' % i['id'], i['name'], nested=True)
  print_log('FloatX', floatX)
  print_log('Epsilon', epsilon)
  print_log('CNMEM', cnmem)
  print_log('SEED', seed)
  print_log('Log-devices', debug)
  print_log('TF-log-level', log_level)
  # ====== Return global objects ====== #
  CONFIG = AttributeDict()
  CONFIG.update({
      'ncpu': dev['ncpu'],
      'ngpu': dev['ngpu'],
      'nthread': dev['nthread'],
      'gpu_indices': dev['gpu_indices'],
      'device_info': dev,
      'floatX': floatX,
      'epsilon': epsilon,
      'cnmem': cnmem,
      'seed': seed,
      'debug': debug
  })
  _RNG_GENERATOR = np.random.RandomState(seed=seed)
  with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=ImportWarning)
    import tensorflow as tf
    the_seed = _RNG_GENERATOR.randint(0, 10e8)
    tf.set_random_seed(seed=the_seed)
    np.random.seed(seed=the_seed)
  # ====== set the random seed for everything ====== #
  return CONFIG

# ===========================================================================
# Getter
# ===========================================================================
def __validate_config():
  if CONFIG is None:
    raise Exception("auto_config has not been called.")

def get_rng():
  """
    Return
    ------
    the numpy RandomState as a "Randomness Generator"
  """
  global _RNG_GENERATOR
  if _RNG_GENERATOR is None:
    _RNG_GENERATOR = np.random.RandomState(seed=120825)
  return _RNG_GENERATOR

get_random_state = get_rng

def randint(low=0, high=10e8, size=None, dtype='int32'):
  """Randomly generate and integer seed for any stochastic function."""
  return _RNG_GENERATOR.randint(low=low, high=high, size=size,
    dtype=dtype)

def get_ncpu():
  """ Return number of inter_op_parallelism_threads """
  __validate_config()
  return CONFIG['ncpu']

def get_ncpu_native():
  return cpu_count()

def get_ngpu():
  """ Return number of GPU """
  __validate_config()
  return CONFIG['ngpu']

def get_gpu_indices():
  __validate_config()
  return CONFIG['gpu_indices']

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

def get_random_seed():
  __validate_config()
  return CONFIG['seed']
