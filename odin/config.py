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

import numpy


# ===========================================================================
# Helper
# ===========================================================================
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
    dev = {'n': 1,
           # deviceName: [cardName, computeCapability, mem(MB)]
           'dev0': ['Unknown', 3.0, 1024]}
    temp_dir = tempfile.mkdtemp()
    p = os.path.join(temp_dir, 'tmp.txt')
    queried = subprocess.call('deviceQuery > ' + p,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) == 0
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
        dev = {'n': ngpu}
        for i, (name, com, mem) in enumerate(zip(devNames, comCap, totalMems)):
            dev['dev%d' % i] = [name, com, mem]
    else:
        print('[WARNING] Cannot use "deviceQuery" to get GPU information for configuration.')
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
_SESSION = None


def set_session(session):
    global _SESSION
    _SESSION = session


def get_session():
    return _SESSION


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
    if CONFIG is not None:
        warnings.warn('You should not auto_config twice, old configuration already '
                      'existed, and cannot be re-configured.')
        return CONFIG
    # ====== specific pattern ====== #
    valid_cnmem_name = re.compile('(cnmem)[=]?[10]?\.\d*')
    valid_seed = re.compile('seed\D?(\d*)')

    floatX = 'float32'
    epsilon = 1e-8
    device = 'cpu'
    cnmem = 0.
    seed = 1208251813
    debug = False
    if config is None: # load config from flags
        ODIN_FLAGS = os.getenv("ODIN", "")
        s = ODIN_FLAGS.split(',')
        # ====== processing each tag ====== #
        for i in s:
            i = i.lower().strip()
            # ====== Data type ====== #
            if 'float' in i or 'int' in i:
                floatX = i
            # ====== Devices ====== #
            elif 'cpu' in i:
                device = 'cpu'
            elif 'gpu' in i:
                device = 'gpu'
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
    else: # load config from object
        floatX = config['floatX']
        device = config['device']
        cnmem = config['cnmem']
        seed = config['seed']
        debug = config['debug']
    # adject epsilon
    if floatX == 'float16':
        epsilon = 10e-5
    elif floatX == 'float32':
        epsilon = 10e-8
    elif floatX == 'float64':
        epsilon = 10e-12
    # ====== Log the configuration ====== #
    sys.stderr.write('[Auto-Config] Device : %s\n' % device)
    sys.stderr.write('[Auto-Config] FloatX : %s\n' % floatX)
    sys.stderr.write('[Auto-Config] Epsilon: %s\n' % epsilon)
    sys.stderr.write('[Auto-Config] CNMEM  : %s\n' % cnmem)
    sys.stderr.write('[Auto-Config] SEED  : %s\n' % seed)
    sys.stderr.write('[Auto-Config] Debug  : %s\n' % debug)
    if device == 'gpu':
        dev = _query_gpu_info()
    else:
        dev = {'n': cpu_count()}
    # ==================== create theano flags ==================== #
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    # ====== Return global objects ====== #
    global CONFIG
    CONFIG = AttributeDict()
    CONFIG.update({'device': device,
                   'device_info': dev,
                   'floatX': floatX, 'epsilon': epsilon,
                   'cnmem': cnmem, 'seed': seed,
                   'debug': debug})
    global _RNG_GENERATOR
    _RNG_GENERATOR = numpy.random.RandomState(seed=seed)
    # ====== initialize tensorflow session ====== #
    import tensorflow as tf
    global _SESSION
    __session_args = {
        'intra_op_parallelism_threads': CONFIG['device_info']['n'],
        'allow_soft_placement': True,
        'log_device_placement': debug,
    }
    if CONFIG['device'] == 'gpu':
        if CONFIG['cnmem'] > 0:
            __session_args['gpu_options'] = tf.GPUOptions(
                per_process_gpu_memory_fraction=CONFIG['cnmem'],
                allow_growth=False)
        else:
            __session_args['gpu_options'] = tf.GPUOptions(
                allow_growth=True)
    _SESSION = tf.InteractiveSession(config=tf.ConfigProto(**__session_args))
    return CONFIG


# ===========================================================================
# Getter
# ===========================================================================
def __validate_config():
    if CONFIG is None:
        raise Exception("auto_config has not been called.")


def get_rng():
    """return the numpy random state as a Randomness Generator"""
    return _RNG_GENERATOR


def randint(low=0, high=10e8, size=None, dtype='int32'):
    """Randomly generate and integer seed for any stochastic function."""
    return _RNG_GENERATOR.randint(low=low, high=high, size=size,
        dtype=dtype)


def get_device():
    """ Return type of device: cpu or gpu """
    __validate_config()
    return CONFIG['device']


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


def get_epsilon():
    __validate_config()
    return CONFIG['epsilon']


def get_multigpu():
    __validate_config()
    return CONFIG['multigpu']


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
