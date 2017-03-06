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
import subprocess
import warnings
from multiprocessing import cpu_count

import numpy

from odin.utils import TemporaryDirectory


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
    with TemporaryDirectory() as p:
        p = os.path.join(p, 'tmp.txt')
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
    return dev


# ===========================================================================
# Auto config
# ===========================================================================
CONFIG = None


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
    backend = 'tensorflow'
    optimizer = 'fast_run'
    epsilon = 1e-8
    device = 'cpu'
    cnmem = 0.
    seed = 1208251813
    multigpu = False

    if config is None: # load config from flags
        ODIN_FLAGS = os.getenv("ODIN", "")
        s = ODIN_FLAGS.split(',')
        # ====== processing each tag ====== #
        for i in s:
            i = i.lower().strip()
            # ====== Data type ====== #
            if 'float' in i or 'int' in i:
                floatX = i
            # ====== Backend ====== #
            elif 'theano' in i:
                backend = 'theano'
            elif 'tensorflow' in i or 'tf' in i:
                backend = 'tensorflow'
            # ====== Devices ====== #
            elif 'multigpu' in i:
                multigpu = True
                device = 'gpu'
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
            elif 'fast_compile' in i:
                optimizer = 'fast_compile'
            # ====== seed ====== #
            elif 'seed' in i:
                match = valid_seed.match(i)
                if match is None:
                    raise ValueError('Invalid pattern for specifying seed value, '
                                     'you can try: [seed][non-digit][digits]')
                seed = int(match.group(1))
    else: # load config from object
        floatX = config['floatX']
        backend = config['backend']
        optimizer = config['optimizer']
        multigpu = config['multigpu']
        device = config['device']
        cnmem = config['cnmem']
        seed = config['seed']
    # adject epsilon
    if floatX == 'float16':
        epsilon = 10e-5
    elif floatX == 'float32':
        epsilon = 10e-8
    elif floatX == 'float64':
        epsilon = 10e-12
    # ====== Log the configuration ====== #
    sys.stderr.write('[Auto-Config] Device : %s\n' % device)
    sys.stderr.write('[Auto-Config] Multi-GPU : %s\n' % multigpu)
    sys.stderr.write('[Auto-Config] Backend: %s\n' % backend)
    sys.stderr.write('[Auto-Config] Optimizer: %s\n' % optimizer)
    sys.stderr.write('[Auto-Config] FloatX : %s\n' % floatX)
    sys.stderr.write('[Auto-Config] Epsilon: %s\n' % epsilon)
    sys.stderr.write('[Auto-Config] CNMEM  : %s\n' % cnmem)
    sys.stderr.write('[Auto-Config] SEED  : %s\n' % seed)
    if device == 'gpu':
        dev = _query_gpu_info()
        if not multigpu:
            dev = {'n': 1, 'dev0': dev['dev0']}
    else:
        dev = {'n': cpu_count()}
    # ==================== create theano flags ==================== #
    ########## Theano
    if backend == 'theano':
        if multigpu and not _check_package_available('pygpu'):
            raise Exception('"multigpu" option in theano requires installation of '
                            'libgpuarray and pygpu.')
        if device == 'cpu':
            contexts = "device=%s" % device
        else:
            if not _check_package_available('pygpu'): # single gpu
                contexts = 'device=gpu'
            else: # multi gpu
                contexts = "device=cuda"
                # contexts = 'contexts='
                # contexts += ';'.join(["dev%d->cuda%d" % (j, j)
                #                       for j in range(dev['n'])])
        flags = contexts + ",mode=FAST_RUN,floatX=%s" % floatX
        # ====== others ====== #
        # Speedup CuDNNv4
        flags += (',dnn.conv.algo_fwd=time_once' +
                  ',dnn.conv.algo_bwd_filter=time_once' +
                  ',dnn.conv.algo_bwd_data=time_once')
        # CNMEM
        if cnmem > 0. and cnmem <= 1.:
            flags += ',lib.cnmem=%.2f,allow_gc=True' % cnmem
        if len(optimizer) > 0:
            flags += ',optimizer={}'.format(optimizer)
        flags += ',optimizer_including=unsafe'
        flags += ',exception_verbosity=high'
        os.environ['THEANO_FLAGS'] = flags
        import theano
    ########## Tensorflow
    elif backend == 'tensorflow':
        if device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        else:
            pass
        import tensorflow
    else:
        raise ValueError('Unsupport backend: ' + backend)

    # ====== Return global objects ====== #
    class AttributeDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    global CONFIG
    CONFIG = AttributeDict()
    CONFIG.update({'device': device,
                   'device_info': dev,
                   'floatX': floatX, 'epsilon': epsilon,
                   'multigpu': multigpu, 'optimizer': optimizer,
                   'cnmem': cnmem, 'backend': backend,
                   'seed': seed})
    global _RNG_GENERATOR
    _RNG_GENERATOR = numpy.random.RandomState(seed=seed)
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
