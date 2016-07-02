from __future__ import division, absolute_import

import os
import sys
import re
import subprocess
import warnings
from multiprocessing import cpu_count

import numpy

from odin.utils import TemporaryDirectory


# ===========================================================================
# Helper
# ===========================================================================
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
    return dev


# ===========================================================================
# Auto config
# ===========================================================================
RNG_GENERATOR = numpy.random.RandomState()


def auto_config(config=None):
    if 'autoconfig' in globals():
        warnings.warn('You should not auto_config twice, object autoconfig has'
                      'already exists!')
        return autoconfig
    # ====== specific pattern ====== #
    valid_cnmem_name = re.compile('(cnmem)[=]?[10]?\.\d*')
    valid_seed = re.compile('seed\D?(\d*)')

    floatX = 'float32'
    backend = 'theano'
    optimizer = 'fast_run'
    epsilon = 10e-8
    device = []
    cnmem = 0.
    seed = 1208251813

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
            elif 'cpu' == i and len(device) == 0:
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
    sys.stderr.write('[Auto-Config] Backend: %s\n' % backend)
    sys.stderr.write('[Auto-Config] Optimizer: %s\n' % optimizer)
    sys.stderr.write('[Auto-Config] FloatX : %s\n' % floatX)
    sys.stderr.write('[Auto-Config] Epsilon: %s\n' % epsilon)
    sys.stderr.write('[Auto-Config] CNMEM  : %s\n' % cnmem)
    sys.stderr.write('[Auto-Config] SEED  : %s\n' % seed)

    if device == 'gpu':
        dev = _query_gpu_info()
    else:
        dev = {'n': cpu_count()}

    # ==================== create theano flags ==================== #
    if backend == 'theano':
        if device == 'cpu':
            contexts = "device=%s" % device
        else:
            contexts = "contexts="
            contexts += ';'.join(["dev%d->cuda%d" % (j, j)
                                  for j in range(dev['n'])])
            # TODO: bizarre degradation in performance if not specify device=gpu
            # device = 'device=gpu'
        flags = contexts + ",mode=FAST_RUN,floatX=%s" % floatX
        # ====== others ====== #
        flags += ',exception_verbosity=high'
        # Speedup CuDNNv4
        flags += ',dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once'
        # CNMEM
        if cnmem > 0. and cnmem <= 1.:
            flags += ',lib.cnmem=%.2f,allow_gc=True' % cnmem
        os.environ['THEANO_FLAGS'] = flags
        if len(optimizer) > 0:
            flags += ',optimizer={}'.format(optimizer)
        flags += ',optimizer_including=unsafe'
        import theano
    elif backend == 'tensorflow':
        import tensorflow
    else:
        raise ValueError('Unsupport backend: ' + backend)

    # ====== Return global objects ====== #
    class AttributeDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    config = AttributeDict()
    config.update({'device': device, 'floatX': floatX, 'epsilon': epsilon,
                   'optimizer': optimizer, 'cnmem': cnmem,
                   'backend': backend, 'seed': seed})

    global RNG_GENERATOR
    RNG_GENERATOR = numpy.random.RandomState()
    global device
    device = dev
    global autoconfig
    autoconfig = config
    return config
