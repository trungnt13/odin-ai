from __future__ import print_function
import os
import sys
import time
import math
import types
import signal
import shutil
import timeit
import numbers
import subprocess
import tempfile
import contextlib
import platform
import argparse
from multiprocessing import cpu_count, Value, Lock, current_process
from collections import OrderedDict, deque, Iterable

from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError, HTTPError
import tarfile

try:
    from numba import jit, autojit, vectorize, guvectorize
except:
    pass
import numpy
import six

from .profile import *


class ArgController(object):
    """ Simple interface to argparse """

    def __init__(self, version='1.00', print_parsed=True):
        super(ArgController, self).__init__()
        self.parser = None
        self._require_input = False
        self.arg_dict = {}
        self.name = []
        self.version = str(version)
        self.print_parsed = print_parsed

    def _is_positional(self, name):
        if name[0] != '-' and name[:2] != '--':
            return True
        return False

    def _parse_input(self, key, val):
        # ====== search if manual preprocessing available ====== #
        for i, preprocess in self.arg_dict.iteritems():
            if key in i and preprocess is not None:
                return preprocess(val)
        # ====== auto preprocess ====== #
        try:
            val = float(val)
            if int(val) == val:
                val = int(val)
        except:
            val = str(val)
            val_lower = val.lower()
            if val_lower == 'true' or val_lower == 'yes':
                val = True
            elif val_lower == 'false' or val_lower == 'no':
                val = False
        return val

    def add(self, name, help, default=None, preprocess=None):
        """ NOTE: if the default value is not given, the argument is
        required

        Parameters
        ----------
        preprocess: callable
            take in the parsed argument and preprocess it into necessary
            information

        """
        if self.parser is None:
            self.parser = argparse.ArgumentParser(
                description='Automatic argument parser (yes,true > True; no,false > False)',
                version=self.version, add_help=True)

        if default is None:
            if self._is_positional(name):
                self.parser.add_argument(name, help=help, type=str, action="store",
                    metavar='')
            else:
                self.parser.add_argument(name, help=help, type=str, action="store",
                    required=True, metavar='')
            self._require_input = True
        else:
            help += ' (default: %s)' % str(default)
            self.parser.add_argument(name, help=help, type=str, action="store",
                default=str(default), metavar='')

        # store preprocess dictionary
        if not callable(preprocess):
            preprocess = None
        self.arg_dict[name] = preprocess
        return self

    def parse(self):
        if self.parser is None:
            raise Exception('Call add to assign at least 1 argument for '
                            'for the function.')
        exit_now = False
        try:
            if len(sys.argv) == 1 and self._require_input:
                self.parser.print_help()
                exit_now = True
            else:
                args = self.parser.parse_args()
        except:
            # if specfy version or help, don't need to print anything else
            if all(i not in ['-h', '--help', '-v', '--version']
                   for i in sys.argv):
                self.parser.print_help()
            exit_now = True
        if exit_now: exit()
        # parse the arguments
        try:
            args = {i: self._parse_input(i, j)
                    for i, j in args._get_kwargs()}
        except Exception, e:
            print('Error parsing given arguments: "%s"' % str(e))
            self.parser.print_help()
            exit()
        # reset everything
        self.parser = None
        self.arg_dict = {}
        self._require_input = False
        # print the parsed arguments if necessary
        if self.print_parsed:
            max_len = max(len(i) for i in args.keys())
            max_len = '%-' + str(max_len) + 's'
            print('\n******** Parsed arguments ********')
            for i, j in args.iteritems():
                print(max_len % i, ': ', j)
            print('**********************************\n')
        return args


def raise_return(e):
    raise e


def one_hot(y, n_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = numpy.asarray(y, dtype='int32')
    if not n_classes:
        n_classes = numpy.max(y) + 1
    Y = numpy.zeros((len(y), n_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for numpy.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = numpy.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = numpy.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : numpy.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `numpy.pad()`

    Returns
    -------
    data_padded : numpy.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ParameterError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(('Target size ({:d}) must be '
                          'at least input size ({:d})').format(size,
                                                               n))

    return numpy.pad(data, lengths, **kwargs)


def segment_list(l, size=None, n_seg=None):
    '''
    Example
    -------
    >>> segment_list([1,2,3,4,5],2)
    >>> [[1, 2, 3], [4, 5]]
    >>> segment_list([1,2,3,4,5],4)
    >>> [[1], [2], [3], [4, 5]]
    '''
    # by floor, make sure and process has it own job
    if size is None:
        size = int(numpy.ceil(len(l) / float(n_seg)))
    else:
        n_seg = int(numpy.ceil(len(l) / float(size)))
    if size * n_seg - len(l) > size:
        size = int(numpy.floor(len(l) / float(n_seg)))
    # start segmenting
    segments = []
    for i in range(n_seg):
        start = i * size
        if i < n_seg - 1:
            end = start + size
        else:
            end = max(start + size, len(l))
        segments.append(l[start:end])
    return segments


def segment_axis(a, frame_length=2048, hop_length=512,
    axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    This method has been implemented by Anne Archibald,
    as part of the talk box toolkit
    example::

        segment_axis(arange(10), 4, 2)
        array([[0, 1, 2, 3],
           ( [2, 3, 4, 5],
             [4, 5, 6, 7],
             [6, 7, 8, 9]])

    :param a: the array to segment
    :param length: the length of each frame
    :param overlap: the number of array elements by which the frames should overlap
    :param axis: the axis to operate on; if None, act on the flattened array
    :param end: what to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value

    :param endvalue: the value to use for end='pad'

    :return: a ndarray

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').

    Note
    ----
    Modified work and error fixing Copyright (c) TrungNT

    """
    if axis is None:
        a = numpy.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]
    overlap = frame_length - hop_length

    if overlap >= frame_length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or frame_length <= 0:
        raise ValueError("overlap must be nonnegative and length must" +
                         "be positive")

    if l < frame_length or (l - frame_length) % (frame_length - overlap):
        if l > frame_length:
            roundup = frame_length + (1 + (l - frame_length) // (frame_length - overlap)) * (frame_length - overlap)
            rounddown = frame_length + ((l - frame_length) // (frame_length - overlap)) * (frame_length - overlap)
        else:
            roundup = frame_length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (frame_length - overlap) \
        or (roundup == frame_length and rounddown == 0)
        a = a.swapaxes(-1, axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad', 'wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = numpy.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]
            a = b
        a = a.swapaxes(-1, axis)
        l = a.shape[0] # update l

    if l == 0:
        raise ValueError("Not enough data points to segment array " +
                "in 'cut' mode; try 'pad' or 'wrap'")
    assert l >= frame_length
    assert (l - frame_length) % (frame_length - overlap) == 0
    n = 1 + (l - frame_length) // (frame_length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, frame_length) + a.shape[axis + 1:]
    newstrides = a.strides[:axis] + ((frame_length - overlap) * s, s) + a.strides[axis + 1:]

    try:
        return numpy.ndarray.__new__(numpy.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((frame_length - overlap) * s, s) \
        + a.strides[axis + 1:]
        return numpy.ndarray.__new__(numpy.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


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


def as_shape_tuple(shape):
    if isinstance(shape, numbers.Number):
        shape = (int(shape),)
    if not isinstance(shape, (tuple, list)):
        raise ValueError('We only accept shape in tuple or list form.')
    shape = tuple([int(i) if i is not None and i >= 0 else None for i in shape])
    if len([i for i in shape if i is None]) >= 2:
        raise Exception('Shape tuple can only have 1 unknown dimension.')
    return shape


def as_tuple(x, N=None, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).

    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
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
    if not isinstance(x, tuple):
        if isinstance(x, (types.GeneratorType, types.ListType)):
            x = tuple(x)
        else:
            x = (x,)
    # ====== check length ====== #
    if isinstance(N, numbers.Number):
        N = int(N)
        if len(x) == 1:
            x = x * N
        elif len(x) != N:
            raise ValueError('x has length=%d, but required length N=%d' %
                             (len(x), N))
    # ====== check type ====== #
    if (t is not None) and not all(isinstance(v, t) for v in x):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))
    return x


def as_list(x, N=None, t=None):
    return list(as_tuple(x, N, t))


# ===========================================================================
# Python
# ===========================================================================
class struct(object):

    '''Flexible object can be assigned any attribtues'''

    def __getitem__(self, x):
        return getattr(self, str(x))


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
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v
            self._inv[v] = k

    def __delitem__(self, key):
        del self._inv[super(bidict, self).__getitem__(key)]
        return dict.__delitem__(self, key)


class queue(object):

    """ FIFO, fast, NO thread-safe queue
    put : append to end of list
    append : append to end of list
    pop : remove data from end of list
    get : remove data from end of list
    empty : check if queue is empty
    clear : remove all data in queue
    """

    def __init__(self):
        super(queue, self).__init__()
        self._data = []
        self._idx = 0

    # ====== queue ====== #
    def put(self, value):
        self._data.append(value)

    def append(self, value):
        self._data.append(value)

    # ====== dequeue ====== #
    def pop(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    def get(self):
        if self._idx == len(self._data):
            raise ValueError('Queue is empty')
        self._idx += 1
        return self._data[self._idx - 1]

    # ====== dqueue with default ====== #
    def pop_default(self, default=None):
        if self._idx == len(self._data):
            return default
        self._idx += 1
        return self._data[self._idx - 1]

    def get_default(self, default=None):
        if self._idx == len(self._data):
            return default
        self._idx += 1
        return self._data[self._idx - 1]

    def empty(self):
        if self._idx == len(self._data):
            return True
        return False

    def clear(self):
        del self._data
        self._data = []
        self._idx = 0

    def __len__(self):
        return len(self._data) - self._idx


class Progbar(object):

    '''
    This function is adpated from: https://github.com/fchollet/keras
    Original work Copyright (c) 2014-2015 keras contributors
    Modified work Copyright 2016-2017 TrungNT
    '''

    def __init__(self, target, title=''):
        '''
            @param target: total number of steps expected
        '''
        self.width = 39
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.title = title

    def update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()

        prev_total_width = self.total_width
        sys.stdout.write("\b" * prev_total_width)
        sys.stdout.write("\r")

        numdigits = int(numpy.floor(numpy.log10(self.target))) + 1
        barstr = '%s %%%dd/%%%dd [' % (self.title, numdigits, numdigits)
        bar = barstr % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width - 1))
            if current < self.target:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
        sys.stdout.write(bar)
        self.total_width = len(bar)

        if current:
            time_per_unit = (now - self.start) / current
        else:
            time_per_unit = 0
        eta = time_per_unit * (self.target - current)
        info = ''
        if current < self.target:
            info += ' - ETA: %ds' % eta
        else:
            info += ' - %ds' % (now - self.start)
        for k in self.unique_values:
            info += ' - %s:' % k
            if type(self.sum_values[k]) is list:
                avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                if abs(avg) > 1e-3:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            else:
                info += ' %s' % self.sum_values[k]

        self.total_width += len(info)
        if prev_total_width > self.total_width:
            info += ((prev_total_width - self.total_width) * " ")

        sys.stdout.write(info)
        if current >= self.target:
            if "Linux" in platform.platform():
                sys.stdout.write("\n\n")
            else:
                sys.stdout.write("\n")
        sys.stdout.flush()

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


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


def get_file(fname, origin, untar=False):
    '''
    This function is adpated from: https://github.com/fchollet/keras
    Original work Copyright (c) 2014-2015 keras contributors
    Modified work Copyright 2016-2017 TrungNT
    '''
    datadir = get_datasetpath()
    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print('Downloading data from', origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count * block_size)

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
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath

    return fpath


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


def get_module_from_path(identifier, prefix='', suffix='', path='.', exclude='',
                  prefer_compiled=False):
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

    Returns
    -------
    list(object, function, ..) :
        any thing match given identifier in all found script file

    Notes
    -----
    File with multiple . character my procedure wrong results
    If the script run this this function match the searching process, a
    infinite loop may happen!
    '''
    import re
    import imp
    from inspect import getmembers
    # ====== validate input ====== #
    if exclude == '': exclude = []
    if type(exclude) not in (list, tuple, numpy.ndarray):
        exclude = [exclude]
    prefer_flag = -1
    if prefer_compiled: prefer_flag = 1
    # ====== create pattern and load files ====== #
    pattern = re.compile('^%s.*%s\.pyc?' % (prefix, suffix)) # py or pyc
    files = os.listdir(path)
    files = [f for f in files
             if pattern.match(f) and
             sum([i in f for i in exclude]) == 0]
    # ====== remove duplicated pyc files ====== #
    files = sorted(files, key=lambda x: prefer_flag * len(x)) # pyc is longer
    # .pyc go first get overrided by .py
    files = sorted({f.split('.')[0]: f for f in files}.values())
    # ====== load all modules ====== #
    modules = []
    for f in files:
        try:
            if '.pyc' in f:
                modules.append(
                    imp.load_compiled(f.split('.')[0],
                                      os.path.join(path, f))
                )
            else:
                modules.append(
                    imp.load_source(f.split('.')[0],
                                    os.path.join(path, f))
                )
        except:
            pass
    # ====== Find all identifier in modules ====== #
    ids = []
    for m in modules:
        for i in getmembers(m):
            if identifier in i:
                ids.append(i[1])
    # remove duplicate py
    return ids


def is_path(path):
    if isinstance(path, str):
        path = os.path.abspath(path)
        if os.path.exists(path):
            #the file is there
            return True
        elif os.access(os.path.dirname(path), os.W_OK):
            #the file does not exists but write privileges are given
            return True
    return False


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
# Misc
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


def get_datasetpath(name=None, override=False):
    datadir_base = os.path.expanduser(os.path.join('~', '.odin'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.odin')
    datadir = os.path.join(datadir_base, 'datasets')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    # ====== check given path with name ====== #
    if isinstance(name, types.StringType):
        datadir = os.path.join(datadir, name)
        if os.path.exists(datadir) and override:
            shutil.rmtree(datadir)
    return datadir


def get_modelpath(name=None, override=False):
    """ Default model path for saving ODIN networks """
    datadir_base = os.path.expanduser(os.path.join('~', '.odin'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.odin')
    datadir = os.path.join(datadir_base, 'models')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    # ====== check given path with name ====== #
    if isinstance(name, types.StringType):
        datadir = os.path.join(datadir, name)
        if os.path.exists(datadir) and override:
            os.remove(datadir)
    return datadir


def exec_commands(cmds):
    ''' Exec commands in parallel in multiple process
    (as much as we have CPU)

    Return
    ------
    failed: list of failed command
    '''
    if not cmds: return [] # empty list
    if not isinstance(cmds, (list, tuple)):
        cmds = [cmds]

    def done(p):
        return p.poll() is not None

    def success(p):
        return p.returncode == 0

    max_task = cpu_count()
    processes = []
    processes_map = {}
    failed = []
    while True:
        while cmds and len(processes) < max_task:
            task = cmds.pop()
            p = subprocess.Popen(task, shell=True)
            processes.append(p)
            processes_map[p] = task

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

    data = numpy.asarray(data)
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


class SharedCounter(object):
    """ A multiprocessing syncrhonized counter """

    def __init__(self):
        self.val = Value('i', 0)
        self.lock = Lock()

    def add(self, value=1):
        with self.lock:
            self.val.value += value

    @property
    def value(self):
        with self.lock:
            return self.val.value


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
