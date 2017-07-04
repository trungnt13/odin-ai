from __future__ import print_function
import os
import sys
# import uuid
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
from multiprocessing import cpu_count, Lock, current_process
from collections import OrderedDict, deque, Iterable, Iterator
from itertools import islice, tee, chain

from six import string_types
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError, HTTPError
import tarfile

try:
    from numba import jit, autojit, vectorize, guvectorize
except:
    pass
import numpy
import six

from . import progbar
from .progbar import Progbar
from .mpi import SelfIterator, segment_list, SharedCounter, async
from .profile import *
from .path_utils import *
from .cache_utils import *
from .python_utils import *
from . import mpi
from . import shape_calculation


# ===========================================================================
# Basics
# ===========================================================================
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


def is_lambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


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


def batching(n, batch_size):
    return [(i, min(i + batch_size, n))
            for i in range(0, n + batch_size, batch_size) if i < n]


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


class _LogWrapper():

    def __init__(self, stream):
        self.stream = stream

    def write(self, message):
        # no backtrack for writing to file
        self.stream.write(message.replace("\b", ''))
        sys.__stdout__.write(message)

    def flush(self):
        self.stream.flush()
        sys.__stdout__.flush()

    def close(self):
        try:
            self.stream.close()
        except:
            pass


def stdio(path=None, suppress=False, stderr=True):
    """
    Parameters
    ----------
    path: None, str
        if str, specified path for saving all stdout (and stderr)
        if None, reset stdout and stderr back to normal
    suppress: boolean
        totally turn-off all stdout (and stdeer)
    stderr:
        apply output file with stderr also

    """
    # turn off stdio
    if suppress:
        f = open(os.devnull, "w")
    # reset
    elif path is None:
        f = None
    # redirect to a file
    else:
        f = _LogWrapper(open(path, "w"))
    # ====== assign stdio ====== #
    if f is None: # reset
        if isinstance(sys.stdout, _LogWrapper):
            sys.stdout.close()
        if isinstance(sys.stderr, _LogWrapper):
            sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    else: # redirect to file
        sys.stdout = f
        if stderr:
            sys.stderr = f

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


# ===========================================================================
# ArgCOntrol
# ===========================================================================
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
        return val

    def add(self, name, help, default=None, preprocess=None):
        """ NOTE: if the default value is not given, the argument is
        required

        Parameters
        ----------
        name: str
            [-name] for optional parameters
            [name] for positional parameters
        help: str
            description of the argument
        default: str
            default value for the argument
        preprocess: callable
            take in the parsed argument and preprocess it into necessary
            information
        """
        if self.parser is None:
            self.parser = argparse.ArgumentParser(
                description='Automatic argument parser (yes,true > True; no,false > False)',
                version=self.version, add_help=True)
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

        # store preprocess dictionary
        if not callable(preprocess):
            preprocess = None
        self.arg_dict[name] = preprocess
        return self

    def parse(self):
        if self.parser is None:
            raise Exception('Call add to assign at least 1 argument for '
                            'for the function.')
        # TODO fix bug here
        exit_now = False
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
        if exit_now: exit()
        # parse the arguments
        try:
            args = {i: self._parse_input(i, j)
                    for i, j in args._get_kwargs()}
        except Exception as e:
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
        # convert it to struct
        return struct(args)


# ===========================================================================
# Simple math and processing
# ===========================================================================
def one_hot(y, n_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy

    Note
    ----
    if any class index in y is smaller than 0, then all of its one-hot
    values is 0.
    '''
    y = numpy.asarray(y, dtype='int32')
    if not n_classes:
        n_classes = numpy.max(y) + 1
    Y = numpy.zeros((len(y), n_classes), dtype='int32')
    for i, j in enumerate(y):
        if j >= 0:
            Y[i, j] = 1
    return Y


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
    if is_number(N):
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


def flatten_list(x, level=1):
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
        for i, j in self.iteritems():
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

    Return
    ------
    file path of the downloaded file
    '''
    datadir = get_datasetpath()
    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print('Downloading data from', origin)
        global _progbar
        _progbar = None

        def dl_progress(count, block_size, total_size):
            global _progbar
            if _progbar is None:
                _progbar = Progbar(total_size)
            else:
                _progbar.update(count * block_size)

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
        _progbar = None

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
    ''' Recurrsively get all extension of files in the given path '''
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
    return list(set(file_list))


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


def get_module_from_path(identifier, path='.', prefix='', suffix='', exclude='',
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
    * This function try to import each modules and find desire function,
    it may mess up something.

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


def _get_managed_path(folder, name, override, is_folder=False, root='~'):
    if root == '~':
        root = os.path.expanduser('~')
    datadir_base = os.path.join(root, '.odin')
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


def get_datasetpath(name=None, override=False, root='~'):
    return _get_managed_path('datasets', name, override, is_folder=True, root=root)


def get_modelpath(name=None, override=False, root='~'):
    """ Default model path for saving ODIN networks """
    return _get_managed_path('models', name, override, is_folder=True, root=root)


def get_logpath(name=None, override=False, root='~'):
    return _get_managed_path('logs', name, override, is_folder=False, root=root)


# ===========================================================================
# Misc
# ===========================================================================
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
        prog = Progbar(target=len(cmds), title="Execute command:")
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
