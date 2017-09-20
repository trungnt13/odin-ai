# ===========================================================================
# Class handle extreme large numpy ndarray
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import re
import marshal
from math import ceil
from six import add_metaclass
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from six.moves import range, zip, zip_longest
from collections import OrderedDict, defaultdict

import numpy as np
from numpy import ndarray

from odin.utils.decorators import autoattr
from odin.utils import (queue, struct, as_tuple,
                        cache_memory, is_string)

__all__ = [
    'as_data',
    'open_hdf5',
    'close_all_hdf5',
    'get_all_hdf_dataset',

    'Data',
    'NdarrayData',
    'MmapData',
    'Hdf5Data',
    'DataIterator',
    'DataMerge'
]


# ===========================================================================
# Helper function
# ===========================================================================
def as_data(x):
    """ make sure x is Data """
    if isinstance(x, Data):
        return x
    if isinstance(x, np.ndarray):
        return NdarrayData(x)
    if isinstance(x, (tuple, list)):
        return DataIterator(x)
    raise ValueError('Cannot create Data object from given object:{}'.format(x))


def _get_chunk_size(shape, size):
    if isinstance(size, int):
        return (2**int(np.ceil(np.log2(size))),) + shape[1:]
    elif size is None:
        return False
    return True


def _validate_operate_axis(axis):
    ''' as we iterate over first dimension, it is prerequisite to
    have 0 in the axis of operator
    '''
    if not isinstance(axis, (tuple, list)):
        axis = [axis]
    axis = tuple(int(i) for i in axis)
    if 0 not in axis:
        raise ValueError('Expect 0 in the operating axis because we always'
                         ' iterate data over the first dimension.')
    return axis


# x can be percantage or number of samples
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)


# ===========================================================================
# Data
# ===========================================================================
@add_metaclass(ABCMeta)
class Data(object):

    """ Note for overriding `Data` class:
    * If `_new_args` is not None it will be return at __getnewargs__
      during unpickling.
    * `_new_args` must be picklable, and also be pickled.
    """
    def __init__(self):
        # batch information
        self._batch_size = 256
        self._start = 0.
        self._end = 1.
        self._seed = None
        self._shuffle_level = 0
        # ====== main data ====== #
        # object that have shape, dtype ...
        self._data = None
        self._path = None
        # ====== special flags ====== #
        # to detect if cPickle called with protocol >= 2
        self._new_args_called = False
        self._new_args = None
        # flag show that array valued changed
        self._status = 0

    # ==================== basic properties ==================== #
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def data(self):
        return self._data

    @property
    def array(self):
        if isinstance(self._data, (tuple, list)):
            return [dat[:] for dat in self._data]
        return self._data[:]

    def tolist(self):
        array = self.array
        if isinstance(array, (tuple, list)):
            array = [i.tolist() for i in array]
        else:
            array = array.tolist()
        return array

    @property
    def path(self):
        if is_string(self._path):
            return os.path.abspath(self._path)
        return self._path

    # ==================== For pickling ==================== #
    @abstractmethod
    def _restore_data(self):
        raise NotImplementedError

    def __getstate__(self):
        if not self._new_args_called:
            raise RuntimeError(
                "You must use argument `protocol=cPickle.HIGHEST_PROTOCOL` "
                "when using `pickle` or `cPickle` to be able pickling Data.")
        self._new_args_called = False
        return (self._batch_size, self._start, self._end, self._seed,
                self._shuffle_level, self._new_args, self._path)

    def __setstate__(self, states):
        (self._batch_size, self._start, self._end, self._seed,
         self._shuffle_level, self._new_args, self._path) = states
        self._status = 0
        self._new_args_called = False
        # ====== restore data ====== #
        self._restore_data()
        if not hasattr(self, '_data') or self._data is None:
            raise RuntimeError("The `_data` attribute is None, and have not "
                               "been restored after pickling, you must properly "
                               "implement function `restore_data` for class '%s'"
                               % self.__class__.__name__)

    def __getnewargs__(self):
        self._new_args_called = True
        if self._new_args is not None:
            return as_tuple(self._new_args)
        return ()

    # ==================== internal utilities ==================== #
    ''' BigData instance store large dataset that need to be iterate over to
    perform any operators.
    '''
    @cache_memory('_status')
    def _iterating_operator(self, ops, axis, merge_func=sum, init_val=0.):
        '''Execute a list of ops on X given the axis or axes'''
        # ====== validate arguments ====== #
        if axis is not None:
            axis = _validate_operate_axis(axis)
        if not isinstance(ops, (tuple, list)):
            ops = [ops]
        # ====== trick to process list of Data ====== #
        old_data = self._data
        data = old_data if isinstance(old_data, (tuple, list)) else (old_data,)
        # ====== Processing ====== #
        results = []
        for dat in data:
            self._data = dat
            # init values all zeros
            s = None
            old_seed = self._seed
            old_start = self._start
            old_end = self._end
            # less than million data points, not a big deal
            it = iter(self.set_batch(start=0., end=1., seed=None))
            for X in it:
                if s is None:
                    # list of results for each ops
                    s = [o(X, axis) for o in ops]
                else:
                    # merge all results from last ops to new ops
                    s = [merge_func((i, o(X, axis))) for i, o in zip(s, ops)]
            self.set_batch(start=old_start, end=old_end, seed=old_seed)
            results.append(s)
        # ====== reset and return ====== #
        self._data = old_data
        return results if isinstance(old_data, (tuple, list)) else results[0]

    def _iterate_update(self, y, ops):
        """Support ops:
        add; mul; div; sub; floordiv; pow
        """
        # ====== trick to process list of Data ====== #
        old_data = self._data
        data = old_data if isinstance(old_data, (tuple, list)) else (old_data,)
        # ====== processing ====== #
        for dat in data:
            self._data = dat

            shape = self._data.shape
            # custom batch_size
            idx = list(range(0, shape[0], 1024))
            if idx[-1] < shape[0]:
                idx.append(shape[0])
            idx = list(zip(idx, idx[1:]))
            Y = lambda start, end: (y[start:end] if hasattr(y, 'shape') and
                                    y.shape[0] == shape[0]
                                    else y)
            for i in idx:
                start, end = i
                if 'add' == ops:
                    self._data[start:end] += Y
                elif 'mul' == ops:
                    self._data[start:end] *= Y
                elif 'div' == ops:
                    self._data[start:end] /= Y
                elif 'sub' == ops:
                    self._data[start:end] -= Y
                elif 'floordiv' == ops:
                    self._data[start:end] //= Y
                elif 'pow' == ops:
                    self._data[start:end] **= Y
        # ====== reset ====== #
        self._data = old_data

    # ==================== properties ==================== #
    @property
    def ndim(self):
        if isinstance(self._data, (tuple, list)):
            return [len(dat.shape) for dat in self._data]
        return len(self.shape)

    @property
    def shape(self):
        # auto infer new shape
        if isinstance(self._data, (tuple, list)):
            return tuple([dat.shape for dat in self._data])
        return self._data.shape

    def __len__(self):
        """ len always return 1 number """
        shape = self.shape
        if isinstance(shape[0], (tuple, list)):
            return min([s[0] for s in shape])
        return shape[0]

    @property
    def T(self):
        array = self.array
        if isinstance(array, (tuple, list)):
            return [i.T for i in array]
        return array.T

    @property
    def dtype(self):
        if isinstance(self._data, (tuple, list)):
            return [dat.dtype for dat in self._data]
        return self._data.dtype

    @contextmanager
    def set_batch_context(self, batch_size=None, seed=-1, start=None, end=None,
                          shuffle_level=None):
        _batch_size = self._batch_size
        _seed = self._seed
        _start = self._start
        _end = self._end
        _shuffle_level = self._shuffle_level
        # temporary batch configuration
        self.set_batch(batch_size=_batch_size if batch_size is None else batch_size,
            seed=_seed if seed == -1 else seed,
            start=_start if start is None else start,
            end=_end if end is None else end,
            shuffle_level=_shuffle_level if shuffle_level is None else shuffle_level)
        yield self
        # reset batch to original batch configuration
        self.set_batch(batch_size=_batch_size, seed=_seed,
                       start=_start, end=_end,
                       shuffle_level=_shuffle_level)

    def set_batch(self, batch_size=None, seed=-1, start=None, end=None,
                  shuffle_level=None):
        """
        Parameters
        ----------
        batch_size: int
            size of each batch return when iterate this Data
        seed: None, int
            if None, no shuffling is performed while iterating,
            if < 0, do not change the current seed
            if >= 0, enable randomization with given seed
        start: int, float
            if int, start indicates the index of starting data points to
            iterate. If float, start is the percentage of data to start.
        end: int, float
            ending point of the interation
        shuffle_level: int
            0: only shuffle the order of each batch
            1: shuffle the order of batches and inside each batch as well.
            2: includes level 0 and 1, and custom shuffling (strongest form)
        """
        if isinstance(batch_size, int) and batch_size > 0:
            self._batch_size = batch_size
        if seed is None or seed >= 0:
            self._seed = seed
        if start is not None and start > 0. - 1e-12:
            self._start = start
        if end is not None and end > 0. - 1e-12:
            self._end = end
        if shuffle_level is not None:
            self._shuffle_level = min(max(int(shuffle_level), 0), 2)
        return self

    # ==================== Slicing methods ==================== #
    def __getitem__(self, y):
        if isinstance(self._data, (tuple, list)):
            return [dat.__getitem__(y) for dat in self._data]
        return self._data.__getitem__(y)

    @autoattr(_status=lambda x: x + 1)
    def __setitem__(self, x, y):
        if isinstance(self._data, (tuple, list)):
            for dat in self._data:
                dat.__setitem__(x, y)
            return
        return self._data.__setitem__(x, y)

    # ==================== iteration ==================== #
    def __iter__(self):
        def create_iteration():
            # TODO: iter support _data is a list of Data
            batch_size = int(self._batch_size)
            shape = self.shape
            seed = self._seed
            shuffle_level = int(self._shuffle_level)
            self._seed = None
            # custom batch_size
            start = _apply_approx(shape[0], self._start)
            end = _apply_approx(shape[0], self._end)
            if start > shape[0] or end > shape[0]:
                raise ValueError('`start`={} or `end`={} excess `data_size`={}'
                                 ''.format(start, end, shape[0]))
            # ====== create batch ====== #
            idx = list(range(start, end, batch_size))
            if idx[-1] < end:
                idx.append(end)
            idx = list(zip(idx, idx[1:]))
            # ====== shuffling the batch ====== #
            if seed is None:
                permutation_func = lambda x: x
            else:
                rand = np.random.RandomState(seed=seed)
                rand.shuffle(idx)
                if shuffle_level > 0: # shuffle with higher level
                    permutation_func = lambda x: x[rand.permutation(x.shape[0])]
                else: # no need for higher level shuffling
                    permutation_func = lambda x: x
            # this dummy return to make everything initialized
            yield None
            for start, end in idx:
                x = self._data[start:end]
                yield permutation_func(x)
        # ====== create, init, and return the iteration ====== #
        it = create_iteration()
        it.next()
        return it

    # ==================== Strings ==================== #
    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

    # ==================== manipulation ==================== #
    @autoattr(_status=lambda x: x + 1)
    def append(self, *arrays):
        # TODO: support list of data
        accepted_arrays = []
        new_size = 0
        shape = self._data.shape

        for a in arrays:
            if hasattr(a, 'shape'):
                if a.shape[1:] == shape[1:]:
                    accepted_arrays.append(a)
                    new_size += a.shape[0]
        old_size = shape[0]
        # special case, Mmap is init with temporary size = 1 (all zeros)
        if old_size == 1 and np.sum(np.abs(self._data[:1])) == 0.:
            old_size = 0
        # resize and append data
        self.resize(old_size + new_size) # resize only once will be faster
        for a in accepted_arrays:
            self._data[old_size:old_size + a.shape[0]] = a
            old_size = old_size + a.shape[0]
        return self

    @autoattr(_status=lambda x: x + 1)
    def prepend(self, *arrays):
        # TODO: support list of data
        accepted_arrays = []
        new_size = 0
        shape = self._data.shape

        for a in arrays:
            if hasattr(a, 'shape'):
                if a.shape[1:] == shape[1:]:
                    accepted_arrays.append(a)
                    new_size += a.shape[0]
        if new_size > shape[0]:
            self.resize(new_size) # resize only once will be faster
        size = 0
        for a in accepted_arrays:
            self._data[size:size + a.shape[0]] = a
            size = size + a.shape[0]
        return self

    # ==================== abstract ==================== #
    @abstractmethod
    def resize(self, shape):
        raise NotImplementedError

    def flush(self):
        pass

    def close(self):
        pass

    # ==================== high-level operators ==================== #
    @abstractmethod
    def sum(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def cumsum(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def sum2(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def pow(self, y):
        raise NotImplementedError

    @abstractmethod
    def min(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def argmin(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def max(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def argmax(self, axis=None):
        raise NotImplementedError

    @abstractmethod
    def mean(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def var(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def std(self, axis=0):
        raise NotImplementedError

    @abstractmethod
    def normalize(self, axis, mean=None, std=None):
        raise NotImplementedError


class MutableData(Data):

    ''' Can only read, NO write or modify the values '''

    def __setitem__(self, x, y):
        raise NotImplementedError

    def append(self, *arrays):
        raise NotImplementedError

    def prepend(self, *arrays):
        raise NotImplementedError

    def resize(self, shape):
        raise NotImplementedError

    # ==================== high-level operators ==================== #
    def sum(self, axis=0):
        ops = lambda x, axis: np.sum(x, axis=axis)
        results = self._iterating_operator(ops, axis)
        if isinstance(self._data, (tuple, list)):
            return [i[0] for i in results]
        return results[0]

    def cumsum(self, axis=None):
        if isinstance(self._data, (tuple, list)):
            return [i.cumsum(axis) for i in self.array]
        return self.array.cumsum(axis)

    def sum2(self, axis=0):
        ops = lambda x, axis: np.sum(np.power(x, 2), axis=axis)
        results = self._iterating_operator(ops, axis)
        if isinstance(self._data, (tuple, list)):
            return [i[0] for i in results]
        return results[0]

    def pow(self, y):
        if isinstance(self._data, (tuple, list)):
            return [i.__pow__(y) for i in self.array]
        return self.array.__pow__(y)

    def min(self, axis=None):
        ops = lambda x, axis: np.min(x, axis=axis)
        results = self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] < x[1], x[0], x[1]),
            init_val=float('inf'))

        if isinstance(self._data, (tuple, list)):
            return [i[0] for i in results]
        return results[0]

    def argmin(self, axis=None):
        if isinstance(self._data, (tuple, list)):
            return [i.argmin(axis) for i in self.array]
        return self.array.argmin(axis)

    def max(self, axis=None):
        ops = lambda x, axis: np.max(x, axis=axis)
        results = self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] > x[1], x[0], x[1]),
            init_val=float('-inf'))[0]

        if isinstance(self._data, (tuple, list)):
            return [i[0] for i in results]
        return results[0]

    def argmax(self, axis=None):
        if isinstance(self._data, (tuple, list)):
            return [i.argmax(axis) for i in self.array]
        return self.array.argmax(axis)

    def mean(self, axis=0):
        sum1 = self.sum(axis)

        axis = _validate_operate_axis(axis)

        if isinstance(sum1, (tuple, list)):
            results = []
            for s, shape in zip(sum1, self.shape):
                n = np.prod([shape[i] for i in axis])
                results.append(s / n)
            return results
        else:
            n = np.prod([self.shape[i] for i in axis])
            return sum1 / n

    def var(self, axis=0):
        sum1 = self.sum(axis)
        sum2 = self.sum2(axis)

        axis = _validate_operate_axis(axis)

        if isinstance(sum1, (tuple, list)):
            results = []
            for s1, s2, shape in zip(sum1, sum2, self.shape):
                n = np.prod([shape[i] for i in axis])
                results.append((s2 - np.power(s1, 2) / n) / n)
            return results
        else:
            n = np.prod([self.shape[i] for i in axis])
            return (sum2 - np.power(sum1, 2) / n) / n

    def std(self, axis=0):
        variances = self.var(axis)
        if isinstance(variances, (tuple, list)):
            return [np.sqrt(v) for v in variances]
        return np.sqrt(variances)

    def normalize(self, axis, mean=None, std=None):
        raise NotImplementedError

    # ==================== low-level operator ==================== #
    def __add__(self, y):
        return self.array.__add__(y)

    def __sub__(self, y):
        return self.array.__sub__(y)

    def __mul__(self, y):
        return self.array.__mul__(y)

    def __div__(self, y):
        return self.array.__div__(y)

    def __floordiv__(self, y):
        return self.array.__floordiv__(y)

    def __pow__(self, y):
        return self.array.__pow__(y)

    def __neg__(self):
        return self.array.__neg__()

    def __pos__(self):
        return self.array.__pos__()

    def __iadd__(self, y):
        raise NotImplementedError

    def __isub__(self, y):
        raise NotImplementedError

    def __imul__(self, y):
        raise NotImplementedError

    def __idiv__(self, y):
        raise NotImplementedError

    def __ifloordiv__(self, y):
        raise NotImplementedError

    def __ipow__(self, y):
        raise NotImplementedError


# ===========================================================================
# Array Data
# ===========================================================================
class NdarrayData(Data):
    """docstring for NdarrayData"""

    def __init__(self, array):
        super(NdarrayData, self).__init__()
        if not isinstance(array, np.ndarray):
            raise ValueError('array must be instance of numpy ndarray')
        self._data = array
        self._path = None
        self._new_args = array

    def _restore_data(self):
        self._data = self._new_args

    # ==================== abstract ==================== #
    def resize(self, shape):
        return self._data.resize(shape)

    # ==================== high-level operators ==================== #
    def sum(self, axis=0):
        return self._data.sum(axis=axis)

    def cumsum(self, axis=None):
        return self._data.cumsum(axis=axis)

    def sum2(self, axis=0):
        return (self._data**2).sum(axis=axis)

    def pow(self, y):
        return self._data.pow(y)

    def min(self, axis=None):
        return self._data.min(axis=axis)

    def argmin(self, axis=None):
        return self._data.argmin(axis=axis)

    def max(self, axis=None):
        return self._data.max(axis=axis)

    def argmax(self, axis=None):
        return self._data.argmax(axis=axis)

    def mean(self, axis=0):
        return self._data.mean(axis=axis)

    def var(self, axis=0):
        return self._data.var(axis=axis)

    def std(self, axis=0):
        return self._data.std(axis=axis)

    def normalize(self, axis, mean=None, std=None):
        mean = self._data.mean(axis=axis) if mean is None else mean
        std = self._data.std(axis=axis) if std is None else std
        self._data = (self._data - mean) / std
        return self

    # ==================== low-level operator ==================== #
    def __add__(self, y):
        return self._data.__add__(y)

    def __sub__(self, y):
        return self._data.__sub__(y)

    def __mul__(self, y):
        return self._data.__mul__(y)

    def __div__(self, y):
        return self._data.__div__(y)

    def __floordiv__(self, y):
        return self._data.__floordiv__(y)

    def __pow__(self, y):
        return self._data.__pow__(y)

    def __neg__(self):
        return self._data.__neg__()

    def __pos__(self):
        return self._data.__pos__()

    def __iadd__(self, y):
        self._data.__iadd__(y)
        return self

    def __isub__(self, y):
        self._data.__isub__(y)
        return self

    def __imul__(self, y):
        self._data.__imul__(y)
        return self

    def __idiv__(self, y):
        self._data.__idiv__(y)
        return self

    def __ifloordiv__(self, y):
        self._data.__ifloordiv__(y)
        return self

    def __ipow__(self, y):
        self._data.__ipow__(y)
        return self

# ===========================================================================
# Memmap Data object
# ===========================================================================
MAX_OPEN_MMAP = 120


def _aligned_memmap_offset(dtype):
    header_size = len(MmapData.HEADER) + 8 + MmapData.MAXIMUM_HEADER_SIZE
    type_size = np.dtype(dtype).itemsize
    n = np.ceil(header_size / type_size)
    return int(n * type_size)


class MmapData(Data):

    """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  Numpy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.

    Parameters
    ----------
    path : str
        The file name or file object to be used as the array data buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is `uint8`.
    shape : tuple, optional
        The desired shape of the array. If ``mode == 'r'`` and the number
        of remaining bytes after `offset` is not a multiple of the byte-size
        of `dtype`, you must specify `shape`. By default, the returned array
        will be 1-D with the number of elements determined by file size
        and data-type.

    Note
    ----
    This class always read MmapData with mode=r+
    """
    _INSTANCES = OrderedDict()
    HEADER = 'mmapdata'
    MAXIMUM_HEADER_SIZE = 486

    @staticmethod
    def read_header(path, mode, return_file):
        """ return: dtype, shape
        Necessary information to create numpy.memmap
        """
        if mode not in ('r', 'r+'):
            raise ValueError("Only support 2 modes: 'r' and 'r+'.")
        f = open(path, mode)
        if f.read(len(MmapData.HEADER)) != MmapData.HEADER:
            raise Exception('Invalid header for MmapData.')
        # 8 bytes for size of info
        try:
            size = int(f.read(8))
            dtype, shape = marshal.loads(f.read(size))
        except Exception as e:
            raise Exception('Error reading memmap data file: %s' % str(e))
        # return file object
        if return_file:
            return dtype, shape, f
        # only return header info
        f.close()
        return dtype, shape

    def __new__(clazz, *args, **kwargs):
        path = kwargs.get('path', None)
        if path is None:
            path = args[0]
        if not is_string(path):
            raise ValueError("`path` for MmapData must be string, but given "
                             "object with type: %s" % type(path))
        path = os.path.abspath(path)
        # Found old instance
        if path in MmapData._INSTANCES:
            return MmapData._INSTANCES[path]
        # new MmapData
        # ====== increase memmap count ====== #
        if len(MmapData._INSTANCES) + 1 > MAX_OPEN_MMAP:
            raise ValueError('Only allowed to open maximum of {} memmap file'.format(MAX_OPEN_MMAP))
        # ====== create new instance ====== #
        new_instance = super(MmapData, clazz).__new__(clazz, *args, **kwargs)
        MmapData._INSTANCES[path] = new_instance
        return new_instance

    def __init__(self, path, dtype='float32', shape=None, read_only=False):
        super(MmapData, self).__init__()
        # validate path
        path = os.path.abspath(path)
        mode = 'r' if read_only else 'r+'
        self.read_only = read_only
        # ====== check shape info ====== #
        if shape is not None:
            if not isinstance(shape, (tuple, list, np.ndarray)):
                shape = (shape,)
            shape = tuple([0 if i is None or i < 0 else i for i in shape])
        # read exist file
        if os.path.exists(path):
            dtype, shape, f = MmapData.read_header(path, mode=mode,
                                                   return_file=True)
        # create new file
        else:
            if dtype is None or shape is None:
                raise Exception("First created this MmapData, `dtype` and "
                                "`shape` must NOT be None.")
            f = open(path, 'w+')
            f.write(MmapData.HEADER)
            dtype = str(np.dtype(dtype))
            if isinstance(shape, np.ndarray):
                shape = shape.tolist()
            if not isinstance(shape, (tuple, list)):
                shape = (shape,)
            _ = marshal.dumps([dtype, shape])
            size = len(_)
            if size > MmapData.MAXIMUM_HEADER_SIZE:
                raise Exception('The size of header excess maximum allowed size '
                                '(%d bytes).' % MmapData.MAXIMUM_HEADER_SIZE)
            f.write('%8d' % size)
            f.write(_)
        self._file = f
        self._data = np.memmap(f, dtype=dtype, shape=shape, mode=mode,
                               offset=_aligned_memmap_offset(dtype))
        self._path = path
        self._new_args = path

    def _restore_data(self):
        dtype, shape, f = MmapData.read_header(self.path, mode='r+',
                                               return_file=True)
        self._file = f
        self._data = np.memmap(f, dtype=dtype, shape=shape, mode='r+',
                               offset=_aligned_memmap_offset(dtype))

    def close(self):
        # Check if exist global instance
        if self.path in MmapData._INSTANCES:
            del MmapData._INSTANCES[self.path]
            # flush in read-write mode
            if not self.read_only:
                self.flush()
            # close mmap and file
            self._data._mmap.close()
            del self._data
            self._file.close()

    # ==================== properties ==================== #
    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return os.path.basename(self.path)

    def __str__(self):
        return '<MMAP dataset "%s": shape %s, type "<%s">' % \
        (self.name, self.shape, self.dtype)

    # ==================== High-level operator ==================== #
    @cache_memory('_status')
    def sum(self, axis=0):
        return self._data.sum(axis)

    @cache_memory('_status')
    def cumsum(self, axis=None):
        return self._data.cumsum(axis)

    @cache_memory('_status')
    def sum2(self, axis=0):
        return self._data.__pow__(2).sum(axis)

    @cache_memory('_status')
    def pow(self, y):
        return self._data.__pow__(y)

    @cache_memory('_status')
    def min(self, axis=None):
        return self._data.min(axis)

    @cache_memory('_status')
    def argmin(self, axis=None):
        return self._data.argmin(axis)

    @cache_memory('_status')
    def max(self, axis=None):
        return self._data.max(axis)

    @cache_memory('_status')
    def argmax(self, axis=None):
        return self._data.argmax(axis)

    @cache_memory('_status')
    def mean(self, axis=0):
        sum1 = self.sum(axis)
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        n = np.prod([self._data.shape[i] for i in axis])
        return sum1 / n

    @cache_memory('_status')
    def var(self, axis=0):
        sum1 = self.sum(axis)
        sum2 = self.sum2(axis)
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        n = np.prod([self._data.shape[i] for i in axis])
        return (sum2 - np.power(sum1, 2) / n) / n

    @cache_memory('_status')
    def std(self, axis=0):
        return np.sqrt(self.var(axis))

    @autoattr(_status=lambda x: x + 1)
    def normalize(self, axis, mean=None, std=None):
        mean = mean if mean is not None else self.mean(axis)
        std = std if std is not None else self.std(axis)
        self._data -= mean
        self._data /= std
        return self

    # ==================== Special operators ==================== #
    def __add__(self, y):
        return self._data.__add__(y)

    def __sub__(self, y):
        return self._data.__sub__(y)

    def __mul__(self, y):
        return self._data.__mul__(y)

    def __div__(self, y):
        return self._data.__div__(y)

    def __floordiv__(self, y):
        return self._data.__floordiv__(y)

    def __pow__(self, y):
        return self._data.__pow__(y)

    @autoattr(_status=lambda x: x + 1)
    def __iadd__(self, y):
        self._data.__iadd__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __isub__(self, y):
        self._data.__isub__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __imul__(self, y):
        self._data.__imul__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __idiv__(self, y):
        self._data.__idiv__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ifloordiv__(self, y):
        self._data.__ifloordiv__(y)
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ipow__(self, y):
        return self._data.__ipow__(y)

    def __neg__(self):
        self._data.__neg__()
        return self

    def __pos__(self):
        self._data.__pos__()
        return self

    # ==================== Save ==================== #
    def resize(self, shape):
        if self.read_only:
            return
        # ====== local files ====== #
        f = self._file
        mmap = self._data
        # ====== check new shape ====== #
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        if any(i != j for i, j in zip(shape[1:], mmap.shape[1:])):
            raise ValueError('Resize only support the first dimension, but '
                             '{} != {}'.format(shape[1:], mmap.shape[1:]))
        if shape[0] < mmap.shape[0]:
            raise ValueError('Only support extend memmap, and do not shrink the memory')
        elif shape[0] == self._data.shape[0]: # nothing to resize
            return self
        # ====== flush previous changes ====== #
        # resize by create new memmap and also rename old file
        shape = (shape[0],) + tuple(mmap.shape[1:])
        dtype = str(mmap.dtype)
        # rewrite the header
        f.seek(len(MmapData.HEADER))
        meta = marshal.dumps([dtype, shape])
        size = len(meta)
        f.write('%8d' % size)
        f.write(meta)
        f.flush()
        # extend the memmap
        mmap._mmap.close()
        del self._data
        self._data = np.memmap(self._path, dtype=dtype, shape=shape,
                               mode='r+', offset=_aligned_memmap_offset(dtype))
        return self

    def flush(self):
        if self.read_only:
            return
        self._data.flush()


# ===========================================================================
# Hdf5 Data object
# ===========================================================================
try:
    import h5py
except:
    pass


def get_all_hdf_dataset(hdf, fileter_func=None, path='/'):
    res = []
    # init queue
    q = queue()
    for i in hdf[path].keys():
        q.put(i)
    # get list of all file
    while not q.empty():
        p = q.pop()
        if 'Dataset' in str(type(hdf[p])):
            if fileter_func is not None and not fileter_func(p):
                continue
            res.append(p)
        elif 'Group' in str(type(hdf[p])):
            for i in hdf[p].keys():
                q.put(p + '/' + i)
    return res


_HDF5 = {}


def open_hdf5(path, read_only=False):
    '''
    Parameters
    ----------
    mode : one of the following options
        +------------------------------------------------------------+
        |r        | Readonly, file must exist                        |
        +------------------------------------------------------------+
        |r+       | Read/write, file must exist                      |
        +------------------------------------------------------------+
        |w        | Create file, truncate if exists                  |
        +------------------------------------------------------------+
        |w- or x  | Create file, fail if exists                      |
        +------------------------------------------------------------+
        |a        | Read/write if exists, create otherwise (default) |
        +------------------------------------------------------------+

    check : bool
        if enable, only return openned files, otherwise, None

    Note
    ----
    If given file already open in read mode, mode = 'w' will cause error
    (this is good error and you should avoid this situation)

    '''
    key = os.path.abspath(path)
    mode = 'r' if read_only else 'a'

    if key in _HDF5:
        f = _HDF5[key]
        if 'Closed' in str(f):
            f = h5py.File(path, mode=mode)
    else:
        f = h5py.File(path, mode=mode)
        _HDF5[key] = f
    return f


def close_all_hdf5():
    import gc
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed


class Hdf5Data(Data):

    def __init__(self, dataset, hdf=None, dtype=None, shape=None):
        super(Hdf5Data, self).__init__()
        raise Exception("Hdf5Data is under-maintanance!")
        # default chunks size is 32 (reduce complexity of the works)
        self._chunk_size = 32
        if isinstance(hdf, str):
            hdf = open_hdf5(hdf)
        if hdf is None and not isinstance(dataset, h5py.Dataset):
            raise ValueError('Cannot initialize dataset without hdf file')

        if isinstance(dataset, h5py.Dataset):
            self._data = dataset
            self._hdf = dataset.file
        else:
            if dataset not in hdf: # not created dataset
                if dtype is None or shape is None:
                    raise ValueError('dtype and shape must be specified if '
                                     'dataset has not created in hdf5 file.')
                shape = tuple([0 if i is None else i for i in shape])
                hdf.create_dataset(dataset, dtype=dtype,
                    chunks=_get_chunk_size(shape, self._chunk_size),
                    shape=shape, maxshape=(None, ) + shape[1:])

            self._data = hdf[dataset]
            if shape is not None and self._data.shape[1:] != shape[1:]:
                raise ValueError('Shape mismatch between predefined dataset '
                                 'and given shape, {} != {}'
                                 ''.format(shape, self._data.shape))
            self._hdf = hdf

    # ==================== properties ==================== #
    @property
    def path(self):
        return self._hdf.filename

    @property
    def name(self):
        _ = self._data.name
        if _[0] == '/':
            _ = _[1:]
        return _

    @property
    def hdf5(self):
        return self._hdf

    # ==================== High-level operator ==================== #
    @cache_memory('_status')
    def sum(self, axis=0):
        ops = lambda x, axis: np.sum(x, axis=axis)
        return self._iterating_operator(ops, axis)[0]

    @cache_memory('_status')
    def cumsum(self, axis=None):
        return self._data[:].cumsum(axis)

    @cache_memory('_status')
    def sum2(self, axis=0):
        ops = lambda x, axis: np.sum(np.power(x, 2), axis=axis)
        return self._iterating_operator(ops, axis)[0]

    @cache_memory('_status')
    def pow(self, y):
        return self._data[:].__pow__(y)

    @cache_memory('_status')
    def min(self, axis=None):
        ops = lambda x, axis: np.min(x, axis=axis)
        return self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] < x[1], x[0], x[1]),
            init_val=float('inf'))[0]

    @cache_memory('_status')
    def argmin(self, axis=None):
        return self._data[:].argmin(axis)

    @cache_memory('_status')
    def max(self, axis=None):
        ops = lambda x, axis: np.max(x, axis=axis)
        return self._iterating_operator(ops, axis,
            merge_func=lambda x: np.where(x[0] > x[1], x[0], x[1]),
            init_val=float('-inf'))[0]

    @cache_memory('_status')
    def argmax(self, axis=None):
        return self._data[:].argmax(axis)

    @cache_memory('_status')
    def mean(self, axis=0):
        sum1 = self.sum(axis)

        axis = _validate_operate_axis(axis)
        n = np.prod([self._data.shape[i] for i in axis])
        return sum1 / n

    @cache_memory('_status')
    def var(self, axis=0):
        sum1 = self.sum(axis)
        sum2 = self.sum2(axis)

        axis = _validate_operate_axis(axis)
        n = np.prod([self._data.shape[i] for i in axis])
        return (sum2 - np.power(sum1, 2) / n) / n

    @cache_memory('_status')
    def std(self, axis=0):
        return np.sqrt(self.var(axis))

    @autoattr(_status=lambda x: x + 1)
    def normalize(self, axis, mean=None, std=None):
        mean = mean if mean is not None else self.mean(axis)
        std = std if std is not None else self.std(axis)
        self._iterate_update(mean, 'sub')
        self._iterate_update(std, 'div')
        return self

    # ==================== low-level operator ==================== #
    def __add__(self, y):
        return self._data.__add__(y)

    def __sub__(self, y):
        return self._data.__sub__(y)

    def __mul__(self, y):
        return self._data.__mul__(y)

    def __div__(self, y):
        return self._data.__div__(y)

    def __floordiv__(self, y):
        return self._data.__floordiv__(y)

    def __pow__(self, y):
        return self._data.__pow__(y)

    @autoattr(_status=lambda x: x + 1)
    def __iadd__(self, y):
        self._iterate_update(y, 'add')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __isub__(self, y):
        self._iterate_update(y, 'sub')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __imul__(self, y):
        self._iterate_update(y, 'mul')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __idiv__(self, y):
        self._iterate_update(y, 'div')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ifloordiv__(self, y):
        self._iterate_update(y, 'floordiv')
        return self

    @autoattr(_status=lambda x: x + 1)
    def __ipow__(self, y):
        self._iterate_update(y, 'pow')
        return self

    def __neg__(self):
        self._data.__neg__()
        return self

    def __pos__(self):
        self._data.__pos__()
        return self

    # ==================== Save ==================== #
    def resize(self, shape):
        if self._hdf.mode == 'r':
            return

        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        if any(i != j for i, j in zip(shape[1:], self._data.shape[1:])):
            raise ValueError('Resize only support the first dimension, but '
                             '{} != {}'.format(shape[1:], self._data.shape[1:]))
        if shape[0] < self._data.shape[0]:
            raise ValueError('Only support extend memmap, and do not shrink the memory')
        elif shape[0] == self._data.shape[0]:
            return self

        self._data.resize(shape[0], axis=0)
        return self

    def flush(self):
        try:
            if self._hdf.mode == 'r':
                return
            self._hdf.flush()
        except:
            pass

    def close(self):
        try:
            self._hdf.close()
        except:
            pass


# ===========================================================================
# data iterator
# ===========================================================================
def _approximate_continuos_by_discrete(distribution):
    '''original distribution: [ 0.47619048  0.38095238  0.14285714]
       best approximated: [ 5.  4.  2.]
    '''
    if len(distribution) == 1:
        return distribution

    inv_distribution = 1 - distribution
    x = np.round(1 / inv_distribution)
    x = np.where(distribution == 0, 0, x)
    return x.astype(int)


class DataIterator(MutableData):

    ''' Vertically merge several data object for iteration
    '''

    def __init__(self, data):
        super(DataIterator, self).__init__()
        if not isinstance(data, (tuple, list)):
            data = (data,)
        # ====== validate args ====== #
        if any(not isinstance(i, Data) for i in data):
            raise ValueError('data must be instance of MmapData or Hdf5Data, '
                             'but given data have types: {}'
                             ''.format(map(lambda x: str(type(x)).split("'")[1],
                                          data)))
        shape = data[0].shape[1:]
        if any(i.shape[1:] != shape for i in data):
            raise ValueError('all data must have the same trial dimension, but'
                             'given shape of all data as following: {}'
                             ''.format([i.shape for i in data]))
        # ====== defaults parameters ====== #
        self._data = data
        self._sequential = False
        self._distribution = [1.] * len(data)

    # ==================== properties ==================== #
    @property
    def shape(self):
        return (len(self),) + self._data[0].shape[1:]

    @property
    def array(self):
        start = self._start
        end = self._end
        idx = [(_apply_approx(i.shape[0], start), _apply_approx(i.shape[0], end))
               for i in self._data]
        idx = [(j[0], int(round(j[0] + i * (j[1] - j[0]))))
               for i, j in zip(self._distribution, idx)]
        return np.vstack([i[j[0]:j[1]] for i, j in zip(self._data, idx)])

    def __len__(self):
        start = self._start
        end = self._end
        return sum(round(i * (_apply_approx(j.shape[0], end) - _apply_approx(j.shape[0], start)))
                   for i, j in zip(self._distribution, self._data))

    @property
    def distribution(self):
        return self._distribution

    def __str__(self):
        s = ['====== Iterator: ======']
        # ====== Find longest string ====== #
        longest_name = 0
        longest_shape = 0
        for d in self._data:
            name = d.name
            dtype = d.dtype
            shape = d.shape
            longest_name = max(len(name), longest_name)
            longest_shape = max(len(str(shape)), longest_shape)
        # ====== return print string ====== #
        format_str = ('Name:%-' + str(longest_name) + 's  '
                      'dtype:%-7s  '
                      'shape:%-' + str(longest_shape) + 's  ')
        for d in self._data:
            name = d.name
            dtype = d.dtype
            shape = d.shape
            s.append(format_str % (name, dtype, shape))
        # ====== batch configuration ====== #
        s.append('Batch: %d' % self._batch_size)
        s.append('Sequential: %r' % self._sequential)
        s.append('Distibution: %s' % str(self._distribution))
        s.append('Seed: %s' % str(self._seed))
        s.append('Range: [%.2f, %.2f]' % (self._start, self._end))
        return '\n'.join(s)

    def __repr__(self):
        return self.__str__()

    # ==================== batch configuration ==================== #
    def set_mode(self, distribution=None, sequential=None):
        '''
        Parameters
        ----------
        distribution : str, list or float
            'up', 'over': over-sampling all Data
            'down', 'under': under-sampling all Data
            list: percentage of each Data in the iterator will be iterated
            float: the same percentage for all Data
        sequential : bool
            if True, read each Data one-by-one, otherwise, mix all Data

        '''
        if sequential is not None:
            self._sequential = sequential
        if distribution is not None:
            # upsampling or downsampling
            if isinstance(distribution, str):
                distribution = distribution.lower()
                if 'up' in distribution or 'over' in distribution:
                    n = max(i.shape[0] for i in self._data)
                elif 'down' in distribution or 'under' in distribution:
                    n = min(i.shape[0] for i in self._data)
                else:
                    raise ValueError("Only upsampling (keyword: up, over) "
                                     "or undersampling (keyword: down, under) "
                                     "are supported.")
                self._distribution = [n / i.shape[0] for i in self._data]
            # real values distribution
            elif isinstance(distribution, (tuple, list)):
                if len(distribution) != len(self._data):
                    raise ValueError('length of given distribution must equal '
                                     'to number of data in the iterator, but '
                                     'len_data={} != len_distribution={}'
                                     ''.format(len(self._data), len(self._distribution)))
                self._distribution = distribution
            # all the same value
            elif isinstance(distribution, float):
                self._distribution = [distribution] * len(self._data)
        return self

    # ==================== main logic of batch iterator ==================== #
    def __iter__(self):
        def create_iteration():
            seed = self._seed; self._seed = None
            if seed is not None:
                rng = np.random.RandomState(seed)
            else: # deterministic RandomState
                rng = struct()
                rng.randint = lambda x: None
                rng.permutation = lambda x: slice(None, None)
            # ====== easy access many private variables ====== #
            sequential = self._sequential
            start, end = self._start, self._end
            batch_size = self._batch_size
            distribution = np.asarray(self._distribution)
            # shuffle order of data (good for sequential mode)
            idx = rng.permutation(len(self._data))
            data = self._data[idx] if isinstance(idx, slice) else [self._data[i] for i in idx]
            distribution = distribution[idx]
            shape = [i.shape[0] for i in data]
            # ====== prepare distribution information ====== #
            # number of sample should be traversed
            n = np.asarray([i * (_apply_approx(j, end) - _apply_approx(j, start))
                            for i, j in zip(distribution, shape)])
            n = np.round(n).astype(int)
            # normalize the distribution (base on new sample n of each data)
            distribution = n / n.sum()
            distribution = _approximate_continuos_by_discrete(distribution)
            # somehow heuristic, rescale distribution to get more benifit from cache
            if distribution.sum() <= len(data):
                distribution = distribution * 3
            # distribution now the actual batch size of each data
            distribution = (batch_size * distribution).astype(int)
            assert distribution.sum() % batch_size == 0, 'wrong distribution size!'
            # predefined (start,end) pair of each batch (e.g (0,256), (256,512))
            idx = list(range(0, batch_size + distribution.sum(), batch_size))
            idx = list(zip(idx, idx[1:]))
            # Dummy return to initialize everything
            yield None
            #####################################
            # 1. optimized parallel code.
            if not sequential:
                # first iterators
                it = [iter(dat.set_batch(bs, seed=rng.randint(10e8),
                                         start=start, end=end,
                                         shuffle_level=self._shuffle_level))
                      for bs, dat in zip(distribution, data)]
                # iterator
                while sum(n) > 0:
                    batch = []
                    for i, x in enumerate(it):
                        if n[i] <= 0:
                            continue
                        try:
                            x = x.next()[:n[i]]
                            n[i] -= x.shape[0]
                            batch.append(x)
                        except StopIteration: # one iterator stopped
                            it[i] = iter(data[i].set_batch(distribution[i],
                                seed=rng.randint(10e8), start=start, end=end,
                                shuffle_level=self._shuffle_level))
                            x = it[i].next()[:n[i]]
                            n[i] -= x.shape[0]
                            batch.append(x)
                    # got final batch
                    batch = np.vstack(batch)
                    # no idea why random permutation is much faster than shuffle
                    if self._shuffle_level > 0:
                        batch = batch[rng.permutation(batch.shape[0])]
                    # return the iterations
                    for i, j in idx[:int(ceil(batch.shape[0] / batch_size))]:
                        yield batch[i:j]
            #####################################
            # 2. optimized sequential code.
            else:
                # first iterators
                batch_size = distribution.sum()
                it = [iter(dat.set_batch(batch_size, seed=rng.randint(10e8),
                                         start=start, end=end,
                                         shuffle_level=self._shuffle_level))
                      for dat in data]
                current_data = 0
                # iterator
                while sum(n) > 0:
                    if n[current_data] <= 0:
                        current_data += 1
                    try:
                        x = it[current_data].next()[:n[current_data]]
                        n[current_data] -= x.shape[0]
                    except StopIteration: # one iterator stopped
                        it[current_data] = iter(data[current_data].set_batch(batch_size, seed=rng.randint(10e8),
                                            start=start, end=end,
                                            shuffle_level=self._shuffle_level))
                        x = it[current_data].next()[:n[current_data]]
                        n[current_data] -= x.shape[0]
                    # shuffle x
                    if self._shuffle_level > 0:
                        x = x[rng.permutation(x.shape[0])]
                    for i, j in idx[:int(ceil(x.shape[0] / self._batch_size))]:
                        yield x[i:j]
        # ====== create and return the iteration ====== #
        it = create_iteration()
        it.next()
        return it

    # ==================== Slicing methods ==================== #
    def __getitem__(self, y):
        start = self._start
        end = self._end
        idx = [(_apply_approx(i.shape[0], start), _apply_approx(i.shape[0], end))
               for i in self._data]
        idx = [(j[0], int(round(j[0] + i * (j[1] - j[0]))))
               for i, j in zip(self._distribution, idx)]
        size = np.cumsum([i[1] - i[0] for i in idx])
        if isinstance(y, int):
            idx = _get_closest_id(size, y)
            return self._data[idx][y]
        elif isinstance(y, slice):
            return self.array[y]
        else:
            raise ValueError('No support for indices type={}'.format(type(y)))


def _get_closest_id(size, y):
    idx = 0
    for i, j in enumerate(size):
        if y >= j:
            idx = i + 1
    return idx


# ===========================================================================
# DataMerge
# ===========================================================================
class DataMerge(MutableData):

    '''
    Parameters
    ----------
    data : list
        list of Data objects
    merge_func : __call__
        function take a list of Data as argument (i.e func([data1, data2]))

    Note
    ----
    First data in the list will be used as root to infer the shape after merge
    '''

    def __init__(self, data, merge_func):
        super(DataMerge, self).__init__()

        if not isinstance(data, (tuple, list)):
            data = (data,)
        self._data = [i for i in data if isinstance(i, Data)]
        if len(self._data) == 0:
            raise ValueError('Cannot find any instance of Data from given argument.')

        if not callable(merge_func):
            raise ValueError('Merge operator must be callable and accept at '
                             'least one argument.')
        self._merge_func = merge_func

    # ==================== properties ==================== #
    @property
    def shape(self):
        shape = [i.shape for i in self._data]
        raise NotImplementedError
        return shape, self._merge_func(x)

    @property
    def dtype(self):
        n = (12 + 8) // 10 # lucky number :D
        tmp = [np.ones((n,) + i.shape[1:]).astype(i.dtype) for i in self._data]
        return self._merge_func(tmp).dtype

    @property
    def array(self):
        return self._merge_func([i[:] for i in self._data])

    # ==================== Slicing methods ==================== #
    def __getitem__(self, y):
        n = self._data[0].shape[0]
        data = [i.__getitem__(y) if len(i.shape) > 0 and i.shape[0] == n else i
                for i in self._data]
        x = self._merge_func(data)
        return x

    # ==================== iteration ==================== #
    def _iter(self):
        batch_size = self._batch_size
        seed = self._seed; self._seed = None
        # ====== prepare root first ====== #
        shape = self._data[0].shape
        # custom batch_size
        start = _apply_approx(shape[0], self._start)
        end = _apply_approx(shape[0], self._end)
        if start > shape[0] or end > shape[0]:
            raise ValueError('start={} or end={} excess data_size={}'
                             ''.format(start, end, shape[0]))

        idx = list(range(start, end, batch_size))
        if idx[-1] < end:
            idx.append(end)
        idx = list(zip(idx, idx[1:]))
        rng = None
        if seed is not None:
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
        idx = [slice(i[0], i[1]) for i in idx]
        none_idx = [slice(None, None)] * len(idx)
        # ====== check other data ====== #
        batches = [idx]
        for d in self._data[1:]:
            if len(d.shape) > 0 and d.shape[0] == shape[0]:
                batches.append(idx)
            else:
                batches.append(none_idx)

        yield None # dummy return for initialize everything
        for b in zip(*batches):
            data = self._merge_func([i[j] for i, j in zip(self._data, b)])
            if self._shuffle_level > 0 and rng is not None:
                data = data[rng.permutation(data.shape[0])]
            yield data
