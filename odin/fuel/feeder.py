"""
MIT License
===========

Copyright (c) 2012 TrungNT (email: [name]@imito.ai)

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function, division, absolute_import

import os
import shutil
import inspect
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from six.moves import zip, zip_longest, range, cPickle
from collections import Mapping

import numpy as np

from odin.utils import (segment_list, one_hot, flatten_list, is_string,
                        Progbar, UnitTimer, get_system_status, batching,
                        get_process_status, SharedCounter, as_tuple, ctext,
                        is_number, is_primitives)
from odin.utils.mpi import MPI, async

from .data import MutableData, as_data
from .dataset import Dataset


# ===========================================================================
# Recipes
# ===========================================================================
@add_metaclass(ABCMeta)
class FeederRecipe(object):
    """ All method of this function a called in following order
    preprocess_indices(indices): return new_indices
    init(ntasks, batch_size, seed): right before create the iter

    [multi-process] process(*x): x->(name, data)
                                 return (if iterator, it will be iterated to
                                         get a list of results)
    [multi-process] group(x): x->(object from group(x))
                              return iterator

    Note
    ----
    This class should not store big amount of data, or the data
    will be replicated to all processes
    """

    def __init__(self):
        super(FeederRecipe, self).__init__()
        self._nb_data = 0
        self._nb_desc = 0

    # ==================== basic properties ==================== #
    def set_feeder_info(self, nb_data=None, nb_desc=None):
        if nb_data is not None:
            self._nb_data = int(nb_data)
        if nb_desc is not None:
            self._nb_desc = int(nb_desc)
        return self

    @property
    def nb_data(self):
        if self._nb_data == 0:
            raise RuntimeError("`nb_data` have not been set, using method "
                               "`set_feeder_info` to set operating information "
                               "for this recipe.")
        return self._nb_data

    @property
    def nb_desc(self):
        if self._nb_desc == 0:
            raise RuntimeError("`nb_data` have not been set, using method "
                               "`set_feeder_info` to set operating information "
                               "for this recipe.")
        return self._nb_desc

    # ==================== abstract ==================== #
    def shape_transform(self, shapes):
        """
        Parameters
        ----------
        shapes: list of [(shape0, indices0), (shape1, indices1), ...]
            list of data shape tuple and indices, the indices is list
            of tuple (name, length)

        Return
        ------
        new shape that transformed by this Recipe
        new indices
        """
        return shapes

    @abstractmethod
    def process(self, name, X, y):
        """
        Parameters
        ----------
        name: string
            the name of file in indices
        X: list of data
            list of all features given in DataDescriptor(s)
        y: list of labels
            list of all labels extracted or provided
        """
        raise NotImplementedError

    def __str__(self):
        # ====== get all attrs ====== #
        all_attrs = dir(self)
        print_attrs = {}
        for name in all_attrs:
            if '_' != name[0] and (len(name) >= 2 and '__' != name[:2]) and\
            name not in ('nb_data', 'nb_desc'):
                attr = getattr(self, name)
                if is_primitives(attr):
                    print_attrs[name] = str(attr)
                elif inspect.isfunction(attr):
                    print_attrs[name] = "(f)" + attr.func_name
        print_attrs = sorted(print_attrs.iteritems(), key=lambda x: x[0])
        print_attrs = [('#data', self.nb_data), ('#desc', self.nb_desc)] + print_attrs
        print_attrs = ' '.join(["%s:%s" % (ctext(key, 'yellow'), val)
                                for key, val in print_attrs])
        # ====== format the output ====== #
        s = '<%s %s>' % (ctext(self.__class__.__name__, 'cyan'), print_attrs)
        return s

    def __repr__(self):
        return self.__str__()


class RecipeList(FeederRecipe):

    def __init__(self, *recipes):
        super(RecipeList, self).__init__()
        self._recipes = recipes

    # ==================== List methods ==================== #
    def __iter__(self):
        return self._recipes.__iter__()

    def __len__(self):
        return len(self._recipes)

    # ==================== Override ==================== #
    def set_recipes(self, *recipes):
        # filter out None value
        recipes = flatten_list(as_tuple(recipes))
        recipes = [rcp for rcp in recipes
                   if rcp is not None and isinstance(rcp, FeederRecipe)]
        # ====== set the new recipes ====== #
        if len(recipes) > 0:
            self._recipes = recipes
            for rcp in self._recipes:
                rcp.set_feeder_info(self.nb_data, self.nb_desc)
        return self

    def set_feeder_info(self, nb_data=None, nb_desc=None):
        super(RecipeList, self).set_feeder_info(nb_data, nb_desc)
        for rcp in self._recipes:
            rcp.set_feeder_info(nb_data, nb_desc)
        return self

    def process(self, name, X, y, **kwargs):
        for i, f in enumerate(self._recipes):
            # return iterator (iterate over all of them)
            args = f.process(name, X, y)
            # break the chain if one of the recipes get error,
            # and return None
            if args is None:
                return None
            name, X, y = args
        return name, X, y

    def shape_transform(self, shapes):
        """
        Parameters
        ----------
        shapes: list of [(shape0, indices0), (shape1, indices1), ...]
            list of data shape tuple and indices, the indices is list
            of tuple (name, length)

        Return
        ------
        new shape that transformed by this Recipe
        new indices
        """
        for i in self._recipes:
            shapes = i.shape_transform(shapes)
            # ====== check returned ====== #
            if not all((isinstance(shp, (tuple, list)) and
                        all(is_number(s) for s in shp) and
                        is_string(ids[0][0]) and is_number(ids[0][1]))
                       for shp, ids in shapes):
                raise RuntimeError("Returned `shapes` must be the list of pair "
                                   "`(shape, indices)`, where `indices` is the "
                                   "list of (name, length(int)).")
        return shapes


# ===========================================================================
# Helper for grouping
# ===========================================================================
def _to_numpy_array(self, x):
    if not is_string(x[0]) and len(set(i.shape[1:] for i in x)) == 1:
        return np.concatenate(x, axis=0)
    return np.array(x)


def _batch_grouping(batch, batch_size, rng, batch_filter):
    """ batch: contains
        [
            (name, [list of data], [list of others]),
            (name, [list of data], [list of others]),
            (name, [list of data], [list of others]),
            ...
        ]

    Note
    ----
    We assume the shape[0] (or length) of all "data" and "others" are
    the same
    """
    if len(batch) == 0:
        yield None
    else:
        # create batch of indices for each file (indices is the start
        # index of each batch)
        indices = [list(range(0, X[0].shape[0], batch_size))
                   for name, X, y in batch]
        # shuffle if possible
        if rng is not None:
            [rng.shuffle(i) for i in indices]
        # ====== create batch of data ====== #
        for idx in zip_longest(*indices):
            ret = []
            for start, (name, X, y) in zip(idx, batch):
                # skip if the one data that is not enough
                if start is None: continue
                # pick data from each given input
                end = start + batch_size
                _ = [x[start:end] for x in X] + [i[start:end] for i in y]
                ret.append(_)
            ret = [np.concatenate(x, axis=0) for x in zip(*ret)]
            # shuffle 1 more time
            N = list(set([r.shape[0] for r in ret]))
            if len(N) > 1:
                raise ValueError("The shape[0] of Data is different, found "
                                 "%d different length: %s" % (len(N), str(N)))
            N = N[0]
            if rng is not None:
                permutation = rng.permutation(N)
                ret = [r[permutation] for r in ret]
            # return the batches
            for start in range(0, N, batch_size):
                end = start + batch_size
                _ = batch_filter([x[start:end] for x in ret])
                # always return tuple or list
                if _ is not None:
                    yield _ if isinstance(_, (tuple, list)) else (ret,)


def _file_grouping(batch, batch_size, rng, batch_filter):
    """ Return: [(name, index, data...), ...]
        NOTE: each element in batch is one file
    """
    # ====== shuffle the file ====== #
    if rng is not None:
        rng.shuffle(batch)
    # ====== return batched files with index for ordering ====== #
    for name, X, Y in batch:
        n = X[0].shape[0]
        ret = list(X) + list(Y)
        for i, (start, end) in enumerate(batching(n, batch_size)):
            r = [name, i] + [j[start:end] for j in ret]
            yield tuple(batch_filter(r))


def _weird_grouping(batch):
    pass

# ===========================================================================
# DataDescriptor
# ===========================================================================
_apply_approx = lambda n, x: int(round(n * x)) if x < 1. + 1e-12 else int(x)
_indices_dtype = [('name', 'object'), ('start', 'i4'), ('end', 'i4')]


def _preprocessing_indices(indices):
    """ Three different kind of indices:
    * file: load from csv file with ' ' (i.e. space as separator)
    * array: numpy array
    * nosql: instance of NoSQL
    """
    # store information for reloading the indices
    indices_info = None
    # indices always sorted in [(name, start, end), ...]
    if isinstance(indices, str) and os.path.isfile(indices):
        indices = {str(name): (start, end) for name, start, end in
                   np.genfromtxt(indices, dtype=_indices_dtype, delimiter=' ')}
        indices_info = ('file', str(indices))
    # list or tuple form: (name, (start, end)) or dictionary
    else:
        if isinstance(indices, (tuple, list, np.ndarray)):
            if len(indices[0]) == 2:
                indices = {name: (int(start), int(end))
                           for name, (start, end) in indices}
            else:
                indices = {name: (int(start), int(end))
                           for name, start, end in indices}
        # dictionary: name -> (start, end) or (name, start, end)
        elif isinstance(indices, Mapping):
            pass
        else:
            raise ValueError('Unsupport `indices` type: "%s".' % type(indices))
        indices_info = ('mapping', indices)
    return indices, indices_info


class DataDescriptor(MutableData):

    def __init__(self, data, indices):
        super(DataDescriptor, self).__init__()
        # ====== load indices ====== #
        self._indices_loader = async(_preprocessing_indices,
            callback=lambda result: self._loaded_callback())(indices)
        self._indices_info = None
        self._indices = None
        # ====== Load data ====== #
        if not isinstance(data, (tuple, list)):
            data = (data,)
        data = tuple([as_data(d) for d in data])
        # check all data have the same shape[0]
        length = len(data[0])
        if any(d.shape[0] != length for d in data):
            raise ValueError('All Data must have the same length '
                             '(i.e. shape[0]), the given data have '
                             'shape: %s' % str([d.shape for d in data]))
        self._data = data
        # ====== states variables ====== #
        self._length = None
        # if True return name during __iter__
        self._return_name = False

    # ==================== Properties ==================== #
    @property
    def indices_info(self):
        if self._indices_info is None:
            self._indices, self._indices_info = self._indices_loader.get()
        return self._indices_info

    @property
    def indices(self):
        if self._indices is None:
            self._indices, self._indices_info = self._indices_loader.get()
        return self._indices

    @property
    def nb_files(self):
        return len(self._indices)

    @property
    def nb_data(self):
        return len(self._data)

    # ==================== pickling ==================== #
    def _loaded_callback(self):
        self._new_args = (self.indices_info, self.data,
                          self._length, self._return_name)

    def _restore_data(self):
        if self._new_args is None:
            raise RuntimeError("Indices have not been loaded before calling "
                               "cPickle.dump on this class.")
        (self._indices_info, self._data,
            self._length, self._return_name) = self._new_args
        # deserialize indices
        ids_type, info = self._indices_info
        if ids_type == 'mapping':
            self._indices = info
        elif ids_type == 'file':
            self._indices = np.genfromtxt(info, dtype=_indices_dtype,
                                          delimiter=' ')

    # ==================== override from Data ==================== #
    @property
    def shape(self):
        """ This is just an "UPPER" estimation, some data points might be lost
        during preprocessing each indices by recipes.
        """
        if self._length is None:
            self._length = sum((end - start)
                               for name, (start, end) in self.indices)
        ret_shape = [(self._length,) + dat.shape[1:]
                     for dat in self.data]
        return ret_shape[0] if len(ret_shape) == 1 else tuple(ret_shape)

    def __str__(self):
        name = ctext('DataDescriptor', 'cyan')
        s = '<%s: Indices(type:"%s" length:%d)>\n' % \
            (name, self.indices_info[0], len(self.indices))
        for dat in self.data:
            s += '   (%s)%s: %s %s\n' % \
                (dat.__class__.__name__,
                    ctext(str(dat.path), 'yellow'),
                    dat.shape, str(dat.dtype))
        return s[:-1]

    # ==================== Strings ==================== #
    def set_return_name(self, return_name):
        self._return_name = bool(return_name)
        return self

    def __iter__(self):
        def _create_iter():
            ret_name = bool(self._return_name)
            yield None # just return for initialize the iteration
            for name, (start, end) in self.indices:
                dat = [d[start: end] for d in self.data]
                if ret_name:
                    dat = [name] + dat
                yield dat[0] if len(dat) == 1 else dat
        it = _create_iter()
        it.next()
        return it

    def __del__(self):
        pass


# ===========================================================================
# Multiprocessing Feeder
# ===========================================================================
def _dummy_batch_filter(x):
    return x


class Feeder(MutableData):
    """ multiprocessing Feeder to 1 comsumer
    Process1    Process2 ...    Process3
        |          |     |          |
         ------- Map Function ------
        |          |     |          |
         ----- Reduce Function -----
         \            |            /
           ---- Return batches ---

    This feeder return a non-deterministic order of data, hence,
    cannot be reproducible

    map_function: (name, x, transcription)
    reduce_function: (list of objects returned from map_function)

    Parameters
    ----------
    indices: path(csv file), list, ndarray, dict
        indices represent following information: [name, start_id, end_id]
        if indices is dictionary, it must in the form: {name: (start, end)}
    dtype: string or numpy.dtype
        all data return from this feeder are converted to given dtype
        if None, original dtype is kept
    batch_filter: callable
        must be a function has take a list of np.ndarray as first arguments
        ([X]) or ([X, y]), you can return None to ignore given batch, return the
        data for accepting the batch
    batch_mode: 'batch' or 'file' (string type)
        'batch' mode return shuffling and return everything in small batches
        'file' mode return [(file_name, order_index, data...), ...]
    ncpu: int
        number of CPU used for multiprocessing
    buffer_size: int
        A process will perform processing on a group of `buffer_size` number of
        data points, then, a list of results are returned to the main process.
        The higher this number the more powerful batch shuffling.
    maximum_queue_size: int (default: 66)
        maximum number of batch will be cached in Queue before main process
        get it and feed to the GPU (if there are too many results in Queue, a
        deadlock will happen)

    Example
    -------
    >>> ds = F.Dataset(os.path.join(temppath, 'ds'), read_only=True)
    >>> feeder = F.Feeder(ds['X'], indices=ds['indices.csv'],
    >>>                   ncpu=2, buffer_size=2, maximum_queue_size=12)
    >>> feeder.set_recipes([
    >>>     F.recipes.TransLoader(ds['transcription.dict'], dtype='int32'),
    >>>     F.recipes.CreateBatch()
    >>> ])
    >>> for i, j in feeder.set_batch(12, seed=1208251813, shuffle_level=2):
    >>>     for x, y in zip(i, j):
    >>>         print(transcription_test[str(x.tolist())] == y)

    Note
    ----
    set(ncpu=1) if you want a reproducible results
     - Memory transferring in Queue is always the bottleneck of multiprocessing
    3 supporting mode for shuffling:
     - shuffle_level=0: only shuffling the indices
     - shuffle_level=1: shuffle the buffered batch (e.g. 12 files in the indices)
     - shuffle_level=2: shuffle each returned batch
    * you must balance 2 number: buffer_size and maximum_queue_size, so the
    amount of data cached by all processed does not excess the RAM

    """

    def __init__(self, data_desc, dtype=None,
                 batch_filter=lambda x: x, batch_mode='batch',
                 ncpu=1, buffer_size=8, maximum_queue_size=66):
        super(Feeder, self).__init__()
        # ====== load indices ====== #
        self._data = as_tuple(data_desc, t=DataDescriptor)
        # find intersection of all indices in DataDescriptor
        self._indices_keys = async(
            lambda: np.array(
                list(set.intersection(*[set(dat.indices.keys())
                                        for dat in self._data])),
                dtype=str)
        )()
        # ====== desire dtype ====== #
        if dtype is None:
            self._output_types = ()
            for dat in self._data:
                self._output_types += as_tuple(dat.dtype)
        else:
            self._output_types = tuple(
                [np.dtype(t) for t in as_tuple(dtype, N=self.nb_data)])
        # ====== Set default recipes ====== #
        self._recipes = RecipeList()
        self._recipes.set_feeder_info(self.nb_data, len(self._data))
        self.set_multiprocessing(ncpu, buffer_size, maximum_queue_size)
        # ====== cache shape information ====== #
        # store first dimension
        self._cache_shape = None
        # if the recipes changed the shape need to be recalculated
        self._recipes_changed = False
        # ====== Iteration information ====== #
        self._running_iter = []
        # ====== batch mode ====== #
        if batch_filter is None:
            batch_filter = _dummy_batch_filter
        elif not callable(batch_filter):
            raise ValueError('batch_filter must be a function has 1 or 2 '
                             'parameters (X) or (X, y).')
        # check if batch_filter Picklable
        try:
            cPickle.dumps(batch_filter, protocol=2)
        except Exception:
            raise ValueError("`batch_filter` must be pickle-able, which must be "
                             "top-level function.")

        self._batch_filter = batch_filter
        # check batch_mode
        batch_mode = str(batch_mode).lower()
        if batch_mode not in ("batch", 'file'):
            raise ValueError("Only support `batch_mode`: 'file'; 'batch', but "
                             "given value: '%s'" % batch_mode)
        self._batch_mode = batch_mode
        # ====== for pickling ====== #
        self._new_args = (self._data, self._recipes,
                          self._output_types, self._cache_shape,
                          self._batch_mode, self._batch_filter,
                          self.ncpu, self.buffer_size, self.maximum_queue_size)

    # ==================== pickling ==================== #
    def _restore_data(self):
        (self._data, self._recipes,
         self._output_types, self._cache_shape,
         self._batch_mode, self._batch_filter,
         self.ncpu, self.buffer_size, self.maximum_queue_size) = self._new_args
        # find intersection of all indices in DataDescriptor
        self._indices_keys = async(
            lambda: np.array(
                list(set.intersection(*[set(dat.indices.keys())
                                        for dat in self._data])),
                dtype=str)
        )()
        # ====== basic attributes ====== #
        self._recipes_changed = False
        self._running_iter = []

    # ==================== multiprocessing ==================== #
    def set_multiprocessing(self, ncpu=None, buffer_size=None,
                            maximum_queue_size=None):
        if ncpu is not None:
            self.ncpu = int(ncpu)
        if buffer_size is not None:
            self.buffer_size = int(buffer_size)
        if maximum_queue_size is not None:
            self.maximum_queue_size = int(maximum_queue_size)
        return self

    def set_batch(self, batch_size=None, batch_filter=None, batch_mode=None,
                  seed=-1, start=None, end=None, shuffle_level=None):
        # ====== check batch_filter ====== #
        if batch_filter is not None:
            if not callable(batch_filter):
                raise ValueError('batch_filter must be a function has 1 or 2 '
                                 'parameters (X) or (X, y).')
            self._batch_filter = batch_filter
        # ====== check batch_mode ====== #
        if batch_mode is not None:
            batch_mode = str(batch_mode).lower()
            if batch_mode not in ("batch", 'file'):
                raise ValueError("Only support `batch_mode`: 'file'; 'batch', but "
                                 "given value: '%s'" % batch_mode)
            self._batch_mode = batch_mode
        return super(Feeder, self).set_batch(batch_size=batch_size, seed=seed,
                                             start=start, end=end,
                                             shuffle_level=shuffle_level)

    def set_recipes(self, *recipes):
        self._recipes_changed = True
        self._recipes.set_recipes(recipes)
        return self

    # ==================== override from Data ==================== #
    @property
    def nb_files(self):
        return len(self.indices_keys)

    @property
    def nb_data(self):
        return sum(dat.nb_data for dat in self._data)

    @property
    def indices_keys(self):
        if not isinstance(self._indices_keys, np.ndarray):
            self._indices_keys = self._indices_keys.get()
        return self._indices_keys

    @property
    def dtype(self):
        return self._output_types

    @property
    def shape(self):
        """ This is just an "UPPER" estimation, some data points might be lost
        during preprocessing each indices by recipes.
        """
        # ====== first time calculate the shape ====== #
        if self._cache_shape is None or self._recipes_changed:
            # for each Descriptor, create list of pairs: (name, length)
            shapes_indices = []
            for dat in self._data:
                indices = []
                length = 0
                for name in self.indices_keys:
                    start, end = dat.indices[name]
                    lng = end - start
                    length += lng
                    indices.append((name, lng))
                # modify shapes by estimted length from indices
                shapes = (dat.shape,) if is_number(dat.shape[0]) \
                    else dat.shape
                # NOTE: the indices is copy for each shape (i.e. data),
                # hence, it will create some overhead in shape_transform
                for shp in [(length,) + shp[1:] for shp in shapes]:
                    shapes_indices.append((shp, list(indices)))
            # Recipes shape_transform
            shapes = tuple([
                shp for shp, ids in self._recipes.shape_transform(shapes_indices)
            ])
            del shapes_indices
            self._cache_shape = shapes[0] if len(shapes) == 1 \
                else shapes
            self._recipes_changed = False
        # ====== get the cached shape ====== #
        return tuple(self._cache_shape)

    def __str__(self):
        s = '<%s: #keys:%d #iter:%d #CPU:%d #Buffer:%d mode:"%s" dtype:%s>\n' % \
            (ctext('Feeder', 'cyan'), len(self.indices_keys),
                len(self._running_iter), self.ncpu, self.buffer_size,
                self._batch_mode,
                '|'.join((str(dt) for dt in self.dtype))
            )
        # ====== print recipes ====== #
        s += '   ' + ctext('Recipes:', 'magenta') + '\n'
        for recipe in self._recipes:
            s += '\n'.join(['\t' + i for i in str(recipe).split('\n')])
            s += '\n'
        # ====== print data descriptor ====== #
        s += '   ' + ctext('Descriptor:', 'magenta') + '\n'
        for desc in self._data:
            s += '\n'.join(['\t' + i for i in str(desc).split('\n')])
            s += '\n'
        return s[:-1]

    # ==================== Strings ==================== #
    def __iter__(self):
        # ====== get start and end for indices ====== #
        start = _apply_approx(self.nb_files, self._start)
        end = _apply_approx(self.nb_files, self._end)
        all_keys = self.indices_keys[start:end]
        # ====== shuffle the indices ====== #
        rng = None
        shuffle_level = self._shuffle_level
        if self._seed is not None:
            rng = np.random.RandomState(self._seed)
            all_keys = all_keys[rng.permutation(self.nb_files)]
            if shuffle_level < 1:
                rng = None
            # reset the seed
            self._seed = None
        batch_size = self._batch_size
        batch_filter = self._batch_filter
        process_func = self._recipes.process
        # ====== prepare data, indices and dtype ====== #
        data_indices_dtype = []
        i = 0
        for dat in self._data:
            for d in dat._data:
                data_indices_dtype.append(
                    (d, dat.indices, self.dtype[i]))
                i += 1

        # ====== create wrapped functions ====== #
        def map_func(jobs):
            batch = []
            for name in jobs:
                X = []
                for dat, ids, dtype in data_indices_dtype:
                    start, end = ids[name]
                    # data can be list of Data, or just 1 Data
                    dat = dat[start:end]
                    if dat.dtype != dtype:
                        dat = dat.astype(dtype)
                    X.append(dat)
                X = process_func(name, X, [])
                # ignore None returned result
                if X is not None:
                    batch.append(X)
            # choose grouping function
            if self._batch_mode == 'batch':
                return _batch_grouping(batch, batch_size, rng, batch_filter)
            elif self._batch_mode == 'file':
                return _file_grouping(batch, batch_size, rng, batch_filter)

        def reduce_func(results):
            # perform batch level permutation
            if rng is not None:
                permutation = rng.permutation(results[0].shape[0])
                results = [r[permutation] for r in results]
            # convert batch to tuple object if possible
            if isinstance(results, (tuple, list)) and len(results) == 1:
                results = results[0]
            elif isinstance(results, list):
                results = tuple(results)
            return results
        # ====== track and return ====== #
        it = MPI(all_keys, map_func, reduce_func,
                 ncpu=self.ncpu,
                 buffer_size=self.buffer_size,
                 maximum_queue_size=self.maximum_queue_size,
                 chunk_scheduler=True)
        self._running_iter.append(it)
        return it

    def save_cache(self, path, datatype='memmap'):
        """ Save all preprocessed data to a Dataset """
        if not isinstance(path, str) or os.path.isfile(path):
            raise ValueError('path must be string path to a folder.')
        if os.path.exists(path):
            print('Remove old dataset at path:', path)
            shutil.rmtree(path)

        ds = Dataset(path)
        # ====== start caching ====== #
        prog = Progbar(target=self.shape[0], name='Caching',
                       print_report=True, print_summary=True)
        for X in self:
            if not isinstance(X, (tuple, list)):
                X = (X,)
            # saving preprocessed data
            for i, x in enumerate(X):
                name = 'data%d' % i
                if name in ds: ds[name].append(x)
                else: ds[(name, datatype)] = x
            # print progress
            prog.add(X[0].shape[0])
        ds.flush()
        ds.close()
        # end
        return self

    def stop_all(self):
        """ Call this method to stop all processes in case you
        spamming to many iteration
        """
        for i in self._running_iter:
            i.stop()
        self._running_iter = []

    def __del__(self):
        self.stop_all()
