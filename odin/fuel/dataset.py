from __future__ import print_function, division, absolute_import

import os
import shutil
import cPickle
from types import StringType
from collections import OrderedDict
from six.moves import zip, range

import numpy as np

from .data import MmapData, Hdf5Data, open_hdf5, get_all_hdf_dataset, MAX_OPEN_MMAP, Data
from .utils import MmapDict

from odin.utils import get_file, Progbar
from odin.utils.decorators import singleton


__all__ = [
    'Dataset',
    'load_mnist',
    'load_cifar10',
    'load_cifar100',
    'load_mspec_test',
    'load_imdb',
    'load_digit_wav'
]


# ===========================================================================
# dataset
# ===========================================================================
def _parse_data_descriptor(path, name, read_only):
    """ Return mapping: path/name -> (dtype, shape, Data) """
    path = os.path.join(path, name)
    if not os.path.isfile(path):
        return None

    # ====== check if a file is Data ====== #
    try:
        dtype, shape = MmapData.read_header(path)
        # shape[1:], because first dimension can be resize afterward
        return [(os.path.basename(path), (dtype, shape, path))]
    except: # cannot read the header of MmapData, maybe Hdf5
        try:
            f = open_hdf5(path, read_only=read_only)
            ds = get_all_hdf_dataset(f)
            data = [Hdf5Data(dataset=i, hdf=f) for i in ds]
            return [(str(i.name), (str(i.dtype), i.shape, i)) for i in data]
        except:
            pass
    # ====== check if a file is Dict ====== #
    f = open(path, 'rb')
    try:
        data = cPickle.load(f)
        if isinstance(data, dict):
            return [(name, ('dict', len(data), data))]
    except:
        pass
    f.close()
    # ====== load memmap dict ====== #
    try:
        data = MmapDict(path)
        return [(name, ('memdict', len(data), data))]
    except:
        pass
    return [(name, ('unknown', 'unknown', path))]


@singleton
class Dataset(object):

    """
    Note
    ----
    for developer:
    _data_map contains: name -> (dtype, shape, Data or pathtoData)
    """

    def __init__(self, path, read_only=False):
        path = os.path.abspath(path)
        self.read_only = read_only
        if path is not None:
            if os.path.isfile(path) and '.zip' in os.path.basename(path):
                self._load_archive(path,
                                   extract_path=path.replace(os.path.basename(path), ''))
            else:
                self._set_path(path)

    def _set_path(self, path):
        # all files are opened with default_mode=r+
        self._data_map = OrderedDict()
        self._path = os.path.abspath(path)
        self._default_hdf5 = os.path.basename(self._path) + '_default.h5'

        if not os.path.exists(path):
            os.mkdir(path)
            return # not thing to do more
        elif not os.path.isdir(path):
            raise ValueError('Dataset path must be a folder.')

        # ====== load all Data ====== #
        files = os.listdir(path)
        for f in files:
            data = _parse_data_descriptor(path, f, self.read_only)
            if data is None: continue
            for key, d in data:
                if key in self._data_map:
                    raise ValueError('Found duplicated data with follow info: '
                                     '{}'.format(key))
                else:
                    self._data_map[key] = d

    # ==================== archive loading ==================== #
    def _load_archive(self, path, extract_path):
        from zipfile import ZipFile, ZIP_DEFLATED
        try:
            zfile = ZipFile(path, mode='r', compression=ZIP_DEFLATED)
            allfile = zfile.namelist()
            # validate extract_path
            if not os.path.isdir(extract_path):
                raise ValueError('Extract path must be path folder, but path'
                                 '={} is a file'.format(extract_path))
            extract_path = os.path.join(extract_path,
                                        os.path.basename(path).replace('.zip', ''))
            # found the extracted dir, use it
            if os.path.isdir(extract_path) and \
               set(os.listdir(extract_path)) == set(allfile):
                self._set_path(extract_path)
                return
            # decompress everything
            if not os.path.exists(extract_path):
                os.mkdir(extract_path)
            maxlen = max([len(i) for i in allfile])
            progbar = Progbar(len(allfile))
            for i, f in enumerate(allfile):
                zfile.extract(f, path=extract_path)
                progbar.title = ('Unarchiving: %-' + str(maxlen) + 's') % f
                progbar.update(i + 1)
            # ====== finally set path ====== #
            self._set_path(extract_path)
        except IOError, e:
            raise IOError('Error loading archived dataset, path:{}, error:{}'
                          '.'.format(path, e))
        return None

    # ==================== properties ==================== #
    @property
    def path(self):
        return self._path

    @property
    def archive_path(self):
        """Return default archive path, which is:
            ../[dataset_name].zip
        """
        name = os.path.basename(self._path)
        return os.path.join(self._path, '..', name + '.zip')

    @property
    def size(self):
        """ return size in MegaByte"""
        size_bytes = 0
        for name, (dtype, shape, data) in self._data_map.iteritems():
            size = np.dtype(dtype).itemsize
            shape = data.shape if hasattr(data, 'shape') else shape
            n = np.prod(shape)
            size_bytes += size * n
        return size_bytes / 1024. / 1024.

    def keys(self):
        """
        Return
        ------
        name of all Data
        """
        return self._data_map.keys()

    def values(self):
        """
        Return
        ------
        (dtype, shape, data) of Data
        """
        return self._data_map.values()

    def archive(self):
        from zipfile import ZipFile, ZIP_DEFLATED
        path = self.archive_path
        zfile = ZipFile(path, mode='w', compression=ZIP_DEFLATED)

        files = []
        for name, (dtype, shape, data) in self._data_map.iteritems():
            files.append(data.path if hasattr(data, 'path') else data)
        files = set(files)

        progbar = Progbar(len(files), title='Archiving:')
        maxlen = max([len(os.path.basename(i)) for i in files])
        for i, f in enumerate(files):
            zfile.write(f, os.path.basename(f))
            progbar.title = ('Archiving: %-' + str(maxlen) + 's') % os.path.basename(f)
            progbar.update(i + 1)
        zfile.close()
        return path

    def flush(self):
        for v in self._data_map.values():
            if isinstance(v, Data):
                v.flush()

    def close(self, name=None):
        if name is None: # close all files
            for name, (dtype, shape, data) in self._data_map.items():
                if isinstance(data, Data):
                    data.close()
                del data
                del self._data_map[name]
            self.dispose() # Singleton class to dispose an instance
        elif name in self._data_map: # close a particular file
            (dtype, shape, data) = self._data_map[name]
            if isinstance(data, Data):
                data.close()
            del data
            del self._data_map[name]

    # ==================== Some info ==================== #
    def _validate_memmap_max_open(self, name):
        # ====== check if MmapData excess limit, close 1 files ====== #
        if MmapData.COUNT > MAX_OPEN_MMAP:
            for i, (_dtype, _shape, _data) in self._data_map.iteritems():
                path = _data.path
                if isinstance(_data, MmapData) and i != name:
                    self.close(name=i)
                    self._data_map[i] = (_dtype, _shape, path)
                    break

    def __contains__(self, key):
        return key in self._data_map

    def __getitem__(self, key):
        if isinstance(key, StringType):
            if key not in self._data_map:
                raise KeyError('%s not found in this dataset' % key)
            dtype, shape, data = self._data_map[key]
            # return type is just a descriptor, create MmapData for it
            if isinstance(data, StringType) and \
            dtype is not 'unknown' and shape is not 'unknown':
                data = MmapData(data, read_only=self.read_only)
                self._data_map[key] = (data.dtype, data.shape, data)
                self._validate_memmap_max_open(key)
            return data
        raise ValueError('Only accept key type is string.')

    def __setitem__(self, key, value):
        """
        Parameters
        ----------
        key : str or tuple
            if tuple is specified, it contain the key and the datatype
            which must be "memmap", "hdf5"
        """
        if not isinstance(key, StringType) and not isinstance(key, (tuple, list)):
            raise ValueError('"key" is the name for Data and must be String or '
                             'tuple specified the name and datatype (memmap, hdf5).')
        # ====== check datatype ====== #
        datatype = 'memmap'
        if isinstance(key, (tuple, list)):
            key, datatype = key
            if datatype != 'memmap' and datatype != 'hdf5':
                raise ValueError('datatype can only be "memmap" or "hdf5", but '
                                 'the given data type is "%s"' % datatype)
        # ====== do nothing ====== #
        if key in self._data_map:
            return
        # ====== dict ====== #
        path = os.path.join(self.path, key)
        if isinstance(value, dict) or isinstance(value, MmapDict):
            if os.path.exists(path):
                raise Exception('File with path=%s already exist.' % path)
            d = MmapDict(path)
            for i, j in value.iteritems():
                d[i] = j
            d.flush()
            # store new dict
            self._data_map[key] = ('memdict', len(d), d)
        # ====== ndarray ====== #
        else:
            dtype, shape = value.dtype, value.shape
            if datatype == 'memmap':
                data = MmapData(path, dtype=dtype, shape=shape)
            else:
                f = open_hdf5(os.path.join(self.path, self._default_hdf5))
                data = Hdf5Data(key, hdf=f, dtype=dtype, shape=shape)
            # store new key
            self._data_map[key] = (data.dtype, data.shape, data)
            data.prepend(value)
            # check maximum opened memmap
            self._validate_memmap_max_open(key)

    def __iter__(self):
        for name, (dtype, shape, data) in self._data_map.iteritems():
            if isinstance(data, (Data, dict, MmapDict)):
                yield data
            else:
                yield self[name]

    def __str__(self):
        s = ['==========  Dataset:%s Total:%d  ==========' %
             (self.path, len(self._data_map))]
        # ====== Find longest string ====== #
        longest_name = 0
        longest_shape = 0
        longest_file = 0
        print_info = []
        for name, (dtype, shape, data) in self._data_map.iteritems():
            shape = data.shape if hasattr(data, 'shape') else shape
            longest_name = max(len(name), longest_name)
            longest_shape = max(len(str(shape)), longest_shape)
            if isinstance(data, dict):
                data = '<dictionary>'
            elif hasattr(data, 'path'):
                data = data.path
            longest_file = max(len(str(data)), longest_file)
            print_info.append([name, dtype, shape, data])
        # ====== return print string ====== #
        format_str = ('Name:%-' + str(longest_name) + 's  '
                      'dtype:%-7s  '
                      'shape:%-' + str(longest_shape) + 's  '
                      'file:%-' + str(longest_file) + 's')
        for name, dtype, shape, path in print_info:
            s.append(format_str % (name, dtype, shape, path))
        return '\n'.join(s)

    # ==================== Pickle ==================== #
    def __getstate__(self):
        return self.path

    def __setstate__(self, path):
        self._set_path(path)


# ===========================================================================
# Predefined dataset
# ===========================================================================
def _load_data_from_path(datapath, create_dataset=True):
    from zipfile import ZipFile, ZIP_DEFLATED
    if not os.path.isdir(datapath):
        datapath_tmp = datapath.replace('.zip', '') + '.tmp'
        os.rename(datapath, datapath_tmp)
        zf = ZipFile(datapath_tmp, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=datapath)
        zf.close()
        os.remove(datapath_tmp)
    if create_dataset:
        ds = Dataset(datapath)
        return ds
    return datapath


def load_mnist(path='https://s3.amazonaws.com/ai-datasets/MNIST.zip'):
    """
    path : str
        local path or url to hdf5 datafile
    """
    datapath = get_file('MNIST', path)
    return _load_data_from_path(datapath)


def load_cifar10(path='https://s3.amazonaws.com/ai-datasets/cifar10.zip'):
    """
    path : str
        local path or url to hdf5 datafile
    """
    datapath = get_file('cifar10', path)
    return _load_data_from_path(datapath)


def load_cifar100(path='https://s3.amazonaws.com/ai-datasets/cifar100.zip'):
    """
    path : str
        local path or url to hdf5 datafile
    """
    datapath = get_file('cifar100', path)
    return _load_data_from_path(datapath)


def load_mspec_test():
    """
    path : str
        local path or url to hdf5 datafile
    """
    path = 'https://s3.amazonaws.com/ai-datasets/mspec_test.zip'
    datapath = get_file('mspec_test', path)
    return _load_data_from_path(datapath)


def load_imdb(nb_words=None, maxlen=None):
    """ The preprocessed imdb dataset with following configuraiton:
     - nb_words=88587
     - length=2494
     - NO skip for any top popular word
     - Word_IDX=1 for beginning of sequences
     - Word_IDX=2 for ignored word (OOV)
     - Other word start from 3
     - padding='pre' with value=0
    """
    path = 'https://s3.amazonaws.com/ai-datasets/imdb.zip'
    datapath = get_file('imdb', path)
    ds = _load_data_from_path(datapath)
    X_train, y_train, X_test, y_test = \
        ds['X_train'], ds['y_train'], ds['X_test'], ds['y_test']
    # create new data with new configuration
    if maxlen is not None or nb_words is not None:
        nb_words = max(min(88587, nb_words), 3)
        path = ds.path + '_tmp'
        if os.path.exists(path):
            shutil.rmtree(path)
        ds = Dataset(path)
        # preprocess data
        if maxlen is not None:
            # for X_train
            _X, _y = [], []
            for i, j in zip(X_train[:], y_train[:]):
                if i[-maxlen] == 0 or i[-maxlen] == 1:
                    _X.append([k if k < nb_words else 2 for k in i[-maxlen:]])
                    _y.append(j)
            X_train = np.array(_X, dtype=X_train.dtype)
            y_train = np.array(_y, dtype=y_train.dtype)
            # for X_test
            _X, _y = [], []
            for i, j in zip(X_test[:], y_test[:]):
                if i[-maxlen] == 0 or i[-maxlen] == 1:
                    _X.append([k if k < nb_words else 2 for k in i[-maxlen:]])
                    _y.append(j)
            X_test = np.array(_X, dtype=X_test.dtype)
            y_test = np.array(_y, dtype=y_test.dtype)
        ds['X_train'] = X_train
        ds['y_train'] = y_train
        ds['X_test'] = X_test
        ds['y_test'] = y_test
        ds.flush()
    return ds


def load_digit_wav():
    from zipfile import ZipFile, ZIP_DEFLATED
    path = 'https://s3.amazonaws.com/ai-datasets/digit_wav.zip'
    datapath = get_file('digit_wav.zip', path)
    try:
        outpath = datapath.replace('.zip', '')
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        zf = ZipFile(datapath, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=outpath + '/../'); zf.close()
    except:
        # remove downloaded zip files
        os.remove(datapath)
        import traceback; traceback.print_exc()
    return outpath
