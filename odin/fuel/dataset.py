from __future__ import print_function, division, absolute_import

import os
import shutil
from collections import OrderedDict
from six.moves import zip, range, cPickle

import numpy as np

from .data import MmapData, Hdf5Data, open_hdf5, get_all_hdf_dataset, MAX_OPEN_MMAP, Data
from .utils import MmapDict

from odin.utils import get_file, Progbar, is_string
from odin.utils.decorators import singleton


__all__ = [
    'Dataset',
    'load_mnist',
    'load_cifar10',
    'load_cifar100',
    'load_mspec_test',
    'load_imdb',
    'load_iris',
    'load_digit_audio',
    'load_tiwave',
]


# ===========================================================================
# dataset
# ===========================================================================
def _parse_data_descriptor(path, read_only):
    """ Return mapping: name -> (dtype, shape, Data, path) """
    if not os.path.isfile(path):
        return None

    # ====== check if a file is Data ====== #
    try:
        dtype, shape = MmapData.read_header(path)
        # shape[1:], because first dimension can be resize afterward
        return [(os.path.basename(path), (dtype, shape, None, path))]
    except: # cannot read the header of MmapData, maybe Hdf5
        try:
            f = open_hdf5(path, read_only=read_only)
            ds = get_all_hdf_dataset(f)
            data = [Hdf5Data(dataset=i, hdf=f) for i in ds]
            return [(str(i.name), (str(i.dtype), i.shape, i, i.path)) for i in data]
        except:
            pass
    # ====== try to load pickle file if possible ====== #
    name = os.path.basename(path)
    try:
        with open(path, 'rb') as f:
            data = cPickle.load(f)
            return [(name,
            (type(data).__name__, len(data) if hasattr(data, '__len__') else 0, data, path))]
    except:
        pass
    # ====== load memmap dict ====== #
    try:
        data = MmapDict(path)
        return [(name, ('memdict', len(data), data, path))]
    except:
        pass
    return [(name, ('unknown', 'unknown', None, path))]


class Dataset(object):
    """ This Dataset can automatically parse memmap (created by MmapData),
    MmapDict, pickled dictionary and hdf5 files and keep tracking the changes.

    Any file name with "readme" prefix will be parsed as text and showed as
    readme.

    Note
    ----
    for developer: _data_map contains: name -> (dtype, shape, Data or pathtoData)
    readme included with the dataset should contain license information
    """

    __INSTANCES = {}

    def __new__(clazz, *args, **kwargs):
        path = kwargs.get('path', None)
        if path is None:
            path = args[0]
        if not is_string(path):
            raise ValueError("`path` for Dataset must be string, but given "
                             "object with type: %s" % type(path))
        path = os.path.abspath(path)
        # Found old instance
        if path in Dataset.__INSTANCES:
            return Dataset.__INSTANCES[path]
        # new Dataset
        new_instance = super(Dataset, clazz).__new__(clazz, *args, **kwargs)
        Dataset.__INSTANCES[path] = new_instance
        return new_instance

    def __init__(self, path, read_only=False, override=False):
        path = os.path.abspath(path)
        self.read_only = read_only
        self._readme_info = ['README:', '------', ' No information!']
        self._readme_path = None
        # parse all data from path
        if path is not None:
            if override and os.path.exists(path) and os.path.isdir(path):
                shutil.rmtree(path)
                print('Overrided old dataset at path:', path)
            if os.path.isfile(path) and '.zip' in os.path.basename(path):
                self._load_archive(path,
                                   extract_path=path.replace(os.path.basename(path), ''))
            else:
                self._set_path(path)
        else:
            raise ValueError('Invalid path for Dataset: %s' % path)

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
        for fname in files:
            # found README
            if 'readme' == fname[:6].lower():
                readme_path = os.path.join(path, fname)
                with open(readme_path, 'r') as readme_file:
                    readme = readme_file.readlines()[:3]
                    readme = [' ' + i[:-1] for i in readme if len(i) > 0 and i != '\n']
                    readme.append(' For more information: ' + readme_path)
                    self._readme_info = ['README:', '------'] + readme
                    self._readme_path = readme_path
            # parse data
            data = _parse_data_descriptor(os.path.join(path, fname), self.read_only)
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
            pb = Progbar(target=len(allfile), name="[Dataset] Loading Archive",
                         print_summary=True, print_report=True)
            for i, f in enumerate(allfile):
                zfile.extract(f, path=extract_path)
                pb['File'] = ('Unarchiving: %-' + str(maxlen) + 's') % f
                pb.add(1)
            # ====== finally set path ====== #
            self._set_path(extract_path)
        except IOError as e:
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
        for name, (dtype, shape, data, path) in self._data_map.iteritems():
            try:
                size_bytes += os.path.getsize(path) # in bytes
            except:
                pass
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
        (dtype, shape, data, path) of Data
        """
        return self._data_map.values()

    def archive(self):
        from zipfile import ZipFile, ZIP_DEFLATED
        path = self.archive_path
        zfile = ZipFile(path, mode='w', compression=ZIP_DEFLATED)

        files = set([_[-1] for _ in self._data_map.itervalues()])

        prog = Progbar(target=len(files), name="[Dataset] Archiving",
                       print_report=True, print_summary=True)
        maxlen = max([len(os.path.basename(i)) for i in files])
        for i, f in enumerate(files):
            zfile.write(f, os.path.basename(f))
            prog['Data'] = ('Archiving: %-' + str(maxlen) + 's') \
                % os.path.basename(f)
            prog.add(1)
        zfile.close()
        return path

    def flush(self):
        for dtype, shape, data, path in self._data_map.itervalues():
            if hasattr(data, 'flush'):
                data.flush()
            elif data is not None:
                with open(path, 'wb') as f:
                    cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def close(self, name=None):
        # ====== close all Data ====== #
        if name is None: # close all files
            for name, (dtype, shape, data, path) in self._data_map.items():
                if hasattr(data, 'close'):
                    data.close()
                del data
                del self._data_map[name]
            # Check if exist global instance
            if self.path in Dataset.__INSTANCES:
                del Dataset.__INSTANCES[self.path]
        # ====== close a particular file ====== #
        elif name in self._data_map:
            (dtype, shape, data, path) = self._data_map[name]
            if hasattr(data, 'close'):
                data.close()
            del data
            del self._data_map[name]

    # ==================== Some info ==================== #
    def _validate_memmap_max_open(self, name):
        # ====== check if MmapData excess limit, close 1 files ====== #
        if MmapData.COUNT > MAX_OPEN_MMAP:
            for i, (_dtype, _shape, _data, _path) in self._data_map.iteritems():
                if isinstance(_data, MmapData) and i != name:
                    self.close(name=i)
                    self._data_map[i] = (_dtype, _shape, _path)
                    break

    def __contains__(self, key):
        return key in self._data_map

    def __getitem__(self, key):
        if is_string(key):
            if key not in self._data_map:
                raise KeyError('%s not found in this dataset' % key)
            dtype, shape, data, path = self._data_map[key]
            # return type is just a descriptor, create MmapData for it
            if data is None and \
            dtype is not 'unknown' and shape is not 'unknown':
                data = MmapData(path, read_only=self.read_only)
                self._data_map[key] = (data.dtype, data.shape, data, path)
                self._validate_memmap_max_open(key)
            return path if data is None else data
        raise ValueError('Only accept key type is string.')

    def __setitem__(self, key, value):
        """
        Parameters
        ----------
        key : str or tuple
            if tuple is specified, it contain the key and the datatype
            which must be "memmap", "hdf5"
            for example: ds[('X', 'hdf5')] = numpy.ones((8, 12))
        """
        if not is_string(key) and not isinstance(key, (tuple, list)):
            raise ValueError('"key" is the name for Data and must be String or '
                             'tuple specified the name and datatype (memmap, hdf5).')
        # ====== check datatype ====== #
        datatype = 'memmap' # default datatype
        if isinstance(key, (tuple, list)):
            key, datatype = key
            datatype = datatype.lower()
            if datatype not in ('memmap', 'hdf5'):
                raise ValueError('datatype can only be "memmap" or "hdf5", but '
                                 'the given data type is "%s"' % datatype)
        # ====== do nothing ====== #
        if key in self._data_map:
            return
        # ====== dict ====== #
        path = os.path.join(self.path, key)
        if isinstance(value, dict):
            if os.path.exists(path):
                raise Exception('File with path=%s already exist.' % path)
            d = MmapDict(path)
            for i, j in value.iteritems():
                d[i] = j
            d.flush()
            # store new dict
            self._data_map[key] = (type(d).__name__, len(d), d, path)
        # ====== ndarray ====== #
        elif isinstance(value, np.ndarray):
            dtype, shape = value.dtype, value.shape
            if datatype == 'memmap':
                data = MmapData(path, dtype=dtype, shape=shape)
            else:
                path = os.path.join(self.path, self._default_hdf5)
                f = open_hdf5(path)
                data = Hdf5Data(key, hdf=f, dtype=dtype, shape=shape)
            # store new key
            self._data_map[key] = (data.dtype, data.shape, data, path)
            data.prepend(value)
            # check maximum opened memmap
            self._validate_memmap_max_open(key)
        # ====== other types ====== #
        else:
            if os.path.exists(path):
                raise Exception('File with path=%s already exist.' % path)
            with open(path, 'wb') as f:
                cPickle.dump(value, f, protocol=cPickle.HIGHEST_PROTOCOL)
            # store new dict
            self._data_map[key] = (type(value).__name__,
                                   len(value) if hasattr(value, '__len__') else 0,
                                   value, path)

    def __iter__(self):
        for name, (dtype, shape, data) in self._data_map.iteritems():
            if isinstance(data, (Data, dict, MmapDict)):
                yield data
            else:
                yield self[name]

    def __str__(self):
        s = ['==========  Dataset:%s Total:%d  ==========' %
             (self.path, len(self._data_map))]
        s += self._readme_info
        s += ['Data:', '----']
        # ====== Find longest string ====== #
        longest_name = 0
        longest_shape = 0
        longest_dtype = 0
        longest_file = 0
        print_info = []
        for name, (dtype, shape, data, path) in self._data_map.iteritems():
            shape = data.shape if hasattr(data, 'shape') else shape
            longest_name = max(len(name), longest_name)
            longest_dtype = max(len(str(dtype)), longest_dtype)
            longest_shape = max(len(str(shape)), longest_shape)
            longest_file = max(len(str(path)), longest_file)
            print_info.append([name, dtype, shape, path])
        # ====== return print string ====== #
        format_str = (' Name:%-' + str(longest_name) + 's  '
                      'dtype:%-' + str(longest_dtype) + 's '
                      'shape:%-' + str(longest_shape) + 's  '
                      'file:%-' + str(longest_file) + 's')
        for name, dtype, shape, path in print_info:
            s.append(format_str % (name, dtype, shape, path))
        return '\n'.join(s)

    @property
    def readme(self):
        """ return text string of README of this dataset """
        if self._readme_path is not None:
            with open(self._readme_path, 'r') as f:
                readme = f.read()
        else:
            readme = self._readme_info[-1]
        return readme

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
        ds = Dataset(datapath, read_only=True)
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


def load_iris():
    path = "https://s3.amazonaws.com/ai-datasets/iris.zip"
    datapath = get_file('iris', path)
    return _load_data_from_path(datapath)


def load_digit_audio():
    path = 'https://s3.amazonaws.com/ai-datasets/digit.zip'
    name = 'digit'
    datapath = get_file(name, path)
    return _load_data_from_path(datapath)


def load_tiwave():
    path = 'https://s3.amazonaws.com/ai-datasets/tiwave.zip'
    name = 'tiwave'
    datapath = get_file(name, path)
    return _load_data_from_path(datapath)
