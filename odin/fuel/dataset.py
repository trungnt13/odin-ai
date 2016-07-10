from __future__ import print_function, division, absolute_import

import os
from collections import OrderedDict
from six.moves import zip, range

import numpy as np

from .data import MmapData, Hdf5Data, open_hdf5, get_all_hdf_dataset, MAX_OPEN_MMAP, Data

from odin.utils import get_file, Progbar
from odin.utils.decorators import singleton


__all__ = [
    'Dataset',
    'load_mnist'
]


# ===========================================================================
# dataset
# ===========================================================================
def _parse_data_descriptor(path, name):
    """ Return mapping: path/name -> (dtype, shape, Data) """
    path = os.path.join(path, name)
    if not os.path.isfile(path):
        return None

    try:
        dtype, shape = MmapData.read_header(path)
        # shape[1:], because first dimension can be resize afterward
        return [(os.path.basename(path), (dtype, shape, path))]
    except: # cannot read the header of MmapData, maybe Hdf5
        try:
            f = open_hdf5(path)
            ds = get_all_hdf_dataset(f)
            data = [Hdf5Data(dataset=i, hdf=f) for i in ds]
            return [(str(i.name), (str(i.dtype), i.shape, i)) for i in data]
        except:
            pass
    return None


@singleton
class Dataset(object):

    """
    Note
    ----
    for developer: _data_map contains, key=(name, dtype shape); value=Data

    """

    def __init__(self, path):
        path = os.path.abspath(path)
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
            data = _parse_data_descriptor(path, f)
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

    # ==================== manipulate data ==================== #
    def get_data(self, name, dtype=None, shape=None, datatype='memmap'):
        """
        Parameters
        ----------
        datatype : memmap, hdf5
            Only support memmap numpy array of hdf5 via h5py
        """
        datatype = datatype.lower()
        if datatype not in ['memmap', 'hdf5']:
            raise ValueError("Only support 'memmap' or 'hdf5' datatype.")

        return_data = None
        # ====== find defined data ====== #
        if name in self._data_map:
            _dtype, _shape, _data = self._data_map[name]

            # return type is just a descriptor, create MmapData for it
            if not isinstance(_data, Data):
                return_data = MmapData(_data)
                self._data_map[name] = (return_data.dtype, return_data.shape, return_data)
            else: # for hdf5 Data, return directly
                return_data = _data
        # ====== auto create new data, if cannot find any match ====== #
        if return_data is None and dtype is not None and shape is not None:
            if datatype == 'memmap':
                return_data = MmapData(os.path.join(self.path, name),
                                       dtype=dtype, shape=shape)
            else:
                f = open_hdf5(os.path.join(self.path, self._default_hdf5))
                return_data = Hdf5Data(name, hdf=f, dtype=dtype, shape=shape)
            # store new key
            self._data_map[name] = (return_data.dtype, return_data.shape, return_data)
        # data still None
        if return_data is None:
            raise ValueError('Cannot find or create data with name={}, dtype={} '
                             'shape={}, and datatype={}'
                             ''.format(name, dtype, shape, datatype))
        # ====== check if excess limit, close 1 files ====== #
        if MmapData.COUNT > MAX_OPEN_MMAP:
            for i, (_dtype, _shape, _data) in self._data_map.iteritems():
                if isinstance(_data, MmapData) and i != name:
                    self.close(name=i)
                    self._data_map[i] = (_dtype, _shape, _data.path)
                    break
        return return_data

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
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_data(name=key)
        raise ValueError('Only accept key type is string.')

    def __iter__(self):
        for name, (dtype, shape, data) in self._data_map.iteritems():
            if isinstance(data, Data):
                yield data
            else:
                yield self.get_data(name)

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
            if hasattr(data, 'path'):
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
def _load_data_from_path(datapath):
    from zipfile import ZipFile, ZIP_DEFLATED
    if not os.path.isdir(datapath):
        datapath_tmp = datapath.replace('.zip', '') + '.tmp'
        os.rename(datapath, datapath_tmp)
        zf = ZipFile(datapath_tmp, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=datapath)
        zf.close()
        os.remove(datapath_tmp)
    ds = Dataset(datapath)
    return ds


def load_mnist(path='https://s3.amazonaws.com/ai-datasets/MNIST.zip'):
    """
    path : str
        local path or url to hdf5 datafile
    """
    datapath = get_file('MNIST', path)
    return _load_data_from_path(datapath)
