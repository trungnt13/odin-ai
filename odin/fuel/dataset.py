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
    path = os.path.join(path, name)
    if not os.path.isfile(path):
        return None

    mmap_match = MmapData.PATTERN.match(name)
    # memmap files
    if mmap_match is not None and \
       len([i for i in mmap_match.groups() if i is not None]) == 4:
        name, dtype, shape = mmap_match.group(1), mmap_match.group(2), mmap_match.group(3)
        dtype = np.dtype(dtype)
        shape = eval(shape)
        # shape[1:], because first dimension can be resize afterward
        return [((name, dtype, shape[1:]), (shape[0], None))]
    # hdf5 files
    elif any(i in name for i in Hdf5Data.SUPPORT_EXT):
        try:
            f = open_hdf5(path)
            ds = get_all_hdf_dataset(f)
            data = [Hdf5Data(i, f) for i in ds]
            return [((i.name, i.dtype, i.shape[1:]), i) for i in data]
        except Exception, e:
            import traceback; traceback.print_exc()
            raise ValueError('Error loading hdf5 data, error:{}, file:{} '
                             ''.format(e, path))
    return None


@singleton
class Dataset(object):

    '''
    Note
    ----
    for developer: _data_map contains, key=(name, dtype shape); value=Data

    '''

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

        if not os.path.exists(path):
            os.mkdir(path)
        elif not os.path.isdir(path):
            raise ValueError('Dataset path must be folder.')

        files = os.listdir(path)
        for f in files:
            data = _parse_data_descriptor(path, f)
            if data is None:
                continue
            for key, d in data:
                if key in self._data_map:
                    raise ValueError('Found duplicated data with follow info: '
                                     '{}'.format(key))
                else:
                    self._data_map[key] = d

        self._path = path
        self._name = os.path.basename(path)
        if len(self._name) == 1:
            self._name = os.path.basename(os.path.abspath(path))
        self._default_hdf5 = self.name + '_default.h5'

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
        return os.path.join(self._path, '..', self._name + '.zip')

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        ''' return size in MegaByte'''
        size_bytes = 0
        for (name, dtype, shape), value in self._data_map.iteritems():
            size = np.dtype(dtype).itemsize
            if hasattr(value, 'shape'):
                shape = value.shape
            else: # memmap descriptor
                shape = (value[0],) + shape
            n = np.prod(shape)
            size_bytes += size * n
        return size_bytes / 1024. / 1024.

    def keys(self):
        '''
        Return
        ------
        (name, dtype, shape): tuple
        '''
        return [(name, dtype, value.shape) if hasattr(value, 'shape')
                else (name, dtype, (value[0],) + shape)
                for (name, dtype, shape), value in self._data_map.iteritems()]

    @property
    def info(self):
        '''
        Return
        ------
        (name, dtype, shape): tuple
        '''
        return [(name, dtype, value.shape, type(value))
                if hasattr(value, 'shape')
                else (name, dtype, (value[0],) + shape, type(MmapData))
                for (name, dtype, shape), value in self._data_map.iteritems()]

    # ==================== manipulate data ==================== #
    def get_data(self, name, dtype=None, shape=None, datatype='mmap', value=None):
        """
        Parameters
        ----------
        value : np.ndarray
            if this is the first time Data is initialized, we assign Data to
            the value
            otherwise, we append value to data.
        """
        datatype = '.' + datatype.lower() if '.' not in datatype else datatype.lower()
        if datatype not in MmapData.SUPPORT_EXT and \
           datatype not in Hdf5Data.SUPPORT_EXT:
            raise ValueError("No support for data type: {}, following formats "
                             " are supported: {} and {}"
                             "".format(
                            datatype, Hdf5Data.SUPPORT_EXT, MmapData.SUPPORT_EXT))
        dtype = value.dtype if value is not None and dtype is None else dtype
        shape = value.shape if value is not None and shape is None else shape
        return_data = None
        return_key = None
        # ====== find defined data ====== #
        for k in self._data_map.keys():
            _name, _dtype, _shape = k
            if name == _name:
                if dtype is not None and np.dtype(_dtype) != np.dtype(dtype):
                    continue
                if shape is not None and shape[1:] != _shape:
                    continue
                return_data = self._data_map[k]
                return_key = k
                # return type is just a descriptor, create MmapData for it
                if not isinstance(return_data, Data):
                    return_data = MmapData(os.path.join(self.path, _name),
                        dtype=_dtype, shape=(return_data[0],) + _shape)
                    self._data_map[return_key] = return_data
                # append value
                if value is not None and value.shape[1:] == _shape:
                    return_data.append(value)
                    return_data.flush()
                break
        # ====== auto create new data, if cannot find any match ====== #
        if return_data is None and dtype is not None and shape is not None:
            if datatype in MmapData.SUPPORT_EXT:
                return_data = MmapData(os.path.join(self.path, name), dtype=dtype, shape=shape)
            else:
                f = open_hdf5(os.path.join(self.path, self._default_hdf5))
                return_data = Hdf5Data(name, f, dtype=dtype, shape=shape)
            # first time create the dataset, assign init value
            if value is not None and value.shape == return_data.shape:
                return_data.prepend(value)
                return_data.flush()
            # store new key
            return_key = (return_data.name, return_data.dtype, return_data.shape[1:])
            self._data_map[return_key] = return_data
        # data still None
        if return_data is None:
            raise ValueError('Cannot find or create data with name={}, dtype={} '
                             'shape={}, and datatype={}'
                             ''.format(name, dtype, shape, datatype))
        # ====== check if excess limit, close 1 files ====== #
        if MmapData.COUNT > MAX_OPEN_MMAP:
            for i, j in self._data_map.iteritems():
                if isinstance(j, MmapData) and i != return_key:
                    break
            n = j.shape[0]
            del self._data_map[i]
            self._data_map[i] = (n, None)
        return return_data

    def create_iter(self, names,
        batch_size=256, shuffle=True, seed=None, start=0., end=1., mode=0):
        pass

    def archive(self):
        from zipfile import ZipFile, ZIP_DEFLATED
        path = self.archive_path
        zfile = ZipFile(path, mode='w', compression=ZIP_DEFLATED)

        files = []
        for key, value in self._data_map.iteritems():
            if hasattr(value, 'path'):
                files.append(value.path)
            else: # unloaded data
                name, dtype, shape = key
                n = value[0]
                name = MmapData.info_to_name(name, (n,) + shape, dtype)
                files.append(os.path.join(self.path, name))
        files = set(files)
        progbar = Progbar(len(files), title='Archiving:')

        maxlen = max([len(os.path.basename(i)) for i in files])
        for i, f in enumerate(files):
            zfile.write(f, os.path.basename(f))
            progbar.title = ('Archiving: %-' + str(maxlen) + 's') % os.path.basename(f)
            progbar.update(i + 1)
        zfile.close()
        return path

    def flush(self, name=None, dtype=None, shape=None):
        if name is None: # flush all files
            for v in self._data_map.values():
                if isinstance(v, Data):
                    v.flush()
        else: # flush a particular file
            for (n, d, s), j in self._data_map.items():
                if not isinstance(j, Data): continue
                if name == n:
                    if dtype is not None and np.dtype(dtype) != np.dtype(d):
                        continue
                    if shape is not None and shape[1:] != s:
                        continue
                    self._data_map[(n, d, s)].flush()

    def close(self, name=None, dtype=None, shape=None):
        if name is None: # close all files
            for k in self._data_map.keys():
                del self._data_map[k]
            try:
                self.dispose()
            except:
                pass
        else: # close a particular file
            for (n, d, s), j in self._data_map.items():
                if name == n:
                    if dtype is not None and np.dtype(dtype) != np.dtype(d):
                        continue
                    if shape is not None and shape[1:] != s:
                        continue
                    del self._data_map[(n, d, s)]

    # ==================== Some info ==================== #
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_data(name=key)
        params = {}
        if isinstance(key, (tuple, list)):
            for i in key:
                if isinstance(i, str):
                    try:
                        params['dtype'] = np.dtype(i)
                    except:
                        params['name'] = i
                elif isinstance(i, (tuple, list)):
                    params['shape'] = i
        elif isinstance(key, dict):
            params = key
        return self.get_data(**params)

    def __str__(self):
        s = ['====== Dataset:%s Total:%d ======' %
             (self.path, len(self._data_map))]
        # ====== Find longest string ====== #
        longest_name = 0
        longest_shape = 0
        longest_file = len(str('not loaded'))
        for (name, dtype, _), data in self._data_map.iteritems():
            shape = data.shape if hasattr(data, 'shape') else (data[0],) + _
            longest_name = max(len(name), longest_name)
            longest_shape = max(len(str(shape)), longest_shape)
            if isinstance(data, Data):
                longest_file = max(len(str(data.path)), longest_file)
        # ====== return print string ====== #
        format_str = ('Name:%-' + str(longest_name) + 's  '
                      'dtype:%-7s  '
                      'shape:%-' + str(longest_shape) + 's  '
                      'file:%-' + str(longest_file) + 's')
        for (name, dtype, _), data in self._data_map.iteritems():
            shape = data.shape if hasattr(data, 'shape') else (data[0],) + _
            path = data.path if isinstance(data, Data) else 'not loaded'
            s.append(format_str % (name, dtype, shape, path))
        return '\n'.join(s)

    # ==================== Pickle ==================== #
    def __getstate__(self):
        config = OrderedDict()
        # convert to byte
        config['path'] = self.path
        return config

    def __setstate__(self, config):
        self._set_path(config['path'])


# ===========================================================================
# Predefined dataset
# ===========================================================================
def _load_data_from_path(datapath):
    from zipfile import ZipFile, ZIP_DEFLATED
    if not os.path.isdir(datapath):
        datapath_tmp = datapath + '.tmp'
        os.rename(datapath, datapath_tmp)
        zf = ZipFile(datapath_tmp, mode='r', compression=ZIP_DEFLATED)
        zf.extractall(path=datapath)
        zf.close()
        os.remove(datapath_tmp)
    ds = Dataset(datapath)
    return ds


def load_mnist(path='https://s3.amazonaws.com/ai-datasets/MNIST'):
    '''
    path : str
        local path or url to hdf5 datafile
    '''
    datapath = get_file('MNIST', path)
    return _load_data_from_path(datapath)
