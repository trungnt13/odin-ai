from __future__ import print_function, division, absolute_import

import os
import mmap
import marshal
from six.moves import cPickle
from collections import OrderedDict

import numpy as np

from odin.config import get_rng
from odin.utils import UnitTimer, async


class MmapDict(dict):
    """ MmapDict
    Handle enormous dictionary (up to thousand terabytes of data) in
    memory mapped dictionary, extremely fast to load, and randomly access.

    Note
    ----
    Only support (key, value) types = (str, primitive_type)

    """
    HEADER = 'mmapdict'
    SIZE_BYTES = 48

    def __init__(self, path, read_only=False):
        super(MmapDict, self).__init__()
        self.__init(path, read_only)
        self.read_only = read_only

    def __init(self, path, read_only):
        # ====== already exist ====== #
        if os.path.exists(path) and os.path.getsize(path) > 0:
            file = open(str(path), mode='r+')
            if file.read(len(MmapDict.HEADER)) != MmapDict.HEADER:
                raise Exception('Given file is not in the right format '
                                'for MmapDict.')
            # 48 bytes for the file size
            self._max_position = int(file.read(MmapDict.SIZE_BYTES))
            # length of pickled indices dictionary
            dict_size = int(file.read(MmapDict.SIZE_BYTES))
            # read dictionary
            file.seek(self._max_position)
            pickled_indices = file.read(dict_size)
            self._indices_dict = async(lambda: cPickle.loads(pickled_indices))()
        # ====== create new file from scratch ====== #
        else:
            if read_only:
                raise Exception('File at path:"%s" does not exist '
                                '(read-only mode).' % path)
            self._indices_dict = OrderedDict()
            # max position is header, include start and length of indices dict
            self._max_position = len(MmapDict.HEADER) + MmapDict.SIZE_BYTES * 2
            file = open(str(path), mode='w+')
            file.write(MmapDict.HEADER) # just write the header
            file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % self._max_position)
            _ = cPickle.dumps(self._indices_dict, protocol=cPickle.HIGHEST_PROTOCOL)
            # write the length of Pickled indices dictionary
            file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % len(_))
            file.write(_)
            file.flush()
        self._path = path
        self._mmap = mmap.mmap(file.fileno(), length=0, offset=0,
                               # access=mmap.ACCESS_READ,
                               flags=mmap.MAP_SHARED)
        # ignore the header
        self._file = file
        self._write_value = ''
        self._new_dict = {}# store all the (key, value) recently added

    # ==================== pickling ==================== #
    def __setstate__(self, states):
        path, read_only = states
        self.__init(path, read_only)

    def __getstate__(self):
        return self._path, self.read_only

    # ==================== I/O methods ==================== #
    @property
    def indices(self):
        if not isinstance(self._indices_dict, dict):
            self._indices_dict = self._indices_dict.get()
        return self._indices_dict

    @property
    def path(self):
        return self._path

    def flush(self):
        if self.read_only:
            raise Exception('Cannot flush to path:"%s" in read-only mode' % self._path)
        # ====== flush the data ====== #
        self._mmap.flush()
        self._mmap.close()
        self._file.close()
        # ====== write new data ====== #
        # save new data
        file = open(self._path, mode='r+')
        # get old position
        file.seek(len(MmapDict.HEADER))
        old_position = int(file.read(MmapDict.SIZE_BYTES))
        # write new max size
        file.seek(len(MmapDict.HEADER))
        file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % self._max_position)
        # length of Pickled indices dictionary
        _ = cPickle.dumps(self.indices, protocol=cPickle.HIGHEST_PROTOCOL)
        file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % len(_))
        # write new values
        file.seek(old_position)
        file.write(self._write_value)
        # write the indices dictionary
        file.write(_)
        file.flush()
        # store new information
        self._file = file
        self._mmap = mmap.mmap(self._file.fileno(),
                               length=0, offset=0,
                               flags=mmap.MAP_SHARED)
        # reset some values
        self._max_position = old_position + len(self._write_value)
        del self._write_value
        self._write_value = ''
        del self._new_dict
        self._new_dict = {}

    def close(self):
        if not self.read_only:
            self.flush()
        self._mmap.close()
        self._file.close()

    def __del__(self):
        if hasattr(self, '_mmap') and self._mmap is not None and \
        self._file is not None:
            self._mmap.close()
            self._file.close()

    def __str__(self):
        return str(self.__class__) + ':' + self._path + ':' + str(len(self.indices))

    def __repr__(self):
        return str(self)

    # ==================== Dictionary ==================== #
    def __setitem__(self, key, value):
        if key in self.indices:
            raise Exception('This dictionary do not support update.')
        # we using marshal so this only support primitive value
        value = marshal.dumps(value)
        self.indices[key] = (self._max_position, len(value))
        self._max_position += len(value)
        self._write_value += value
        # store newly added value for fast query
        self._new_dict[key] = value
        if len(self._write_value) > 48000:
            self.flush()

    def __iter__(self):
        return self.iteritems()

    def __getitem__(self, key):
        if key in self._new_dict:
            return marshal.loads(self._new_dict[key])
        start, size = self.indices[key]
        self._mmap.seek(start)
        return marshal.loads(self._mmap.read(size))

    def __contains__(self, key):
        return key in self.indices

    def __len__(self):
        return len(self.indices)

    def __delitem__(self, key):
        del self.indices[key]

    def __cmp__(self, dict):
        if isinstance(dict, MmapDict):
            return cmp(self.indices, dict._dict)
        else:
            return cmp(self.indices, dict)

    def keys(self, shuffle=False):
        k = self.indices.keys()
        if shuffle:
            get_rng().shuffle(k)
        return k

    def iterkeys(self, shuffle=False):
        if shuffle:
            return (k for k in self.keys(shuffle))
        return self.indices.iterkeys()

    def values(self, shuffle=False):
        return list(self.itervalues(shuffle))

    def itervalues(self, shuffle=False):
        for k in self.iterkeys(shuffle):
            yield self[k]

    def items(self, shuffle=False):
        return list(self.iteritems(shuffle))

    def iteritems(self, shuffle=False):
        # ====== shuffling if required ====== #
        if shuffle:
            it = self.indices.items()
            get_rng().shuffle(it)
        else:
            it = self.indices.iteritems()
        # ====== iter over items ====== #
        for key, (start, size) in it:
            if key in self._new_dict:
                value = self._new_dict[key]
            else:
                self._mmap.seek(start)
                value = self._mmap.read(size)
            yield key, marshal.loads(value)

    def clear(self):
        self.indices.clear()

    def copy(self):
        raise NotImplementedError

    def has_key(self, key):
        return key in self.indices

    def update(*args, **kwargs):
        raise NotImplementedError


class MmapList(object):
    """docstring for MmapList"""

    def __init__(self, arg):
        super(MmapList, self).__init__()
        self.arg = arg
