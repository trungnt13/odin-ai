from __future__ import print_function, division, absolute_import

import mmap
import os
import cPickle
import marshal


class MmapDict(dict):
    """ MmapDict
    Handle enormous dictionary (up to thousand terabytes of data) in
    memory mapped dictionary, extremely fast to load, and randomly access.

    Note
    ----
    Only support (key, value) types = (str, str)

    """
    HEADER = 'mmapdict'
    SIZE_BYTES = 48

    def __init__(self, path):
        super(MmapDict, self).__init__()
        self.__init(path)

    def __init(self, path):
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
            self._dict = cPickle.loads(file.read(dict_size))
        else:
            self._dict = {}
            # max position is header, include start and length of indices dict
            self._max_position = len(MmapDict.HEADER) + MmapDict.SIZE_BYTES * 2
            file = open(str(path), mode='w+')
            file.write(MmapDict.HEADER) # just write the header
            file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % self._max_position)
            _ = cPickle.dumps(self._dict, protocol=cPickle.HIGHEST_PROTOCOL)
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

    # ==================== I/O methods ==================== #
    @property
    def path(self):
        return self._path

    def flush(self):
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
        _ = cPickle.dumps(self._dict, protocol=cPickle.HIGHEST_PROTOCOL)
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
        self.flush()
        self._mmap.close()
        self._file.close()

    def __del__(self):
        if self._mmap is not None and self._file is not None:
            self._mmap.close()
            self._file.close()

    def __str__(self):
        return str(self.__class__) + ':' + self._path + ':' + str(len(self._dict))

    def __repr__(self):
        return str(self)

    # ==================== Dictionary ==================== #
    def __setitem__(self, key, value):
        if key in self._dict:
            raise Exception('This dictionary do not support update.')
        # we using marshal so this only support primitive value
        value = marshal.dumps(value)
        self._dict[key] = (self._max_position, len(value))
        self._max_position += len(value)
        self._write_value += value
        # store newly added value for fast query
        self._new_dict[key] = value
        if len(self._write_value) > 48000:
            self.flush()

    def __getitem__(self, key):
        if key in self._new_dict:
            return marshal.loads(self._new_dict[key])
        start, size = self._dict[key]
        self._mmap.seek(start)
        return marshal.loads(self._mmap.read(size))

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def __delitem__(self, key):
        del self._dict[key]

    def __cmp__(self, dict):
        if isinstance(dict, MmapDict):
            return cmp(self._dict, dict._dict)
        else:
            return cmp(self._dict, dict)

    def keys(self):
        return self._dict.keys()

    def iterkeys(self):
        return self._dict.iterkeys()

    def values(self):
        return list(self.itervalues())

    def itervalues(self):
        for k in self._dict.iterkeys():
            yield self[k]

    def items(self):
        return list(self.iteritems())

    def iteritems(self):
        for key, (start, size) in self._dict.iteritems():
            if key in self._new_dict:
                value = self._new_dict[key]
            else:
                self._mmap.seek(start)
                value = self._mmap.read(size)
            yield key, marshal.loads(value)

    def clear(self):
        self._dict.clear()

    def copy(self):
        raise NotImplementedError

    def has_key(self, key):
        return key in self._dict

    def update(*args, **kwargs):
        raise NotImplementedError

    # ==================== pickling ==================== #
    def __setstate__(self, states):
        self.__init(states)

    def __getstate__(self):
        return self._path


class MmapList(object):
    """docstring for MmapList"""

    def __init__(self, arg):
        super(MmapList, self).__init__()
        self.arg = arg
