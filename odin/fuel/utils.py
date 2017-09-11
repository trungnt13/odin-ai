from __future__ import print_function, division, absolute_import

import os
import mmap
import marshal
import sqlite3
from six.moves import cPickle
from collections import OrderedDict, Iterator

import numpy as np

from odin.config import get_rng
from odin.utils import async, is_string


class MmapDict(dict):
    """ MmapDict
    Handle enormous dictionary (up to thousand terabytes of data) in
    memory mapped dictionary, extremely fast to load, and for randomly access.
    The alignment of saved files:
        |'mmapdict'|48-bytes(max_pos)|48-bytes(dict_size)|MmapData|indices-dict
    * The first 48-bytes number: is ending position of the Mmmap
    * The next 48-bytes number: is length of pickled indices (i.e. the
    indices start from max_pos to max_pos + dict_size)

    Note
    ----
    Only support (key, value) types = (str, primitive_type)

    """
    HEADER = 'mmapdict'
    SIZE_BYTES = 48
    __INSTANCES = {}

    def __new__(clazz, *args, **kwargs):
        path = kwargs.get('path', None)
        if path is None:
            path = args[0]
        if not is_string(path):
            raise ValueError("`path` for MmapDict must be string, but given "
                             "object with type: %s" % type(path))
        path = os.path.abspath(path)
        # Found old instance
        if path in MmapDict.__INSTANCES:
            return MmapDict.__INSTANCES[path]
        # new MmapDict
        new_instance = super(MmapDict, clazz).__new__(clazz, *args, **kwargs)
        MmapDict.__INSTANCES[path] = new_instance
        return new_instance

    def __init__(self, path, read_only=False):
        super(MmapDict, self).__init__()
        self.__init(path, read_only)
        self.read_only = read_only
        self._is_closed = False

    def __init(self, path, read_only):
        self._path = path
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
            _ = cPickle.dumps(self._indices_dict,
                              protocol=cPickle.HIGHEST_PROTOCOL)
            # write the length of Pickled indices dictionary
            file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % len(_))
            file.write(_)
            file.flush()
        # ====== create Mmap from offset file ====== #
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
    def loaded(self):
        if not isinstance(self._indices_dict, dict):
            return self._indices_dict.finished
        return True

    @property
    def is_closed(self):
        return self._is_closed

    @property
    def path(self):
        return self._path

    def flush(self):
        # check if closed
        if self._is_closed:
            return
        # check if in read only mode
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
        del self._write_value; self._write_value = ''
        del self._new_dict; self._new_dict = {}

    def close(self):
        # check if closed
        if self._is_closed:
            return
        # check if in read only mode
        if not self.read_only:
            self.flush()
        # remove global instance
        del MmapDict.__INSTANCES[os.path.abspath(self.path)]
        self._mmap.close()
        self._file.close()
        self._is_closed = True

    def __del__(self):
        # check if closed
        if self._is_closed:
            return
        if hasattr(self, '_mmap') and self._mmap is not None and \
        self._file is not None:
            path = os.path.abspath(self.path)
            if path in MmapDict.__INSTANCES and \
            id(self) == id(MmapDict.__INSTANCES[path]):
                del MmapDict.__INSTANCES[path]
            self._mmap.close()
            self._file.close()

    def __str__(self):
        return str(self.__class__) + ':' + self._path + ':' + str(len(self.indices))

    def __repr__(self):
        return str(self)

    # ==================== Dictionary ==================== #
    def __setitem__(self, key, value):
        if key in self.indices:
            raise Exception('This dictionary do not support update, i.e. cannot '
                            'update the value of key: %s' % key)
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


class SQLiteDict(dict):
    """ Using SQLite in key-value pair manner
    Note
    ---
    """

    _DEFAULT_TABLE = '_default_'
    __INSTANCES = {}

    def __new__(clazz, *args, **kwargs):
        path = kwargs.get('path', None)
        if path is None:
            path = args[0]
        if not is_string(path):
            raise ValueError("`path` for MmapDict must be string, but given "
                             "object with type: %s" % type(path))
        path = os.path.abspath(path)
        # Found old instance
        if path in SQLiteDict.__INSTANCES:
            return SQLiteDict.__INSTANCES[path]
        # new MmapDict
        new_instance = super(SQLiteDict, clazz).__new__(clazz, *args, **kwargs)
        SQLiteDict.__INSTANCES[path] = new_instance
        return new_instance

    def __init__(self, path, cache_size=250, read_only=False):
        super(SQLiteDict, self).__init__()
        path = os.path.abspath(path)
        self._path = path
        self.read_only = read_only
        self._is_closed = False
        # ====== cache mechanism ====== #
        self._cache_size = int(cache_size)
        self._cache = {}
        # ====== db manager ====== #
        self._conn = sqlite3.connect(path)
        self._conn.text_factory = str
        self._cursor = self._conn.cursor()
        # ====== create default table ====== #
        self.set_table(SQLiteDict._DEFAULT_TABLE)
        self._current_table = SQLiteDict._DEFAULT_TABLE

    @property
    def connection(self):
        return self._conn

    @property
    def cursor(self):
        return self._cursor

    def set_cache_size(self, cache):
        self._cache = int(cache)
        return self

    def set_table(self, table_name):
        table_name = str(table_name)
        if not self.is_table_exist(table_name):
            query = """CREATE TABLE {tb} (
                            key text NOT NULL,
                            value text NOT NULL,
                            PRIMARY KEY (key)
                        );"""
            self.cursor.execute(query.format(tb=table_name))
            query = """CREATE UNIQUE INDEX IX_fast ON {tb} (key);"""
            self.cursor.execute(query.format(tb=table_name))
            self.connection.commit()
        self._current_table = table_name
        return self

    def drop_table(self, table_name=None):
        if self.read_only:
            return
        if table_name is None:
            table_name = self._current_table
            self._current_table = SQLiteDict._DEFAULT_TABLE
        else:
            table_name = str(table_name)
        if table_name == SQLiteDict._DEFAULT_TABLE:
            raise ValueError("Cannot drop default table.")
        self.cursor.execute("""DROP TABLE {tb};""".format(tb=table_name))
        return self

    def is_table_exist(self, table_name):
        try:
            self.cursor.execute('SELECT 1 FROM %s LIMIT 1;' % table_name)
            self.cursor.fetchone()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return False
            else:
                raise e
        return True

    def flush(self):
        if not self.read_only and len(self._cache) > 0:
            self.cursor.executemany(
                "INSERT INTO {tb} VALUES (?, ?)".format(tb=self._current_table),
                [(str(k), marshal.dumps(v.tolist()) if isinstance(v, np.ndarray)
                  else marshal.dumps(v))
                 for k, v in self._cache.iteritems()])
            self.connection.commit()
        return self

    def close(self):
        # check if closed
        if self._is_closed:
            return
        # check if in read only model
        if not self.read_only:
            self.flush()
        # remove global instance
        del SQLiteDict.__INSTANCES[self.path]
        self._conn.close()
        self._is_closed = True

    def __del__(self):
        self.close()

    # ==================== pickling ==================== #
    def __setstate__(self, states):
        raise NotImplementedError

    def __getstate__(self):
        return self._path, self._cache, self.read_only

    @property
    def path(self):
        return self._path

    @property
    def is_closed(self):
        return self._is_closed

    # ==================== Dictionary ==================== #
    def __setitem__(self, key, value):
        key = str(key)
        self._cache[key] = value
        if len(self._cache) >= self._cache_size:
            self.flush()
            self._cache.clear()

    def __getitem__(self, key):
        # ====== multiple keys select ====== #
        if isinstance(key, (tuple, list, np.ndarray)):
            query = """SELECT value FROM {tb}
                       WHERE key IN {keyval};"""
            keyval = '(' + ', '.join([str(k) for k in key]) + ')'
            self.cursor.execute(
                query.format(tb=self._current_table, keyval=keyval))
            results = self.cursor.fetchall()
            # check if any not found keys
            if len(results) != len(key):
                raise KeyError("Cannot find all `key`='%s' in the dictionary." % keyval)
            # load binary data
            results = [marshal.loads(r[0]) for r in results]
        # ====== single key select ====== #
        else:
            key = str(key)
            if key in self._cache:
                return self._cache[key]
            query = """SELECT value FROM {tb} WHERE key={keyval} LIMIT 1;"""
            self.cursor.execute(
                query.format(tb=self._current_table, keyval=key))
            results = self.cursor.fetchone()
            if results is None:
                raise KeyError("Cannot find `key`='%s' in the dictionary." % key)
            results = marshal.loads(results[0])
        return results

    def update(self, items):
        if self.read_only:
            return
        query = """UPDATE {tb} SET value=(?) WHERE key=(?);"""
        if isinstance(items, dict):
            items = items.iteritems()
        # ====== check if update is in cache ====== #
        db_update = []
        for key, value in items:
            key = str(key)
            if key in self._cache:
                self._cache[key] = value
            else:
                db_update.append((marshal.dumps(value), key))
        # ====== perform DB update ====== #
        self.cursor.executemany(query.format(tb=self._current_table), db_update)
        self.connection.commit()
        return self

    def __iter__(self):
        return self.iteritems()

    def __contains__(self, key):
        key = str(key)
        # check in cache
        if key in self._cache:
            return True
        # check in database
        query = """SELECT 1 FROM {tb} WHERE key={keyval} LIMIT 1;"""
        self.cursor.execute(
            query.format(tb=self._current_table, keyval=key))
        if self.cursor.fetchone() is None:
            return False
        return True

    def has_key(self, key):
        return self.__contains__(key)

    def __len__(self):
        query = """SELECT COUNT(1) FROM {tb}""".format(tb=self._current_table)
        self.cursor.execute(query)
        n = self.cursor.fetchone()[0]
        return n + len(self._cache)

    def __delitem__(self, key):
        if self.read_only:
            return
        query = """DELETE FROM {tb} WHERE {cond};"""
        if isinstance(key, (tuple, list, Iterator, np.ndarray)):
            key = [str(k) for k in key]
        else:
            key = [str(key)]
        # ====== check if key in cache ====== #
        db_key = []
        for k in key:
            if k in self._cache:
                del self._cache[k]
            else:
                db_key.append(k)
        # ====== remove key from db ====== #
        self.cursor.execute(
            query.format(tb=self._current_table, cond='key IN (%s)' % ', '.join(db_key)))
        self.connection.commit()

    def __cmp__(self, dict):
        raise NotImplementedError

    def keys(self):
        return list(self.iterkeys())

    def iterkeys(self, shuffle=False):
        for k in self.cursor.execute(
            """SELECT key from {tb};""".format(tb=self._current_table)):
            yield k[0]
        for k in self._cache.iterkeys():
            yield k

    def values(self):
        return list(self.itervalues())

    def itervalues(self):
        for val in self.cursor.execute(
            """SELECT value from {tb};""".format(tb=self._current_table)):
            yield marshal.loads(val[0])
        for v in self._cache.itervalues():
            yield v

    def items(self):
        return list(self.iteritems())

    def iteritems(self):
        for item in self.cursor.execute(
            """SELECT key, value from {tb};""".format(tb=self._current_table)):
            yield (item[0], marshal.loads(item[1]))
        for k, v in self._cache.iteritems():
            yield k, v

    def clear(self):
        if self.read_only:
            return
        self.cursor.execute("""DELETE FROM {tb};""".format(tb=self._current_table))
        self.connection.commit()
        self._cache.clear()
        return self

    def copy(self):
        raise NotImplementedError
