from __future__ import print_function, division, absolute_import

import os
import mmap
import marshal
import sqlite3
from six.moves import cPickle
from contextlib import contextmanager
from collections import OrderedDict, Iterator, defaultdict
import io

import numpy as np

from odin.config import get_rng
from odin.utils import async, is_string, ctext


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

    def update(*args, **kwargs):
        raise NotImplementedError


# ===========================================================================
# SQLiteDict
# ===========================================================================
def _adapt_array(arr):
    """ Converts np.array to TEXT when inserting
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text):
    """ Converts TEXT to np.array when selecting """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


class TableDict(dict):

    def __init__(self, sqlite, table_name):
        if not isinstance(sqlite, SQLiteDict):
            raise ValueError("`sqlite` must be instance of SQLiteDict")
        self._sqlite = sqlite
        self._table_name = str(table_name)

    @contextmanager
    def table_context(self):
        """Return temporary context that switch the SQLite to given table"""
        curr_tab = self._sqlite.current_table
        self._sqlite.set_table(self._table_name)
        yield None
        self._sqlite.set_table(curr_tab)

    @property
    def sqlite(self):
        return self._sqlite

    @property
    def name(self):
        return self._table_name

    def set_cache_size(self, cache_size):
        self._sqlite.set_cache_size(cache_size)
        return self

    def flush(self):
        with self.table_context():
            self._sqlite.flush()

    # ==================== pickling and properties ==================== #
    def __setstate__(self, states):
        raise NotImplementedError

    def __getstate__(self):
        return self.path

    def __str__(self):
        return '<' + ctext('TableDict:', 'red') + ' "%s" db:%d cache:%d>' % \
        (self.path, len(self), len(self.sqlite._cache[self.name]))

    def __repr__(self):
        return str(self)

    @property
    def connection(self):
        return self._sqlite.connection

    @property
    def path(self):
        return self._sqlite.path

    @property
    def is_closed(self):
        return self._sqlite.is_closed

    # ==================== Dictionary ==================== #
    def set(self, key, value):
        self.__setitem__(key, value)
        return self

    def __setitem__(self, key, value):
        with self.table_context():
            return self._sqlite.__setitem__(key, value)

    def __getitem__(self, key):
        with self.table_context():
            return self._sqlite.__getitem__(key)

    def update(self, items):
        with self.table_context():
            self._sqlite.update(items)
        return self

    def __iter__(self):
        with self.table_context():
            return self._sqlite.__iter__()

    def __contains__(self, key):
        with self.table_context():
            return self._sqlite.__contains__(key)

    def __len__(self):
        with self.table_context():
            return self._sqlite.__len__()

    def __delitem__(self, key):
        with self.table_context():
            return self._sqlite.__delitem__(key)

    def __cmp__(self, dict):
        if isinstance(dict, TableDict):
            return self._sqlite is dict._sqlite and \
                self._table_name == dict._table_name
        return False

    def keys(self):
        return list(self.iterkeys())

    def iterkeys(self):
        with self.table_context():
            return self._sqlite.iterkeys()

    def values(self):
        return list(self.itervalues())

    def itervalues(self):
        with self.table_context():
            return self._sqlite.itervalues()

    def items(self):
        return list(self.iteritems())

    def iteritems(self):
        with self.table_context():
            return self._sqlite.items()

    def clear(self):
        with self.table_context():
            self._sqlite.clear()
        return self


class SQLiteDict(dict):
    """ Using SQLite in key-value pair manner

    Example
    -------
    >>> path = '/tmp/tmp.db'
    >>> db = SQLiteDict(path)
    >>> db1 = SQLiteDict(path)
    >>> print(id(db) == id(db1)) # True
    >>> def sett(i):
    ...     db = SQLiteDict(path)
    ...     db[i] = 'Process%d' % i
    ...     db.close()
    >>> def get(i):
    ...     db = SQLiteDict(path)
    ...     print("Get", db[i])
    >>> pros = [Process(target=sett, args=(i,))
    ...         for i in range(10)]
    >>> [p.start() for p in pros]
    >>> [p.join() for p in pros]
    ...
    >>> db = SQLiteDict(path)
    >>> print(db.items())
    >>> print(db[0])
    ...
    >>> pros = [Process(target=get, args=(i,))
    ...         for i in range(10)]
    >>> [p.start() for p in pros]
    >>> [p.join() for p in pros]

    Note
    ----
    numpy.ndarray will be converted to list before dumped to the database,
    hence, the speed of serialize and deserialize ndarray is extremely slow.

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
        self._cache = defaultdict(dict)
        # ====== db manager ====== #
        # detect_types=sqlite3.PARSE_DECLTYPES
        self._conn = sqlite3.connect(path)
        self._conn.text_factory = str
        # ====== create default table ====== #
        self._current_table = SQLiteDict._DEFAULT_TABLE
        self.set_table(SQLiteDict._DEFAULT_TABLE)

    # ==================== DB manager ==================== #
    def set_cache_size(self, cache_size):
        self._cache_size = int(cache_size)
        return self

    def as_table(self, table_name):
        return TableDict(self, table_name)

    def set_table(self, table_name):
        table_name = str(table_name)
        if not self.is_table_exist(table_name):
            query = """CREATE TABLE {tb} (
                            key text NOT NULL,
                            value text NOT NULL,
                            PRIMARY KEY (key)
                        );"""
            self.connection.execute(query.format(tb=table_name))
            query = """CREATE UNIQUE INDEX IX_{tb} ON {tb} (key);"""
            self.connection.execute(query.format(tb=table_name))
            self.connection.commit()
        # set the new table
        self._current_table = table_name
        return self

    @property
    def current_table(self):
        return self._current_table

    @property
    def current_cache(self):
        return self._cache[self.current_table]

    def get_all_tables(self):
        query = """SELECT name FROM sqlite_master where type='table';"""
        self.connection.execute(query)
        return [table[0] for table in self.connection.fetchall()]

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
        self.connection.execute("""DROP TABLE {tb};""".format(tb=table_name))
        return self

    def is_table_exist(self, table_name):
        try:
            self.connection.execute('SELECT 1 FROM %s LIMIT 1;' % table_name)
            self.connection.fetchone()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return False
            else:
                raise e
        return True

    def flush(self, all_tables=False):
        curr_tab = self.current_table
        tables = self.get_all_tables() if all_tables else [curr_tab]
        for tab in tables:
            self.set_table(tab)
            if not self.read_only and len(self.current_cache) > 0:
                self.connection.executemany(
                    "INSERT INTO {tb} VALUES (?, ?)".format(tb=tab),
                    [(str(k), marshal.dumps(v.tolist()) if isinstance(v, np.ndarray)
                      else marshal.dumps(v))
                     for k, v in self.current_cache.iteritems()])
                self.connection.commit()
                self.current_cache.clear()
        return self.set_table(curr_tab)

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

    # ==================== pickling and properties ==================== #
    def __setstate__(self, states):
        raise NotImplementedError

    def __getstate__(self):
        return self._path

    def __str__(self):
        curr_tab = self.current_table
        s = '<' + ctext('SQLiteDB:', 'red') + '%s>\n' % self.path
        all_tables = self.get_all_tables()
        name_fmt = '%' + str(max(len(i) for i in all_tables)) + 's'
        for tab in all_tables:
            self.set_table(tab)
            s += ' ' + ctext(name_fmt % tab, 'yellow') + ': db=%d cache=%d\n' % \
                (len(self), len(self.current_cache))
        self.set_table(curr_tab) # back to original table
        return s[:-1]

    def __repr__(self):
        return str(self)

    @property
    def connection(self):
        return self._conn

    @property
    def path(self):
        return self._path

    @property
    def is_closed(self):
        return self._is_closed

    # ==================== Dictionary ==================== #
    def set(self, key, value):
        self.__setitem__(key, value)
        return self

    def __setitem__(self, key, value):
        if self.read_only:
            raise RuntimeError("Cannot __setitem__ for this Dict in read_only mode.")
        key = str(key)
        self.current_cache[key] = value
        if len(self.current_cache) >= self._cache_size:
            self.flush()
            self.current_cache.clear()

    def __getitem__(self, key):
        # ====== multiple keys select ====== #
        if isinstance(key, (tuple, list, np.ndarray)):
            query = """SELECT value FROM {tb}
                       WHERE key IN {keyval};"""
            keyval = '(' + ', '.join(['"%s"' % str(k) for k in key]) + ')'
            self.connection.execute(
                query.format(tb=self._current_table, keyval=keyval))
            results = self.connection.fetchall()
            # check if any not found keys
            if len(results) != len(key):
                raise KeyError("Cannot find all `key`='%s' in the dictionary." % keyval)
            # load binary data
            results = [marshal.loads(r[0]) for r in results]
        # ====== single key select ====== #
        else:
            key = str(key)
            if key in self.current_cache:
                return self.current_cache[key]
            query = """SELECT value FROM {tb} WHERE key="{keyval}" LIMIT 1;"""
            self.connection.execute(
                query.format(tb=self._current_table, keyval=key))
            results = self.connection.fetchone()
            if results is None:
                raise KeyError("Cannot find `key`='%s' in the dictionary." % key)
            results = marshal.loads(results[0])
        return results

    def update(self, items):
        if self.read_only:
            return
        query = """UPDATE {tb} SET value=(?) WHERE key=("?");"""
        if isinstance(items, dict):
            items = items.iteritems()
        # ====== check if update is in cache ====== #
        db_update = []
        for key, value in items:
            key = str(key)
            if key in self.current_cache:
                self.current_cache[key] = value
            else:
                db_update.append((marshal.dumps(value), key))
        # ====== perform DB update ====== #
        self.connection.executemany(query.format(tb=self._current_table), db_update)
        self.connection.commit()
        return self

    def __iter__(self):
        return self.iteritems()

    def __contains__(self, key):
        key = str(key)
        # check in cache
        if key in self.current_cache:
            return True
        # check in database
        query = """SELECT 1 FROM {tb} WHERE key="{keyval}" LIMIT 1;"""
        self.connection.execute(
            query.format(tb=self._current_table, keyval=key))
        if self.connection.fetchone() is None:
            return False
        return True

    def __len__(self):
        query = """SELECT COUNT(1) FROM {tb}""".format(tb=self._current_table)
        self.connection.execute(query)
        n = self.connection.fetchone()[0]
        return n + len(self.current_cache)

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
            if k in self.current_cache:
                del self.current_cache[k]
            else:
                db_key.append(k)
        # ====== remove key from db ====== #
        self.connection.execute(
            query.format(tb=self._current_table,
                         cond='key IN ("%s")' % ', '.join(db_key)))
        self.connection.commit()

    def __cmp__(self, dict):
        raise NotImplementedError

    def keys(self):
        return list(self.iterkeys())

    def iterkeys(self):
        for k in self.connection.execute(
            """SELECT key from {tb};""".format(tb=self._current_table)):
            yield k[0]
        for k in self.current_cache.iterkeys():
            yield k

    def values(self):
        return list(self.itervalues())

    def itervalues(self):
        for val in self.connection.execute(
            """SELECT value from {tb};""".format(tb=self._current_table)):
            yield marshal.loads(val[0])
        for v in self.current_cache.itervalues():
            yield v

    def items(self):
        return list(self.iteritems())

    def iteritems(self):
        for item in self.connection.execute(
            """SELECT key, value from {tb};""".format(tb=self._current_table)):
            yield (item[0], marshal.loads(item[1]))
        for k, v in self.current_cache.iteritems():
            yield k, v

    def clear(self):
        if self.read_only:
            return
        self.connection.execute("""TRUNCATE TABLE {tb};""".format(tb=self._current_table))
        self.connection.commit()
        self.current_cache.clear()
        return self

    def copy(self):
        raise NotImplementedError
