from __future__ import print_function, division, absolute_import

import os
import mmap
import marshal
import sqlite3
from six.moves import cPickle
from itertools import chain
from contextlib import contextmanager
from collections import OrderedDict, Iterator, defaultdict
import io

import numpy as np

from odin.config import get_rng
from odin.utils import async, is_string, ctext


# ===========================================================================
# MmapDict
# ===========================================================================
class MmapDict(dict):
    """ MmapDict
    Handle enormous dictionary (up to thousand terabytes of data) in
    memory mapped dictionary, extremely fast to load, and for randomly access.
    The alignment of saved files:

    ==> |'mmapdict'|48-bytes(max_pos)|48-bytes(dict_size)|MmapData|indices-dict|

    * The first 48-bytes number: is ending position of the Mmmap, start from
    the (8 + 48 + 48) bytes.

    * The next 48-bytes number: is length of pickled indices (i.e. the
    indices start from max_pos to max_pos + dict_size)

    Note
    ----
    Only support (key, value) types = (str, primitive_type)
    MmapDict read speed is double faster than SQLiteDict.
    MmapDict also support multiprocessing
    """
    HEADER = 'mmapdict'
    SIZE_BYTES = 48
    # the indices are flushed after it is increased this amount of size
    MAX_INDICES_SIZE = 25 # in megabyte
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

    def __init__(self, path, cache_size=250, read_only=False):
        super(MmapDict, self).__init__()
        self.__init(path, cache_size, read_only)

    def __init(self, path, cache_size, read_only):
        path = os.path.abspath(path)
        self._path = path
        self.read_only = bool(read_only)
        self.cache_size = int(cache_size)
        # ====== already exist ====== #
        if os.path.exists(path) and os.path.getsize(path) > 0:
            file = open(str(path), mode='r+')
            if file.read(len(MmapDict.HEADER)) != MmapDict.HEADER:
                raise Exception('Given file is not in the right format '
                                'for MmapDict.')
            # 48 bytes for the file size
            max_position = int(file.read(MmapDict.SIZE_BYTES))
            # length of pickled indices dictionary
            dict_size = int(file.read(MmapDict.SIZE_BYTES))
            # read dictionary
            file.seek(max_position)
            pickled_indices = file.read(dict_size)
            self._indices_dict = async(lambda: cPickle.loads(pickled_indices))()
        # ====== create new file from scratch ====== #
        else:
            if read_only:
                raise Exception('File at path:"%s" does not exist '
                                '(read-only mode).' % path)
            file = open(str(path), mode='w+')
            file.write(MmapDict.HEADER)
            # just write the header
            file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') %
                       (len(MmapDict.HEADER) + MmapDict.SIZE_BYTES * 2))
            # write the length of Pickled indices dictionary
            file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % 0)
            file.flush()
            # init indices dict
            self._indices_dict = {}
        # ====== create Mmap from offset file ====== #
        self._file = file
        self._mmap = mmap.mmap(file.fileno(), length=0, offset=0,
                               flags=mmap.MAP_SHARED)
        # store all the (key, value) recently added
        self._cache_dict = {}
        self._increased_indices_size = 0.
        self._is_closed = False

    # ==================== pickling ==================== #
    def __setstate__(self, states):
        raise NotImplementedError

    def __getstate__(self):
        return self.path

    # ==================== I/O methods ==================== #
    @property
    def indices(self):
        if not isinstance(self._indices_dict, dict):
            self._indices_dict = self._indices_dict.get()
        return self._indices_dict

    @property
    def is_loaded(self):
        if self.is_closed:
            return False
        if not isinstance(self._indices_dict, dict):
            return self._indices_dict.finished
        return True

    @property
    def is_closed(self):
        return self._is_closed

    @property
    def path(self):
        return self._path

    def flush(self, save_indices=False):
        """
        Parameters
        ----------
        save_indices: bool
            force the indices dictionary to be saved, even though,
            its increased hasn't reach the maximum.
        """
        # check if closed or in read only mode
        if self.is_closed or self.read_only:
            return
        # ====== write new data ====== #
        # get old position
        file = self._file
        # start from header (i.e. "mmapdict")
        file.seek(len(MmapDict.HEADER))
        max_position = int(file.read(MmapDict.SIZE_BYTES))
        # ====== serialize the data ====== #
        # start from old_max_position, append new values
        file.seek(max_position)
        for key, value in self._cache_dict.iteritems():
            value = marshal.dumps(value)
            self.indices[key] = (max_position, len(value))
            max_position += len(value)
            file.write(value)
            # increase indices size (in MegaBytes)
            self._increased_indices_size += (8 + 8 + len(key)) / 1024. / 1024.
        # ====== write the dumped indices ====== #
        indices_length = 0
        if save_indices or \
        self._increased_indices_size > MmapDict.MAX_INDICES_SIZE:
            indices_dump = cPickle.dumps(self.indices,
                                         protocol=cPickle.HIGHEST_PROTOCOL)
            indices_length = len(indices_dump)
            file.write(indices_dump)
            self._increased_indices_size = 0.
        # ====== update the position ====== #
        # write new max size
        file.seek(len(MmapDict.HEADER))
        file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % max_position)
        # update length of pickled indices dictionary
        if indices_length > 0:
            file.write(('%' + str(MmapDict.SIZE_BYTES) + 'd') % indices_length)
        # flush everything
        file.flush()
        # upate the mmap
        self._mmap.close(); del self._mmap
        self._mmap = mmap.mmap(file.fileno(), length=0, offset=0,
                               flags=mmap.MAP_SHARED)
        # reset some values
        del self._cache_dict
        self._cache_dict = {}

    def close(self):
        # check if closed
        if self._is_closed:
            return
        # check if in read only mode
        if not self.read_only:
            self.flush(save_indices=True)
        # remove global instance
        del MmapDict.__INSTANCES[self.path]
        self._mmap.close()
        self._file.close()
        self._is_closed = True
        del self._indices_dict
        del self._cache_dict

    def __del__(self):
        self.close()

    def __str__(self):
        length = None if self.is_closed else \
            str(len(self.indices) + len(self._cache_dict))
        cache_length = 'None' if self.is_closed else \
            str(len(self._cache_dict))
        fmt = '<MmapDict path:"%s", length:%s/%s, loaded:%s, closed:%s>'
        return fmt % (self.path, length, cache_length,
                      self.is_loaded, self.is_closed)

    def __repr__(self):
        return str(self)

    # ==================== Dictionary ==================== #
    def __setitem__(self, key, value):
        if self.read_only:
            return
        key = str(key)
        if key in self.indices:
            raise Exception('MmapDict do NOT support update, i.e. cannot '
                            'update the value of key: "%s"' % key)
        # store newly added value for fast query
        self._cache_dict[key] = value
        if len(self._cache_dict) > self.cache_size:
            self.flush()

    def __iter__(self):
        return self.iteritems()

    def __getitem__(self, key):
        if key in self._cache_dict:
            return self._cache_dict[key]
        # ====== load from mmap ====== #
        start, size = self.indices[key]
        self._mmap.seek(start)
        return marshal.loads(self._mmap.read(size))

    def __contains__(self, key):
        return key in self.indices or key in self._cache_dict

    def __len__(self):
        return len(self.indices) + len(self._cache_dict)

    def __delitem__(self, key):
        if self.read_only:
            return
        if key in self._cache_dict:
            del self._cache_dict
        else:
            del self.indices[key]

    def __cmp__(self, d):
        if isinstance(d, MmapDict):
            return (self.indices == d.indices and
                    self._cache_dict == d._cache_dict)
        else:
            for key, value in self.iteritems():
                if key not in d or d[key] != value:
                    return False
            return True

    def keys(self):
        return list(self.iterkeys())

    def iterkeys(self):
        return chain(self.indices.iterkeys(), self._cache_dict.iterkeys())

    def values(self):
        return list(self.itervalues())

    def itervalues(self, shuffle=False):
        for name, (start, size) in self.indices.iteritems():
            self._mmap.seek(start)
            yield marshal.loads(self._mmap.read(size))
        for val in self._cache_dict.itervalues():
            yield val

    def items(self):
        return list(self.iteritems())

    def iteritems(self):
        for name, (start, size) in self.indices.iteritems():
            self._mmap.seek(start)
            yield name, marshal.loads(self._mmap.read(size))
        for key, val in self._cache_dict.itervalues():
            yield key, val

    def clear(self):
        if self.read_only:
            return
        self.indices.clear()
        self._cache_dict.clear()
        return self

    def copy(self):
        raise NotImplementedError

    def update(self, items):
        if self.read_only:
            return
        if isinstance(items, dict):
            items = items.iteritems()
        # ====== check if update is in cache ====== #
        for key, value in items:
            self.__setitem__(key, value)


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
        return
        '<' + ctext('TableDict: ', 'red') + 'name:"%s" path:"%s" db:%d cache:%d>' % \
        (self.name, self.path, len(self), len(self.sqlite._cache[self.name]))

    def __repr__(self):
        return str(self)

    @property
    def connection(self):
        return self._sqlite.connection

    @property
    def cursor(self):
        return self._sqlite.cursor

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
    This dict is purely performing in multiprocessing
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
        self._cursor = self._conn.cursor()
        # adjust pragma
        # SQLITE_OPEN_EXCLUSIVE
        self.connection.execute('PRAGMA main.locking_mode = EXCLUSIVE;')
        self.connection.execute("PRAGMA main.synchronous = 0;")
        self.connection.execute("PRAGMA journal_mode = MEMORY;")
        self.connection.commit()
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
        if table_name is None:
            table_name = SQLiteDict._DEFAULT_TABLE
        table_name = str(table_name)
        if not self.is_table_exist(table_name):
            query = """CREATE TABLE {tb} (
                            key text NOT NULL,
                            value text NOT NULL,
                            PRIMARY KEY (key)
                        );"""
            self.cursor.execute(query.format(tb=table_name))
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
        self.cursor.execute(query)
        return [table[0] for table in self.cursor.fetchall()]

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

    def flush(self, all_tables=False):
        curr_tab = self.current_table
        tables = self.get_all_tables() if all_tables else [curr_tab]
        for tab in tables:
            self.set_table(tab)
            if not self.read_only and len(self.current_cache) > 0:
                self.cursor.executemany(
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
    def cursor(self):
        return self._cursor

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
            if key in self.current_cache:
                return self.current_cache[key]
            query = """SELECT value FROM {tb} WHERE key="{keyval}" LIMIT 1;"""
            results = self.connection.execute(
                query.format(tb=self._current_table, keyval=key)).fetchone()
            # results = self.cursor.fetchone()
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
        self.cursor.executemany(query.format(tb=self._current_table), db_update)
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
        self.cursor.execute(
            query.format(tb=self._current_table, keyval=key))
        if self.cursor.fetchone() is None:
            return False
        return True

    def __len__(self):
        query = """SELECT COUNT(1) FROM {tb}""".format(tb=self._current_table)
        self.cursor.execute(query)
        n = self.cursor.fetchone()[0]
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
        self.cursor.execute(
            query.format(tb=self._current_table,
                         cond='key IN ("%s")' % ', '.join(db_key)))
        self.connection.commit()

    def __cmp__(self, dict):
        raise NotImplementedError

    def keys(self):
        return list(self.iterkeys())

    def iterkeys(self):
        for k in self.cursor.execute(
            """SELECT key from {tb};""".format(tb=self._current_table)):
            yield k[0]
        for k in self.current_cache.iterkeys():
            yield k

    def values(self):
        return list(self.itervalues())

    def itervalues(self):
        for val in self.cursor.execute(
            """SELECT value from {tb};""".format(tb=self._current_table)):
            yield marshal.loads(val[0])
        for v in self.current_cache.itervalues():
            yield v

    def items(self):
        return list(self.iteritems())

    def iteritems(self):
        for item in self.cursor.execute(
            """SELECT key, value from {tb};""".format(tb=self._current_table)):
            yield (item[0], marshal.loads(item[1]))
        for k, v in self.current_cache.iteritems():
            yield k, v

    def clear(self):
        if self.read_only:
            return
        self.cursor.execute("""TRUNCATE TABLE {tb};""".format(tb=self._current_table))
        self.connection.commit()
        self.current_cache.clear()
        return self

    def copy(self):
        raise NotImplementedError
