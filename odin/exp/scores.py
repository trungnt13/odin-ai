from __future__ import absolute_import, division, print_function

import itertools
import os
import re
import sqlite3
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from numbers import Number

import numpy as np
from six import string_types


# ===========================================================================
# Helpers
# ===========================================================================
def _to_sqltype(x):
  if isinstance(x, Number):
    if isinstance(x, int) or x.is_integer():
      return "INTEGER"
    return "REAL"
  if isinstance(x, np.ndarray):
    return "BLOB"
  return "TEXT"


def _to_pytype(x):
  if x == "INTEGER":
    return int
  if x == "REAL":
    return float
  if x == "BLOB":
    return np.ndarray
  return str


def _data(x):
  if isinstance(x, string_types):
    return x
  elif isinstance(x, Number):
    if isinstance(x, int) or x.is_integer():
      return int(x)
    return float(x)
  b = BytesIO()
  np.savez_compressed(b, x=x)
  b.seek(0)
  data = b.read()
  b.close()
  return data


def _parse(x):
  if isinstance(x, bytes):
    b = BytesIO(x)
    return np.load(b)['x']
  return x


# ===========================================================================
# Main
# ===========================================================================
class ScoreBoard:
  r""" Using SQLite database for storing the scores and configuration of
  multiple experiments.

  Note:
    it might be easier to just use NoSQL, however, we are not dealing with
    performance critical app so SQL still a more intuitive approach.

    All column names are lower case
  """

  def __init__(self, path=":memory:"):
    if ':memory:' not in path:
      path = os.path.abspath(os.path.expanduser(path))
      if os.path.isdir(path):
        raise ValueError("path to %s must be path to a file." % path)
    self.path = path
    self._conn = None
    self._c = None

  @property
  def conn(self) -> sqlite3.Connection:
    if self._conn is None:
      self._conn = sqlite3.connect(self.path)
    return self._conn

  @contextmanager
  def recording(self):
    self._c = self.conn.cursor()
    yield self
    self.conn.commit()
    self._c.close()
    self._c = None

  ######## Good old query
  @contextmanager
  def cursor(self):
    c = self.conn.cursor()
    yield c
    self.conn.commit()
    c.close()

  def is_table_exist(self, name):
    with self.cursor() as c:
      count = c.execute(
          f"""SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{name}'"""
      ).fetchone()
    if count[0] == 0:
      return False
    return True

  def get_all_tables(self, python_type=True):
    r""" Return a dictionary mapping from table name to columns and
    data type """
    with self.cursor() as c:
      name = c.execute(
          f"""SELECT name FROM sqlite_master WHERE type='table'""").fetchall()
      name = [i[0] for i in name]
      tables = {}
      for table_name in name:
        desc = [(i[1], _to_pytype(i[2]) if python_type else i[2]) for i in
                c.execute(f"""PRAGMA table_info({table_name})""").fetchall()]
        tables[table_name] = OrderedDict(desc)
    return tables

  def get_nrow(self, table):
    table = str(table).strip().lower()
    return self.select(f"SELECT count() FROM {table}")[0]

  def get_table(self, table, where="", distinct=False, newest_first=True):
    r""" Get all rows from given table, exclude the column 'timestamp'

    Example
    ```
    get_table("t1", where="a=1 and b=2", distinct=True)
    ```
    """
    order = ""
    if newest_first:
      order = "ORDER BY timestamp DESC"
    if distinct:
      distinct = "DISTINCT"
    else:
      distinct = ""
    where = str(where)
    if len(where) > 0 and 'where' not in where.lower():
      where = "WHERE " + where
    # remove the timestamp
    table = str(table).strip().lower()
    return [
        row[:-1] for row in self.select(
            f"SELECT {distinct} * FROM {table} {where} {order}")
    ]

  def select(self,
             query=None,
             table="def",
             keys='*',
             where="",
             join="",
             order="",
             group=""):
    r""" If `query` is given run the query directly, otherwise, infer the
    appropriate query from all given information

    The format of the query is:
    `SELECT {keys} FROM {table} {join} {where} {order} {group};`

    An example query:
    ```
    SELECT DISTINCT t1.a,t2.b
    FROM t1
      JOIN t2 ON t1.c=t2.c
    WHERE t1.d>1 and t2.d<3
    ORDER BY
      t1.e asc, t1.f desc
    GROUP BY
      t1.a, t2.g
    HAVING t1.a=1;
    ```

    Note:
      All table has the 'timestamp' column, select * will return the timestamp
      as well. For selecting the newest inserted rows, `ORDER BY timestamp desc`


    Example:
    ```
    select(table='t1', keys='t1.c', where="", join="INNER JOIN t2 on t1.a=t2.a")
    # is the same as
    select(query="SELECT t1.c FROM t1 INNER JOIN t2 on t1.a=t2.a;")
    # also same as
    select(query="SELECT t1.c FROM t1, t2 WHERE t1.a=t2.a;")
    ```
    """
    if query is None:
      # join
      join = str(join)
      # where clause
      where = str(where).strip().lower()
      if len(where) > 0 and "where" not in where.lower():
        where = "WHERE %s" % where
      # order
      order = str(order)
      if len(order) > 0 and "order by" not in order.lower():
        order = "ORDER BY %s" % order
      # group
      group = str(group)
      if len(group) > 0 and "group by" not in group.lower():
        group = "GROUP BY %s" % group
      # table
      if isinstance(table, (tuple, list)):
        table = ','.join([str(i) for i in table])
      table = str(table).strip().lower()
      # keys
      if isinstance(keys, (tuple, list)):
        keys = ','.join([str(k).strip().lower() for k in keys])
      else:
        keys = str(keys)
      keys = ','.join(
          [table + '.' + k if '.' not in k else k \
            for k in keys.split(',')])
      # final query
      query = f"""SELECT {keys} FROM {table} {join} {where} {order} {group};"""
    # execute
    with self.cursor() as c:
      try:
        rows = c.execute(str(query)).fetchall()
      except sqlite3.OperationalError as e:
        print(query)
        raise e
      rows = [[_parse(x) for x in r] for r in rows]
      rows = [r[0] if len(r) == 1 else r for r in rows]
    return rows

  ######## Create and insert
  def _create_table(self, c, name, row, unique):
    keys = []
    keys_name = []
    for k, v in row.items():
      k = str(k).strip().lower()
      keys_name.append(k)
      keys.append([k, _to_sqltype(v)])
    keys = ", ".join([" ".join(i) for i in keys])
    if unique:
      # no timestamp
      unique = ", UNIQUE (%s)" % ','.join(keys_name[:-1])
    else:
      unique = ""
    query = f""" CREATE TABLE IF NOT EXISTS {name} ({keys}{unique});"""
    try:
      c.execute(query)
    except sqlite3.OperationalError as e:
      print(query)
      raise e

  def _write_table(self, c, table, unique, **row):
    row['timestamp'] = datetime.now().timestamp()
    self._create_table(c, table, row, unique)
    table_name = str(table).strip().lower()
    cols = ",".join([str(k).strip().lower() for k in row.keys()])
    fmt = ','.join(['?'] * len(row))
    try:
      c.execute(f"""INSERT INTO {table_name} ({cols}) VALUES({fmt});""",
                [_data(v) for v in row.values()])
    except sqlite3.IntegrityError as e:
      if unique:
        pass
      else:
        raise e

  def write(self, table, unique=False, **row):
    if self._c is None:
      with self.cursor() as c:
        self._write_table(c, table=table, unique=unique, **row)
    else:
      self._write_table(self._c, table=table, unique=unique, **row)
    return self

  ######## others
  def __repr__(self):
    return self.__str__()

  def __str__(self):
    text = "ScoreBoard: %s\n" % self.path
    for tab, attrs in self.get_all_tables(python_type=False).items():
      text += " Table: '%s' %d(rows)\n" % (tab, self.get_nrow(tab))
      for k, t in attrs.items():
        text += "  (%-7s) %s\n" % (t, k)
    return text[:-1]

  def close(self):
    if self._c is not None:
      self._c.close()
    if self._conn is not None:
      self._conn.close()
    self._c = None
    self._conn = None

  def __del__(self):
    self.close()
