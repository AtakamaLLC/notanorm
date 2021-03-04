"""Handle common database functionality.

NOTE: Make sure to close the db handle when you are done.
"""

import time
import threading
import logging
from collections import defaultdict
from abc import ABC, abstractmethod

from .errors import OperationalError
from .model import DbModel, DbTable
from . import errors as err


log = logging.getLogger(__name__)


def is_list(obj):
    """Determine if object is list-like, as opposed to string or bytes-like."""
    return isinstance(obj, (list, set, tuple))


def del_all(mapping, to_remove):
    """Remove list of elements from mapping."""
    for key in to_remove:
        del mapping[key]


class CIKey(str):
    def __eq__(self, other):
        return other.lower() == self.lower()

    def __hash__(self):
        return hash(self.lower())

class DbRow(dict):
    """Default row factory.

    Elements accessible via string or int key getters.
    Elements accessible as attributes
    Case insensitive access
    Case preserving setters

    For access to the case-preserved keys use:
        row.items()
        row.keys()
        or row._asdict()

    For case-insensitive access for "key" use:
        row.key
        row["key"]

    Access to __dict__ is deprecated (and slow, it makes a case-preserved copy)
    """
    __vals = None

    # noinspection PyDefaultArgument
    def __init__(self, dct={}):             # pylint: disable=dangerous-default-value
        super().__init__()
        for k, v in dct.items():
            super().__setitem__(CIKey(k), v)

    @property
    def __dict__(self):
        return self._asdict()

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __getitem__(self, key):
        if type(key) is int:                # pylint: disable=unidiomatic-typecheck
            return self._aslist()[key]
        return super().__getitem__(CIKey(key))

    def __setitem__(self, key, val):
        return super().__setitem__(CIKey(key), val)

    def __contains__(self, key):
        return super().__contains__(CIKey(key))

    def _asdict(self):
        """Warning: this is inefficient.   But also it's not needed.  Just access the container itself."""
        return {k: v for k, v in self.__items()}

    def __items(self):
        return ((str(k), v) for k, v in super().items() if k[0:2] != '__')

    def items(self):
        return list(self.__items())

    def keys(self):
        return list(k for k, _v in self.__items())

    def values(self):
        return list(v for _k, v in self.__items())

    def _aslist(self):
        if not self.__vals:
            self["__vals"] = self.values()
        return self["__vals"]


# noinspection PyProtectedMember
class DbTxGuard:
    def __init__(self, db: "DbBase"):
        self.db = db
        self.lock = self.db.r_lock

    def __enter__(self):
        if not self.lock.acquire(timeout=self.db.timeout):
            # raise the same sort of error
            raise OperationalError("database table is locked")
        self.db._commit(self.db._conn())
        self.db._begin(self.db._conn())
        self.db._transaction += 1
        return self.db

    def __exit__(self, exc_type, value, _traceback):
        self.db._transaction -= 1
        if not self.db._transaction:  # pylint: disable=protected-access
            if exc_type:
                self.db._rollback(self.db._conn())
            else:
                self.db._commit(self.db._conn())
        self.lock.release()


# noinspection PyMethodMayBeStatic
class DbBase(ABC):                          # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Abstract base class for database connections."""
    placeholder = '?'
    max_reconnect_attempts = 5
    reconnect_backoff_start = 0.1  # seconds
    reconnect_backoff_factor = 2
    debug_sql = None
    debug_args = None
    use_pooled_locks = False
    __lock_pool = defaultdict(threading.RLock)

    @property
    def timeout(self):
        # total timeout for connections == geometric sum
        return self.reconnect_backoff_start * ((1 - self.reconnect_backoff_factor ** self.max_reconnect_attempts) / (
                    1 - self.reconnect_backoff_factor))

    def _lock_key(self, *args, **kws):
        raise NotImplementedError("define _lock_key in your subclass if use_pooled_locks is enabled")

    def __init__(self, *args, **kws):
        assert self.reconnect_backoff_factor > 1
        self.__conn_p = None
        self._conn_args = args
        self._conn_kws = kws
        if self.use_pooled_locks:
            self.r_lock = self.__lock_pool[self._lock_key(*args, **kws)]
        else:
            self.r_lock = threading.RLock()
        self.__primary_cache = {}
        self.__classes = {}
        self._transaction = 0
        self._conn()

    def transaction(self):
        return DbTxGuard(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        self.close()

    def _begin(self, conn):
        """Override if needed, just calls begin() on the connection."""
        conn.begin()

    def _commit(self, conn):
        """Override if needed, just calls commit() on the connection."""
        conn.commit()

    def _rollback(self, conn):
        """Override if needed, just calls rollback() on the connection."""
        conn.rollback()

    # must implement
    @abstractmethod
    def _get_primary(self, table):
        pass

    @abstractmethod
    def _connect(self, *args, **kws):
        pass

    @staticmethod
    def translate_error(exp):
        return exp

    # can override
    def _cursor(self, conn):
        return conn.cursor()

    @property
    def connection_args(self):
        return self._conn_args, self._conn_kws

    def _conn(self):
        if not self.__conn_p:
            self.__conn_p = self._connect(*self._conn_args, **self._conn_kws)
            if not self.__conn_p:
                raise ValueError("No connection returned by _connect for %s" % type(self))
        return self.__conn_p

    def model(self) -> DbModel:
        raise NotImplementedError("Generic models not supported")

    def create_model(self, model: DbModel):
        for name, schema in model.items():
            self.create_table(name, schema)

    def create_table(self, name, schema: DbTable):
        raise NotImplementedError("Generic models not supported")

    def execute(self, sql, parameters=()):
        with self.r_lock:
            backoff = self.reconnect_backoff_start
            for tries in range(self.max_reconnect_attempts):
                try:
                    cursor = self._cursor(self._conn())
                    cursor.execute(sql, parameters)
                    break
                except Exception as exp:                  # pylint: disable=broad-except
                    was = exp
                    exp = self.translate_error(exp)
                    log.debug("exception %s -> %s", repr(was), repr(exp))
                    if isinstance(exp, err.DbConnectionError):
                        self.__conn_p = None
                        if tries == self.max_reconnect_attempts - 1:
                            raise
                        time.sleep(backoff)
                        backoff *= self.reconnect_backoff_factor
                    else:
                        raise exp
        return cursor

    def close(self):
        with self.r_lock:
            if self.__conn_p:
                self.__conn_p.close()
                self.__conn_p = None

    # probably don't override these

    def __is_primary(self, table, field):
        if table not in self.__primary_cache:
            self.__primary_cache[table] = self._get_primary(table)
        return field in self.__primary_cache[table]

    class RetList(list):
        rowcount = None
        lastrowid = None

    def register_class(self, table, cls):
        """Class will be used instead of Row object.  Must accept kw args for every table col"""
        self.__classes[table] = cls

    def unregister_class(self, table):
        """Class will no longer be used"""
        self.__classes.pop(table, None)

    def query(self, sql, *args, factory=None):
        """Run sql, pass args, optionally use factory for each row (cols passed as kwargs)"""
        # for debugging....
        self.debug_sql = sql + ";"
        self.debug_args = args
        log.debug("SQL: %s, ARGS%s", sql, str(args))

        fetch = None

        ret = self.RetList()

        with self.r_lock:
            try:
                fetch = self.execute(sql, tuple(args))
                rows = fetch.fetchall() if fetch else []
            except Exception as ex:
                debug_str = "SQL: " + sql + ", ARGS" + str(args)
                log.debug("sql error %s", repr(ex))
                raise type(ex)(str(ex) + ", " + debug_str) from ex
            finally:
                if fetch:
                    fetch.close()

        for row in rows:
            if factory:
                row = factory(**row)
            else:
                if type(row) is not DbRow:
                    row = DbRow(row)
            ret.append(row)

        ret.rowcount = fetch.rowcount
        ret.lastrowid = fetch.lastrowid

        return ret

    def insert(self, table, ins=None, **vals):
        if ins:
            vals.update(ins)

        sql = "insert into " + table

        sql += '('
        sql += ','.join([self.quote_keys(k) for k in vals.keys()])
        sql += ')'

        sql += " values ("
        sql += ",".join([self.placeholder for _ in vals.keys()])
        sql += ")"

        return self.query(sql, *vals.values())

    def quote_key(self, key):
        return '"' + key + '"'

    def quote_keys(self, key):
        return ".".join([self.quote_key(k) for k in key.split(".")])

    def _where(self, where):
        if not where:
            return "", ()

        none_keys = [key for key, val in where.items() if val is None]
        listKeys = [(key, val) for key, val in where.items() if is_list(val)]

        del_all(where, none_keys)
        del_all(where, (k[0] for k in listKeys))

        sql = " and ".join([self.quote_keys(key) + "=" + self.placeholder for key in where.keys()])

        if none_keys:
            if sql:
                sql += " and "
            sql += " and ".join([self.quote_keys(key) + " is NULL" for key in none_keys])

        vals = where.values()
        if listKeys:
            vals = list(vals)
            for key, lst in listKeys:
                placeholders = ",".join([self.placeholder] * len(lst))
                if sql:
                    sql += " and "
                sql += self.quote_keys(key) + " in (" + placeholders + ")"
                for val in lst:
                    vals.append(val)
        return " where " + sql, vals

    def select(self, table, fields=None, dict_where=None, **where):
        """ Select from table (or join) using fields (or *) and where (vals can be list or none).
            __class keyword optionally replaces Row obj.
        """
        sql = "select "

        no_from = False
        if table[0:len(sql)].lower() == sql and "from" in table.lower():
            sql = ""
            no_from = True

        if (isinstance(fields, dict) or dict_where) and where:
            raise ValueError("Dict where cannot be mixed with kwargs")

        if isinstance(fields, dict):
            dict_where = fields
            fields = None

        if dict_where:
            where = dict_where

        if fields and no_from:
            raise ValueError("Specify field list or select statement, not both")

        factory = None
        if not no_from:
            factory = where.pop("__class", self.__classes.get(table))

            if not fields:
                sql += "*"
            else:
                sql += ",".join(fields)

            sql += " from " + table

        where, vals = self._where(where)
        sql += where

        return self.query(sql, *vals, factory=factory)

    def count(self, table, where=None, **kws):
        if where and kws:
            raise ValueError("Dict where cannot be mixed with kwargs")

        if not where:
            where = kws

        sql = "select count(*) as k from " + table
        where, vals = self._where(where)
        sql += where
        return self.query(sql, *vals)[0]["k"]

    def delete(self, table, **where):
        sql = "delete "
        sql += " from " + table

        where, vals = self._where(where)
        if not where:
            raise ValueError("Use delete_all to delete all rows from a table")

        sql += where

        return self.query(sql, *vals)

    def delete_all(self, table):
        sql = "delete "
        sql += " from " + table
        return self.query(sql)

    def infer_where(self, table, where, vals):
        if not where:
            where = {}

            for key in vals.keys():
                if self.__is_primary(table, key):
                    where[key] = vals[key]

            for key in where:
                del vals[key]

            if not where:
                log.debug("PRIMARY CACHE: %s", self.__primary_cache)
                raise Exception("Unable to determine update key for table %s" % table)

        return where

    def update(self, table, where=None, upd=None, **vals):
        where = self.infer_where(table, where, vals)

        if upd:
            vals.update(upd)

        sql = "update " + table + " set "

        none_keys = [key for key, val in where.items() if val is None]
        del_all(where, none_keys)

        sql += ", ".join([self.quote_keys(key) + "=" + self.placeholder for key in vals])
        sql += " where "
        sql += " and ".join([self.quote_keys(key) + "=" + self.placeholder for key in where])
        if where and none_keys:
            sql += " and "
        sql += " and ".join([self.quote_keys(key) + " is NULL" for key in none_keys])

        vals = list(vals.values()) + list(where.values())

        return self.query(sql, *vals)

    def update_all(self, table, **vals):
        sql = "update " + table + " set "
        sql += ", ".join([self.quote_keys(key) + "=" + self.placeholder for key in vals])
        return self.query(sql, *vals.values())

    def upsert_all(self, table, **vals):
        has = self.select(table)
        if not has:
            self.insert(table, **vals)
        else:
            self.update_all(table, **vals)

    def upsert(self, table, where=None, **vals):
        # get where dict from values and primary key
        where = self.infer_where(table, where, vals)

        # find existing row
        with self.transaction():
            has = self.select(table, **where)
            if not has:
                # restore value dict
                vals.update(where)
                return self.insert(table, **vals)
            else:
                return self.update(table, where, **vals)

    def upsert_non_null(self, table, where=None, **vals):
        remove = []
        for key, val in vals.items():
            if val is None:
                remove.append(key)

        for key in remove:
            del vals[key]

        self.upsert(table, where, **vals)

    def select_one(self, table, fields=None, **where):
        ret = self.select(table, fields, **where)
        assert len(ret) <= 1
        if ret:
            return ret[0]
        return None
