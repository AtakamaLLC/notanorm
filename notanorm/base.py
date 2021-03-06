"""Handle common database functionality.

NOTE: Make sure to close the db handle when you are done.
"""

import time
import threading
import logging

from abc import ABC, abstractmethod

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


class DbRow(dict):
    """Default row factory: elements accessible via string or int keys."""
    __vals = None

    def __init__(self, dct={}):             # pylint: disable=dangerous-default-value
        self.__dict__ = dct.copy()
        super().__init__({k.lower(): v for k, v in dct.items()})

    def __repr__(self):
        return "DbRow(" + super().__repr__() + ")"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __getattr__(self, key):
        return self[key.lower()]

    def __setattr__(self, key, val):
        if key in self:
            self[key] = val
            self.__dict__[key] = val
        else:
            super().__setattr__(key, val)

    def __getitem__(self, key):
        if type(key) is int:                # pylint: disable=unidiomatic-typecheck
            return self._aslist()[key]
        return super().__getitem__(key)

    def _asdict(self):
        return dict(self)

    def _aslist(self):
        if not self.__vals:
            self.__vals = list(self.__dict__.values())
        return self.__vals


class DbBase(ABC):                          # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Abstract base class for database connections."""
    placeholder = '?'
    max_reconnect_attempts = 5
    reconnect_backoff_start = 0.1  # seconds
    reconnect_backoff_factor = 2
    debug_sql = None
    debug_args = None

    def __init__(self, *args, **kws):
        self.__conn_p = None
        self.__conn_args = args
        self.__conn_kws = kws
        self.__lock = threading.RLock()
        self.__primary_cache = {}
        self.__classes = {}

        self._conn()

    # must implement
    @abstractmethod
    def _get_primary(self, table):
        pass

    @abstractmethod
    def _connect(self):
        pass

    @staticmethod
    def translate_error(exp):
        return exp

    # can override
    def _cursor(self, conn):
        return conn.cursor()

    def _conn(self):
        if not self.__conn_p:
            self.__conn_p = self._connect(*self.__conn_args, **self.__conn_kws)
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
        with self.__lock:
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
        with self.__lock:
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
        "Class will be used instead of Row object.  Must accept kw args for every table col"
        self.__classes[table] = cls

    def unregister_class(self, table):
        "Class will no longer be used"
        self.__classes.pop(table, None)

    def query(self, sql, *args, factory=None):
        "Run sql, pass args, optionally use factory for each row (cols passed as kwargs)"
        # for debugging....
        self.debug_sql = sql + ";"
        self.debug_args = args
        log.debug("SQL: %s, ARGS%s", sql, str(args))

        fetch = None

        ret = self.RetList()

        with self.__lock:
            try:
                fetch = self.execute(sql, tuple(args))
                rows = fetch.fetchall()
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
        sql += ",".join([self.placeholder for k in vals.keys()])
        sql += ")"

        return self.query(sql, *vals.values())

    def quote_key(self, key):
        return '"' + key + '"'

    def quote_keys(self, key):
        return ".".join([self.quote_key(k) for k in key.split(".")])

    def _where(self, table, where):
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

        where, vals = self._where(table, where)
        sql += where

        return self.query(sql, *vals, factory=factory)

    def count(self, table, where=None, **kws):
        if where and kws:
            raise ValueError("Dict where cannot be mixed with kwargs")

        if not where:
            where = kws

        sql = "select count(*) as k from " + table
        where, vals = self._where(table, where)
        sql += where
        return self.query(sql, *vals)[0]["k"]

    def delete(self, table, **where):
        sql = "delete "
        sql += " from " + table

        where, vals = self._where(table, where)
        if not where:
            raise ValueError("Use delete_all to delete all rows from a table")

        sql += where

        return self.query(sql, vals)

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
        has = self.select(table, **where)
        if not has:
            # restore value dict
            vals.update(where)
            self.insert(table, **vals)
        else:
            self.update(table, where, **vals)

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
