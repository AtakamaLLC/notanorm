"""Handle common database functionality.

NOTE: Make sure to close the db handle when you are done.
"""
import contextlib
import time
import threading
import logging
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any, Tuple, Generator

from .errors import OperationalError, MoreThanOneError, DbClosedError
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


class FakeCursor:
    rowcount = 0
    lastrowid = 0

    @staticmethod
    def fetchall():
        return []

    @staticmethod
    def close():
        pass


class CIKey(str):
    def __eq__(self, other):
        return other.lower() == self.lower()

    def __hash__(self):
        return hash(self.lower())


class Op:
    def __init__(self, op, val):
        self.op = op
        self.val = val


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
    def __init__(self, dct={}):  # pylint: disable=dangerous-default-value
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
        if type(key) is int:  # pylint: disable=unidiomatic-typecheck
            return self._aslist()[key]
        return super().__getitem__(CIKey(key))

    def __setitem__(self, key, val):
        return super().__setitem__(CIKey(key), val)

    def __getstate__(self):
        return self._asdict()

    def __setstate__(self, state):
        for k, v in state.items():
            super().__setitem__(CIKey(k), v)

    def __contains__(self, key):
        return super().__contains__(CIKey(key))

    def _asdict(self):
        """Warning: this is inefficient.   But also it's not needed.  Just access the container itself."""
        return {k: v for k, v in self.__items()}

    def __items(self):
        return ((str(k), v) for k, v in super().items() if k[0:2] != "__")

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
        if not self.db._transaction:
            self.db._begin(self.db._conn_p)
        self.db._transaction += 1
        return self.db

    def __exit__(self, exc_type, value, _traceback):
        self.db._transaction -= 1
        if not self.db._transaction:  # pylint: disable=protected-access
            # if the connection dropped... we can't roll back or commit
            if self.db._conn_p:
                if exc_type:
                    self.db._rollback(self.db._conn_p)
                else:
                    self.db._commit(self.db._conn_p)
        self.lock.release()


# noinspection PyMethodMayBeStatic
class DbBase(
    ABC
):  # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Abstract base class for database connections."""

    __known_drivers = {}
    uri_name = None
    uri_conn_func = None
    placeholder = "?"
    default_values = "default values"
    max_reconnect_attempts = 5
    reconnect_backoff_start = 0.1  # seconds
    reconnect_backoff_factor = 2
    debug_sql = None
    debug_args = None
    r_lock = None
    use_pooled_locks = False
    use_collation_locks = False
    __lock_pool = defaultdict(threading.RLock)

    @property
    def timeout(self):
        # total timeout for connections == geometric sum
        return self.reconnect_backoff_start * (
            (1 - self.reconnect_backoff_factor**self.max_reconnect_attempts)
            / (1 - self.reconnect_backoff_factor)
        )

    def _lock_key(self, *args, **kws):
        raise RuntimeError(
            "define _lock_key in your subclass if use_pooled_locks is enabled"
        )

    def __init__(self, *args, **kws):
        self.__capture = None
        self.__capture_exec = None
        self.__capture_stmts = []
        assert self.reconnect_backoff_factor > 1
        self.__closed = False
        self._conn_p = None
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

    def __init_subclass__(cls: "DbBase", **kwargs):
        if cls.uri_name:
            cls.__known_drivers[cls.uri_name] = cls
        cls.__known_drivers[cls.__name__] = cls

    @classmethod
    def get_driver_by_name(cls, name) -> Type["DbBase"]:
        return cls.__known_drivers.get(name)

    @classmethod
    def uri_adjust(cls, args: List, kws: Dict):
        """Modify the url-parsed and keywords before they are passed to the driver.

        For example, a keyword: `?port=50` might need conversion to an integer.

        Or a parameter might need to be positional.

        Or positional args (url path parameters) might need conversion to keywords.

        Ideally, adjustment of the URI parsed args and keywords should be minimal,
        so the user can rely on the documentation of the underlying database
        connection.
        """

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
        if not self._conn_p:
            if self.__closed:
                raise DbClosedError
            self._conn_p = self._connect(*self._conn_args, **self._conn_kws)
            if not self._conn_p:
                raise ValueError(
                    "No connection returned by _connect for %s" % type(self)
                )
        return self._conn_p

    @property
    def closed(self) -> bool:
        return self.__closed

    def model(self) -> DbModel:
        raise RuntimeError("Generic models not supported")

    def ddl_stmts_from_model(self, model: DbModel):
        with self.capture_sql() as cap:
            self.create_model(model)
        return cap

    def ddl_from_model(self, model: DbModel):
        res = ""
        for sql, params in self.ddl_stmts_from_model(model):
            assert not params, "Ddl parameter expansion is not supported"
            res += sql + ";\n"
        return res

    @staticmethod
    def simplify_model(model):
        """Override if you want to allow model comparisons.

        For example, if you have a model that defines a fixed-width char, but your db ignores
        fixed-with chars, you can remove or normalize the fixed-width flag in the model.
        """
        return model

    @contextlib.contextmanager
    def capture_sql(self, execute=False) -> Generator[List[Tuple[str, Tuple[Any, ...]]], None, None]:
        self.__capture = True
        self.__capture_exec = execute
        self.__capture_stmts = []
        try:
            yield self.__capture_stmts
        finally:
            self.__capture = False

    def create_model(self, model: DbModel):
        for name, schema in model.items():
            self.create_table(name, schema)

    def create_table(self, name, schema: DbTable):
        raise RuntimeError("Generic models not supported")

    def executescript(self, sql):
        self.execute(sql, _script=True)

    @staticmethod
    def _executemany(cursor, sql):
        return cursor.execute(sql)

    def execute(self, sql, parameters=(), _script=False):
        with self.r_lock:
            if self.__capture:
                self.__capture_stmts.append((sql, parameters))
                if not self.__capture_exec:
                    return FakeCursor()

            backoff = self.reconnect_backoff_start
            for tries in range(self.max_reconnect_attempts):
                cursor = None

                try:
                    cursor = self._cursor(self._conn())
                    if _script:
                        assert not parameters, "Script isn't compatible with parameters"
                        self._executemany(cursor, sql)
                    else:
                        cursor.execute(sql, parameters)
                    break
                except Exception as exp:  # pylint: disable=broad-except
                    if cursor:
                        # cursor will be automatically closed on del, but better to do it explicitly
                        # Some tools, like pytest, will capture locals, which may keep the cursor
                        # alive indefinitely
                        try:
                            cursor.close()
                        except Exception as close_exc:
                            log.debug("Failed to close temp cursor: %r", close_exc)

                    was = exp
                    exp = self.translate_error(exp)
                    log.debug("exception %s -> %s", repr(was), repr(exp))
                    if isinstance(exp, err.DbConnectionError):
                        self._conn_p = None
                        if tries == self.max_reconnect_attempts - 1:
                            raise
                        time.sleep(backoff)
                        backoff *= self.reconnect_backoff_factor
                    else:
                        raise exp
        return cursor

    def close(self):
        if self.r_lock:
            with self.r_lock:
                if self._conn_p:
                    self._conn_p.close()
                    self.__closed = True
                    self._conn_p = None

    # probably don't override these

    def __is_primary(self, table, field):
        return field in self.primary_fields(table)

    def primary_fields(self, table):
        if table not in self.__primary_cache:
            self.__primary_cache[table] = self._get_primary(table)
        return self.__primary_cache[table]

    class RetList(list):
        rowcount = None
        lastrowid = None

    def register_class(self, table, cls):
        """Class will be used instead of Row object.  Must accept kw args for every table col"""
        self.__classes[table] = cls

    def unregister_class(self, table):
        """Class will no longer be used"""
        self.__classes.pop(table, None)

    def __debug_sql(self, sql, args):
        self.debug_sql = sql + ";"
        self.debug_args = args
        log.debug("SQL: %s, ARGS%s", sql, str(args))

    def query_gen(self, sql: str, *args, factory=None):
        """Same as query, but returns a generator."""
        self.__debug_sql(sql, args)

        fetch = None

        with self.r_lock:
            try:
                fetch = self.execute(sql, tuple(args))
            except Exception as ex:
                log.debug("sql query %s, error %s", sql, repr(ex))
                raise

        try:
            while True:
                if self.use_collation_locks:
                    with self.r_lock:
                        row = fetch.fetchone()
                else:
                    row = fetch.fetchone()
                if row is None:
                    break
                if factory:
                    row = factory(**row)
                else:
                    if type(row) is not DbRow:
                        row = DbRow(row)
                yield row
            fetch = None
        finally:
            if fetch:
                fetch.close()

    def query(self, sql: str, *args, factory=None):
        """Run sql, pass args, optionally use factory for each row (cols passed as kwargs)"""
        self.__debug_sql(sql, args)

        fetch = None

        ret = self.RetList()

        with self.r_lock:
            try:
                done = False
                fetch = self.execute(sql, tuple(args))
                rows = fetch.fetchall() if fetch else []
                done = True
            except Exception as ex:
                log.debug("sql %s, error %s", sql, repr(ex))
                raise
            finally:
                if fetch and not done:
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

    def _insql(self, table: str, ins=None, **vals):
        if ins:
            vals.update(ins)

        sql = "insert into " + self.quote_key(table)

        if vals:
            sql += "("
            sql += ",".join([self.quote_keys(k) for k in vals.keys()])
            sql += ")"

            sql += " values ("
            sql += ",".join([self.placeholder for _ in vals.keys()])
            sql += ")"
        else:
            sql += " " + self.default_values

        return sql, vals.values()

    def insert(self, table: str, ins=None, **vals):
        sql, vals = self._insql(table, ins, **vals)
        return self.query(sql, *vals)

    @classmethod
    def quote_key(cls, key):
        return '"' + key + '"'

    def quote_keys(self, key):
        return ".".join([self.quote_key(k) for k in key.split(".")])

    @staticmethod
    def _op_from_val(val):
        if isinstance(val, Op):
            return val
        return Op("=", val)

    def _where(self, where):
        if not where:
            return "", ()

        none_keys = [key for key, val in where.items() if val is None]
        list_keys = [(key, val) for key, val in where.items() if is_list(val)]

        del_all(where, none_keys)
        del_all(where, (k[0] for k in list_keys))

        sql = " and ".join(
            [
                self.quote_keys(key) + self._op_from_val(val).op + self.placeholder
                for key, val in where.items()
            ]
        )

        if none_keys:
            if sql:
                sql += " and "
            sql += " and ".join(
                [self.quote_keys(key) + " is NULL" for key in none_keys]
            )

        vals = [self._op_from_val(val).val for val in where.values()]
        if list_keys:
            vals = list(vals)
            for key, lst in list_keys:
                placeholders = ",".join([self.placeholder] * len(lst))
                if sql:
                    sql += " and "
                sql += self.quote_keys(key) + " in (" + placeholders + ")"
                for val in lst:
                    vals.append(val)
        return " where " + sql, vals

    def __select_to_query(self, table, *, fields, dict_where, order_by, **where):
        sql = "select "

        no_from = False
        if table[0 : len(sql)].lower() == sql and "from" in table.lower():
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

            if " join " not in table.lower():
                sql += " from " + self.quote_key(table)
            else:
                sql += " from " + table

        where, vals = self._where(where)
        sql += where

        if order_by:
            if isinstance(order_by, str):
                order_by = [order_by]
            order_by_fd = ",".join(order_by)
            # todo: limit order_by more strictly
            assert ";" not in order_by_fd
            sql += " order by " + order_by_fd

        return sql, vals, factory

    def select(self, table, fields=None, dict_where=None, order_by=None, **where):
        """Select from table (or join) using fields (or *) and where (vals can be list or none).
        __class keyword optionally replaces Row obj.
        """
        sql, vals, factory = self.__select_to_query(
            table, fields=fields, dict_where=dict_where, order_by=order_by, **where
        )
        return self.query(sql, *vals, factory=factory)

    def select_gen(self, table, fields=None, dict_where=None, order_by=None, **where):
        """Same as select, but returns a generator."""
        sql, vals, factory = self.__select_to_query(
            table, fields=fields, dict_where=dict_where, order_by=order_by, **where
        )
        return self.query_gen(sql, *vals, factory=factory)

    def count(self, table, where=None, **kws):
        if where and kws:
            raise ValueError("Dict where cannot be mixed with kwargs")

        if not where:
            where = kws

        sql = "select count(*) as k from " + self.quote_key(table)
        where, vals = self._where(where)
        sql += where
        return self.query(sql, *vals)[0]["k"]

    def delete(self, table, **where):
        """Delete all rows in a table that match the supplied value(s).


        For example:  db.delete("table_name", column_name="matching value")
        """
        sql = "delete "
        sql += " from " + self.quote_key(table)

        where, vals = self._where(where)
        if not where:
            raise ValueError("Use delete_all to delete all rows from a table")

        sql += where

        return self.query(sql, *vals)

    def delete_all(self, table):
        """Delete all rows in a table."""
        sql = "delete "
        sql += " from " + self.quote_key(table)
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

    def _setsql(self, table, where, upd, vals):
        none_keys = [key for key, val in where.items() if val is None]
        del_all(where, none_keys)

        sql = ", ".join([self.quote_keys(key) + "=" + self.placeholder for key in vals])
        sql += " where "
        sql += " and ".join(
            [self.quote_keys(key) + "=" + self.placeholder for key in where]
        )
        if where and none_keys:
            sql += " and "
        sql += " and ".join([self.quote_keys(key) + " is NULL" for key in none_keys])

        vals = list(vals.values()) + list(where.values())

        return sql, vals

    def update(self, table, where=None, upd=None, **vals):
        if upd:
            vals.update(upd)
        where = self.infer_where(table, where, vals)
        if not vals:
            return
        set_sql, vals = self._setsql(table, where, upd, vals)
        sql = "update " + self.quote_key(table) + " set " + set_sql
        return self.query(sql, *vals)

    def update_all(self, table, **vals):
        """Update all rows in a table to the same values."""
        sql = "update " + self.quote_key(table) + " set "
        sql += ", ".join(
            [self.quote_keys(key) + "=" + self.placeholder for key in vals]
        )
        return self.query(sql, *vals.values())

    def upsert_all(self, table, **vals):
        """Update all rows in a table to the same values, or insert if not present."""
        with self.transaction():
            has = self.select(table)
            if not has:
                self.insert(table, **vals)
            else:
                self.update_all(table, **vals)

    def upsert(self, table, where=None, _insert_only=None, **vals):
        """Select a row, and if present, update it, otherwise insert."""

        # insert only fields
        _insert_only = _insert_only or {}

        if hasattr(self, "_upsert_sql"):
            # _upsert_sql is a function that takes two sql statements and joins them into one

            # insert statement + values
            tmp = vals.copy()
            if where is not None:
                tmp.update(where)
            tmp.update(_insert_only)
            ins_sql, in_vals = self._insql(table, **tmp)

            # discard vals, remove where clause stuff
            self.infer_where(table, where, vals)

            # set non-primary key values
            set_sql = ", ".join(
                [self.quote_keys(key) + "=" + self.placeholder for key in vals]
            )
            sql, vals = self._upsert_sql(
                table, ins_sql, in_vals, set_sql, vals.values()
            )

            return self.query(sql, *vals)

        where = self.infer_where(table, where, vals)

        # find existing row
        with self.transaction():
            has = self.select(table, **where)
            if not has:
                # restore value dict
                vals.update(where)
                vals.update(_insert_only)
                return self.insert(table, **vals)
            else:
                return self.update(table, where, **vals)

    def upsert_non_null(self, table, where=None, **vals):
        """Same as upsert, but values with None in them are ignored."""
        remove = []
        for key, val in vals.items():
            if val is None:
                remove.append(key)

        for key in remove:
            del vals[key]

        self.upsert(table, where, **vals)

    def select_one(self, table, fields=None, **where):
        """Select one row.

        Returns None if not found.

        Raises MoreThanOneError if there is more than one result.
        """
        ret = self.select(table, fields, **where)
        if len(ret) > 1:
            raise MoreThanOneError
        if ret:
            return ret[0]
        return None
