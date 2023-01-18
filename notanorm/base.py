"""Handle common database functionality.

NOTE: Make sure to close the db handle when you are done.
"""
import contextlib
import time
import random
import threading
import logging
from dataclasses import dataclass
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Type,
    Any,
    Tuple,
    Generator,
    TypeVar,
    Optional,
    Callable,
    Union,
)

from .errors import (
    OperationalError,
    MoreThanOneError,
    DbClosedError,
    UnknownPrimaryError,
    UnknownColumnError,
)
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


def parse_bool(field_name: str, value: str) -> bool:
    value_lower = value.lower()

    truthy = value_lower == "true"
    falsey = value_lower == "false"

    if not (truthy or falsey):
        raise ValueError(f"{field_name} must be a boolean, not {value}")

    return truthy


@dataclass
class ReconnectionArgs:
    failure_callback: Optional[Callable[[], None]] = None
    max_reconnect_attempts: int = 8
    reconnect_backoff_start: float = 0.1  # seconds
    reconnect_backoff_factor: float = 2
    jitter: bool = True


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
    def __init__(self, op: str, val: Any):
        self.op = op
        self.val = val

    def __repr__(self) -> str:
        return f"Op({self.op!r}, {self.val!r})"

    def __eq__(self, o: Any) -> bool:
        if not isinstance(o, Op):
            return False

        return self.op == o.op and self.val == o.val


class BaseQ(ABC):
    _next = 0
    fields = []
    field_map = {}

    def __init__(self, db):
        self.db = db

    @classmethod
    def next_num(cls):
        cls._next = cls._next + 1 % 100000
        return cls._next

    @classmethod
    def reset_num(cls) -> None:
        cls._next = 0

    @abstractmethod
    def resolve_field(self, field: str):
        ...

    def field_sql(self):
        if not self.fields:
            return "*"

        if not self.field_map:
            return ",".join(self.db.quote_key(fd) for fd in self.fields)

        sql = ""
        for name, qual_name in self.field_map.items():
            if self.db.auto_quote(qual_name) != self.db.quote_key(name):
                sql += (
                    ", "
                    + self.db.auto_quote(qual_name)
                    + " as "
                    + self.db.quote_key(name)
                )
            else:
                sql += ", " + self.db.auto_quote(qual_name)
        return sql.lstrip(",")


BaseQType = TypeVar("BaseQType", bound=BaseQ)


class SubQ(BaseQ):
    def __init__(
        self,
        db: "DbBase",
        table: Union[str, "SubQ"],
        sql: str,
        vals: List[Any] = (),
        alias=None,
        fields=None,
    ):
        alias = alias or (table if table is str else "subq") + "_" + str(
            self.next_num()
        )

        super().__init__(db)

        self.sql = sql
        self.vals = vals
        self.alias = alias
        self.fields = fields or []

        if type(table) is SubQ:
            self.table = table.table
        else:
            self.table = table

    def resolve_field(self, field: str):
        if self.fields and field not in self.fields:
            raise UnknownColumnError(f"{field} not found in {self.fields}")
        return self.alias + "." + field


class JoinQ(BaseQ):
    _next = 0

    def __init__(
        self,
        db: "T",
        join_type: str,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        on: dict,
        field_map: dict,
    ):
        super().__init__(db)

        self.join_type = join_type
        self.tab1 = tab1
        self.tab2 = tab2
        self.on = on
        self.vals = []
        self.__sql = ""
        self.__field_map = field_map
        self.__fields = []

    @property
    def sql(self) -> str:
        if not self.__sql:
            self.resolve()
        return self.__sql

    @property
    def fields(self) -> list:
        if not self.__sql:
            self.resolve()
        return self.__fields

    @property
    def field_map(self) -> dict:
        if not self.__sql:
            self.resolve()
        return self.__field_map

    def flat_tabs(self):
        for tab in [self.tab1, self.tab2]:
            if tab is self.tab1:
                join = None
            else:
                join = self

            if type(tab) is JoinQ:
                for tab, subj in tab.flat_tabs():
                    yield tab, subj or join
            else:
                yield tab, join

    def resolve_field(self, field: str):
        if field in self.db.get_subq_col_names(self.tab1):
            return self.tab1 + "." + field
        if field in self.db.get_subq_col_names(self.tab2):
            return self.tab2 + "." + field
        raise UnknownColumnError(f"{field} not found in {self.tab1} or {self.tab2}")

    @staticmethod
    def tab_to_sql(tab) -> [str, list]:
        if type(tab) is JoinQ:
            return tab.sql, []
        if type(tab) is SubQ:
            return "(" + tab.sql + ") as " + tab.alias, tab.vals
        return tab, []

    def resolve(self, ambig_cols=None):
        flat_tabs = list(self.flat_tabs())

        if ambig_cols is None:
            all_cols = defaultdict(lambda: 0)

            for tab, _ in flat_tabs:
                for col in self.db.get_subq_col_names(tab):
                    all_cols[col] += 1
            ambig_cols = {k for k, v in all_cols.items() if v > 1}

        sql = ""
        vals = []
        for tab, join in flat_tabs:
            if type(tab) is JoinQ:
                tab.resolve(ambig_cols)
            tab_sql, tab_vals = self.tab_to_sql(tab)
            if not sql:
                sql = tab_sql
            else:
                sql += " " + join.join_type + " join " + tab_sql
                on_sql = join.get_on_sql()
                sql += " on " + on_sql

            vals += tab_vals

        self.vals = vals

        self.__sql = sql

        if self.field_map:
            self.__fields = list(self.field_map.keys())
            return

        fields = []
        field_map = {}

        wild = not ambig_cols
        if not wild:
            for tab, _ in flat_tabs:
                cols = self.db.get_subq_col_names(tab)
                for col in cols:
                    if col in ambig_cols:
                        alias = (tab if type(tab) is str else tab.alias) + "." + col
                        if col in self.on and type(tab) is str:
                            field_map[col] = alias
                            field = col
                        else:
                            field_map[alias] = alias
                            field = alias
                    else:
                        field_map[col] = col
                        field = col
                    fields.append(field)
        self.__fields = fields
        self.__field_map = field_map

    def get_on_sql(self):
        fd1 = (
            self.db.get_subq_col_names(self.tab1)
            if type(self.tab1) in (SubQ, JoinQ)
            else []
        )
        fd2 = (
            self.db.get_subq_col_names(self.tab2)
            if type(self.tab2) in (SubQ, JoinQ)
            else []
        )

        on_sql = ""
        for k, v in self.on.items():
            on_sql += " and " if on_sql else ""
            if "." not in k:
                if type(self.tab1) is str:
                    k = self.tab1 + "." + k
                else:
                    k = self.tab1.resolve_field(k)

            if "." not in v:
                if type(self.tab2) is str:
                    v = self.tab2 + "." + v
                else:
                    v = self.tab2.resolve_field(v)

            if k in fd1 and type(self.tab1) is SubQ:
                k = self.db.quote_key(k)
            else:
                k = self.db.quote_keys(k)

            if v in fd2 and type(self.tab2) is SubQ:
                v = self.db.quote_key(v)
            else:
                v = self.db.quote_keys(v)

            on_sql += k + "=" + v
        return on_sql


class AlreadyAliased(str):
    pass


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
        try:
            return self[key]
        except KeyError:
            # allow user to refer to table.field as table__field
            return self[key.replace("__", ".")]

    def __setattr__(self, key, val):
        alt_key = key.replace("__", ".")
        if alt_key in self:
            key = alt_key
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


T = TypeVar("T", bound="DbBase")


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

    def __init__(
        self, *args, reconnection_args: Optional[ReconnectionArgs] = None, **kws
    ):
        recon_args = reconnection_args or ReconnectionArgs()
        self.max_reconnect_attempts = recon_args.max_reconnect_attempts
        self.reconnect_backoff_start = recon_args.reconnect_backoff_start
        self.reconnect_backoff_factor = recon_args.reconnect_backoff_factor
        self.recon_failure_cb = recon_args.failure_callback
        self.recon_jitter = recon_args.jitter

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
        self.__model_cache: Optional[DbModel] = None
        self.__classes = {}
        self._transaction = 0
        self._conn()

    def __init_subclass__(cls: "DbBase", **kwargs):
        if cls.uri_name and cls.uri_name not in cls.__known_drivers:
            cls.register_driver(cls, cls.uri_name)

        if cls.__name__ not in cls.__known_drivers:
            cls.register_driver(cls, cls.__name__)

    @classmethod
    def get_driver_by_name(cls, name) -> Type["DbBase"]:
        return cls.__known_drivers.get(name)

    @classmethod
    def register_driver(cls, sub, name):
        cls.__known_drivers[name] = sub

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

    @property
    def uri(self) -> str:
        """Uri that represents a copy of my connection"""
        return (
            self.uri_name
            + ":"
            + ",".join(str(v) for v in self._conn_args)
            + ",".join(k + "=" + str(v) for k, v in self._conn_kws.items())
        )

    def clone(self: T) -> T:
        """Make a copy of my connection"""
        return type(self)(*self._conn_args, **self._conn_kws)

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
    def simplify_model(model: DbModel) -> DbModel:
        """Override if you want to allow model comparisons.

        For example, if you have a model that defines a fixed-width char, but your db ignores
        fixed-with chars, you can remove or normalize the fixed-width flag in the model.
        """
        return model

    @contextlib.contextmanager
    def capture_sql(
        self, execute=False
    ) -> Generator[List[Tuple[str, Tuple[Any, ...]]], None, None]:
        self.__capture = True
        self.__capture_exec = execute
        self.__capture_stmts = []
        try:
            yield self.__capture_stmts
        finally:
            self.__capture = False

    def create_model(self, model: DbModel, ignore_existing=False):
        for name, schema in model.items():
            self.create_table(name, schema, ignore_existing)

    def create_table(self, name, schema: DbTable, ignore_existing=False):
        raise RuntimeError("Generic models not supported")

    def create_indexes(self, name, schema: DbTable, ignore_existing=False):
        raise RuntimeError("Generic models not supported")

    def executescript(self, sql):
        self.execute(sql, _script=True)

    @staticmethod
    def _executemany(cursor, sql: str):
        return cursor.execute(sql)

    @staticmethod
    def _executeone(cursor, sql: str, parameters: Tuple[Any, ...]):
        return cursor.execute(sql, parameters)

    def execute_ddl(self, sql: str, *dialect: str, ignore_existing=True):
        # import here cuz not always avail
        dialect = dialect or ("mysql",)
        from notanorm.ddl_helper import model_from_ddl

        model = model_from_ddl(sql, *dialect)
        self.create_model(model, ignore_existing=ignore_existing)

        return model

    def execute(self, sql: str, parameters=(), _script=False, write=True):
        if "alter " in sql.lower() or "create " in sql.lower():
            self.__model_cache = None

        self.__debug_sql(sql, parameters)

        if self.__capture:
            self.__capture_stmts.append((sql, parameters))
            if not self.__capture_exec:
                return FakeCursor()

        backoff = self.reconnect_backoff_start
        for tries in range(self.max_reconnect_attempts):
            cursor = None

            try:
                with self.r_lock:
                    SubQ.reset_num()
                    cursor = self._cursor(self._conn())
                    if _script:
                        assert not parameters, "Script isn't compatible with parameters"
                        self._executemany(cursor, sql)
                    else:
                        self._executeone(cursor, sql, parameters)
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
                        if self.recon_failure_cb is not None:
                            try:
                                self.recon_failure_cb()
                            except Exception:
                                log.exception("Exception in recon_failure_cb")
                        raise
                    sleep_time = backoff
                    if self.recon_jitter:
                        sleep_time = random.uniform(sleep_time * 0.5, sleep_time * 1.5)
                    time.sleep(sleep_time)
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
        fetch = None

        try:
            fetch = self.execute(sql, tuple(args), write=False)
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
        fetch = None

        ret = self.RetList()

        with self.r_lock:
            try:
                done = False
                fetch = self.execute(sql, tuple(args), write=False)
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
        return self.execute(sql, tuple(vals))

    def auto_quote(self, key: str):
        return (
            self.quote_key(key) if type(key) is AlreadyAliased else self.quote_keys(key)
        )

    @classmethod
    def quote_key(cls, key: str) -> str:
        return '"' + key + '"'

    def quote_keys(self, key):
        return ".".join([self.quote_key(k) for k in key.split(".")])

    @staticmethod
    def _op_from_val(val):
        if isinstance(val, Op):
            return val
        return Op("=", val)

    def _where(self, where, field_map=None):
        if not where:
            return "", ()

        if is_list(where):
            sql = ""
            vals = []
            for ent in where:
                sub_sql, sub_vals = self._where_base(ent, field_map)
                if sql:
                    sql = sql + " or " + sub_sql
                else:
                    sql = sub_sql
                vals = vals + sub_vals
        else:
            sql, vals = self._where_base(where, field_map)

        return " where " + sql, vals

    def _where_base(self, where, field_map):
        none_keys = [key for key, val in where.items() if val is None]
        list_keys = [(key, val) for key, val in where.items() if is_list(val)]
        subq_keys = [(key, val) for key, val in where.items() if type(val) is SubQ]

        del_all(where, none_keys)
        del_all(where, (k[0] for k in list_keys))
        del_all(where, (k[0] for k in subq_keys))

        where = {k.replace("__", "."): v for k, v in where.items()}

        field_map = field_map or {}
        sql = " and ".join(
            [
                (
                    self.quote_key(field_map[key])
                    if key in field_map and type(field_map[key]) is AlreadyAliased
                    else self.quote_keys(field_map[key])
                    if key in field_map
                    else self.quote_keys(key)
                )
                + self._op_from_val(val).op
                + self.placeholder
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
        if subq_keys:
            for key, subq in subq_keys:
                if sql:
                    sql += " and "
                sql += self.quote_keys(key) + " in (" + subq.sql + ")"
                for val in subq.vals:
                    vals.append(val)
        return sql, vals

    def __select_to_query(
        self, table: Union[str, BaseQType], *, fields, dict_where, order_by, **where
    ):
        sql = "select "

        base_table = table.table if type(table) is SubQ else table

        no_from = False
        if (
            type(table) is str
            and table[0 : len(sql)].lower() == sql
            and "from" in table.lower()
        ):
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

        vals = []
        factory = None
        field_map = None
        if not no_from:
            fac = self.__classes.get(base_table)
            factory = where.pop("__class", fac) if not is_list(where) else fac

            if not fields:
                if type(table) in (JoinQ, SubQ) and table.field_map:
                    sql += table.field_sql()
                else:
                    sql += "*"
            else:
                sql += ",".join(fields)

            if type(table) is JoinQ:
                sql += " from " + table.sql
                vals += table.vals
                field_map = table.field_map
            elif type(table) is SubQ:
                sql += " from (" + table.sql + ") as " + table.alias
                vals += table.vals
                field_map = table.field_map
            elif " join " not in table.lower():
                sql += " from " + self.quote_key(table)
            else:
                sql += " from " + table

        where, where_vals = self._where(where, field_map)
        sql += where
        vals += where_vals

        if order_by:
            if isinstance(order_by, str):
                order_by = [order_by]
            order_by_fd = ",".join(order_by)
            # todo: limit order_by more strictly
            assert ";" not in order_by_fd
            sql += " order by " + order_by_fd

        return sql, vals, factory

    def select(
        self,
        table: Union[str, BaseQType],
        fields=None,
        _where=None,
        order_by=None,
        **where,
    ):
        """Select from table (or join) using fields (or *) and where (vals can be list or none).
        __class keyword optionally replaces Row obj.
        """
        sql, vals, factory = self.__select_to_query(
            table, fields=fields, dict_where=_where, order_by=order_by, **where
        )
        return self.query(sql, *vals, factory=factory)

    def subq(
        self,
        table: Union[str, BaseQType],
        fields=None,
        _where=None,
        order_by=None,
        _alias=None,
        **where,
    ):
        """Subquery from table (or join) using fields (or *) and where (vals can be list or none)."""
        sql, vals, factory = self.__select_to_query(
            table, fields=fields, dict_where=_where, order_by=order_by, **where
        )
        return SubQ(self, table, sql, vals, _alias, fields=fields)

    def join(
        self,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        _on=None,
        *,
        field_map=None,
        **on,
    ):
        return self._join("inner", tab1, tab2, _on, field_map=field_map, **on)

    def left_join(
        self,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        _on=None,
        *,
        field_map=None,
        **on,
    ):
        return self._join("left", tab1, tab2, _on, field_map=field_map, **on)

    def right_join(
        self,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        _on=None,
        *,
        field_map=None,
        **on,
    ):
        return self._join("right", tab1, tab2, _on, field_map=field_map, **on)

    def _join_to_sql(self, tab: Union[str, BaseQType], as_subq=False):
        sel = (
            self.quote_key(tab)
            if type(tab) is str
            else "(" + tab.sql + ") as " + tab.alias
            if type(tab) is SubQ
            else f"(select {tab.field_sql()} from " + tab.sql + ") as " + tab.alias
            if type(tab) is JoinQ and as_subq
            else tab.sql
        )
        name = tab.alias if type(tab) in (SubQ, JoinQ) else tab
        vals = tab.vals if type(tab) in (SubQ, JoinQ) else []

        return sel, name, vals

    def _join(
        self,
        join_type: str,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        _on: Optional[Dict[str, str]] = None,
        *,
        field_map=None,
        **on,
    ):
        """Subquery from table (or join) using fields (or *) and where (vals can be list or none)."""
        on.update(_on if _on else {})
        return JoinQ(self, join_type, tab1, tab2, on=on, field_map=field_map)

    def select_gen(
        self,
        table: Union[str, BaseQType],
        fields=None,
        _where=None,
        order_by=None,
        **where,
    ):
        """Same as select, but returns a generator."""
        sql, vals, factory = self.__select_to_query(
            table, fields=fields, dict_where=_where, order_by=order_by, **where
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

    def delete(self, table, where=None, **kws):
        """Delete all rows in a table that match the supplied value(s).


        For example:  db.delete("table_name", column_name="matching value")
        """
        sql = "delete "
        sql += " from " + self.quote_key(table)

        if where and kws:
            raise ValueError("Dict where cannot be mixed with kwargs")

        if not where:
            where = kws

        where, vals = self._where(where)
        if not where:
            raise ValueError("Use delete_all to delete all rows from a table")

        sql += where

        return self.execute(sql, tuple(vals))

    def delete_all(self, table):
        """Delete all rows in a table."""
        sql = "delete "
        sql += " from " + self.quote_key(table)
        return self.execute(sql)

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
                raise UnknownPrimaryError(
                    "Unable to determine update key for table %s" % table
                )

        return where

    def _set_sql(self, where, vals):
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
        set_sql, vals = self._set_sql(where, vals)
        sql = "update " + self.quote_key(table) + " set " + set_sql
        return self.execute(sql, tuple(vals))

    def update_all(self, table, **vals):
        """Update all rows in a table to the same values."""
        sql = "update " + self.quote_key(table) + " set "
        sql += ", ".join(
            [self.quote_keys(key) + "=" + self.placeholder for key in vals]
        )
        return self.execute(sql, tuple(vals.values()))

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

            return self.execute(sql, tuple(vals))

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

    def _get_table_cols(self, tab: str):
        if not self.__model_cache:
            self.__model_cache = self.model()
        tab_mod = self.__model_cache[tab]
        return tab_mod.columns

    def get_subq_col_names(self, tab: Union[str, BaseQType]):
        if getattr(tab, "fields", None):
            if type(tab) is SubQ:
                return [AlreadyAliased(fd) for fd in tab.fields]
            else:
                return tab.fields

        if type(tab) is SubQ:
            tab = tab.table
            return self.get_subq_col_names(tab)

        if type(tab) is str:
            return [col.name for col in self._get_table_cols(tab)]
