"""Handle common database functionality.

NOTE: Make sure to close the db handle when you are done.
"""
import contextlib
import os
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
    Iterable,
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


def is_dict(obj):
    """Determine if object is dict-like."""
    return isinstance(obj, (dict,))


def del_all(mapping, to_remove):
    """Remove list of elements from mapping."""
    for key in to_remove:
        del mapping[key]


def prune_keys(tuple_list: List[Tuple[str, Any]], to_remove: set):
    """Return list without elements"""
    return [(key, val) for key, val in tuple_list if key not in to_remove]


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
    fields: List[str]
    field_map: Dict[str, str]

    def __init__(self, db: "DbBase"):
        self.db = db

    @classmethod
    def unique_name(cls):
        return os.urandom(16).hex()

    @abstractmethod
    def resolve_field(self, field: str):
        ...

    def field_sql(self):
        if not self.fields:
            return "*"

        if not self.field_map:
            return ",".join(self.db.quote_key(fd) for fd in self.fields)

        return self.db.field_sql_from_map(self.field_map)


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
        alias = alias or (table if table is str else "subq") + "_" + self.unique_name()

        super().__init__(db)

        self.sql = sql
        self.vals = vals
        self.alias = alias
        self.fields = fields or []
        self.field_map = {}

        if type(table) is SubQ:
            self.table = table.table
        else:
            self.table = table

    def resolve_field(self, field: str):
        if self.fields and field not in self.fields:
            raise UnknownColumnError(f"{field} not found in {self.fields}")
        return self.alias + "." + field


class JoinQ(BaseQ):
    def __init__(
        self,
        db: "T",
        join_type: str,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        on: dict,
        fields: Union[dict, list],
    ):
        super().__init__(db)

        self.join_type = join_type
        self.tab1 = tab1
        self.tab2 = tab2
        self.on = on
        self.vals = []
        self.__sql = ""
        self.__field_map = fields if type(fields) is dict else {}
        self.__fields = fields if type(fields) is list else []

    def __resolve_if_needed(self):
        if not self.__sql:
            self.resolve()

    @property
    def sql(self) -> str:
        self.__resolve_if_needed()
        return self.__sql

    @property
    def fields(self) -> list:
        self.__resolve_if_needed()
        return self.__fields

    @property
    def field_map(self) -> dict:
        self.__resolve_if_needed()
        return self.__field_map

    def flat_tabs(self) -> Generator[Tuple[Union[str, SubQ], "JoinQ"], None, None]:
        """Returns an inordered generator of all tables in all joins.

        Second part of the tuple is the join that referenced them."""
        for tab in [self.tab1, self.tab2]:
            if tab is self.tab1:
                join = None
            else:
                join = self

            if type(tab) is JoinQ:
                for sub_tab, subj in tab.flat_tabs():
                    yield sub_tab, subj or join
            else:
                yield tab, join

    def resolve_field(self, field: str):
        ret = None
        for tab in (self.tab1, self.tab2):
            if field in self.db.get_subq_col_names(tab):
                if ret is not None:
                    raise UnknownColumnError(
                        f"{field} found both in {self.tab1} and {self.tab2}, ambiguous join!"
                    )
                ret = tab + "." + field
        if ret is not None:
            return ret
        raise UnknownColumnError(f"{field} not found in {self.tab1} or {self.tab2}")

    @staticmethod
    def tab_to_sql(tab) -> [str, list]:
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
        on_sql = ""
        for k, v in self.on.items():
            on_sql += " and " if on_sql else ""
            k = self.__resolve_on_field(k, self.tab1)
            v = self.__resolve_on_field(v, self.tab2)
            on_sql += k + "=" + v
        return on_sql

    def __resolve_on_field(self, k, tab):
        k = k.replace("__", ".")
        if "." not in k:
            if type(tab) is str:
                k = tab + "." + k
            else:
                k = tab.resolve_field(k)
        k = self.db.auto_quote(k)
        return k


QueryValueType = Union[Op, SubQ, List["QueryValueType"], str, int, float, bytes, None]
QueryDictType = Dict[str, QueryValueType]
QueryListType = List[QueryDictType]
WhereClauseType = Union[QueryDictType, QueryListType]
WhereKwargsType = Union[QueryValueType, QueryListType]
LimitArgType = Union[int, Tuple[int, int]]
GroupByArgType = Union[str, Iterable[str]]
OrderByArgType = Union[str, Iterable[str]]


class And(QueryListType):
    """This list is "AND"'d in the resulting query:

    Dict is field, op
    """


class Or(QueryListType):
    """This list is "OR"'d in the resulting query:

    Dict is field, op
    """


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

    def clear_model_cache(self):
        self.__model_cache = None

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
                            except Exception:  # noqa
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

    def query_gen(self, sql: str, *args, factory=None) -> Generator[DbRow, None, None]:
        """Same as query, but returns a generator."""
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

    def query(self, sql: str, *args, factory=None) -> List[DbRow]:
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
            self.quote_field_or_func(key)
            if type(key) is AlreadyAliased
            else self.quote_keys(key)
        )

    @classmethod
    def quote_field_or_func(cls, key: str) -> str:
        if "(" in key or "*" in key:
            return key
        return cls.quote_key(key)

    @classmethod
    def quote_key(cls, key: str) -> str:
        return '"' + key + '"'

    @classmethod
    def quote_keys(cls, key):
        return ".".join([cls.quote_field_or_func(k) for k in key.split(".")])

    @staticmethod
    def _op_from_val(val):
        if isinstance(val, Op):
            return val
        return Op("=", val)

    def _where(self, where, field_map=None):
        sql, vals = self._where_base(where, field_map)
        if not sql:
            return "", ()
        return " where " + sql, vals

    def _where_base(self, where, field_map=None, is_and=False):
        if not where:
            return "", []

        if type(where) is And:
            sql, vals = self._where_base(Or(where), field_map, is_and=True)
        elif is_list(where):
            sql = ""
            vals = []
            for ent in where:
                sub_sql, sub_vals = self._where_base(ent, field_map)
                op = "and" if is_and else "or"
                if sql:
                    sql = sql + " " + op + " (" + sub_sql + ")"
                else:
                    sql = sub_sql
                vals = vals + sub_vals
        else:
            sql, vals = self._where_items(list(where.items()), field_map)

        return sql, vals

    def _where_items(self, where_items: List[Tuple[str, QueryValueType]], field_map):
        none_keys = [key for key, val in where_items if val is None]
        list_keys = [(key, val) for key, val in where_items if is_list(val)]
        subq_keys = [(key, val) for key, val in where_items if type(val) is SubQ]

        where_items = prune_keys(
            where_items,
            set(none_keys)
            | set(k[0] for k in list_keys)
            | set(k[0] for k in subq_keys),
        )

        where = [(k.replace("__", "."), v) for k, v in where_items]

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
                for key, val in where
            ]
        )

        if none_keys:
            if sql:
                sql += " and "
            sql += " and ".join(
                [self.quote_keys(key) + " is NULL" for key in none_keys]
            )

        vals = [self._op_from_val(val).val for _, val in where]
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

    def select_to_query(
        self,
        table: Union[str, BaseQType],
        *,
        fields: Union[Dict[str, str], List[str]],
        dict_where: WhereClauseType,
        _order_by,
        _limit: Optional[LimitArgType],
        _group_by,
        **where: WhereKwargsType,
    ):
        sql = "select "

        base_table = table.table if type(table) is SubQ else table

        if isinstance(fields, dict) and not where and dict_where is None:
            dict_where = fields
            fields = None

        if dict_where is not None and where:
            raise ValueError("Dict where cannot be mixed with kwargs")

        if dict_where is not None:
            where = dict_where

        vals = []

        fac = self.__classes.get(base_table)
        factory = where.pop("__class", fac) if is_dict(where) else fac

        field_map = None
        if not fields:
            if type(table) in (JoinQ, SubQ):
                sql += table.field_sql()
            else:
                sql += "*"
        else:
            if isinstance(fields, dict):
                field_map = fields
                sql += self.field_sql_from_map(fields)
            else:
                sql += ",".join(self.auto_quote(key) for key in fields)

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

        if _group_by is not None:
            sql += " " + self.group_by_query(_group_by)

        if _order_by:
            sql += " " + self.order_by_query(_order_by)

        if _limit is not None:
            sql += " " + self.limit_query(_limit)

        return sql, vals, factory

    def order_by_query(self, _order_by: OrderByArgType):
        if isinstance(_order_by, str):
            _order_by = [_order_by]
        order_by_fd = ",".join(_order_by)
        # todo: limit order_by more strictly
        assert ";" not in order_by_fd
        return "order by " + order_by_fd

    def group_by_query(self, group_by: GroupByArgType):
        gb = ",".join([group_by] if type(group_by) is str else group_by)
        return f"group by {gb}"

    def limit_query(self, limit: LimitArgType):
        try:
            offset, rows = limit
            return f"limit {offset}, {rows}"
        except TypeError:
            return f"limit {limit}"

    def select(
        self,
        table: Union[str, BaseQType],
        fields=None,
        _where=None,
        *,
        order_by=None,
        _order_by: OrderByArgType = None,
        _limit: Optional[LimitArgType] = None,
        _group_by: Optional[GroupByArgType] = None,
        **where: WhereKwargsType,
    ) -> List[DbRow]:
        """Select from table (or join) using fields (or *) and where (vals can be list or none).
        __class keyword optionally replaces Row obj.

        Special params:

        fields: list of fields or dict of field-mappings*
        _where: dict of condition clauses, can be used instead of keyword args
        order_by: string, "colname asc" or "colname desc"
        _limit: number limiting rows or tuple of (offset, limit)
        _group_by: group_by clause


        If a dict is used as a positional in the 2nd arg, and there are no other where clauses,
        this is a where clause, not a fields arg.
        """
        sql, vals, factory = self.select_to_query(
            table,
            fields=fields,
            dict_where=_where,
            _order_by=order_by or _order_by,
            _limit=_limit,
            _group_by=_group_by,
            **where,
        )
        return self.query(sql, *vals, factory=factory)

    def subq(
        self,
        table: Union[str, BaseQType],
        fields=None,
        _where=None,
        *,
        order_by=None,
        _order_by: OrderByArgType = None,
        _limit: Optional[LimitArgType] = None,
        _group_by: Optional[GroupByArgType] = None,
        _alias=None,
        **where: WhereKwargsType,
    ) -> SubQ:
        """Subquery from table (or join) using fields (or *) and where (vals can be list or none).
        Same params as select.
        """
        sql, vals, factory = self.select_to_query(
            table,
            fields=fields,
            dict_where=_where,
            _order_by=order_by or _order_by,
            _limit=_limit,
            _group_by=_group_by,
            **where,
        )
        return SubQ(
            self,
            table,
            sql,
            vals,
            _alias,
            fields=fields or getattr(table, "fields", None),
        )

    def join(
        self,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        _on=None,
        *,
        fields=None,
        **on,
    ) -> JoinQ:
        return self._join("inner", tab1, tab2, _on, fields=fields, **on)

    def left_join(
        self,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        _on=None,
        *,
        fields=None,
        **on,
    ):
        return self._join("left", tab1, tab2, _on, fields=fields, **on)

    def right_join(
        self,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        _on=None,
        *,
        fields=None,
        **on,
    ):
        return self._join("right", tab1, tab2, _on, fields=fields, **on)

    def _join(
        self,
        join_type: str,
        tab1: Union[str, BaseQType],
        tab2: Union[str, BaseQType],
        _on: Optional[Dict[str, str]] = None,
        *,
        fields=None,
        **on,
    ):
        """Subquery from table (or join) using fields (or *) and where (vals can be list or none)."""
        on.update(_on if _on else {})
        return JoinQ(self, join_type, tab1, tab2, on=on, fields=fields)

    def select_gen(
        self,
        table: Union[str, BaseQType],
        fields=None,
        _where=None,
        *,
        order_by=None,
        _order_by: OrderByArgType = None,
        _limit: Optional[LimitArgType] = None,
        _group_by: Optional[GroupByArgType] = None,
        **where: WhereKwargsType,
    ) -> Generator[DbRow, None, None]:
        """Same as select, but returns a generator."""
        sql, vals, factory = self.select_to_query(
            table,
            fields=fields,
            dict_where=_where,
            _order_by=order_by or _order_by,
            _limit=_limit,
            _group_by=_group_by,
            **where,
        )
        return self.query_gen(sql, *vals, factory=factory)

    @abstractmethod
    def version(self):
        ...

    def aggregate(
        self,
        table,
        agg_map_or_str,
        where=None,
        _group_by: Optional[GroupByArgType] = None,
        _order_by: OrderByArgType = None,
        _order: Optional[str] = None,  # used only for "simplified" aggregates
        _limit: Optional[LimitArgType] = None,
        **kws,
    ):
        if where and kws:
            raise ValueError("Dict where cannot be mixed with kwargs")

        if not where:
            where = kws

        simple_result = type(agg_map_or_str) is str

        # when using simple results, the caller doesn't have access to result field names
        # instead they specify "_order"
        if _order:
            assert simple_result, "_order kw is only valid when doing simple aggregates"
            _order_by = "k " + _order

        if simple_result:
            agg_map = {"k": agg_map_or_str}
        else:
            agg_map = agg_map_or_str

        aggs = ",".join(
            aggval + " as " + self.quote_key(alias) for alias, aggval in agg_map.items()
        )

        sql = "select " + aggs
        if _group_by:
            sql += "," + ",".join(_group_by)
        sql += " from " + self.quote_key(table)
        where, vals = self._where(where)
        sql += where

        if _group_by:
            sql += " " + self.group_by_query(_group_by)

        if _order_by:
            sql += " " + self.order_by_query(_order_by)

        if _limit:
            sql += " " + self.limit_query(_limit)

        if _group_by:
            ret = {}
            for row in self.query(sql, *vals):
                index = tuple(row[field] for field in _group_by)
                if len(index) == 1:
                    index = index[0]
                ret[index] = {}
                for alias in agg_map:
                    ret[index][alias] = row[alias]
            if simple_result:
                ret = {k: v["k"] for k, v in ret.items()}
        else:
            ret = self.query(sql, *vals)[0]
            if simple_result:
                ret = ret["k"]

        return ret

    def count(self, table, where=None, *, _group_by=None, **kws):
        return self.aggregate(
            table, "count(*)", where=where, _group_by=_group_by, **kws
        )

    def sum(self, table, field, where=None, _group_by=None, **kws):
        return self.aggregate(
            table,
            "sum(" + self.quote_key(field) + ")",
            where=where,
            _group_by=_group_by,
            **kws,
        )

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

    def select_one(
        self, table, fields=None, **where: WhereKwargsType
    ) -> Optional[DbRow]:
        """Select one row.

        Returns None if not found.

        Raises MoreThanOneError if there is more than one result.
        """
        ret = self.select(table, fields, _limit=2, **where)
        if len(ret) > 1:
            raise MoreThanOneError
        if ret:
            return ret[0]
        return None

    def select_any_one(
        self, table, fields=None, **where: WhereKwargsType
    ) -> Optional[DbRow]:
        """Select one row.

        Returns None if not found.

        Returns the first one found if there is more than one result.
        """
        ret = self.select_gen(table, fields, _limit=1, **where)
        try:
            return next(ret)
        except StopIteration:
            return None

    def _get_table_cols(self, tab: str):
        if self.__model_cache is None:
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

    def field_sql_from_map(self, field_map: Dict[str, str]):
        sql = ""
        for name, qual_name in field_map.items():
            if self.auto_quote(qual_name) != self.quote_key(name):
                sql += ", " + self.auto_quote(qual_name) + " as " + self.quote_key(name)
            else:
                sql += ", " + self.auto_quote(qual_name)
        return sql.lstrip(",")
