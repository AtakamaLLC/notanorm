import sqlite3

from .base import DbBase, DbRow
from .model import DbModel
from . import errors as err

import logging

log = logging.getLogger(__name__)
sqlite_version = tuple(int(v) for v in sqlite3.sqlite_version.split('.'))


class SqliteDb(DbBase):
    uri_name = "sqlite"
    placeholder = "?"
    use_pooled_locks = True

    @classmethod
    def uri_adjust(cls, args, kws):
        for nam, typ in [("timeout", float), ("check_same_thread", bool), ("cached_statements", int), ("detect_types", int)]:
            if nam in kws:
                kws[nam] = typ(kws[nam])

    def _lock_key(self, *args, **kws):
        return args[0]

    def _begin(self, conn):
        conn.execute("BEGIN IMMEDIATE")

    if sqlite_version >= (3, 35, 0):  # pragma: no cover
        # this only works in newer versions, we have no good way of testing different sqlites right now (todo!)
        def _upsert_sql(self, table, inssql, insvals, setsql, setvals):
            if not setvals:
                return inssql + " ON CONFLICT DO NOTHING", insvals
            else:
                return inssql + " ON CONFLICT DO UPDATE SET " + setsql, (*insvals, *setvals)
    elif sqlite_version >= (3, 24, 0):
        def _upsert_sql(self, table, inssql, insvals, setsql, setvals):
            fds = ",".join(self.primary_fields(table))
            if not setvals:
                return inssql + f" ON CONFLICT({fds}) DO NOTHING", insvals
            else:
                return inssql + f" ON CONFLICT({fds}) DO UPDATE SET " + setsql, (*insvals, *setvals)

    @staticmethod
    def translate_error(exp):
        msg = str(exp)
        if isinstance(exp, sqlite3.OperationalError):
            if "no such table" in str(exp):
                return err.TableNotFoundError(msg)
            if "readonly" in str(exp):
                return err.DbReadOnlyError(msg)
            if "no column" in str(exp):
                return err.NoColumnError(msg)
            return err.OperationalError(msg)
        if isinstance(exp, sqlite3.IntegrityError):
            return err.IntegrityError(msg)
        if isinstance(exp, sqlite3.ProgrammingError):
            if "closed database" in str(exp):
                return err.DbConnectionError(msg)
        return exp

    def __init__(self, *args, **kws):
        if "timeout" in kws:
            self.__timeout = kws["timeout"]
        else:
            self.__timeout = super().timeout
        if args[0] == ":memory:":
            # never try to reconnect to memory dbs!
            self.max_reconnect_attempts = 1
        super().__init__(*args, **kws)

    @property
    def timeout(self):
        return self.__timeout

    @timeout.setter
    def timeout(self, val):
        self.__timeout = val

    def model(self):
        """Get sqlite db model: dict of tables, each a dict of rows, each with type, unique, autoinc, primary"""
        res = self.query("SELECT sql from sqlite_master")
        ddl = ""
        for ent in res:
            ddl += ent.sql + ";\n"
        return DbModel(ddl, dialect="sqlite")

    def create_model(self, model: DbModel):
        for ent in model.to_sql("sqlite"):
            self.execute(ent)

    @staticmethod
    def _obj_factory(cursor, row):
        d = DbRow()
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def _connect(self, *args, **kws):
        kws["check_same_thread"] = False
        if "isolation_level" not in kws:
            # enable autocommit mode
            kws["isolation_level"] = None
        conn = sqlite3.connect(*args, **kws)
        conn.row_factory = self._obj_factory
        return conn

    def _get_primary(self, table):
        info = self.query("pragma table_info(\"" + table + "\");")
        prim = set()
        for x in info:
            if x.pk:
                prim.add(x.name)
        return prim
