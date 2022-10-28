from collections import defaultdict

try:
    import MySQLdb
    import MySQLdb.cursors
    InterfaceError = type(None)
except ImportError:
    import pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb
    import MySQLdb.cursors
    from pymysql.err import InterfaceError


from .base import DbBase
from .model import DbModel
from . import errors as err
import re

import logging as log
# driver for mysql


class MySqlDb(DbBase):
    uri_name = "mysql"

    placeholder = "%s"
    default_values = ' () values ()'

    def _begin(self, conn):
        conn.cursor().execute("START TRANSACTION")

    @classmethod
    def uri_adjust(cls, args, kws):
        for nam, typ in [("port", int), ("use_unicode", bool), ("autocommit", bool)]:
            if nam in kws:
                kws[nam] = typ(kws[nam])

        if args:
            kws["host"] = args[0]
            args.clear()

    def _upsert_sql(self, table, inssql, insvals, setsql, setvals):
        if not setvals:
            fields = self.primary_fields(table)
            f0 = next(iter(fields))
            return inssql + f" ON DUPLICATE KEY UPDATE `{f0}`=`{f0}`", insvals
        return inssql + " ON DUPLICATE KEY UPDATE " + setsql, (*insvals, *setvals)

    @staticmethod
    def translate_error(exp):
        try:
            err_code = exp.args[0]
        except (TypeError, AttributeError, IndexError):  # pragma: no cover
            err_code = 0

        msg = str(exp)

        if isinstance(exp, MySQLdb.OperationalError):
            if err_code in (1054, ):
                return err.NoColumnError(msg)
            if err_code in (1075, 1212, 1239, 1293):   # pragma: no cover
                # this error is very hard to support and we should probably drop it
                # it's used as a base class for TableError and other stuff
                # using the base here is odd
                return err.SchemaError(msg)
            if err_code in (1792, ):
                return err.DbReadOnlyError(msg)
            if err_code >= 2000:
                # client connection issues
                return err.DbConnectionError(msg)
            return err.OperationalError(msg)
        if isinstance(exp, InterfaceError):
            return err.DbConnectionError(msg)
        if isinstance(exp, MySQLdb.IntegrityError):
            return err.IntegrityError(msg)
        if isinstance(exp, MySQLdb.ProgrammingError):
            if err_code == 1146:
                return err.TableNotFoundError(exp.args[1])
            return err.OperationalError(msg)

        return exp

    def _connect(self, *args, **kws):
        conn = MySQLdb.connect(*args, **kws)
        conn.autocommit(True)
        conn.cursor().execute("SET SESSION sql_mode = 'ANSI';")
        return conn

    def _cursor(self, conn):
        return conn.cursor(MySQLdb.cursors.DictCursor)

    def quote_key(self, key):
        return '`' + key + '`'

    def _get_primary(self, table):
        info = self.query("SHOW KEYS FROM " + self.quote_key(table) + " WHERE Key_name = 'PRIMARY'")
        prim = set()
        for x in info:
            prim.add(x.column_name)
        return prim

    def model(self):
        self.query("set session sql_mode=default")
        tabs = self.query("show tables")
        ddl = ""
        for tab in tabs:
            ddl += self.table_ddl(tab[0]) + ";\n"
        self.query("set session sql_mode='ansi'")
        ret = DbModel(ddl, dialect="mysql")
        return ret

    def table_ddl(self, tab):
        res = self.query("show create table `" + tab + "`")[0][1]
        # fix bug in sqlglot
        res = re.sub(r" KEY `([^`]+)` \([^)]+\)", "", res)
        res = re.sub(r"\) ENGINE=.*", ");", res)
        return res