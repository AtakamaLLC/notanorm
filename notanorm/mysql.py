import MySQLdb
import MySQLdb.cursors

from .base import DbBase
from .model import DbType, DbModel, DbTable, DbCol
from . import errors as err

import logging as log
# driver for mysql


class MySqlDb(DbBase):
    placeholder = "%s"

    @staticmethod
    def translate_error(exp):
        try:
            err_code = exp[0]
        except TypeError:
            err_code = 0

        msg = str(exp)

        if isinstance(exp, MySQLdb.OperationalError):
            if err_code in (1075, 1212, 1239, 1293):
                return exp.SchemaError(msg)
            return err.DbConnectionError(msg)
        if isinstance(exp, MySQLdb.IntegrityError):
            return err.IntegrityError(msg)

        return err

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
        info = self.query("SHOW KEYS FROM " + table + " WHERE Key_name = 'PRIMARY'")
        prim = set()
        for x in info:
            prim.add(x.Column_name)
        return prim

    _type_map = {
        DbType.TEXT: "text",
        DbType.BLOB: "blob",
        DbType.INTEGER: "integer",
        DbType.FLOAT: "float",
        DbType.DOUBLE: "double",
        DbType.ANY: "",
    }
    _type_map_inverse = {v: k for k, v in _type_map.items()}

    def create_table(self, name, schema):
        coldefs = []
        for col in schema.columns:
            coldef = "`" + col.name + "`"
            if col.size and col.typ == DbType.TEXT:
                if col.fixed:
                    typ = "char"
                else:
                    typ = "varchar"
                typ += '(%s)' % col.size
            else:
                typ = self._type_map[col.typ]

            if not typ:
                raise err.DbSchemaError("mysql doesn't supprt ANY type")
            coldef += " " + typ
            if col.notnull:
                coldef += " not null"
            if col.default:
                coldef += " default(" + col.default + ")"
            if col.autoinc:
                coldef += " auto_increment"
            coldefs.append(coldef)
        create = "create table " + name + "("
        create += ",".join(coldefs)
        create += ")"
        log.error(create)
        self.query(create)

    def model(self):
        tabs = self.query("show tables")
        ret = DbModel()
        for tab in tabs:
            ret[tab] = self.table_model(tab)

    def table_model(self, tab):
        res = self.query("describe table " + tab)
        cols = []
        for col in res:
            cols.append(self.column_model(col))
        return DbTable(columns=tuple(cols))

    def column_model(self, info):
        typ = self._type_map_inverse[info.type]
        return DbCol(info.name, typ)
