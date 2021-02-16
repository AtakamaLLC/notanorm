from collections import defaultdict

import MySQLdb
import MySQLdb.cursors

from .base import DbBase
from .model import DbType, DbModel, DbTable, DbCol, DbIndex
from . import errors as err
import re

import logging as log
# driver for mysql


class MySqlDb(DbBase):
    placeholder = "%s"

    def _begin(self, conn):
        conn.cursor().execute("START TRANSACTION")

    @staticmethod
    def translate_error(exp):
        try:
            err_code = exp.args[0]
        except (TypeError, AttributeError):  # pragma: no cover
            err_code = 0

        msg = str(exp)

        if isinstance(exp, MySQLdb.OperationalError):
            if err_code in (1075, 1212, 1239, 1293):
                return err.SchemaError(msg)
            return err.DbConnectionError(msg)
        if isinstance(exp, MySQLdb.IntegrityError):
            return err.IntegrityError(msg)

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
        info = self.query("SHOW KEYS FROM " + table + " WHERE Key_name = 'PRIMARY'")
        prim = set()
        for x in info:
            prim.add(x.column_name)
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
        primary_fields = []
        for idx in schema.indexes:
            if idx.primary:
                primary_fields = idx.fields

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
                raise err.SchemaError("mysql doesn't supprt ANY type")
            coldef += " " + typ
            if col.notnull:
                coldef += " not null"
            if (col.name, ) == primary_fields:
                coldef += " primary key"
            if col.default:
                coldef += " default(" + col.default + ")"
            if col.autoinc:
                if (col.name, ) != primary_fields:
                    raise err.SchemaError("auto increment only works on primary key")
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
            ret[tab[0]] = self.table_model(tab[0])
        return ret

    def table_model(self, tab):
        res = self.query("describe `" + tab + "`")
        cols = []
        for col in res:
            cols.append(self.column_model(col))
        res = self.query("show index from  `" + tab + "`")

        idxmap = defaultdict(lambda: [])
        for idxinfo in res:
            idxmap[idxinfo["key_name"]].append(idxinfo["column_name"])

        indexes = []
        for name, fds in idxmap.items():
            indexes.append(DbIndex(tuple(fds), primary=(name == "PRIMARY")))

        return DbTable(columns=tuple(cols), indexes=tuple(indexes))

    def column_model(self, info):
        if info.type == "int(11)":
            info.type = "integer"
        fixed = False
        size = 0
        match = re.match(r"(varchar|char)\((\d+)\)", info.type)

        if match:
            typ = DbType.TEXT
            fixed = match[1] == 'char'
            size = int(match[2])
        else:
            typ = self._type_map_inverse[info.type]

        ret = DbCol(info.field, typ,
                    fixed=fixed,
                    size=size,
                    notnull=info.null == "NO", default=info.default,
                    autoinc=info.extra == "auto_increment")

        return ret
