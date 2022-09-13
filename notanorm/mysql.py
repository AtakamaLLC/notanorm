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
from .model import DbType, DbModel, DbTable, DbCol, DbIndex
from . import errors as err
import re

import logging as log
# driver for mysql


class MySqlDb(DbBase):
    placeholder = "%s"
    default_values = ' () values ()'

    def _begin(self, conn):
        conn.cursor().execute("START TRANSACTION")

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
            elif col.size and col.typ == DbType.BLOB:
                if col.fixed:
                    typ = "binary"
                else:
                    typ = "varbinary"
                typ += '(%s)' % col.size
            else:
                typ = self._type_map[col.typ]

            if not typ:
                raise err.SchemaError(f"mysql doesn't supprt ANY type: {col.name}")
            coldef += " " + typ
            if col.notnull:
                coldef += " not null"
            if col.default:
                coldef += " default " + col.default
            if col.autoinc:
                if (col.name, ) != primary_fields:
                    raise err.SchemaError(f"auto increment only works on primary key: {col.name}")
                coldef += " auto_increment"
            coldefs.append(coldef)
        keys = ["`" + k + "`" for k in primary_fields]
        if keys:
            coldefs.append("primary key(" + ",".join(keys) + ")")
        create = "create table " + name + "("
        create += ",".join(coldefs)
        create += ")"
        log.error(create)
        self.query(create)

        for idx in schema.indexes:
            if not idx.primary:
                index_name = "ix_" + name + "_" + "_".join(idx.fields)
                unique = "unique " if idx.unique else ""
                icreate = "create " + unique + "index " + index_name + " on " + name + " ("
                icreate += ",".join(idx.fields)
                icreate += ")"
                self.query(icreate)

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

        idxunique = {}
        idxmap = defaultdict(lambda: [])
        for idxinfo in res:
            unique = not idxinfo["non_unique"]
            idxunique[idxinfo["key_name"]] = unique
            idxmap[idxinfo["key_name"]].append(idxinfo["column_name"])

        indexes = []
        for name, fds in idxmap.items():
            primary = (name == "PRIMARY")
            unique = idxunique[name] and not primary
            indexes.append(DbIndex(tuple(fds), primary=primary, unique=unique))

        return DbTable(columns=tuple(cols), indexes=set(indexes))

    def column_model(self, info):
        if info.type == "int(11)" or info.type == "int":
            # depends on specific mysql version
            info.type = "integer"
        fixed = False
        size = 0
        match_t = re.match(r"(varchar|char|text)\((\d+)\)", info.type)
        match_b = re.match(r"(varbinary|binary|blob)\((\d+)\)", info.type)

        if match_t:
            typ = DbType.TEXT
            fixed = match_t[1] == 'char'
            size = int(match_t[2])
        elif match_b:
            typ = DbType.BLOB
            fixed = match_b[1] == 'binary'
            size = int(match_b[2])
        else:
            typ = self._type_map_inverse[info.type]

        ret = DbCol(info.field, typ,
                    fixed=fixed,
                    size=size,
                    notnull=info.null == "NO", default=info.default,
                    autoinc=info.extra == "auto_increment")

        return ret
