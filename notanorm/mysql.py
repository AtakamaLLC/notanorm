from .base import DbBase, parse_bool
from .model import (
    DbType,
    DbModel,
    DbTable,
    DbCol,
    DbIndex,
    DbIndexField,
    DbColCustomInfo,
)
from . import errors as err

import re
from collections import defaultdict
from functools import partial
from typing import Callable, Tuple, List, Dict, Any
import logging as log

MySQLLib = None
try:
    import MySQLdb

    MySQLLib = MySQLdb
    import MySQLdb.cursors

    InterfaceError = type(None)
    pymysql_force_flags = 0
except (ImportError, NameError):
    # known issue with mysql c++ client is a "nameerror" can be thrown if a dll import fails
    import pymysql

    MySQLLib = pymysql

    import pymysql.cursors
    from pymysql.err import InterfaceError
    import pymysql.constants.CLIENT

    pymysql_force_flags = pymysql.constants.CLIENT.MULTI_STATEMENTS


class MySqlDb(DbBase):
    uri_name = "mysql"

    placeholder = "%s"
    default_values = " () values ()"
    max_index_name = 64

    def _begin(self, conn):
        conn.cursor().execute("START TRANSACTION")

    @classmethod
    def uri_adjust(cls, args, kws):
        # adjust to appropriate types
        typ: Callable[[Any], Any]
        for nam, typ in [
            ("port", int),
            ("use_unicode", bool),
            ("autocommit", bool),
            ("buffered", bool),
            ("ssl_verify_cert", bool),
            ("ssl_verify_identity", bool),
            ("compress", bool),
            ("pool_size", int),
            ("client_flag", int),
            ("raise_on_warnings", bool),
        ]:
            if nam in kws:
                if typ is bool:
                    typ = partial(parse_bool, nam)

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

        if isinstance(exp, MySQLLib.OperationalError):
            if err_code in (1054,):
                return err.NoColumnError(msg)
            if err_code in (1050,):
                return err.TableExistsError(msg)
            if err_code in (1075, 1212, 1239, 1293):  # pragma: no cover
                # this error is very hard to support and we should probably drop it
                # it's used as a base class for TableError and other stuff
                # using the base here is odd
                return err.SchemaError(msg)
            if err_code in (1792,):
                return err.DbReadOnlyError(msg)
            if err_code >= 2000:
                # client connection issues
                return err.DbConnectionError(msg)
            if err_code == 1051:
                return err.TableNotFoundError(msg)
            if err_code == 1048:
                return err.IntegrityError(msg)
            return err.OperationalError(msg)
        if isinstance(exp, InterfaceError):
            return err.DbConnectionError(msg)
        if isinstance(exp, MySQLLib.IntegrityError):
            return err.IntegrityError(msg)
        if isinstance(exp, MySQLLib.ProgrammingError):
            if err_code == 1146:
                return err.TableNotFoundError(exp.args[1])
            return err.OperationalError(msg)

        return exp

    def _connect(self, *args, **kws):
        if pymysql_force_flags:
            kws["client_flag"] = kws.get("client_flag", 0) | pymysql_force_flags
        conn = MySQLLib.connect(*args, **kws)
        conn.autocommit(True)
        conn.cursor().execute("SET SESSION sql_mode = 'ANSI';")
        return conn

    def _cursor(self, conn):
        return conn.cursor(MySQLLib.cursors.DictCursor)

    @classmethod
    def quote_key(cls, key):
        return "`" + key + "`"

    def _get_primary(self, table):
        info = self.query(
            "SHOW KEYS FROM " + self.quote_key(table) + " WHERE Key_name = 'PRIMARY'",
        )
        prim = set()
        for x in info:
            prim.add(x.column_name)
        return prim

    _type_map = {
        DbType.TEXT: "longtext",
        DbType.BLOB: "blob",
        DbType.INTEGER: "bigint",
        DbType.BOOLEAN: "boolean",
        DbType.FLOAT: "float",
        DbType.DOUBLE: "double",
        DbType.ANY: "",
    }
    _type_map_inverse = {v: k for k, v in _type_map.items()}
    _type_map_inverse.update(
        {
            "integer": DbType.INTEGER,
            "smallint": DbType.INTEGER,
            "bigint": DbType.INTEGER,
            "tinyblob": DbType.BLOB,
            "mediumblob": DbType.BLOB,
            "longblob": DbType.BLOB,
            "tinytext": DbType.TEXT,
            "mediumtext": DbType.TEXT,
            "text": DbType.TEXT,
        }
    )
    _int_map = {1: "tinyint", 2: "smallint", 4: "integer", 8: "bigint"}
    _type_map_custom = {
        "mediumtext": DbColCustomInfo("mysql", "medium"),
        "mediumblob": DbColCustomInfo("mysql", "medium"),
        "text": DbColCustomInfo("mysql", "small"),
    }

    def create_table(
        self, name, schema: DbTable, ignore_existing=False, create_indexes: bool = True
    ):
        coldefs = []
        primary_fields: Tuple[str, ...] = ()
        for idx in schema.indexes:
            if idx.primary:
                primary_fields = tuple(f.name for f in idx.fields)

        for col in schema.columns:
            coldef = "`" + col.name + "`"
            if col.custom and col.typ == DbType.TEXT and col.custom.dialect == "mysql":
                if col.custom.info == "medium":
                    typ = "mediumtext"
                elif col.custom.info == "small":
                    typ = "text"
                else:  # pragma: no cover
                    assert False, "unknown custom info"
            elif col.size and col.typ == DbType.TEXT:
                if col.fixed:
                    typ = "char"
                else:
                    typ = "varchar"
                typ += "(%s)" % col.size
            elif col.size and col.typ == DbType.BLOB:
                if col.fixed:
                    typ = "binary"
                else:
                    typ = "varbinary"
                typ += "(%s)" % col.size
            elif (
                col.custom and col.typ == DbType.BLOB and col.custom.dialect == "mysql"
            ):
                typ = "mediumblob"
            elif col.size and col.typ == DbType.INTEGER and col.size in self._int_map:
                typ = self._int_map[col.size]
            else:
                typ = self._type_map[col.typ]

            if not typ:
                raise err.SchemaError(f"mysql doesn't support ANY type: {col.name}")
            coldef += " " + typ
            if col.notnull:
                coldef += " not null"
            if col.default:
                coldef += " default " + col.default
            if col.autoinc:
                if (col.name,) != primary_fields:
                    raise err.SchemaError(
                        f"auto increment only works on primary key: {col.name}"
                    )
                coldef += " auto_increment"
            coldefs.append(coldef)
        keys = ["`" + k + "`" for k in primary_fields]
        if keys:
            coldefs.append("primary key(" + ",".join(keys) + ")")

        ignore = "if not exists " if ignore_existing else ""
        create = "create table " + ignore + name + "("
        create += ",".join(coldefs)
        create += ")"
        self.query(create)

        if create_indexes:
            self.create_indexes(name, schema)

    def _create_index(self, table_name, index_name, idx):
        if not idx.primary:
            unique = "unique " if idx.unique else ""
            icreate = (
                "create " + unique + "index " + index_name + " on " + table_name + " ("
            )
            icreate += ",".join(
                f.name if f.prefix_len is None else f"{f.name}({f.prefix_len})"
                for f in idx.fields
            )
            icreate += ")"
            self.execute(icreate)

    def model(self, no_capture=False):
        tabs = self.query("show tables", no_capture=no_capture)
        ret = DbModel()
        for tab in tabs:
            ret[tab[0]] = self.table_model(tab[0], no_capture=no_capture)
        return ret

    def drop_index_by_name(self, table: str, index_name):
        sql = (
            "drop index " + self.quote_key(index_name) + " on " + self.quote_key(table)
        )
        self.execute(sql)

    def table_model(self, tab, no_capture):
        res = self.query("show index from  `" + tab + "`", no_capture=no_capture)

        idxunique = {}
        idxmap: Dict[str, List[Dict[str, Any]]] = defaultdict(lambda: [])
        for idxinfo in res:
            unique = not idxinfo["non_unique"]
            idxunique[idxinfo["key_name"]] = unique
            idxmap[idxinfo["key_name"]].append(
                {"name": idxinfo["column_name"], "prefix_len": idxinfo.get("sub_part")}
            )

        indexes = []
        for name, fds in idxmap.items():
            primary = name == "PRIMARY"
            unique = idxunique[name] and not primary
            indexes.append(
                DbIndex(
                    tuple(DbIndexField(**f) for f in fds),
                    primary=primary,
                    unique=unique,
                    name=name,
                )
            )

        res = self.query("describe `" + tab + "`")
        cols = []
        for col in res:
            primary = [
                tuple(f.name for f in idx.fields) for idx in indexes if idx.primary
            ]
            in_primary = primary and col.field in primary[0]
            dbcol = self.column_model(col, in_primary)
            cols.append(dbcol)

        if len(set(indexes)) != len(indexes):
            log.warning("duplicate indexes in table %s", tab)
        return DbTable(columns=tuple(cols), indexes=set(indexes))

    @staticmethod
    def simplify_model(model: DbModel):
        model2 = DbModel()
        primary_fields: Tuple[str, ...] = ()
        for nam, tab in model.items():
            for index in tab.indexes:
                if index.primary:
                    primary_fields = tuple(f.name for f in index.fields)
            cols = []
            for col in tab.columns:
                d = col._asdict()
                if col.typ == DbType.INTEGER and not col.size:
                    # defaults to 8 if unspecified
                    d["size"] = 8
                if col.name in primary_fields:
                    d["notnull"] = True
                if col.custom and col.custom.dialect == "mysql":
                    d["custom"] = col.custom
                col = DbCol(**d)
                cols.append(col)
            model2[nam] = DbTable(columns=tuple(cols), indexes=tab.indexes)

        return model2

    def column_model(self, info, in_primary):
        # depends on specific mysql version, these are often display width hints
        size = 0
        if info.type in ("int(11)", "integer", "int"):
            # whether you see this depends on the version of mysql
            info.type = "integer"
            size = 4
        elif info.type == "tinyint(1)":
            info.type = "boolean"
        elif info.type.startswith("tinyint"):
            info.type = "integer"
            size = 1
        elif info.type.startswith("bigint"):
            info.type = "bigint"
            size = 8
        elif info.type.startswith("smallint"):
            info.type = "smallint"
            size = 2
        fixed = False
        match_t = re.match(r"(varchar|char|text)\((\d+)\)", info.type)
        match_b = re.match(r"(varbinary|binary|blob)\((\d+)\)", info.type)

        if match_t:
            typ = DbType.TEXT
            fixed = match_t[1] == "char"
            size = int(match_t[2])
        elif match_b:
            typ = DbType.BLOB
            fixed = match_b[1] == "binary"
            size = int(match_b[2])
        else:
            typ = self._type_map_inverse[info.type]

        custom = self._type_map_custom.get(info.type, None)

        autoinc_primary = in_primary and info.extra == "auto_increment"

        ret = DbCol(
            info.field,
            typ,
            fixed=fixed,
            size=size,
            notnull=not autoinc_primary and info.null == "NO",
            default=info.default,
            autoinc=info.extra == "auto_increment",
            custom=custom,
        )

        return ret

    def version(self):
        return self.query("select version() as ver")[0].ver
