import sqlite3
import re

from .base import DbBase, DbRow
from .model import DbType, DbCol, DbTable, DbIndex, DbModel
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

    def __columns(self, table):
        self.query("SELECT name, type from sqlite_master")

        tinfo = self.query("PRAGMA table_info(" + table + ")")
        if len(tinfo) == 0:
            raise KeyError(f"Table {table} not found in db {self}")

        one_pk = True
        for col in tinfo:
            if col.pk > 1:
                one_pk = False

        cols = []
        for col in tinfo:
            col.type = col.type.lower()
            col.autoinc = False
            if col.type == "integer" and col.pk == 1 and one_pk:
                col.autoinc = True
            cols.append(self.__info_to_model(col))
        return tuple(cols)

    def __indexes(self, table):
        tinfo = self.query("PRAGMA table_info(" + table + ")")
        pks = []
        for col in tinfo:
            if col.pk:
                pks.append((col.pk, col.name))
        pks = [p[1] for p in sorted(pks)]

        clist = []
        res = self.query("PRAGMA index_list(" + table + ")")
        for row in res:
            res = self.query("PRAGMA index_info(" + row.name + ")")
            clist.append(self.__info_to_index(row, res))

        if pks:
            if not any(c.primary for c in clist):
                clist.append(DbIndex(fields=tuple(pks), primary=True))
        return set(clist)

    @staticmethod
    def __info_to_index(index, cols):
        primary = index.origin == "pk"
        unique = bool(index.unique) and not primary
        field_names = [ent.name for ent in sorted(cols, key=lambda col: col.seqno)]
        return DbIndex(fields=tuple(field_names), primary=primary, unique=unique)

    @classmethod
    def __info_to_model(cls, info):
        size = 0
        fixed = False
        match_t = re.match(r"(varchar|character)\((\d+)\)", info.type, re.I)
        match_b = re.match(r"(varbinary|binary)\((\d+)\)", info.type, re.I)
        if match_t:
            typ = DbType.TEXT
            fixed = match_t[1] == "character"
            size = int(match_t[2])
        elif match_b:
            typ = DbType.BLOB
            fixed = match_b[1] == "binary"
            size = int(match_b[2])
        else:
            try:
                typ = cls._type_map_inverse[info.type.lower()]
            except KeyError:
                typ = DbType.ANY

        return DbCol(name=info.name, typ=typ, notnull=bool(info.notnull),
                     default=info.dflt_value, autoinc=info.autoinc,
                     size=size, fixed=fixed)

    def model(self):
        """Get sqlite db model: dict of tables, each a dict of rows, each with type, unique, autoinc, primary"""
        res = self.query("SELECT name, type from sqlite_master")
        model = DbModel()
        for tab in res:
            if tab.name == "sqlite_sequence":
                continue
            if tab.type == "table":
                cols = self.__columns(tab.name)
                indxs = self.__indexes(tab.name)
                model[tab.name] = DbTable(cols, indxs)
        return model

    _type_map = {
        DbType.TEXT: "text",
        DbType.BLOB: "blob",
        DbType.INTEGER: "integer",
        DbType.FLOAT: "float",
        DbType.DOUBLE: "double",
        DbType.BOOLEAN: "boolean",
        DbType.ANY: "",
    }
    _type_map_inverse = {v: k for k, v in _type_map.items()}

    # allow "double/float" reverse map
    _type_map_inverse.update({
        "real": DbType.DOUBLE,
        "int": DbType.INTEGER,
        "smallint": DbType.INTEGER,
        "tinyint": DbType.INTEGER,
        "bigint": DbType.INTEGER,
        "clob": DbType.TEXT,
        "bool": DbType.BOOLEAN,
    })

    @classmethod
    def _column_def(cls, col: DbCol, single_primary: str):
        coldef = col.name
        typ = cls._type_map[col.typ]
        if col.size and col.typ == DbType.TEXT:
            if col.fixed:
                typ = "character"
            else:
                typ = "varchar"
            typ += '(%s)' % col.size

        if col.size and col.typ == DbType.BLOB:
            if col.fixed:
                typ = "binary"
            else:
                typ = "varbinary"
            typ += '(%s)' % col.size

        if typ:
            coldef += " " + typ
        if col.notnull:
            coldef += " not null"
        if col.default:
            coldef += " default(" + col.default + ")"
        if single_primary and single_primary.lower() == col.name.lower():
            coldef += " primary key"
        if col.autoinc:
            if single_primary.lower() == col.name.lower():
                coldef += " autoincrement"
            else:
                raise err.SchemaError("sqlite only supports autoincrement on integer primary keys")
        return coldef

    def create_table(self, name, schema):
        coldefs = []
        single_primary = None
        for idx in schema.indexes:
            if idx.primary:
                single_primary = idx.fields[0] if len(idx.fields) == 1 else None

        for col in schema.columns:
            coldefs.append(self._column_def(col, single_primary))
        for idx in schema.indexes:
            if idx.primary and not single_primary:
                coldef = "primary key (" + ",".join(idx.fields) + ")"
                coldefs.append(coldef)
        create = "create table " + name + "("
        create += ",".join(coldefs)
        create += ")"
        log.info(create)
        self.query(create)
        for idx in schema.indexes:
            if not idx.primary:
                index_name = "ix_" + name + "_" + "_".join(idx.fields)
                unique = "unique " if idx.unique else ""
                icreate = "create " + unique + "index " + index_name + " on " + name + " ("
                icreate += ",".join(idx.fields)
                icreate += ")"
                self.query(icreate)

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
        info = self.query("pragma table_info(" + table + ");")
        prim = set()
        for x in info:
            if x.pk:
                prim.add(x.name)
        return prim
